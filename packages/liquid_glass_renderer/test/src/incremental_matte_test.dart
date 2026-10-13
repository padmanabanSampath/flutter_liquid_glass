import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/internal/flutter_gpu_geometry_renderer.dart';

const _expectFallback = bool.fromEnvironment('EXPECT_FLUTTER_GPU_FALLBACK');
const _bundle = 'build/shaderbundles/liquid_glass_renderer.shaderbundle';

/// An axis-aligned primitive in matte pixels, scaled by [scaleX] and
/// [scaleY] as a stretched shape is.
class _Shape {
  const _Shape(
    this.type,
    this.x,
    this.y,
    this.width,
    this.height, {
    this.radius = 0,
    this.scaleX = 1,
    this.scaleY = 1,
    this.rse = const [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  });

  /// A living blob at [time].
  _Shape.alive(double x, double y, double time)
    : this(
        4,
        x,
        y,
        120,
        110,
        rse: LiquidBlob.alive(time: time).shaderParameters,
      );

  final double type;
  final double x;
  final double y;
  final double width;
  final double height;
  final double radius;
  final double scaleX;
  final double scaleY;

  /// The superellipse slots, which carry a blob's waves.
  final List<double> rse;

  _Shape moved(double dx, double dy) => _Shape(
    type,
    x + dx,
    y + dy,
    width,
    height,
    radius: radius,
    scaleX: scaleX,
    scaleY: scaleY,
    rse: rse,
  );

  _Shape stretched(double sx, double sy) => _Shape(
    type,
    x,
    y,
    width,
    height,
    radius: radius,
    scaleX: sx,
    scaleY: sy,
    rse: rse,
  );
}

/// Shapes split into groups: a negative blend starts a standalone shape or a
/// new group.
typedef _Scene = List<(_Shape, double)>;

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('incremental matte updates', skip: _expectFallback, () {
    tearDown(() => FlutterGpuGeometryRenderer.incrementalMatteUpdates = true);

    ({int width, int height}) render(
      FlutterGpuGeometryRenderer renderer,
      _Scene scene, {
      double offsetX = 0,
      double offsetY = 0,
      int width = 640,
      int height = 480,
      double groupBlend = 30,
    }) {
      final result = renderer.render(
        width: width,
        height: height,
        shapeData: [
          for (final (shape, blend) in scene) ...[
            shape.type, shape.width, shape.height, shape.radius, //
            1 / shape.scaleX, 0, 0, 1 / shape.scaleY, //
            shape.x,
            shape.y,
            math.min(shape.scaleX, shape.scaleY),
            if (blend < 0) -(groupBlend + 1) else groupBlend,
          ],
        ],
        numShapes: scene.length,
        rseData: [for (final (shape, _) in scene) ...shape.rse],
        boundsData: [
          for (final (shape, _) in scene) ...[
            shape.x - shape.width / 2 * shape.scaleX,
            shape.y - shape.height / 2 * shape.scaleY,
            shape.x + shape.width / 2 * shape.scaleX,
            shape.y + shape.height / 2 * shape.scaleY,
          ],
        ],
        refractionHeight: 12,
        refractionAmount: 20,
        offsetX: offsetX,
        offsetY: offsetY,
      );
      return (width: result.width, height: result.height);
    }

    // A group of nine tiles in a grid, gaps below the blend radius, as in a
    // quick settings panel, and two standalone pebbles.
    final base = <(_Shape, double)>[
      for (var row = 0; row < 3; row++)
        for (var column = 0; column < 3; column++)
          (
            _Shape(
              3,
              120.0 + column * 150,
              90.0 + row * 110,
              138,
              98,
              radius: 30,
            ),
            row == 0 && column == 0 ? -1.0 : 1.0,
          ),
      (const _Shape(2, 560, 420, 60, 52), -1.0),
      (const _Shape(2, 60, 420, 44, 44), -1.0),
    ];

    Future<void> expectSameAsFullRender(
      FlutterGpuGeometryRenderer incremental,
      _Scene scene, {
      double offsetX = 0,
      double offsetY = 0,
      int width = 640,
      int height = 480,
      double groupBlend = 30,
      // Whether the render must be incremental; null checks pixels only.
      bool? updates = true,
    }) async {
      final count = incremental.debugIncrementalRenderCount;
      final updated = render(
        incremental,
        scene,
        offsetX: offsetX,
        offsetY: offsetY,
        width: width,
        height: height,
        groupBlend: groupBlend,
      );
      final full = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(full.dispose);
      final reference = render(
        full,
        scene,
        offsetX: offsetX,
        offsetY: offsetY,
        width: width,
        height: height,
        groupBlend: groupBlend,
      );
      expect(
        (updated.width, updated.height),
        (reference.width, reference.height),
      );
      final a = await _bytes(incremental);
      final b = await _bytes(full);
      var differing = 0;
      for (var y = 0; y < reference.height; y++) {
        for (var x = 0; x < reference.width; x++) {
          final index = (y * a.stride + x) * 4;
          final referenceIndex = (y * b.stride + x) * 4;
          for (var c = 0; c < 4; c++) {
            if (a.data.getUint8(index + c) !=
                b.data.getUint8(referenceIndex + c)) {
              differing++;
              break;
            }
          }
        }
      }
      expect(differing, 0, reason: 'pixels differing from a full render');
      if (updates == null) return;
      if (updates) {
        expect(
          incremental.debugIncrementalRenderCount,
          count + 1,
          reason: 'drawn incrementally',
        );
      } else {
        expect(incremental.debugIncrementalRenderCount, count);
      }
    }

    test('moving or stretching one shape redraws only around it', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      render(renderer, base);

      var scene = [...base];
      // A tile in the middle of the group, moved toward its neighbor.
      scene[1] = (scene[1].$1.moved(7, 3), scene[1].$2);
      await expectSameAsFullRender(renderer, scene);
      expect(
        renderer.debugLastRedrawnPixels,
        lessThan(640 * 480 * 0.35),
        reason: 'only the region the tile reaches is drawn again',
      );

      // Stretched like LiquidStretch does.
      scene = [...scene];
      scene[4] = (scene[4].$1.stretched(1.2, 0.85).moved(-5, 2), scene[4].$2);
      await expectSameAsFullRender(renderer, scene);

      // A standalone pebble.
      scene = [...scene];
      scene[9] = (scene[9].$1.moved(-12, -9), scene[9].$2);
      await expectSameAsFullRender(renderer, scene);
      expect(renderer.debugLastRedrawnPixels, lessThan(640 * 480 * 0.1));

      // Two shapes at once, one of them merging into the group.
      scene = [...scene];
      scene[0] = (scene[0].$1.moved(4, 9), scene[0].$2);
      scene[10] = (scene[10].$1.moved(20, -40), scene[10].$2);
      await expectSameAsFullRender(renderer, scene);
    });

    test('a living blob redraws only around itself', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      // Close enough to the group that its reach overlaps a tile.
      _Scene at(double time) => [
        ...base.take(9),
        (_Shape.alive(560, 300, time), -1.0),
      ];
      render(renderer, at(0));
      for (final time in [0.25, 0.5, 1.5]) {
        await expectSameAsFullRender(renderer, at(time));
        expect(renderer.debugLastRedrawnPixels, lessThan(640 * 480 * 0.2));
      }
    });

    test('many shapes springing out at once match full renders', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      // A floating action button and its menu items, which grow out of it
      // and move up together, as an expanding menu does.
      _Scene at(double t) => [
        (const _Shape(3, 560, 420, 64, 64, radius: 20), -1.0),
        for (var i = 0; i < 5; i++)
          (
            _Shape(
              3,
              560 - 40 * t,
              420 - (i + 1) * 62 * t,
              24 + 96 * t,
              20 + 26 * t,
              radius: 14,
            ),
            -1.0,
          ),
      ];
      // The layer's matte covers the shapes' union, in whole pixels, so it
      // moves and resizes with them.
      ({double x, double y, int width, int height}) matteOf(_Scene scene) {
        var left = double.infinity;
        var top = double.infinity;
        var right = -double.infinity;
        var bottom = -double.infinity;
        for (final (shape, _) in scene) {
          left = math.min(left, shape.x - shape.width / 2 - 8);
          top = math.min(top, shape.y - shape.height / 2 - 8);
          right = math.max(right, shape.x + shape.width / 2 + 8);
          bottom = math.max(bottom, shape.y + shape.height / 2 + 8);
        }
        final x = left.floorToDouble();
        final y = top.floorToDouble();
        return (
          x: x,
          y: y,
          width: (right - x).ceil(),
          height: (bottom - y).ceil(),
        );
      }

      for (final scene in [
        for (final t in [0.0, 0.1, 0.25, 0.45, 0.7, 0.9, 1.0, 0.8, 0.4, 0.05])
          at(t),
      ]) {
        // Shapes stay in layer pixels; the matte starts at its offset.
        final matte = matteOf(scene);
        await expectSameAsFullRender(
          renderer,
          scene,
          offsetX: matte.x,
          offsetY: matte.y,
          width: matte.width,
          height: matte.height,
          updates: null,
        );
      }
    });

    test('a sequence of drags matches full renders frame by frame', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      render(renderer, base);
      final random = math.Random(7);
      var scene = [...base];
      for (var frame = 0; frame < 12; frame++) {
        final index = random.nextInt(scene.length);
        scene = [...scene];
        scene[index] = (
          scene[index].$1.moved(
            random.nextDouble() * 16 - 8,
            random.nextDouble() * 16 - 8,
          ),
          scene[index].$2,
        );
        await expectSameAsFullRender(renderer, scene);
      }
    });

    test('a whole-pixel matte move copies the matte along', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      render(renderer, base, offsetX: 10, offsetY: 6);
      final scene = [...base];
      scene[2] = (scene[2].$1.moved(3, 0), scene[2].$2);
      await expectSameAsFullRender(
        renderer,
        scene,
        offsetX: 14,
        offsetY: 2,
      );
      // A matte that grows to the next size bucket.
      await expectSameAsFullRender(
        renderer,
        scene,
        offsetX: 14,
        offsetY: 2,
        width: 720,
        height: 520,
      );
    });

    test('an unchanged matte is copied without drawing', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      render(renderer, base);
      await expectSameAsFullRender(renderer, base);
      expect(renderer.debugLastRedrawnPixels, 0);
    });

    test('changed optics or blend draw the whole matte', () async {
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      render(renderer, base);
      await expectSameAsFullRender(
        renderer,
        base,
        groupBlend: 18,
        updates: false,
      );
      // A fractional matte move cannot be copied exactly.
      await expectSameAsFullRender(
        renderer,
        base,
        groupBlend: 18,
        offsetX: 0.5,
        updates: false,
      );
    });

    testWidgets('uniforms outlive frames the GPU has yet to run', (
      tester,
    ) async {
      // The shared host buffer cycles through four slots, one per frame. A
      // slot the GPU has not read yet must not be rewound into: the passes
      // would draw a later frame's shapes, and the next update would keep
      // them. In widget tests completions arrive only when the test yields
      // to the event loop, so these frames run ahead of the GPU.
      final renderer = (await tester.runAsync(
        () => FlutterGpuGeometryRenderer.fromAsset(_bundle),
      ))!;
      addTearDown(renderer.dispose);
      final full = (await tester.runAsync(
        () => FlutterGpuGeometryRenderer.fromAsset(_bundle),
      ))!;
      addTearDown(full.dispose);

      final deferred =
          FlutterGpuGeometryRenderer.debugDeferredHostBufferResetCount;
      var scene = base;
      Future<void> nextFrame() {
        tester.binding.scheduleFrame();
        return tester.pump(const Duration(milliseconds: 16));
      }

      for (var frame = 0; frame < 8; frame++) {
        await nextFrame();
        scene = [...scene];
        scene[frame % 4 + 1] = (
          scene[frame % 4 + 1].$1.moved(3, frame.isEven ? 2 : -2),
          scene[frame % 4 + 1].$2,
        );
        render(renderer, scene);
      }
      expect(
        FlutterGpuGeometryRenderer.debugDeferredHostBufferResetCount,
        greaterThan(deferred),
        reason: 'frames ahead of the GPU keep their slot',
      );
      render(full, scene);
      final a = (await tester.runAsync(() => _bytes(renderer)))!;
      final b = (await tester.runAsync(() => _bytes(full)))!;
      expect(a.stride, b.stride);
      var differing = 0;
      for (var i = 0; i < a.data.lengthInBytes; i += 4) {
        if (a.data.getUint32(i) != b.data.getUint32(i)) differing++;
      }
      expect(differing, 0, reason: 'pixels differing from a full render');

      // Once the GPU has caught up, the next frame rewinds again.
      for (var i = 0; i < 200; i++) {
        if (FlutterGpuGeometryRenderer.debugPendingHostBufferReaders == 0) {
          break;
        }
        await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 5)),
        );
      }
      expect(FlutterGpuGeometryRenderer.debugPendingHostBufferReaders, 0);
      final stalled =
          FlutterGpuGeometryRenderer.debugDeferredHostBufferResetCount;
      await nextFrame();
      render(renderer, scene);
      expect(
        FlutterGpuGeometryRenderer.debugDeferredHostBufferResetCount,
        stalled,
      );
    });

    test('a full slot starts a fresh host buffer', () async {
      // Each slot's block holds 32 emplacements. More within one frame, or
      // while the GPU holds the next slot, must not take HostBuffer's
      // overflow path, which binds the whole block (past the 64 KB uniform
      // range many Vulkan GPUs allow) and allocates a block every time.
      // Outside widget frames the frame timestamp stays put, so all these
      // renders share one slot.
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      final replaced =
          FlutterGpuGeometryRenderer.debugHostBufferReplacementCount;
      var scene = base;
      for (var step = 0; step < 40; step++) {
        scene = [...scene];
        scene[step % 4 + 1] = (
          scene[step % 4 + 1].$1.moved(2, step.isEven ? 1 : -1),
          scene[step % 4 + 1].$2,
        );
        render(renderer, scene);
      }
      expect(
        FlutterGpuGeometryRenderer.debugHostBufferReplacementCount,
        greaterThan(replaced),
        reason: 'a full slot is replaced, not overflowed',
      );
      scene = [...scene];
      scene[1] = (scene[1].$1.moved(3, 2), scene[1].$2);
      await expectSameAsFullRender(renderer, scene);
    });

    test('can be turned off', () async {
      FlutterGpuGeometryRenderer.incrementalMatteUpdates = false;
      final renderer = await FlutterGpuGeometryRenderer.fromAsset(_bundle);
      addTearDown(renderer.dispose);
      render(renderer, base);
      final scene = [...base];
      scene[1] = (scene[1].$1.moved(7, 3), scene[1].$2);
      await expectSameAsFullRender(renderer, scene, updates: false);
    });
  });
}

Future<({ByteData data, int stride})> _bytes(
  FlutterGpuGeometryRenderer renderer,
) async {
  final image = renderer.debugMatteImage!;
  final data = (await image.toByteData())!;
  return (data: data, stride: image.width);
}
