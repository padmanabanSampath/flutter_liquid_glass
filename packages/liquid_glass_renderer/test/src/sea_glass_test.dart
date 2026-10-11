import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/internal/fake_glass_color.dart';

import 'shared.dart';

const _dpr = 2.0;
const _sceneSize = Size(240, 140);
const _tileSize = Size(200, 100);

/// No lens, blur or lighting, so the face is the sea glass stage alone.
LiquidGlassSettings _flat(SeaGlassFinish finish) => LiquidGlassSettings(
  refractionHeight: 0,
  refractionAmount: 0,
  frost: 0,
  highlight: 0,
  contourStrength: 0,
  bevelShadowStrength: 0,
  seaGlassFinish: finish,
);

/// The body alone: no scatter, rim glow, grain, sparkle or pitting.
const _bodyOnly = SeaGlassFinish(
  scatter: 0,
  rimGlow: 0,
  grain: 0,
  sparkle: 0,
  pitting: 0,
);

/// One tile centered over [backdrop].
Widget _scene(
  LiquidGlassSettings settings,
  LiquidGlassAppearance appearance, {
  Widget backdrop = const ColoredBox(color: Colors.white),
}) => Directionality(
  textDirection: TextDirection.ltr,
  child: Stack(
    children: [
      Positioned.fill(child: backdrop),
      Center(
        child: LiquidGlassLayer(
          settings: settings,
          child: LiquidGlass(
            appearance: appearance,
            shape: const LiquidRoundedSuperellipse(borderRadius: 30),
            child: SizedBox.fromSize(size: _tileSize),
          ),
        ),
      ),
    ],
  ),
);

/// Renders [child] and returns a reader of logical-pixel RGB values.
Future<List<int> Function(double x, double y)> _render(
  WidgetTester tester,
  Widget child,
) async {
  tester.view
    ..devicePixelRatio = _dpr
    ..physicalSize = _sceneSize * _dpr;
  addTearDown(tester.view.reset);
  final key = GlobalKey();
  await tester.pumpWidget(RepaintBoundary(key: key, child: child));
  for (var frame = 0; frame < 20; frame++) {
    await tester.pump(const Duration(milliseconds: 16));
  }
  final boundary =
      key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
  final image = await boundary.toImage(pixelRatio: _dpr);
  final bytes = (await tester.runAsync(image.toByteData))!;
  final width = image.width;
  image.dispose();
  return (double x, double y) {
    final offset = ((y * _dpr).round() * width + (x * _dpr).round()) * 4;
    return bytes.buffer.asUint8List(bytes.offsetInBytes + offset, 3).toList();
  };
}

Future<Uint8List> _renderBytes(WidgetTester tester, Widget child) async {
  tester.view
    ..devicePixelRatio = _dpr
    ..physicalSize = _sceneSize * _dpr;
  addTearDown(tester.view.reset);
  final key = GlobalKey();
  await tester.pumpWidget(RepaintBoundary(key: key, child: child));
  for (var frame = 0; frame < 20; frame++) {
    await tester.pump(const Duration(milliseconds: 16));
  }
  final boundary =
      key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
  final image = await boundary.toImage(pixelRatio: _dpr);
  final bytes = (await tester.runAsync(image.toByteData))!;
  image.dispose();
  return bytes.buffer.asUint8List();
}

double _luminance(List<int> rgb) =>
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

void main() {
  group('SeaGlassFinish', () {
    test('none is disabled and the default is enabled', () {
      expect(const SeaGlassFinish.none().isEnabled, isFalse);
      expect(const SeaGlassFinish().isEnabled, isTrue);
      expect(const SeaGlassFinish.frosted().isEnabled, isTrue);
      expect(const SeaGlassFinish.clear().isEnabled, isTrue);
      expect(const SeaGlassFinish(strength: -1).isEnabled, isFalse);
      expect(const SeaGlassFinish.none().samplingReach, 0);
      expect(const SeaGlassFinish(pitting: 3).samplingReach, 3);
    });

    test('effective values are clamped', () {
      const finish = SeaGlassFinish(
        strength: 2,
        depth: -4,
        scatter: 3,
        rimGlow: -1,
        grain: 9,
        grainSize: 0,
        sparkle: -2,
        pitting: -5,
      );
      expect(finish.effectiveStrength, 1);
      expect(finish.effectiveDepth, .1);
      expect(finish.effectiveScatter, 1);
      expect(finish.effectiveRimGlow, 0);
      expect(finish.effectiveGrain, 1);
      expect(finish.effectiveGrainSize, .25);
      expect(finish.effectiveSparkle, 0);
      expect(finish.effectivePitting, 0);
    });

    test('copyWith and JSON preserve every control', () {
      final finish = const SeaGlassFinish.frosted().copyWith(
        strength: .5,
        depth: 30,
        grainSize: 2.5,
      );
      expect(finish.strength, .5);
      expect(finish.depth, 30);
      expect(finish.grainSize, 2.5);
      expect(finish.scatter, const SeaGlassFinish.frosted().scatter);
      expect(SeaGlassFinish.fromJson(finish.toJson()), finish);
      expect(SeaGlassFinish.fromJson(null), const SeaGlassFinish.none());
    });

    test('shadows are tinted by the body and vanish without elevation', () {
      expect(
        SeaGlassFinish.shadows(SeaGlassColors.cobalt, elevation: 0),
        isEmpty,
      );
      final shadows = SeaGlassFinish.shadows(SeaGlassColors.cobalt);
      expect(shadows, isNotEmpty);
      final tinted = shadows.first.color;
      expect(tinted.b, greaterThan(tinted.r));
      expect(shadows.first.offset.dy, greaterThan(0));
    });

    test('the palette is opaque', () {
      expect(SeaGlassColors.values, hasLength(11));
      for (final color in SeaGlassColors.values) {
        expect(color.a, 1);
      }
    });
  });

  group('LiquidGlassSettings', () {
    test('regular glass has no finish and serializes as before', () {
      const settings = LiquidGlassSettings();
      expect(settings.seaGlassFinish, const SeaGlassFinish.none());
      expect(settings.toJson().containsKey('seaGlassFinish'), isFalse);
    });

    test('the sea glass preset is a flat, frosted, finished glass', () {
      final settings = LiquidGlassSettings.seaGlass();
      expect(settings.seaGlassFinish, const SeaGlassFinish());
      expect(settings.bevelShadowStrength, 0);
      expect(settings.refractionAmount, lessThan(20));
      expect(settings.frost, greaterThan(0));
      expect(
        LiquidGlassSettings.seaGlass(
          finish: const SeaGlassFinish.frosted(),
          frost: 10,
        ).seaGlassFinish,
        const SeaGlassFinish.frosted(),
      );
    });

    test('the finish takes part in equality, copyWith and JSON', () {
      final settings = LiquidGlassSettings.seaGlass();
      expect(
        settings,
        isNot(settings.copyWith(seaGlassFinish: const SeaGlassFinish.none())),
      );
      expect(LiquidGlassSettings.fromJson(settings.toJson()), settings);
      // A disabled finish with custom controls survives a round trip too.
      final disabled = settings.copyWith(
        seaGlassFinish: const SeaGlassFinish(strength: 0, grain: .9),
      );
      expect(LiquidGlassSettings.fromJson(disabled.toJson()), disabled);
    });
  });

  test('sea glass appearance carries the body color as its tint', () {
    const appearance = LiquidGlassAppearance.seaGlass(
      color: SeaGlassColors.amber,
    );
    expect(appearance.tint, SeaGlassColors.amber);
    expect(appearance.colorModel, const LiquidGlassColorModel.direct());
    expect(appearance.visibility, 1);
    expect(
      const LiquidGlassAppearance.seaGlass().tint,
      SeaGlassColors.seafoam,
    );
  });

  group('FakeGlass', () {
    List<double> apply(List<double> m, List<double> rgb) => [
      for (var row = 0; row < 3; row++)
        m[row * 5] * rgb[0] +
            m[row * 5 + 1] * rgb[1] +
            m[row * 5 + 2] * rgb[2] +
            m[row * 5 + 4] / 255,
    ];

    test('colorless glass without scatter passes the backdrop through', () {
      final matrix = fakeGlassSeaGlassMatrix(
        finish: const SeaGlassFinish(scatter: 0),
        body: const Color(0x00FFFFFF),
      );
      for (final rgb in [
        [0.0, 0.0, 0.0],
        [.2, .5, .9],
        [1.0, 1.0, 1.0],
      ]) {
        final out = apply(matrix, rgb);
        for (var i = 0; i < 3; i++) {
          expect(out[i], closeTo(rgb[i], 1e-9));
        }
      }
    });

    test('the body filters white to its color at mean thickness', () {
      const body = SeaGlassColors.cobalt;
      final out = apply(
        fakeGlassSeaGlassMatrix(
          finish: const SeaGlassFinish(scatter: 0),
          body: body,
        ),
        [1, 1, 1],
      );
      expect(out[0], closeTo(math.pow(body.r, .8), 1e-9));
      expect(out[1], closeTo(math.pow(body.g, .8), 1e-9));
      expect(out[2], closeTo(math.pow(body.b, .8), 1e-9));
    });

    test('scatter makes the glass glow over black', () {
      final out = apply(
        fakeGlassSeaGlassMatrix(
          finish: const SeaGlassFinish(scatter: .5),
          body: SeaGlassColors.teal,
        ),
        [0, 0, 0],
      );
      expect(out.every((channel) => channel > .1), isTrue);
      // Tinted by the teal body.
      expect(out[1], greaterThan(out[0]));
    });

    test('the finish adds a backdrop filter even without frost', () {
      const appearance = LiquidGlassAppearance.seaGlass();
      expect(
        fakeGlassBackdropFilter(
          const LiquidGlassSettings(frost: 0),
          appearance,
        ),
        isNull,
      );
      expect(
        fakeGlassBackdropFilter(
          const LiquidGlassSettings(frost: 0, seaGlassFinish: SeaGlassFinish()),
          appearance,
        ),
        isNotNull,
      );
    });
  });

  group('rendering', () {
    testWidgets(
      'a finish at strength 0 renders exactly as regular glass',
      (tester) async {
        Widget scene(SeaGlassFinish finish) => _scene(
          LiquidGlassSettings.seaGlass().copyWith(seaGlassFinish: finish),
          const LiquidGlassAppearance(tint: Color(0x332E5AD2)),
          backdrop: const ColoredBox(
            color: Colors.white,
            child: GridPaper(color: Colors.black),
          ),
        );
        final regular = await _renderBytes(
          tester,
          scene(const SeaGlassFinish.none()),
        );
        final disabled = await _renderBytes(
          tester,
          scene(
            const SeaGlassFinish(
              strength: 0,
              grain: 1,
              sparkle: 1,
              pitting: 4,
              scatter: 1,
              rimGlow: 1,
            ),
          ),
        );
        expect(disabled, regular);
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'the body filters the backdrop, deepest at the center',
      (tester) async {
        final pixel = await _render(
          tester,
          _scene(
            _flat(_bodyOnly),
            const LiquidGlassAppearance.seaGlass(color: SeaGlassColors.cobalt),
          ),
        );
        final center = pixel(120, 70);
        // Over white the full-thickness face is the body color itself.
        expect(center[0], closeTo(0x2E, 6));
        expect(center[1], closeTo(0x5A, 6));
        expect(center[2], closeTo(0xD2, 6));
        // Two points inside the left edge the glass is thinner and paler.
        final rim = pixel(22, 70);
        expect(_luminance(rim), greaterThan(_luminance(center) + 20));
        // Outside, the backdrop is untouched.
        expect(pixel(5, 70), [255, 255, 255]);
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'the glass stays transparent: the backdrop shows through',
      (tester) async {
        final pixel = await _render(
          tester,
          _scene(
            _flat(const SeaGlassFinish(grain: 0, sparkle: 0, pitting: 0)),
            const LiquidGlassAppearance.seaGlass(color: SeaGlassColors.aqua),
            backdrop: const Row(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(child: ColoredBox(color: Colors.black)),
                Expanded(child: ColoredBox(color: Colors.white)),
              ],
            ),
          ),
        );
        final overBlack = pixel(80, 70);
        final overWhite = pixel(160, 70);
        expect(
          _luminance(overWhite) - _luminance(overBlack),
          greaterThan(80),
        );
        // Scattered light keeps the glass visible, in its color, over black.
        expect(_luminance(overBlack), greaterThan(25));
        expect(overBlack[2], greaterThan(overBlack[0]));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'grain textures the frosted face',
      (tester) async {
        Future<double> spread(SeaGlassFinish finish) async {
          final pixel = await _render(
            tester,
            _scene(
              _flat(finish),
              LiquidGlassAppearance.seaGlass(
                color: SeaGlassColors.white.withValues(alpha: .5),
              ),
            ),
          );
          final samples = [
            for (var y = 50.0; y < 90; y += 1.5)
              for (var x = 80.0; x < 160; x += 1.5) _luminance(pixel(x, y)),
          ];
          final mean = samples.reduce((a, b) => a + b) / samples.length;
          final variance =
              samples
                  .map((v) => (v - mean) * (v - mean))
                  .reduce((a, b) => a + b) /
              samples.length;
          return math.sqrt(variance);
        }

        final smooth = await spread(_bodyOnly);
        final grainy = await spread(_bodyOnly.copyWith(grain: 1));
        expect(smooth, lessThan(.75));
        expect(grainy, greaterThan(smooth + 1.5));
      },
      skip: skipProperGlassTests,
    );
  });
}
