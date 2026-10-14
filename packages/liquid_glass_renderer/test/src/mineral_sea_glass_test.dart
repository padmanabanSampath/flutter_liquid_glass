import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/glass_shadow.dart';

import 'shared.dart';

const _dpr = 2.0;
const _sceneSize = Size(240, 140);
const _tileRect = Rect.fromLTWH(20, 20, 200, 100);

/// Mineral glass without its shadow, so the tile is the glass alone.
LiquidGlassSettings _mineral({
  double etch = 0.8,
  double caustic = 0.6,
  SeaGlassFinish Function(SeaGlassFinish finish)? finish,
}) {
  final settings = LiquidGlassSettings.mineralSeaGlass(
    etch: etch,
    caustic: caustic,
  );
  final base = settings.seaGlassFinish.copyWith(shadow: 0, wobble: 0);
  return settings.copyWith(seaGlassFinish: finish?.call(base) ?? base);
}

Widget _scene(
  LiquidGlassSettings settings, {
  Widget backdrop = const ColoredBox(color: Colors.white),
}) => Directionality(
  textDirection: TextDirection.ltr,
  child: Stack(
    children: [
      Positioned.fill(child: backdrop),
      Positioned.fill(
        child: LiquidGlassLayer(
          settings: settings,
          child: Stack(
            children: [
              Positioned.fromRect(
                rect: _tileRect,
                child: LiquidGlass(
                  shape: const LiquidRoundedSuperellipse(borderRadius: 50),
                  appearance: MineralTint.seafoamJade.appearance,
                  child: const SizedBox.expand(),
                ),
              ),
            ],
          ),
        ),
      ),
    ],
  ),
);

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

/// Renders [child] and returns a reader of logical-pixel RGB values.
Future<List<int> Function(double x, double y)> _render(
  WidgetTester tester,
  Widget child,
) async {
  final bytes = await _renderBytes(tester, child);
  final width = (_sceneSize.width * _dpr).round();
  return (double x, double y) {
    final offset = ((y * _dpr).round() * width + (x * _dpr).round()) * 4;
    return bytes.sublist(offset, offset + 3);
  };
}

double _luminance(List<int> rgb) =>
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];

/// Mean luminance over [rect], in logical pixels, sampled every half pixel.
double _meanLuminance(List<int> Function(double, double) pixel, Rect rect) {
  var sum = 0.0;
  var count = 0;
  for (var y = rect.top; y < rect.bottom; y += 0.5) {
    for (var x = rect.left; x < rect.right; x += 0.5) {
      sum += _luminance(pixel(x, y));
      count++;
    }
  }
  return sum / count;
}

void main() {
  group('SeaGlassFinish.mineral', () {
    test('a deeper etch scatters, glitters and pits more', () {
      const light = SeaGlassFinish.mineral(etch: 0);
      const deep = SeaGlassFinish.mineral(etch: 1);
      expect(deep.scatter, greaterThan(light.scatter));
      expect(deep.grain, greaterThan(light.grain));
      expect(deep.sparkle, greaterThan(light.sparkle));
      expect(deep.flecks, greaterThan(light.flecks));
      expect(deep.glitter, greaterThan(light.glitter));
      expect(deep.pitting, greaterThan(light.pitting));
      expect(light.effectiveGlitter, greaterThan(0));
    });

    test('the caustic focuses the glow and lights the surface beneath', () {
      const dim = SeaGlassFinish.mineral(caustic: 0);
      const bright = SeaGlassFinish.mineral(caustic: 1);
      expect(dim.effectiveGlow, 1);
      expect(bright.effectiveGlow, 2);
      expect(dim.effectiveCaustic, 0);
      expect(bright.effectiveCaustic, 1);
      expect(bright.effectiveShadow, 1);
    });

    test('caustic and glitter are off in every other finish', () {
      for (final finish in const [
        SeaGlassFinish(),
        SeaGlassFinish.frosted(),
        SeaGlassFinish.clear(),
        SeaGlassFinish.slab(),
        SeaGlassFinish.none(),
      ]) {
        expect(finish.caustic, 0);
        expect(finish.glitter, 0);
        expect(finish.toJson().containsKey('caustic'), isFalse);
        expect(finish.toJson().containsKey('glitter'), isFalse);
      }
    });

    test('clamps its controls and keeps them through JSON', () {
      const wild = SeaGlassFinish(glow: 5, caustic: 3, glitter: -1);
      expect(wild.effectiveGlow, 2);
      expect(wild.effectiveCaustic, 1);
      expect(wild.effectiveGlitter, 0);
      final disabled = wild.copyWith(strength: 0);
      expect(disabled.effectiveCaustic, 0);
      const mineral = SeaGlassFinish.mineral(etch: 0.3, caustic: 0.4);
      expect(SeaGlassFinish.fromJson(mineral.toJson()), mineral);
      final changed = mineral.copyWith(caustic: 0.9, glitter: 0.2);
      expect(changed.caustic, 0.9);
      expect(changed.glitter, 0.2);
      expect(changed, isNot(mineral));
      expect(SeaGlassFinish.fromJson(changed.toJson()), changed);
    });
  });

  group('LiquidGlassSettings.mineralSeaGlass', () {
    test('frosts more as the etch deepens, lit from above', () {
      final light = LiquidGlassSettings.mineralSeaGlass(etch: 0);
      final deep = LiquidGlassSettings.mineralSeaGlass(etch: 1);
      expect(deep.frost, greaterThan(light.frost * 3));
      expect(deep.lightAngle, 0);
      expect(deep.refractionAmount, greaterThan(0));
      expect(deep.highlight, 0);
      expect(
        deep.seaGlassFinish,
        const SeaGlassFinish.mineral(etch: 1),
      );
      expect(LiquidGlassSettings.fromJson(deep.toJson()), deep);
    });

    test('clamps the etch and caustic', () {
      final wild = LiquidGlassSettings.mineralSeaGlass(etch: 4, caustic: -2);
      expect(
        wild.seaGlassFinish,
        const SeaGlassFinish.mineral(etch: 1, caustic: 0),
      );
    });
  });

  group('MineralTint', () {
    test('six named, opaque tints', () {
      expect(MineralTint.values, hasLength(6));
      expect(MineralTint.values.first, MineralTint.seafoamJade);
      expect(MineralTint.values.map((tint) => tint.name).toSet(), hasLength(6));
      for (final tint in MineralTint.values) {
        expect(tint.swatch.a, 1, reason: tint.name);
        expect(tint.body.a, 1, reason: tint.name);
      }
      expect(
        MineralTint.amber.appearance,
        LiquidGlassAppearance.seaGlass(color: MineralTint.amber.body),
      );
    });
  });

  group('caustic band', () {
    const body = Color(0xFF8ED6EE);
    const center = Offset(200, 200);
    const size = Size(300, 90);

    List<BoxShadow> cast(double caustic) => seaGlassCastShadows(
      LiquidGlassSettings.mineralSeaGlass(caustic: caustic),
      body,
      center,
      size,
    );

    test('without a caustic the band is the slab band, unchanged', () {
      final slab = LiquidGlassSettings.slabSeaGlass().copyWith(
        seaGlassFinish: const SeaGlassFinish.slab().copyWith(elevation: 12),
      );
      final mineral = slab.copyWith(
        seaGlassFinish: slab.seaGlassFinish.copyWith(caustic: 0),
      );
      expect(
        seaGlassCastShadows(mineral, body, center, size),
        seaGlassCastShadows(slab, body, center, size),
      );
    });

    test('a caustic brightens, widens and pushes out the band', () {
      final dim = cast(0).last;
      final bright = cast(1).last;
      double luminance(Color color) =>
          0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
      expect(luminance(bright.color), greaterThan(luminance(dim.color)));
      expect(luminance(bright.color), greaterThan(luminance(body)));
      expect(bright.color.a, greaterThanOrEqualTo(dim.color.a));
      expect(bright.blurRadius, greaterThan(dim.blurRadius));
      expect(bright.offset.dy, greaterThan(dim.offset.dy));
      // Straight down, under a light from above.
      expect(bright.offset.dx, closeTo(0, 1e-9));
    });
  });

  group('rendering', () {
    testWidgets(
      'glitter grains the face with lighter and darker specks',
      (tester) async {
        Future<List<double>> face(double glitter) async {
          final pixel = await _render(
            tester,
            _scene(
              _mineral(
                finish: (finish) => finish.copyWith(
                  glitter: glitter,
                  flecks: 0,
                  sparkle: 0,
                  grain: 0,
                ),
              ),
            ),
          );
          return [
            for (var y = 60.0; y < 80; y += 0.5)
              for (var x = 60.0; x < 180; x += 0.5) _luminance(pixel(x, y)),
          ];
        }

        final plain = await face(0);
        final glittered = await face(1);
        var darker = 0;
        var lighter = 0;
        for (var i = 0; i < plain.length; i++) {
          if (glittered[i] < plain[i] - 6) darker++;
          if (glittered[i] > plain[i] + 6) lighter++;
        }
        expect(darker / plain.length, inInclusiveRange(0.03, 0.4));
        expect(lighter / plain.length, inInclusiveRange(0.03, 0.4));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'a deeper etch hides the backdrop',
      (tester) async {
        Future<double> through(double etch) async {
          final white = await _render(tester, _scene(_mineral(etch: etch)));
          final black = await _render(
            tester,
            _scene(
              _mineral(etch: etch),
              backdrop: const ColoredBox(color: Colors.black),
            ),
          );
          const middle = Rect.fromLTWH(100, 60, 40, 20);
          return _meanLuminance(white, middle) - _meanLuminance(black, middle);
        }

        final clear = await through(0);
        final etched = await through(1);
        expect(clear, greaterThan(etched + 40));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'a caustic glows brighter on the side away from the light',
      (tester) async {
        Future<double> lowerMinusUpper(double caustic) async {
          final pixel = await _render(
            tester,
            _scene(
              _mineral(
                caustic: caustic,
                finish: (finish) => finish.copyWith(
                  glitter: 0,
                  flecks: 0,
                  sparkle: 0,
                  grain: 0,
                ),
              ),
            ),
          );
          const upper = Rect.fromLTWH(80, 34, 80, 10);
          const lower = Rect.fromLTWH(80, 100, 80, 10);
          return _meanLuminance(pixel, lower) - _meanLuminance(pixel, upper);
        }

        final dim = await lowerMinusUpper(0);
        final bright = await lowerMinusUpper(1);
        expect(bright, greaterThan(dim + 2));
        expect(math.max(dim, bright), greaterThan(0));
      },
      skip: skipProperGlassTests,
    );
  });
}
