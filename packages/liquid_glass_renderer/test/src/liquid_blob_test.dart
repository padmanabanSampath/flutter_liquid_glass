import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';

import 'shared.dart';

const _halfSize = Size(60, 45);

void main() {
  group('LiquidBlob', () {
    test('the outline stays within its box', () {
      for (final blob in [
        const LiquidBlob(),
        LiquidBlob.cookie(lobes: 7, depth: .2),
        LiquidBlob.pebble(seed: 5, irregularity: 3),
        LiquidBlob.alive(time: 2.5),
      ]) {
        for (var i = 0; i < 360; i++) {
          final angle = i * math.pi / 180;
          final (radius, _) = blob.radiusAt(angle, _halfSize);
          final point = Offset(math.cos(angle), math.sin(angle)) * radius;
          // Inside the ellipse that fills the box.
          final reach =
              math.pow(point.dx / _halfSize.width, 2) +
              math.pow(point.dy / _halfSize.height, 2);
          expect(reach, lessThanOrEqualTo(1 + 1e-9));
        }
        final rect = Offset.zero & _halfSize * 2;
        final bounds = blob.getOuterPath(rect).getBounds();
        expect(rect.inflate(1e-6).contains(bounds.topLeft), isTrue);
        expect(rect.inflate(1e-6).contains(bounds.bottomRight), isTrue);
      }
    });

    test('without waves it is the ellipse', () {
      const blob = LiquidBlob();
      expect(blob.radiusAt(0, _halfSize).$1, closeTo(60, 1e-9));
      expect(blob.radiusAt(math.pi / 2, _halfSize).$1, closeTo(45, 1e-9));
    });

    test('a cookie reaches the box at its crests', () {
      final cookie = LiquidBlob.cookie(lobes: 4, depth: .3);
      const square = Size(60, 60);
      expect(cookie.radiusAt(0, square).$1, closeTo(60, 1e-9));
      // Between crests the outline dips by twice the depth.
      expect(
        cookie.radiusAt(math.pi / 4, square).$1,
        closeTo(60 * .7 / 1.3, 1e-9),
      );
    });

    test('the radius slope matches the outline', () {
      final blob = LiquidBlob.pebble(seed: 2, irregularity: 2);
      const step = 1e-5;
      for (var i = 0; i < 24; i++) {
        final angle = i * math.pi / 12 + .1;
        final (_, slope) = blob.radiusAt(angle, _halfSize);
        final numeric =
            (blob.radiusAt(angle + step, _halfSize).$1 -
                blob.radiusAt(angle - step, _halfSize).$1) /
            (2 * step);
        expect(slope, closeTo(numeric, 1e-4));
      }
    });

    test('draws at most three waves, kept from folding over', () {
      const blob = LiquidBlob(
        waves: [
          LiquidWave(count: 2, amplitude: .5),
          LiquidWave(count: 3, amplitude: .4),
          LiquidWave(count: 5, amplitude: .3),
          LiquidWave(count: 7, amplitude: .2),
        ],
      );
      final drawn = blob.effectiveWaves;
      expect(drawn.map((wave) => wave.count), [2, 3, 5]);
      final total = drawn.fold<double>(0, (sum, w) => sum + w.amplitude);
      expect(total, closeTo(.6, 1e-9));
      // Count, amplitude and phase of each wave, then the normalization.
      final parameters = blob.shaderParameters;
      expect(parameters, hasLength(12));
      expect(parameters[0], 2);
      expect(parameters[3], 3);
      expect(parameters[9], closeTo(1.6, 1e-9));
      expect(
        const LiquidBlob().shaderParameters,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
      );
    });

    test('pebbles are stable per seed and differ between seeds', () {
      expect(LiquidBlob.pebble(seed: 4), LiquidBlob.pebble(seed: 4));
      expect(LiquidBlob.pebble(seed: 4), isNot(LiquidBlob.pebble(seed: 5)));
      final smooth = LiquidBlob.pebble(seed: 4, irregularity: 0);
      expect(
        smooth.radiusAt(1, _halfSize).$1,
        closeTo(const LiquidBlob().radiusAt(1, _halfSize).$1, 1e-9),
      );
    });

    test('a living blob flows over time', () {
      final now = LiquidBlob.alive(time: 1);
      expect(LiquidBlob.alive(time: 1), now);
      final later = LiquidBlob.alive(time: 1.5);
      expect(later, isNot(now));
      expect(
        later.radiusAt(0, _halfSize).$1,
        isNot(closeTo(now.radiusAt(0, _halfSize).$1, 1e-3)),
      );
      expect(LiquidBlob.alive(time: 1, seed: 1), isNot(now));
    });

    test('copyWith, scale and equality', () {
      final blob = LiquidBlob.cookie(lobes: 6);
      const side = BorderSide(width: 2);
      expect(blob.copyWith(side: side).side, side);
      expect(blob.copyWith(side: side).waves, blob.waves);
      expect(blob.scale(2), blob);
      expect(blob, isNot(const LiquidBlob()));
    });
  });

  group('rendering', () {
    Future<ByteData> capture(WidgetTester tester, {required bool fake}) async {
      tester.view
        ..physicalSize = const Size(160, 160)
        ..devicePixelRatio = 1;
      addTearDown(tester.view.reset);
      final key = GlobalKey();
      await tester.pumpWidget(
        Directionality(
          textDirection: TextDirection.ltr,
          child: RepaintBoundary(
            key: key,
            child: ColoredBox(
              color: Colors.white,
              child: Center(
                child: LiquidGlassLayer(
                  fake: fake,
                  settings: const LiquidGlassSettings(
                    refractionAmount: 0,
                    frost: 0,
                    highlight: 0,
                    contourStrength: 0,
                    bevelShadowStrength: 0,
                  ),
                  defaultAppearance: const LiquidGlassAppearance(
                    tint: Color(0xCCFF0000),
                  ),
                  child: LiquidGlass(
                    shape: LiquidBlob.cookie(lobes: 4, depth: .3),
                    child: const SizedBox.square(dimension: 120),
                  ),
                ),
              ),
            ),
          ),
        ),
      );
      for (var frame = 0; frame < 5; frame++) {
        await tester.pump(const Duration(milliseconds: 16));
      }
      final boundary =
          key.currentContext!.findRenderObject()! as RenderRepaintBoundary;
      final image = await boundary.toImage();
      final bytes = (await tester.runAsync(image.toByteData))!;
      image.dispose();
      return bytes;
    }

    // The cookie's crests reach the box at 0, 90, 180 and 270 degrees; at
    // 45 degrees its outline dips to about 32 px from the center.
    int greenAt(ByteData bytes, Offset point) =>
        bytes.getUint8(((point.dy.round()) * 160 + point.dx.round()) * 4 + 1);
    const center = Offset(80, 80);
    final crest = center + const Offset(52, 0);
    final notch = center + Offset.fromDirection(math.pi / 4, 44);

    for (final fake in [true, false]) {
      testWidgets(
        '${fake ? 'FakeGlass' : 'glass'} follows the outline',
        (tester) async {
          final bytes = await capture(tester, fake: fake);
          expect(greenAt(bytes, crest), lessThan(150), reason: 'tinted glass');
          expect(greenAt(bytes, notch), 255, reason: 'backdrop between lobes');
          expect(greenAt(bytes, center), lessThan(150));
        },
        skip: !fake && skipProperGlassTests,
      );
    }
  });
}
