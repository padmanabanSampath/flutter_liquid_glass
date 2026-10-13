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

/// The Google finish without its shadow, so the tile is the glass alone.
LiquidGlassSettings _google({
  double lightAngle = -1.07,
  GlassLight light = const GlassLight(),
  double wobble = 0,
}) => LiquidGlassSettings.googleSeaGlass(
  finish: const SeaGlassFinish.google().copyWith(shadow: 0, wobble: wobble),
  light: light,
  lightAngle: lightAngle,
);

/// [children] in a layer over [backdrop].
Widget _scene(
  LiquidGlassSettings settings,
  List<Widget> children, {
  Widget backdrop = const ColoredBox(color: Colors.white),
}) => Directionality(
  textDirection: TextDirection.ltr,
  child: Stack(
    children: [
      Positioned.fill(child: backdrop),
      Positioned.fill(
        child: LiquidGlassLayer(
          settings: settings,
          child: Stack(children: children),
        ),
      ),
    ],
  ),
);

Widget _tile({
  Rect rect = _tileRect,
  Color color = const Color(0xFFB9C2F3),
  double visibility = 1,
}) => Positioned.fromRect(
  rect: rect,
  child: LiquidGlass(
    shape: const LiquidRoundedSuperellipse(borderRadius: 30),
    appearance: LiquidGlassAppearance.seaGlass(
      color: color,
      visibility: visibility,
    ),
    child: const SizedBox.expand(),
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

/// Mean luminance over [rect], in logical pixels, sampled every pixel.
double _meanLuminance(List<int> Function(double, double) pixel, Rect rect) {
  var sum = 0.0;
  var count = 0;
  for (var y = rect.top; y < rect.bottom; y++) {
    for (var x = rect.left; x < rect.right; x++) {
      sum += _luminance(pixel(x, y));
      count++;
    }
  }
  return sum / count;
}

void main() {
  group('SeaGlassFinish.google', () {
    test('is a lit slab finish; the other finishes have none of it', () {
      const google = SeaGlassFinish.google();
      expect(google.isEnabled, isTrue);
      expect(google.effectiveShading, 1);
      expect(google.effectiveGlow, 1);
      expect(google.effectiveFlecks, 1);
      expect(google.effectiveSparkle, 0);
      expect(google.effectiveShadow, 1);
      expect(google.effectiveElevation, greaterThan(0));
      expect(google.effectiveWobble, greaterThan(0));
      for (final finish in const [
        SeaGlassFinish(),
        SeaGlassFinish.frosted(),
        SeaGlassFinish.clear(),
        SeaGlassFinish.none(),
      ]) {
        expect(finish.shading, 0);
        expect(finish.glow, 0);
        expect(finish.flecks, 0);
        expect(finish.effectiveFlecks, 0);
        expect(finish.shadow, 0);
        expect(finish.wobble, 0);
        expect(finish.effectiveShadow, 0);
        expect(finish.effectiveWobble, 0);
      }
    });

    test('slab controls are clamped and vanish when disabled', () {
      const wild = SeaGlassFinish(
        shading: 3,
        glow: -1,
        flecks: 2,
        shadow: 2,
        elevation: -5,
        wobble: 9,
      );
      expect(wild.effectiveShading, 1);
      expect(wild.effectiveGlow, 0);
      expect(wild.effectiveFlecks, 1);
      expect(wild.effectiveShadow, 1);
      expect(wild.effectiveElevation, 0);
      expect(wild.effectiveWobble, 4);
      expect(wild.copyWith(wobble: -2).effectiveWobble, 0);
      final disabled = wild.copyWith(strength: 0);
      expect(disabled.effectiveShadow, 0);
      expect(disabled.effectiveWobble, 0);
      expect(disabled.effectiveFlecks, 0);
    });

    test('copyWith and JSON preserve the slab controls', () {
      const google = SeaGlassFinish.google();
      expect(SeaGlassFinish.fromJson(google.toJson()), google);
      final changed = google.copyWith(
        shading: 0.5,
        glow: 0.25,
        flecks: 0.6,
        shadow: 0.75,
        elevation: 40,
        wobble: 1.5,
      );
      expect(changed.shading, 0.5);
      expect(changed.glow, 0.25);
      expect(changed.flecks, 0.6);
      expect(changed.shadow, 0.75);
      expect(changed.elevation, 40);
      expect(changed.wobble, 1.5);
      expect(changed, isNot(google));
      expect(SeaGlassFinish.fromJson(changed.toJson()), changed);
    });
  });

  group('GlassLight', () {
    test('defaults to a far white light without ambient light', () {
      const light = GlassLight();
      expect(light.position, isNull);
      expect(light.reach, isNull);
      expect(light.color, const Color(0xFFFFFFFF));
      expect(light.ambient, const Color(0xFF000000));
      expect(light.intensity, 1);
    });

    test('clamps its height and intensity', () {
      const light = GlassLight(height: -10, intensity: -1);
      expect(light.effectiveHeight, 1);
      expect(light.effectiveIntensity, 0);
    });

    test('copyWith can clear the position and reach', () {
      const light = GlassLight(position: Offset(10, 20), reach: 300);
      expect(light.copyWith(height: 200).position, const Offset(10, 20));
      expect(light.copyWith(clearPosition: true).position, isNull);
      expect(light.copyWith(clearReach: true).reach, isNull);
      expect(light.copyWith(clearReach: true).position, light.position);
    });

    test('round-trips through JSON', () {
      const far = GlassLight(
        color: Color(0xFFFFE2C4),
        intensity: 0.8,
        ambient: Color(0xFF1C120A),
      );
      const point = GlassLight(
        position: Offset(120.5, 64),
        height: 360,
        reach: 560,
      );
      expect(GlassLight.fromJson(far.toJson()), far);
      expect(GlassLight.fromJson(point.toJson()), point);
      expect(GlassLight.fromJson(null), const GlassLight());
    });
  });

  group('GoogleSeaGlass', () {
    test('cycles through four distinctly named, opaque sets', () {
      expect(GoogleSeaGlass.values, hasLength(4));
      expect(GoogleSeaGlass.values.first, GoogleSeaGlass.original);
      expect(
        GoogleSeaGlass.values.map((set) => set.name).toSet(),
        hasLength(4),
      );
      for (final set in GoogleSeaGlass.values) {
        for (final color in [
          set.primary,
          set.secondary,
          set.tertiary,
          set.tertiaryLight,
          set.positive,
          set.positiveLight,
          set.error,
          set.accent,
          set.avatar,
          set.clear,
        ]) {
          expect(color.a, 1, reason: set.name);
        }
      }
    });

    test('colors become sea glass appearances', () {
      expect(
        GoogleSeaGlass.appearance(GoogleSeaGlass.original.primary),
        LiquidGlassAppearance.seaGlass(color: GoogleSeaGlass.original.primary),
      );
    });
  });

  group('LiquidGlassSettings.googleSeaGlass', () {
    test('carries the Google finish and the light', () {
      final settings = LiquidGlassSettings.googleSeaGlass();
      expect(settings.seaGlassFinish, const SeaGlassFinish.google());
      expect(settings.light, const GlassLight());
      // Straight above, as on the concept's phone screens.
      expect(settings.lightAngle, 0);
      // The thin rim bends the content behind it.
      expect(settings.refractionAmount, greaterThan(0));
      expect(settings.highlight, 0);
      expect(LiquidGlassSettings.fromJson(settings.toJson()), settings);
    });

    test('the light takes part in equality, copyWith and JSON', () {
      final settings = LiquidGlassSettings.googleSeaGlass();
      const lamp = GlassLight(position: Offset(40, 30), height: 300);
      final lit = settings.copyWith(light: lamp);
      expect(lit.light, lamp);
      expect(lit, isNot(settings));
      expect(LiquidGlassSettings.fromJson(lit.toJson()), lit);
      expect(const LiquidGlassSettings().light, const GlassLight());
    });
  });

  group('cast shadows', () {
    final settings = LiquidGlassSettings.googleSeaGlass();
    const body = Color(0xFFB9C2F3);
    const center = Offset(200, 200);
    const slab = Size(300, 90);

    test('need a shadow and an elevation', () {
      final regular = LiquidGlassSettings.seaGlass();
      expect(seaGlassCastShadows(regular, body, center, slab), isEmpty);
      final grounded = settings.copyWith(
        seaGlassFinish: const SeaGlassFinish.google().copyWith(elevation: 0),
      );
      expect(seaGlassCastShadows(grounded, body, center, slab), isEmpty);
    });

    test('a far light casts every shadow along its direction', () {
      final turned = settings.copyWith(lightAngle: -1.07);
      final direction = turned.lightDirection;
      final shadows = seaGlassCastShadows(turned, body, center, slab);
      for (final shadow in shadows) {
        final offset = shadow.offset;
        expect(
          offset.dx * direction.dy - offset.dy * direction.dx,
          closeTo(0, 1e-6),
        );
        expect(
          offset.dx * direction.dx + offset.dy * direction.dy,
          isPositive,
        );
      }
      // Higher glass casts farther.
      final raised = turned.copyWith(
        seaGlassFinish: const SeaGlassFinish.google().copyWith(elevation: 44),
      );
      final low = shadows.first;
      final high = seaGlassCastShadows(raised, body, center, slab).first;
      expect(high.offset.distance, greaterThan(low.offset.distance * 1.5));
      // The shadow itself is a faint, cool gray.
      expect(low.color.b, greaterThan(low.color.r));
      expect(low.color.a, lessThan(0.3));
    });

    test('a positioned light casts away from itself, farther as it drops', () {
      LiquidGlassSettings lamp(double height) => settings.copyWith(
        light: GlassLight(position: const Offset(100, 200), height: height),
      );
      final high = seaGlassCastShadows(lamp(600), body, center, slab).first;
      final low = seaGlassCastShadows(lamp(200), body, center, slab).first;
      expect(high.offset.dx, isPositive);
      expect(high.offset.dy, closeTo(0, 1e-9));
      expect(low.offset.dx, greaterThan(high.offset.dx));
    });

    test('light through the glass lands as a band of its color', () {
      final band = seaGlassCastShadows(settings, body, center, slab).last;
      expect(band.color.r, closeTo(body.r * 1.08, 0.01));
      expect(band.color.b, closeTo(1, 0.01));
      // Closer to the glass than the shadow, and narrower.
      final shadow = seaGlassCastShadows(settings, body, center, slab).first;
      expect(band.offset.distance, lessThan(shadow.offset.distance));
      expect(band.spreadRadius, isNegative);
      // Small round pebbles focus more of it.
      final pebble = seaGlassCastShadows(
        settings,
        body,
        center,
        const Size(90, 90),
      ).last;
      expect(pebble.color.a, greaterThan(band.color.a));
    });

    test('bright ambient light fades the shadow', () {
      final dim = settings.copyWith(
        light: const GlassLight(intensity: 0.3, ambient: Color(0xFFFFFFFF)),
      );
      final strong = seaGlassCastShadows(settings, body, center, slab).first;
      final faint = seaGlassCastShadows(dim, body, center, slab).first;
      expect(faint.color.a, lessThan(strong.color.a / 2));
    });

    test('offsets are mapped into the local frame', () {
      final shadows = [const BoxShadow(offset: Offset(10, 4))];
      final scaled = localizeShadowOffsets(
        shadows,
        Matrix4.diagonal3Values(2, 2, 1)..setTranslationRaw(30, 40, 0),
      );
      expect(scaled.single.offset, const Offset(5, 2));
    });
  });

  group('rendering', () {
    testWidgets(
      'light gathers on the side away from it',
      (tester) async {
        // Mean luminance just inside each short side, away from the
        // corners.
        const left = Rect.fromLTWH(26, 50, 10, 40);
        const right = Rect.fromLTWH(204, 50, 10, 40);
        Future<(double, double)> sides(double lightAngle) async {
          final pixel = await _render(
            tester,
            _scene(_google(lightAngle: lightAngle), [_tile()]),
          );
          return (_meanLuminance(pixel, left), _meanLuminance(pixel, right));
        }

        // Light shining rightward comes from the left and glows on the
        // right, and the reverse.
        final (fromLeftL, fromLeftR) = await sides(-math.pi / 2);
        final (fromRightL, fromRightR) = await sides(math.pi / 2);
        expect(fromLeftR, greaterThan(fromLeftL + 6));
        expect(fromRightL, greaterThan(fromRightR + 6));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'a positioned light shades each shape from its own direction',
      (tester) async {
        const left = Rect.fromLTWH(26, 50, 10, 40);
        const right = Rect.fromLTWH(204, 50, 10, 40);
        Future<double> leftMinusRight(Offset light) async {
          final pixel = await _render(
            tester,
            _scene(
              _google(
                light: GlassLight(position: light, height: 160),
              ),
              [_tile()],
            ),
          );
          return _meanLuminance(pixel, left) - _meanLuminance(pixel, right);
        }

        // The side away from the light glows.
        expect(await leftMinusRight(const Offset(-200, 70)), lessThan(-6));
        expect(await leftMinusRight(const Offset(440, 70)), greaterThan(6));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'the thin rim is clear and the middle milky',
      (tester) async {
        Future<List<int> Function(double, double)> over(Color backdrop) =>
            _render(
              tester,
              _scene(_google(), [
                _tile(),
              ], backdrop: ColoredBox(color: backdrop)),
            );
        final white = await over(Colors.white);
        final black = await over(Colors.black);
        // How much of the backdrop comes through: a point on the rim, just
        // inside the outline, and the middle.
        double through(double x, double y) =>
            _luminance(white(x, y)) - _luminance(black(x, y));
        expect(through(21.5, 70), greaterThan(through(120, 70) + 60));
        expect(through(120, 70), lessThan(80));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'flecks chip the face in darker and lighter bits',
      (tester) async {
        Future<List<double>> face(double flecks) async {
          final pixel = await _render(
            tester,
            _scene(
              LiquidGlassSettings.googleSeaGlass(
                finish: const SeaGlassFinish.google().copyWith(
                  shadow: 0,
                  wobble: 0,
                  flecks: flecks,
                ),
              ),
              [_tile()],
            ),
          );
          return [
            for (var y = 60.0; y < 80; y += 0.5)
              for (var x = 60.0; x < 180; x += 0.5) _luminance(pixel(x, y)),
          ];
        }

        final plain = await face(0);
        final flecked = await face(1);
        var darker = 0;
        var lighter = 0;
        for (var i = 0; i < plain.length; i++) {
          if (flecked[i] < plain[i] - 8) darker++;
          if (flecked[i] > plain[i] + 6) lighter++;
        }
        // A few percent of the face each, as on the concept's glass.
        expect(darker / plain.length, inInclusiveRange(0.01, 0.15));
        expect(lighter / plain.length, inInclusiveRange(0.005, 0.15));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'a wobbling outline only ever sinks inside the shape',
      (tester) async {
        Widget scene(double wobble) => _scene(
          _google(wobble: wobble),
          [_tile()],
          backdrop: const ColoredBox(
            color: Colors.white,
            child: GridPaper(color: Colors.black),
          ),
        );
        final smooth = await _renderBytes(tester, scene(0));
        final wobbly = await _renderBytes(tester, scene(4));
        final width = (_sceneSize.width * _dpr).round();
        final box = Rect.fromLTRB(
          _tileRect.left * _dpr,
          _tileRect.top * _dpr,
          _tileRect.right * _dpr,
          _tileRect.bottom * _dpr,
        ).inflate(1);
        var inside = 0;
        for (var i = 0; i < smooth.length; i += 4) {
          final pixel = i ~/ 4;
          final point = Offset(
            (pixel % width).toDouble(),
            (pixel ~/ width).toDouble(),
          );
          var differs = false;
          for (var c = 0; c < 3; c++) {
            if ((smooth[i + c] - wobbly[i + c]).abs() > 2) differs = true;
          }
          if (!differs) continue;
          expect(box.contains(point), isTrue, reason: 'pixel at $point');
          inside++;
        }
        expect(inside, greaterThan(100), reason: 'the outline moves');
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'hidden glass leaves the shading of glass over it alone',
      (tester) async {
        // A hidden pill under the tile's right end, as when a card opens
        // over components that faded out.
        const pill = Rect.fromLTWH(180, 50, 60, 40);
        final alone = await _render(
          tester,
          _scene(_google(), [_tile()]),
        );
        final over = await _render(
          tester,
          _scene(_google(), [
            _tile(rect: pill, color: const Color(0xFFE2F4DD), visibility: 0),
            _tile(),
          ]),
        );
        // Just inside the tile's end, the pill's box would be deeper than
        // the tile's and own the frost and shading there.
        const overlap = Rect.fromLTWH(204, 56, 12, 28);
        expect(
          _meanLuminance(over, overlap),
          closeTo(_meanLuminance(alone, overlap), 2.5),
        );
      },
      skip: skipProperGlassTests,
    );
  });
}
