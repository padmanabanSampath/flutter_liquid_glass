import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/internal/fake_glass_color.dart';
import 'package:liquid_glass_renderer/src/rendering/liquid_glass_layer.dart';
import 'package:liquid_glass_renderer/src/sea_glass.dart';

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

/// Renders [child] and returns a reader of device-pixel RGB values.
Future<List<int> Function(int x, int y)> _renderDevicePixels(
  WidgetTester tester,
  Widget child,
) async {
  final bytes = await _renderBytes(tester, child);
  final width = (_sceneSize.width * _dpr).round();
  return (int x, int y) {
    final offset = (y * width + x) * 4;
    return bytes.sublist(offset, offset + 3);
  };
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
        stochasticFrost: false,
      );
      expect(finish.stochasticFrost, isFalse);
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

    test('larger shapes are thicker glass', () {
      const finish = SeaGlassFinish();
      final small = seaGlassSizeResponse(finish, 36);
      final tile = seaGlassSizeResponse(finish, 64);
      final card = seaGlassSizeResponse(finish, 200);
      expect(small, (path: .85, scatter: .9));
      expect(tile.path, inExclusiveRange(small.path, card.path));
      expect(card.path, closeTo(1.2, 1e-9));
      expect(card.scatter, closeTo(1.3, 1e-9));
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
      // A pebble's whole face curves instead of a bevel fitted to the shape.
      expect(settings.refractionFitsShape, isFalse);
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

  group('lightAngle', () {
    test('the default is the overhead light, exactly', () {
      expect(const LiquidGlassSettings().lightAngle, 0);
      expect(const LiquidGlassSettings().lightDirection, const Offset(0, 1));
      expect(
        const LiquidGlassSettings().toJson().containsKey('lightAngle'),
        isFalse,
      );
    });

    test('turns the light clockwise and round-trips', () {
      const fromRight = LiquidGlassSettings(lightAngle: math.pi / 2);
      expect(fromRight.lightDirection.dx, closeTo(-1, 1e-12));
      expect(fromRight.lightDirection.dy, closeTo(0, 1e-12));
      expect(LiquidGlassSettings.fromJson(fromRight.toJson()), fromRight);
      expect(
        const LiquidGlassSettings().copyWith(lightAngle: 1).lightAngle,
        1,
      );
      expect(fromRight, isNot(const LiquidGlassSettings()));
    });
  });

  test('sea glass stretch is stiffer than the default', () {
    const glass = LiquidStretch(child: SizedBox());
    const pebble = LiquidStretch.seaGlass(child: SizedBox());
    expect(pebble.stretch, lessThan(glass.stretch));
    expect(pebble.interactionScale, lessThan(glass.interactionScale));
    expect(pebble.resistance, greaterThan(glass.resistance));
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

    test("the body filters white to the thin rim's color", () {
      const body = SeaGlassColors.cobalt;
      // The thickness pass deepens it toward the center. A shape twice the
      // default depth across has the smallest size response.
      final out = apply(
        fakeGlassSeaGlassMatrix(
          finish: const SeaGlassFinish(scatter: 0),
          body: body,
          shortSide: 36,
        ),
        [1, 1, 1],
      );
      expect(out[0], closeTo(math.pow(body.r, .3 * .85), 1e-9));
      expect(out[1], closeTo(math.pow(body.g, .3 * .85), 1e-9));
      expect(out[2], closeTo(math.pow(body.b, .3 * .85), 1e-9));
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

  group('scattered frost', () {
    RenderLiquidGlassLayer layerOf(WidgetTester tester) =>
        tester.allRenderObjects.whereType<RenderLiquidGlassLayer>().first;

    Future<RenderLiquidGlassLayer> pumpLayer(
      WidgetTester tester,
      LiquidGlassSettings settings,
    ) async {
      await _renderBytes(
        tester,
        _scene(settings, const LiquidGlassAppearance.seaGlass()),
      );
      return layerOf(tester);
    }

    testWidgets(
      'replaces the blur pass under a sea glass finish',
      (tester) async {
        final seaGlass = LiquidGlassSettings.seaGlass(frost: 6);
        var layer = await pumpLayer(tester, seaGlass);
        expect(layer.frostsInShader, isTrue);
        expect(layer.blurPassSigma, 0);

        // Opting out, disabling the finish and visible dispersion all keep
        // the Gaussian blur pass.
        for (final settings in [
          seaGlass.copyWith(
            seaGlassFinish: const SeaGlassFinish(stochasticFrost: false),
          ),
          seaGlass.copyWith(seaGlassFinish: const SeaGlassFinish.none()),
          seaGlass.copyWith(dispersion: .5),
        ]) {
          layer = await pumpLayer(tester, settings);
          expect(layer.frostsInShader, isFalse, reason: '$settings');
          expect(layer.blurPassSigma, 6, reason: '$settings');
        }
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'blurs the backdrop seen through the glass',
      (tester) async {
        Future<double> edgeMean(double frost) async {
          final pixel = await _render(
            tester,
            _scene(
              _flat(
                _bodyOnly.copyWith(scatter: 0),
              ).copyWith(frost: frost),
              const LiquidGlassAppearance.seaGlass(color: Color(0x00FFFFFF)),
              backdrop: const Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Expanded(child: ColoredBox(color: Colors.black)),
                  Expanded(child: ColoredBox(color: Colors.white)),
                ],
              ),
            ),
          );
          // Just left of the split, over black.
          final samples = [
            for (var y = 50.0; y < 90; y += 2)
              for (var x = 114.0; x < 118; x += .5) _luminance(pixel(x, y)),
          ];
          return samples.reduce((a, b) => a + b) / samples.length;
        }

        final sharp = await edgeMean(0);
        final frosted = await edgeMean(6);
        expect(sharp, lessThan(5));
        expect(frosted, greaterThan(40));
      },
      skip: skipProperGlassTests,
    );
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
        // Over white the full-thickness face is the body color, deepened
        // by the tile's size like the shader's size response.
        const body = SeaGlassColors.cobalt;
        final path = seaGlassSizeResponse(
          _bodyOnly,
          _tileSize.shortestSide,
        ).path;
        for (final (channel, value) in [body.r, body.g, body.b].indexed) {
          expect(
            center[channel],
            closeTo(255 * math.pow(value, path), 6),
            reason: 'channel $channel',
          );
        }
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
            _flat(
              const SeaGlassFinish(
                scatter: .25,
                grain: 0,
                sparkle: 0,
                pitting: 0,
              ),
            ),
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
      'the frost moves with its own shape, not with the layer',
      (tester) async {
        // Two separate shapes in one layer; only the second moves.
        Widget pair(double secondLeft) => Directionality(
          textDirection: TextDirection.ltr,
          child: Stack(
            children: [
              const Positioned.fill(child: ColoredBox(color: Colors.white)),
              Positioned.fill(
                child: LiquidGlassLayer(
                  settings: _flat(
                    _bodyOnly.copyWith(grain: 1, sparkle: 1),
                  ),
                  child: Stack(
                    children: [
                      for (final left in [10.0, secondLeft])
                        Positioned(
                          left: left,
                          top: 20,
                          child: LiquidGlass(
                            appearance: LiquidGlassAppearance.seaGlass(
                              color: SeaGlassColors.white.withValues(
                                alpha: .5,
                              ),
                            ),
                            shape: const LiquidRoundedSuperellipse(
                              borderRadius: 30,
                            ),
                            child: const SizedBox.square(dimension: 100),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );

        final before = await _render(tester, pair(120));
        final after = await _render(tester, pair(130));
        var maxDifference = 0;
        void compare(List<int> a, List<int> b) {
          for (var c = 0; c < 3; c++) {
            maxDifference = math.max(maxDifference, (a[c] - b[c]).abs());
          }
        }

        for (var y = 30.0; y < 110; y += 2.5) {
          // The first shape stays put, frost included.
          for (var x = 20.0; x < 100; x += 2.5) {
            compare(before(x, y), after(x, y));
          }
          // The second carries its frost along.
          for (var x = 130.0; x < 210; x += 2.5) {
            compare(before(x, y), after(x + 10, y));
          }
        }
        expect(maxDifference, lessThanOrEqualTo(2));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'FakeGlass deepens the body toward the center like real glass',
      (tester) async {
        Future<(List<int>, List<int>)> centerAndRim({
          required bool fake,
        }) async {
          final pixel = await _render(
            tester,
            Directionality(
              textDirection: TextDirection.ltr,
              child: Stack(
                children: [
                  const Positioned.fill(
                    child: ColoredBox(color: Colors.white),
                  ),
                  Center(
                    child: LiquidGlassLayer(
                      settings: _flat(_bodyOnly.copyWith(scatter: .25)),
                      fake: fake,
                      child: LiquidGlass(
                        appearance: const LiquidGlassAppearance.seaGlass(
                          color: SeaGlassColors.cobalt,
                        ),
                        shape: const LiquidRoundedSuperellipse(
                          borderRadius: 30,
                        ),
                        child: SizedBox.fromSize(size: _tileSize),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
          return (pixel(120, 70), pixel(22, 70));
        }

        final (realCenter, realRim) = await centerAndRim(fake: false);
        final (fakeCenter, fakeRim) = await centerAndRim(fake: true);
        expect(
          _luminance(fakeRim),
          greaterThan(_luminance(fakeCenter) + 20),
          reason: 'the thin rim is paler than the center',
        );
        for (var c = 0; c < 3; c++) {
          expect(fakeCenter[c], closeTo(realCenter[c], 12), reason: 'center');
          expect(fakeRim[c], closeTo(realRim[c], 24), reason: 'rim');
        }
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'the frost turns with its shape',
      (tester) async {
        // A disc centered on a device pixel corner, so a quarter turn maps
        // pixel centers onto pixel centers.
        Widget disc(double angle) => Directionality(
          textDirection: TextDirection.ltr,
          child: Stack(
            children: [
              const Positioned.fill(child: ColoredBox(color: Colors.white)),
              Positioned.fill(
                child: LiquidGlassLayer(
                  settings: _flat(_bodyOnly.copyWith(grain: 1)),
                  child: Stack(
                    children: [
                      Positioned(
                        left: 70,
                        top: 20,
                        child: Transform.rotate(
                          angle: angle,
                          child: LiquidGlass(
                            appearance: LiquidGlassAppearance.seaGlass(
                              color: SeaGlassColors.white.withValues(
                                alpha: .5,
                              ),
                            ),
                            shape: const LiquidOval(),
                            child: const SizedBox.square(dimension: 100),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        );

        final upright = await _renderDevicePixels(tester, disc(0));
        final turned = await _renderDevicePixels(tester, disc(math.pi / 2));
        // Center (120, 70) pt is device pixel corner (240, 140). The pixel
        // at (i, j) from it moves to (-j - 1, i) under a clockwise quarter
        // turn.
        var maxDifference = 0;
        var grainRange = 0;
        for (var j = -70; j < 70; j += 3) {
          for (var i = -70; i < 70; i += 3) {
            final before = upright(240 + i, 140 + j);
            final after = turned(240 - j - 1, 140 + i);
            for (var c = 0; c < 3; c++) {
              maxDifference = math.max(
                maxDifference,
                (before[c] - after[c]).abs(),
              );
            }
            grainRange = math.max(
              grainRange,
              (before[1] - upright(240, 140)[1]).abs(),
            );
          }
        }
        // The grain is visible, and it turned with the disc.
        expect(grainRange, greaterThan(6));
        expect(maxDifference, lessThanOrEqualTo(2));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'larger glass absorbs more of its body color',
      (tester) async {
        final pixel = await _render(
          tester,
          Directionality(
            textDirection: TextDirection.ltr,
            child: Stack(
              children: [
                const Positioned.fill(
                  child: ColoredBox(color: Colors.white),
                ),
                Positioned.fill(
                  child: LiquidGlassLayer(
                    settings: _flat(_bodyOnly),
                    child: Stack(
                      children: [
                        for (final (left, side) in [
                          (10.0, 60.0),
                          (100.0, 130.0),
                        ])
                          Positioned(
                            left: left,
                            top: 5,
                            child: LiquidGlass(
                              appearance: const LiquidGlassAppearance.seaGlass(
                                color: SeaGlassColors.cobalt,
                              ),
                              shape: const LiquidOval(),
                              child: SizedBox.square(dimension: side),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
        final small = _luminance(pixel(40, 35));
        final large = _luminance(pixel(165, 70));
        expect(large, lessThan(small - 5));
      },
      skip: skipProperGlassTests,
    );

    testWidgets(
      'the glint follows the light',
      (tester) async {
        Future<(double, double)> topAndSide(double angle) async {
          final pixel = await _render(
            tester,
            _scene(
              const LiquidGlassSettings(
                refractionHeight: 0,
                refractionAmount: 0,
                frost: 0,
                highlight: 2,
                contourStrength: 0,
                bevelShadowStrength: 0,
              ).copyWith(lightAngle: angle),
              const LiquidGlassAppearance(tint: Color(0xFF808080)),
            ),
          );
          // Half a point inside the top wall and the left wall.
          return (_luminance(pixel(120, 20.5)), _luminance(pixel(20.5, 70)));
        }

        final (topOverhead, sideOverhead) = await topAndSide(0);
        final (topFromLeft, sideFromLeft) = await topAndSide(-math.pi / 2);
        expect(topOverhead, greaterThan(sideOverhead + 10));
        expect(sideFromLeft, greaterThan(topFromLeft + 10));
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
