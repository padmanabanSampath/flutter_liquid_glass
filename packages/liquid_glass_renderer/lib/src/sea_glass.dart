import 'dart:math' as math;

import 'package:equatable/equatable.dart';
import 'package:flutter/painting.dart';
import 'package:liquid_glass_renderer/src/glass_light.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_appearance.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_settings.dart';
import 'package:liquid_glass_renderer/src/mineral_sea_glass.dart';
import 'package:liquid_glass_renderer/src/slab_sea_glass.dart';
import 'package:meta/meta.dart';

/// The tumbled, frosted surface of sea glass.
///
/// Sea glass is ordinary glass whose surface the sea has pitted into a fine
/// frost. It stays translucent, but the frost scatters part of the light
/// into a soft milky veil, sprinkles the face with tiny glints and turns the
/// rounded rim lighter than the body. The glass itself is colored throughout,
/// so thicker glass absorbs more: the face is deepest in color at its center
/// and pales toward the thin rim.
///
/// Start from [LiquidGlassSettings.seaGlass] (or set
/// [LiquidGlassSettings.seaGlassFinish] yourself) and give each shape its
/// body color with [LiquidGlassAppearance.seaGlass]. The finish is shared by
/// every shape of a layer; the body color can differ per shape.
///
/// [strength] cross-fades the whole finish: `0` is the regular liquid-glass
/// face and costs nothing, `1` is full sea glass. Distances are logical
/// pixels; amounts use the `0` to `1` range.
///
/// See also [SeaGlassColors] for body colors sampled from real sea glass.
class SeaGlassFinish with Equatable {
  /// Creates a sea glass finish.
  ///
  /// The defaults are a light frost that keeps the backdrop readable, which
  /// is clearer than most frosted UI glass.
  const SeaGlassFinish({
    this.strength = 1,
    this.depth = 18,
    this.scatter = 0.12,
    this.rimGlow = 0.35,
    this.grain = 0.5,
    this.grainSize = 1.4,
    this.sparkle = 0.35,
    this.pitting = 0.6,
    this.stochasticFrost = true,
    this.gloss = 0.6,
    this.shading = 0,
    this.glow = 0,
    this.flecks = 0,
    this.shadow = 0,
    this.elevation = 12,
    this.wobble = 0,
    this.colorClarity = 0,
    this.caustic = 0,
    this.glitter = 0,
  });

  /// No sea glass: the regular liquid-glass face.
  const SeaGlassFinish.none()
    : strength = 0,
      depth = 18,
      scatter = 0.12,
      rimGlow = 0.35,
      grain = 0.5,
      grainSize = 1.4,
      sparkle = 0.35,
      pitting = 0.6,
      stochasticFrost = true,
      gloss = 0.6,
      shading = 0,
      glow = 0,
      flecks = 0,
      shadow = 0,
      elevation = 12,
      wobble = 0,
      colorClarity = 0,
      caustic = 0,
      glitter = 0;

  /// A heavily tumbled, milkier pebble: more scatter, coarser grain and a
  /// brighter rim. Pair it with a larger [LiquidGlassSettings.frost].
  const SeaGlassFinish.frosted()
    : strength = 1,
      depth = 22,
      scatter = 0.45,
      rimGlow = 0.5,
      grain = 0.7,
      grainSize = 1.8,
      sparkle = 0.45,
      pitting = 1.6,
      stochasticFrost = true,
      gloss = 0.35,
      shading = 0,
      glow = 0,
      flecks = 0,
      shadow = 0,
      elevation = 12,
      wobble = 0,
      colorClarity = 0,
      caustic = 0,
      glitter = 0;

  /// A barely tumbled shard that is still mostly clear.
  const SeaGlassFinish.clear()
    : strength = 1,
      depth = 14,
      scatter = 0.05,
      rimGlow = 0.25,
      grain = 0.3,
      grainSize = 1.2,
      sparkle = 0.2,
      pitting = 0.4,
      stochasticFrost = true,
      gloss = 0.85,
      shading = 0,
      glow = 0,
      flecks = 0,
      shadow = 0,
      elevation = 12,
      wobble = 0,
      colorClarity = 0,
      caustic = 0,
      glitter = 0;

  /// The milky, candy-like sea glass of the Sea Glass design concept: thick
  /// slabs of nearly opaque pastel glass flecked like terrazzo, brightest
  /// where they are thin and where the light that went through them
  /// gathers, casting soft shadows with a band of colored light, and with
  /// slightly hand-made outlines.
  ///
  /// Use it through [LiquidGlassSettings.slabSeaGlass], which also turns
  /// off the iOS glint and border, and color shapes from [SlabSeaGlass].
  const SeaGlassFinish.slab()
    : strength = 1,
      depth = 14,
      scatter = 0.85,
      rimGlow = 0,
      grain = 0.15,
      grainSize = 1.1,
      sparkle = 0,
      pitting = 0.3,
      stochasticFrost = true,
      gloss = 0,
      shading = 1,
      glow = 1,
      flecks = 1,
      shadow = 1,
      elevation = 18,
      wobble = 1.5,
      colorClarity = 0,
      caustic = 0,
      glitter = 0;

  /// Mineral sea glass: glass tinted by the minerals in it, its surface
  /// etched into a sugary glitter, which focuses the light it lets through
  /// into a bright band on the surface beneath it.
  ///
  /// [etch] runs from barely etched glass, clear enough to read what lies
  /// beneath, bent by the lens (`0`), to a deep etch that glitters and hides
  /// it (`1`): it raises the scatter, grain, sparkle, flecks, glitter and
  /// pitting together. [caustic] runs from the inner glow every slab shows
  /// on the side away from the light (`0`) to a brighter glow and a band of
  /// light on the surface beneath (`1`).
  ///
  /// Use it through [LiquidGlassSettings.mineralSeaGlass], which also frosts
  /// the backdrop by [etch], and color shapes from [MineralTint].
  const SeaGlassFinish.mineral({double etch = 0.7, double caustic = 0.6})
    : strength = 1,
      depth = 20,
      scatter = 0.2 + 0.56 * etch,
      rimGlow = 0.5,
      grain = 0.15 + 0.6 * etch,
      grainSize = 0.7,
      sparkle = 0.1 + 0.8 * etch,
      pitting = 0.3 + 1.2 * etch,
      stochasticFrost = true,
      gloss = 0,
      shading = 1,
      glow = 1 + caustic,
      flecks = 0.15 + 0.85 * etch,
      shadow = 1,
      elevation = 12,
      wobble = 2.5,
      colorClarity = 0,
      caustic = caustic,
      glitter = 0.15 + 0.85 * etch;

  /// Restores a finish produced by [toJson].
  factory SeaGlassFinish.fromJson(Object? json) {
    if (json is! Map<String, Object?>) return const SeaGlassFinish.none();
    const defaults = SeaGlassFinish();
    double number(String key, double fallback) =>
        (json[key] as num?)?.toDouble() ?? fallback;
    return SeaGlassFinish(
      strength: number('strength', defaults.strength),
      depth: number('depth', defaults.depth),
      scatter: number('scatter', defaults.scatter),
      rimGlow: number('rimGlow', defaults.rimGlow),
      grain: number('grain', defaults.grain),
      grainSize: number('grainSize', defaults.grainSize),
      sparkle: number('sparkle', defaults.sparkle),
      pitting: number('pitting', defaults.pitting),
      stochasticFrost:
          json['stochasticFrost'] as bool? ?? defaults.stochasticFrost,
      gloss: number('gloss', defaults.gloss),
      shading: number('shading', defaults.shading),
      glow: number('glow', defaults.glow),
      flecks: number('flecks', defaults.flecks),
      shadow: number('shadow', defaults.shadow),
      elevation: number('elevation', defaults.elevation),
      wobble: number('wobble', defaults.wobble),
      colorClarity: number('colorClarity', defaults.colorClarity),
      caustic: number('caustic', defaults.caustic),
      glitter: number('glitter', defaults.glitter),
    );
  }

  /// How much of the sea glass finish replaces the regular glass face.
  ///
  /// `0` disables the finish entirely, `1` renders full sea glass. Values in
  /// between cross-fade the face, which is useful for transitions.
  final double strength;

  /// How far inside the silhouette the glass reaches its full thickness.
  ///
  /// Sea glass is a rounded pebble: it is thin at the rim and thickens over
  /// this distance, so the body color deepens from a pale rim to a saturated
  /// center. Larger values give a softer, more pebble-like gradient. The
  /// renderer resolves edge distance up to four times
  /// [LiquidGlassSettings.effectiveEdgeDistanceRange]; deeper values are
  /// clamped to that.
  final double depth;

  /// The share of light the frost scatters instead of letting it through.
  ///
  /// Scattered light leaves the glass in its body color, paler at the thin
  /// rim, so scattering glass glows even over dark content. `0` is clear
  /// tinted glass; around `0.45` it reads as a milky frosted pebble. The
  /// glow is brightest over light backdrops.
  final double scatter;

  /// Brightness of the frosted rim, which scatters more light than the face
  /// and catches it most on the side facing the light.
  final double rimGlow;

  /// Strength of the fine sugary grain left by the sea's pitting.
  final double grain;

  /// Size of one grain in logical pixels.
  final double grainSize;

  /// Strength of the tiny glints where single pits catch the light.
  final double sparkle;

  /// How far, in logical pixels, the pits scatter the backdrop seen through
  /// the glass. This gives frosted glass its slightly broken, textured
  /// transmission on top of [LiquidGlassSettings.frost]. `0` disables it.
  final double pitting;

  /// Whether the glass pass renders [LiquidGlassSettings.frost] itself, by
  /// scattering its backdrop samples the way the pits scatter light,
  /// instead of running a separate blur pass.
  ///
  /// Frosted glass blurs because its rough surface sends each ray a random
  /// way, so the frost comes out finely grained, as real sea glass looks.
  /// It skips the blur pass's intermediate textures and render passes, at
  /// the cost of twelve backdrop samples per glass pixel; it is the lean
  /// choice for the light frost sea glass uses. Turn it off for a smooth
  /// Gaussian frost, or for a very heavy frost over a large area.
  final bool stochasticFrost;

  /// How brightly the pebble's rounded surface catches the light.
  ///
  /// A pebble is a low dome: its surface tilts toward the outline. A light
  /// high above glints off the shoulder facing it in a curved highlight, its
  /// reflection inside the glass lights the far wall more softly, and the
  /// outline, where the surface turns away, shows a thin bright line. The
  /// glints follow [LiquidGlassSettings.lightAngle]. `0` turns them off;
  /// tumbled, frostier glass glints less.
  final double gloss;

  /// How strongly the light shades the glass as a thick, milky slab.
  ///
  /// The body color shows where the slab is thickest, just inside its
  /// center toward the light. The light brightens the thin glass along the
  /// outline, except on the wall facing it, and a little of the half facing
  /// it; brightening saturates a color's strongest channels first, so a
  /// blue slab pales toward cyan. The thin rim is clear glass that scatters
  /// little, so content behind shows through it, bent by the lens. Pair it
  /// with a high [scatter], which makes the rest of the glass milky enough
  /// to show its own color over any backdrop. The
  /// light is [LiquidGlassSettings.light]; without a position it shines
  /// along [LiquidGlassSettings.lightAngle]. `0`, the default, keeps the
  /// tumbled pebble face.
  final double shading;

  /// How brightly the light that went through the slab glows where it
  /// gathers, on the side away from the light. Only with [shading]. `1` is
  /// a slab's own glow; up to `2` the glass focuses the light more, as
  /// mineral glass with a [caustic] does.
  final double glow;

  /// How densely the glass is flecked like terrazzo: angular chips, dark
  /// ones of a deeper tone of the body and light, whitish crumbs, about a
  /// [grainSize] or two long. They are anchored to each shape, so they move,
  /// turn and scale with it. `0`, the default, has none.
  final double flecks;

  /// Strength of the shadow each shape casts away from the light.
  ///
  /// The shadow is soft and tinted by the glass, and light that passes
  /// through the body pools inside it as a glow of the body's color. It
  /// lengthens as the light moves aside or comes down
  /// ([GlassLight.height]); [elevation] is how high the glass stands above
  /// the surface. `0`, the default, casts none; shadows given to
  /// `LiquidGlass.shadows` are drawn either way.
  final double shadow;

  /// How high, in logical pixels, the glass stands above the surface it
  /// casts its [shadow] on.
  final double elevation;

  /// How far, in logical pixels, the outline strays from the exact shape,
  /// in gentle undulations different for every shape, as hand-cut glass
  /// does. The outline only sinks inward, so a shape never grows past its
  /// box. `0`, the default, keeps the exact shape. Only full glass draws
  /// it; FakeGlass keeps the exact shape.
  final double wobble;

  /// How much glass of a color clears: it scatters, frosts and pits the
  /// backdrop less, so it shows and bends what is beneath while keeping its
  /// color. Even at `1` it keeps a third of its milkiness and nearly half
  /// its frost, so it still reads as frosted sea glass rather than tinted
  /// film. Only glass of a color clears, and as much as it holds: frosted
  /// white and colorless glass in the same layer keep their frost, and a
  /// bridge between the two clears gradually. `0`, the default, renders
  /// as before.
  final double colorClarity;

  /// How brightly the glass focuses the light it lets through into a band
  /// on the surface beneath it, past the edge away from the light: a caustic,
  /// brighter than the surface around it and in the color of the glass. It
  /// brightens and widens the band of color the [shadow] already carries,
  /// so it needs a [shadow]. `0`, the default, renders as before.
  final double caustic;

  /// How densely a deep etch glitters: granules a grain or so across, some
  /// catching the light and some shadowed, all over the face and moving
  /// with it. Unlike [sparkle] they glitter from every side. `0`, the
  /// default, has none.
  final double glitter;

  /// [strength] clamped to `0` to `1`.
  double get effectiveStrength => strength.clamp(0.0, 1.0);

  /// Whether this finish changes the rendered glass at all.
  bool get isEnabled => effectiveStrength > 0;

  /// [depth], never below a tenth of a logical pixel.
  double get effectiveDepth => math.max(0.1, depth);

  /// [scatter] clamped to `0` to `1`.
  double get effectiveScatter => scatter.clamp(0.0, 1.0);

  /// [rimGlow] clamped to `0` to `1`.
  double get effectiveRimGlow => rimGlow.clamp(0.0, 1.0);

  /// [grain] clamped to `0` to `1`.
  double get effectiveGrain => grain.clamp(0.0, 1.0);

  /// [grainSize], never below a quarter of a logical pixel.
  double get effectiveGrainSize => math.max(0.25, grainSize);

  /// [sparkle] clamped to `0` to `1`.
  double get effectiveSparkle => sparkle.clamp(0.0, 1.0);

  /// [gloss] clamped to `0` to `1`.
  double get effectiveGloss => gloss.clamp(0.0, 1.0);

  /// [pitting]; never negative.
  double get effectivePitting => math.max(0, pitting);

  /// [shading] clamped to `0` to `1`.
  double get effectiveShading => shading.clamp(0.0, 1.0);

  /// [glow] clamped to `0` to `2`.
  double get effectiveGlow => glow.clamp(0.0, 2.0);

  /// [flecks] clamped to `0` to `1`; `0` when the finish is disabled.
  double get effectiveFlecks => isEnabled ? flecks.clamp(0.0, 1.0) : 0;

  /// [shadow] clamped to `0` to `1`; `0` when the finish is disabled.
  double get effectiveShadow => isEnabled ? shadow.clamp(0.0, 1.0) : 0;

  /// [elevation]; never negative.
  double get effectiveElevation => math.max(0, elevation);

  /// [wobble] clamped to `0` to `4` logical pixels; `0` when the finish is
  /// disabled.
  double get effectiveWobble => isEnabled ? wobble.clamp(0.0, 4.0) : 0;

  /// [colorClarity] clamped to `0` to `1`; `0` when the finish is
  /// disabled.
  double get effectiveColorClarity =>
      isEnabled ? colorClarity.clamp(0.0, 1.0) : 0;

  /// [caustic] clamped to `0` to `1`; `0` when the finish is disabled.
  double get effectiveCaustic => isEnabled ? caustic.clamp(0.0, 1.0) : 0;

  /// [glitter] clamped to `0` to `1`; `0` when the finish is disabled.
  double get effectiveGlitter => isEnabled ? glitter.clamp(0.0, 1.0) : 0;

  /// Backdrop reach of the pitting, in logical pixels, that the layer must
  /// add to its capture; `0` when the finish is disabled.
  double get samplingReach => isEnabled ? effectivePitting : 0;

  /// Returns a copy with the supplied controls replaced.
  SeaGlassFinish copyWith({
    double? strength,
    double? depth,
    double? scatter,
    double? rimGlow,
    double? grain,
    double? grainSize,
    double? sparkle,
    double? pitting,
    bool? stochasticFrost,
    double? gloss,
    double? shading,
    double? glow,
    double? flecks,
    double? shadow,
    double? elevation,
    double? wobble,
    double? colorClarity,
    double? caustic,
    double? glitter,
  }) => SeaGlassFinish(
    strength: strength ?? this.strength,
    depth: depth ?? this.depth,
    scatter: scatter ?? this.scatter,
    rimGlow: rimGlow ?? this.rimGlow,
    grain: grain ?? this.grain,
    grainSize: grainSize ?? this.grainSize,
    sparkle: sparkle ?? this.sparkle,
    pitting: pitting ?? this.pitting,
    stochasticFrost: stochasticFrost ?? this.stochasticFrost,
    gloss: gloss ?? this.gloss,
    shading: shading ?? this.shading,
    glow: glow ?? this.glow,
    flecks: flecks ?? this.flecks,
    shadow: shadow ?? this.shadow,
    elevation: elevation ?? this.elevation,
    wobble: wobble ?? this.wobble,
    colorClarity: colorClarity ?? this.colorClarity,
    caustic: caustic ?? this.caustic,
    glitter: glitter ?? this.glitter,
  );

  /// A soft shadow tinted by the glass, as colored glass casts on a light
  /// surface: light passing through the body colors the shadow.
  ///
  /// Pass the result to `LiquidGlass.shadows`. [color] is the body color of
  /// the shape (see [SeaGlassColors]); [elevation] scales the offset and
  /// blur in logical pixels. Keep it small for flat UI.
  static List<BoxShadow> shadows(
    Color color, {
    double elevation = 6,
    Offset direction = const Offset(0.35, 1),
  }) {
    if (elevation <= 0) return const [];
    final hsl = HSLColor.fromColor(color.withValues(alpha: 1));
    final shade = hsl
        .withLightness((hsl.lightness * 0.62).clamp(0.0, 1.0))
        .withSaturation((hsl.saturation * 1.1).clamp(0.0, 1.0))
        .toColor();
    return [
      BoxShadow(
        color: shade.withValues(alpha: 0.22 * color.a),
        offset: direction * elevation,
        blurRadius: elevation * 2.2,
      ),
      BoxShadow(
        color: const Color(0xFF000000).withValues(alpha: 0.05),
        offset: direction * (elevation * 0.4),
        blurRadius: elevation,
      ),
    ];
  }

  /// Serializes the finish for presets and tooling.
  Map<String, Object> toJson() => {
    'strength': strength,
    'depth': depth,
    'scatter': scatter,
    'rimGlow': rimGlow,
    'grain': grain,
    'grainSize': grainSize,
    'sparkle': sparkle,
    'pitting': pitting,
    'stochasticFrost': stochasticFrost,
    'gloss': gloss,
    'shading': shading,
    'glow': glow,
    'flecks': flecks,
    'shadow': shadow,
    'elevation': elevation,
    'wobble': wobble,
    if (colorClarity != 0) 'colorClarity': colorClarity,
    if (caustic != 0) 'caustic': caustic,
    if (glitter != 0) 'glitter': glitter,
  };

  @override
  List<Object?> get props => [
    strength,
    depth,
    scatter,
    rimGlow,
    grain,
    grainSize,
    sparkle,
    pitting,
    stochasticFrost,
    gloss,
    shading,
    glow,
    flecks,
    shadow,
    elevation,
    wobble,
    colorClarity,
    caustic,
    glitter,
  ];
}

/// How a shape's size scales a sea glass [finish]: larger glass behaves like
/// thicker material, as Apple's does, absorbing more of its body color
/// (`path`) and scattering light more softly (`scatter`). Mirrors
/// `seaGlassSizeResponse` in `sea_glass.glsl`.
@internal
({double path, double scatter}) seaGlassSizeResponse(
  SeaGlassFinish finish,
  double shortSide,
) {
  final x = ((shortSide / 2 / finish.effectiveDepth - 1) / 3).clamp(0.0, 1.0);
  final size = x * x * (3 - 2 * x);
  return (path: 0.85 + 0.35 * size, scatter: 0.9 + 0.4 * size);
}

/// Body colors of real sea glass.
///
/// Each color is what the glass shows over white at full thickness. Pass one
/// to [LiquidGlassAppearance.seaGlass]; its alpha is the color density, so
/// `SeaGlassColors.cobalt.withValues(alpha: .5)` is a paler cobalt.
abstract final class SeaGlassColors {
  /// Pale blue-green seafoam, the most common sea glass.
  static const Color seafoam = Color(0xFF8ED3B6);

  /// Deep bottle green.
  static const Color kellyGreen = Color(0xFF3E9B5E);

  /// Bright aqua turquoise.
  static const Color aqua = Color(0xFF2FB8DA);

  /// Soft cornflower blue.
  static const Color cornflower = Color(0xFFA4C3EC);

  /// Saturated cobalt blue.
  static const Color cobalt = Color(0xFF2E5AD2);

  /// Pale lavender, from manganese glass sun-tinted over time.
  static const Color lavender = Color(0xFFD8C7DB);

  /// Yellow-green citron.
  static const Color citron = Color(0xFFC4D37C);

  /// Honey amber.
  static const Color amber = Color(0xFFF2AE43);

  /// Bottle brown, the root beer of beach glass.
  static const Color brown = Color(0xFF94502A);

  /// Deep blue-green teal.
  static const Color teal = Color(0xFF1F7482);

  /// Faintly green white glass.
  static const Color white = Color(0xFFE9F0EC);

  /// Deep red ruby, from rare gold-tinted glass.
  static const Color ruby = Color(0xFFC8444A);

  /// Clear, almost colorless glass with the faintest cool cast.
  static const Color clear = Color(0xFFF4F8F7);

  /// Soft gray mist, smoky glass weathered pale.
  static const Color mist = Color(0xFFB9C4C2);

  /// All colors, in the order above.
  static const List<Color> values = [
    seafoam,
    kellyGreen,
    aqua,
    cornflower,
    cobalt,
    lavender,
    citron,
    amber,
    brown,
    teal,
    white,
    ruby,
    clear,
    mist,
  ];
}
