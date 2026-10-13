import 'dart:math' as math;

import 'package:flutter/painting.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';

/// Every sea glass color, in the order of the palette menu.
const seaGlassPalette = [
  (name: 'Seafoam', color: SeaGlassColors.seafoam),
  (name: 'Aqua', color: SeaGlassColors.aqua),
  (name: 'Teal', color: SeaGlassColors.teal),
  (name: 'Cobalt', color: SeaGlassColors.cobalt),
  (name: 'Cornflower', color: SeaGlassColors.cornflower),
  (name: 'Lavender', color: SeaGlassColors.lavender),
  (name: 'White', color: SeaGlassColors.white),
  (name: 'Citron', color: SeaGlassColors.citron),
  (name: 'Kelly green', color: SeaGlassColors.kellyGreen),
  (name: 'Amber', color: SeaGlassColors.amber),
  (name: 'Brown', color: SeaGlassColors.brown),
  (name: 'Ruby', color: SeaGlassColors.ruby),
  (name: 'Clear', color: SeaGlassColors.clear),
  (name: 'Mist', color: SeaGlassColors.mist),
];

/// The frosted white glass of tiles that are switched off.
const frostedWhite = Color(0x59E9F0EC);

/// Which glass the showcase is made of: our tumbled sea glass, or slab
/// glass, the thick, milky pastel slabs of the Sea Glass design concept
/// (LiquidGlassSettings.slabSeaGlass), flecked like terrazzo, glowing on
/// the side away from the light and casting soft shadows.
///
/// Clear slab is the same pastel slab glass made translucent: it scatters
/// little and its lens is wider, so it shows and bends what is beneath, as
/// our sea glass does, while keeping the slab's glow and grain.
enum ShowcaseMaterial {
  seaGlass('Sea glass'),
  slab('Slab'),
  clearSlab('Clear slab');

  const ShowcaseMaterial(this.label);

  final String label;

  /// The frosted glass of switched-off tiles and panels: frosted white sea
  /// glass, or the milky clear slab of the concept.
  Color get frost => switch (this) {
    seaGlass => frostedWhite,
    slab => _slabClear,
    clearSlab => _clearSlabClear,
  };

  /// Whether this is one of the slab glasses.
  bool get isSlab => this != seaGlass;

  /// The glass after this one, for the quick toggle.
  ShowcaseMaterial get next => values[(index + 1) % values.length];

  /// [color] as this glass shows it: as it is for sea glass; for slab
  /// glass, the concept's clean pastel of the same hue, light and with
  /// enough color to read as candy rather than grey. Near-colorless glass
  /// stays near-colorless.
  Color tint(Color color) {
    if (!isSlab) return color;
    final hsl = HSLColor.fromColor(color.withValues(alpha: 1));
    final saturation = hsl.saturation < .12
        ? hsl.saturation
        : math.max(hsl.saturation, .72);
    return hsl
        .withSaturation(saturation.clamp(0.0, 1.0))
        .withLightness(.8)
        .toColor();
  }

  /// How much of the showcase's transparency this glass takes: slab glass
  /// is milky by nature, so it clears only a little.
  double get clarityShare => this == slab ? .3 : 1;

  static const _slabClear = Color(0xFFEDEFF0);
  static const _clearSlabClear = Color(0x66EDEFF0);
}

/// The slab finish the showcase uses, after the concept's sheet: an
/// almost smooth face with only a fine, sugary grain, a soft shadow that
/// starts at the slab's edge, and outlines whose hand-cut undulations show.
final slabFinish = const SeaGlassFinish.slab().copyWith(
  flecks: .35,
  elevation: 22,
  wobble: 2.5,
);

/// The part a component plays, which picks its slab glass color: each
/// component of the concept's sheet has its own (see SlabSeaGlass).
enum SlabRole {
  /// Prompt bars and primary controls: periwinkle.
  primary,

  /// Toolbars, tab bars and scenes: orchid.
  secondary,

  /// Active devices, hours and levels: aqua.
  tertiary,

  /// Selected chips and toggles: mint.
  positive,

  /// The main action: coral.
  action;

  /// This role's color in [set].
  Color of(SlabSeaGlass set) => switch (this) {
    primary => set.primary,
    secondary => set.secondary,
    tertiary => set.tertiary,
    positive => set.positive,
    action => set.error,
  };
}

/// How tumbled the glass is.
///
/// A fine satin frost, between the renderer's presets and clear glass: the
/// presets' coarse grain, strong sparkle and deep pitting read as a woven,
/// carpet-like texture on wide UI surfaces, while without them the glass
/// loses its sea glass character and reads as plain frosted glass. So the
/// grain is finer and softer, the sparkle a faint glint, and the pitting
/// still breaks the backdrop a little. Frosted keeps its frost narrower and
/// scatters less, so the frost's scattered samples stay fine and the
/// backdrop glows through.
enum ShowcaseFinish {
  clear(
    'Clear',
    SeaGlassFinish(
      depth: 14,
      scatter: .06,
      rimGlow: .3,
      grain: .22,
      grainSize: .9,
      sparkle: .1,
      pitting: .3,
      gloss: .85,
    ),
    1.5,
  ),
  tumbled(
    'Tumbled',
    SeaGlassFinish(
      scatter: .16,
      rimGlow: .42,
      grain: .3,
      grainSize: .9,
      sparkle: .12,
      pitting: .45,
    ),
    2.5,
  ),
  frosted(
    'Frosted',
    SeaGlassFinish(
      depth: 22,
      scatter: .32,
      rimGlow: .5,
      grain: .34,
      grainSize: 1,
      sparkle: .14,
      pitting: .7,
      gloss: .35,
    ),
    4.5,
  );

  const ShowcaseFinish(this.label, this.finish, this.frost);

  final String label;
  final SeaGlassFinish finish;
  final double frost;

  /// This finish, lit as [shine] with the light from [lightAngle], with
  /// its colored glass made [clarity] clearer, from `0` (as tumbled) to `1`.
  ///
  /// Clarity takes the milkiness out of colored glass only
  /// ([SeaGlassFinish.colorClarity]): it scatters, frosts and pits the
  /// backdrop less, so it shows and bends what is beneath while keeping its
  /// color. Frosted white glass beside it stays frosted.
  LiquidGlassSettings settings({
    ShowcaseShine shine = ShowcaseShine.satin,
    double lightAngle = 0,
    double clarity = 0,
    double extraFrost = 0,
  }) => LiquidGlassSettings.seaGlass(
    finish: finish.copyWith(
      gloss: shine.gloss(finish),
      colorClarity: clarity,
    ),
    frost: frost + extraFrost,
    refractionHeight: 36,
    refractionAmount: 16,
  ).copyWith(lightAngle: lightAngle);
}

/// The outline of the round sea glass buttons.
enum ShowcaseShape {
  round('Round'),
  pebble('Pebble'),
  alive('Alive');

  const ShowcaseShape(this.label);

  final String label;

  /// The outline of the button [seed] at [time], in seconds: an oval, a
  /// tumbled pebble of its own, or a living pebble that flows and breathes.
  LiquidShape outline(int seed, double time) => switch (this) {
    round => const LiquidOval(),
    pebble => LiquidBlob.pebble(seed: seed, irregularity: .8),
    alive => LiquidBlob.alive(time: time, seed: seed),
  };
}

/// How the glass catches the light.
///
/// Both are flat, frosted sea glass with a shallow lens at the rim, so
/// tiles and bars read as tumbled glass. The renderer's 100 pt dome makes
/// capsules read as tubes; only the tide pool's living pebbles offer it.
enum ShowcaseShine {
  satin('Satin'),
  glossy('Glossy');

  const ShowcaseShine(this.label);

  final String label;

  /// The gloss of [finish] in this shine: none for satin, a soft glint for
  /// glossy.
  double gloss(SeaGlassFinish finish) => switch (this) {
    satin => 0,
    glossy => finish.gloss * .4,
  };
}

/// The lens of the tide pool's living pebbles, and of nothing else.
enum PebbleLens {
  flat('Flat'),
  dome('Dome'),
  prism('Prism');

  const PebbleLens(this.label);

  final String label;

  /// [settings] with this lens: unchanged when flat; a 100 pt dome that bows
  /// the backdrop like a magnifying glass, with the finish's gloss; and the
  /// dome with rainbow edges, bending blue more as real glass does.
  LiquidGlassSettings apply(
    LiquidGlassSettings settings,
    SeaGlassFinish finish,
  ) => switch (this) {
    flat => settings,
    dome || prism => settings.copyWith(
      refractionHeight: 100,
      refractionAmount: 30,
      dispersion: this == prism ? -.3 : 0,
      seaGlassFinish: settings.seaGlassFinish.copyWith(gloss: finish.gloss),
    ),
  };
}

/// A stable seed for the pebble outline of the control named [label].
int seedOf(String label) =>
    label.codeUnits.fold(7, (hash, unit) => (hash * 31 + unit) & 0xFFFF);

/// How dense the body color is: vivid glass at full strength, or a pastel
/// that lets more light through.
enum ShowcaseTone {
  vivid('Vivid', 1),
  pastel('Pastel', .42);

  const ShowcaseTone(this.label, this.density);

  final String label;
  final double density;

  /// The body of [color] in this tone, at color [strength], in
  /// [material]: slab glass turns it to the concept's pastel.
  Color body(
    Color color, {
    double strength = defaultColorStrength,
    ShowcaseMaterial material = ShowcaseMaterial.seaGlass,
  }) => seaGlassBody(
    material.tint(color),
    density: density,
    strength: strength,
  );
}

/// How clear colored glass is by default: clear enough to show and bend
/// what is beneath it, with its color kept.
const defaultClarity = .5;

/// The blur, in logical pixels, the Frost slider adds at its top: enough
/// to soften a photo behind white glass into color, as frosted glass does.
const maxExtraFrost = 9.0;

/// The color strength that leaves sea glass colors as sampled.
const defaultColorStrength = .6;

/// The body of colored sea glass of [color] at [density], at color
/// [strength] from `0` to `1`.
///
/// [defaultColorStrength] keeps the color as it is. Above it the glass is
/// richer: more saturated and a little deeper. Below it the glass holds
/// less color, so it reads paler and lets more of the backdrop's own color
/// through. Only colored glass takes it; frosted white glass keeps its
/// own body.
Color seaGlassBody(
  Color color, {
  required double density,
  double strength = defaultColorStrength,
}) {
  final richer =
      ((strength - defaultColorStrength) / (1 - defaultColorStrength)).clamp(
        0.0,
        1.0,
      );
  final softer = ((defaultColorStrength - strength) / defaultColorStrength)
      .clamp(0.0, 1.0);
  final hsl = HSLColor.fromColor(color.withValues(alpha: 1));
  return hsl
      .withSaturation((hsl.saturation * (1 + .5 * richer)).clamp(0.0, 1.0))
      .withLightness((hsl.lightness * (1 - .14 * richer)).clamp(0.0, 1.0))
      .toColor()
      .withValues(alpha: density * (1 - .6 * softer));
}

/// Ink that reads on glass with body color [body]: dark on light glass,
/// white on deep glass.
Color inkOn(Color body) {
  final seen = Color.lerp(
    const Color(0xFFFFFFFF),
    body.withValues(alpha: 1),
    body.a,
  )!;
  if (seen.computeLuminance() <= .4) return const Color(0xFFFFFFFF);
  // On light glass, the deep tone of the glass's own color: dark red on
  // pale red, deep blue on pale blue, as colored glass prints its labels.
  final hsl = HSLColor.fromColor(body.withValues(alpha: 1));
  return hsl
      .withLightness(.24)
      .withSaturation((hsl.saturation * .9).clamp(0.0, 1.0))
      .toColor();
}

/// Ink for frosted glass and text directly on a wallpaper.
Color inkFor({required bool darkWallpaper}) =>
    darkWallpaper ? const Color(0xFFFFFFFF) : const Color(0xFF173A40);

/// A soft halo behind text of [ink], in the opposite tone, so the text
/// stays legible over photos and over clear glass: a dark glow behind
/// light ink, a light one behind dark ink.
///
/// A tight halo outlines each letter; a wide one lifts the text off busy
/// detail such as rocks or leaves.
List<Shadow> haloFor(Color ink) {
  final light = ink.computeLuminance() > .5;
  return light
      ? const [
          Shadow(color: Color(0xA6000000), blurRadius: 3),
          Shadow(color: Color(0x73000000), blurRadius: 14),
        ]
      : const [
          Shadow(color: Color(0xD9FFFFFF), blurRadius: 3),
          Shadow(color: Color(0x99FFFFFF), blurRadius: 14),
        ];
}
