import 'package:flutter/widgets.dart';
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
];

/// The frosted white glass of tiles that are switched off.
const frostedWhite = Color(0x59E9F0EC);

/// How tumbled the glass is.
enum ShowcaseFinish {
  clear('Clear', SeaGlassFinish.clear(), 1.5),
  tumbled('Tumbled', SeaGlassFinish(), 2),
  frosted('Frosted', SeaGlassFinish.frosted(), 6);

  const ShowcaseFinish(this.label, this.finish, this.frost);

  final String label;
  final SeaGlassFinish finish;
  final double frost;

  LiquidGlassSettings get settings =>
      LiquidGlassSettings.seaGlass(finish: finish, frost: frost);
}

/// How dense the body color is: vivid glass at full strength, or a pastel
/// that lets more light through.
enum ShowcaseTone {
  vivid('Vivid', 1),
  pastel('Pastel', .42);

  const ShowcaseTone(this.label, this.density);

  final String label;
  final double density;

  /// The sea glass body of [color] in this tone.
  Color body(Color color) => color.withValues(alpha: density);
}

/// Ink that reads on glass with body color [body]: dark on light glass,
/// white on deep glass.
Color inkOn(Color body) {
  final seen = Color.lerp(
    const Color(0xFFFFFFFF),
    body.withValues(alpha: 1),
    body.a,
  )!;
  return seen.computeLuminance() > .4
      ? const Color(0xFF173A40)
      : const Color(0xFFFFFFFF);
}

/// Ink for frosted glass and text directly on a wallpaper.
Color inkFor({required bool darkWallpaper}) =>
    darkWallpaper ? const Color(0xFFFFFFFF) : const Color(0xFF173A40);
