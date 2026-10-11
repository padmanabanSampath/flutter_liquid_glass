import 'package:flutter/cupertino.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer_example/loupe/liquid_glass_loupe.dart';

/// The glass styles the playground starts from.
///
/// Each style is a pair of layer-wide [LiquidGlassSettings] and a per-shape
/// [LiquidGlassAppearance]. Regular and toolbar glass follow the appearance
/// brightness; clear glass, the loupe and sea glass are identical in light
/// and dark.
enum GlassStyle {
  regular('Regular'),
  toolbar('Toolbar'),
  clear('Clear'),
  loupe('Loupe'),
  seaGlass('Sea Glass');

  const GlassStyle(this.label);

  final String label;

  /// Whether the style has distinct light and dark variants.
  bool get followsBrightness => this == regular || this == toolbar;

  /// Whether the Liquid Glass slider changes the style. The iOS 27 loupe and
  /// sea glass are identical at every slider position.
  bool get followsSlider => this != loupe && this != seaGlass;

  LiquidGlassSettings settings({
    required Brightness brightness,
    required double tintAmount,
  }) => switch (this) {
    regular => LiquidGlassSettings(
      tintAmount: tintAmount,
      frost: LiquidGlassSettings.ios27RegularFrost(tintAmount),
    ),
    toolbar => LiquidGlassSettings.ios27Toolbar(
      brightness: brightness,
      tintAmount: tintAmount,
    ),
    clear => LiquidGlassSettings.ios27Clear(tintAmount: tintAmount),
    loupe => loupeSettings,
    seaGlass => LiquidGlassSettings.seaGlass(),
  };

  LiquidGlassAppearance appearance(Brightness brightness) => switch (this) {
    regular => LiquidGlassAppearance.ios27Regular(brightness: brightness),
    toolbar => LiquidGlassAppearance.ios27Toolbar(brightness: brightness),
    clear => const LiquidGlassAppearance.ios27Clear(),
    loupe => const LiquidGlassAppearance(),
    seaGlass => seaGlassAppearance,
  };
}

/// Frosted white sea glass: a pale, light body so the backdrop stays
/// readable through it.
const seaGlassAppearance = LiquidGlassAppearance.seaGlass(
  color: Color(0x8CE9F0EC),
);

/// The glass of the iOS 27 text loupe: a clear, unfrosted lens with a narrow
/// bevel.
const loupeSettings = LiquidGlassLoupe.defaultSettings;
