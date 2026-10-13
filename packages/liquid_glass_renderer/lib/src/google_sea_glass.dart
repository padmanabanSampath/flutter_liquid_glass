import 'package:equatable/equatable.dart';
import 'package:flutter/painting.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_appearance.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_settings.dart';

/// A set of glass colors from Google's Sea Glass concept, by the role each
/// plays in a UI.
///
/// The concept cycles through four sets: [original], [beach], [jewel] and
/// [frost]. Each color is the glass's body where it is thickest; the light
/// brightens its thin edges and the side away from it (see
/// [LiquidGlassSettings.googleSeaGlass]). Pass a color to
/// [LiquidGlassAppearance.seaGlass].
class GoogleSeaGlass with Equatable {
  /// Creates a glass set.
  const GoogleSeaGlass({
    required this.name,
    required this.primary,
    required this.secondary,
    required this.tertiary,
    required this.tertiaryLight,
    required this.positive,
    required this.positiveLight,
    required this.error,
    required this.accent,
    required this.avatar,
    required this.clear,
  });

  /// The name the concept gives this set, such as "Beach glass".
  final String name;

  /// Search bars and primary tiles: periwinkle in the original set.
  final Color primary;

  /// Media cards and toolbars: orchid in the original set.
  final Color secondary;

  /// Navigation bars: aqua in the original set.
  final Color tertiary;

  /// Sliders and other thin tertiary glass: a paler [tertiary].
  final Color tertiaryLight;

  /// Selected segments and switched-on states: mint green in the original
  /// set.
  final Color positive;

  /// Unselected segments next to [positive] ones.
  final Color positiveLight;

  /// Calls to action that create or warn: coral in the original set.
  final Color error;

  /// Floating action buttons and album art: amber in the original set.
  final Color accent;

  /// Avatars and the account button: rose in the original set.
  final Color avatar;

  /// Clear, milky glass: toggles, and chips that mark a selection on other
  /// glass.
  final Color clear;

  /// The concept's own glass: periwinkle, orchid, aqua, mint, coral and
  /// amber pastels.
  static const original = GoogleSeaGlass(
    name: 'Original glass',
    primary: Color(0xFFB9C2F3),
    secondary: Color(0xFFE9AFF4),
    tertiary: Color(0xFFA4E5E1),
    tertiaryLight: Color(0xFFCAF4F0),
    positive: Color(0xFFC2EFBB),
    positiveLight: Color(0xFFE2F4DD),
    error: Color(0xFFEFA4A4),
    accent: Color(0xFFF3D1A7),
    avatar: Color(0xFFECB1B2),
    clear: Color(0xFFEDEFF0),
  );

  /// Greens and sea blues, with a peach call to action and clear accents.
  static const beach = GoogleSeaGlass(
    name: 'Beach glass',
    primary: Color(0xFF9BA5F4),
    secondary: Color(0xFFADECAF),
    tertiary: Color(0xFF8FDDEE),
    tertiaryLight: Color(0xFFBDF7F8),
    positive: Color(0xFFBDF5D2),
    positiveLight: Color(0xFFDDF9EC),
    error: Color(0xFFECB596),
    accent: Color(0xFFF5F3EA),
    avatar: Color(0xFFEBB9A1),
    clear: Color(0xFFEDEFEF),
  );

  /// Deep, saturated violet, indigo, teal and red.
  static const jewel = GoogleSeaGlass(
    name: 'Jewel glass',
    primary: Color(0xFF8F9AF4),
    secondary: Color(0xFFCC9BF6),
    tertiary: Color(0xFF86D9E3),
    tertiaryLight: Color(0xFFB1F5F3),
    positive: Color(0xFFA6EFAC),
    positiveLight: Color(0xFFD1F8D3),
    error: Color(0xFFF1959C),
    accent: Color(0xFFF4C48F),
    avatar: Color(0xFFF09DA2),
    clear: Color(0xFFEDEFF0),
  );

  /// Frosted, nearly white glass with the faintest tints.
  static const frost = GoogleSeaGlass(
    name: 'Frost glass',
    primary: Color(0xFFE6ECF8),
    secondary: Color(0xFFF3EAF6),
    tertiary: Color(0xFFDDF4F2),
    tertiaryLight: Color(0xFFF4F9F8),
    positive: Color(0xFFEAF6E6),
    positiveLight: Color(0xFFF4F8F2),
    error: Color(0xFFF4E8E6),
    accent: Color(0xFFF6F2E6),
    avatar: Color(0xFFF4E9E6),
    clear: Color(0xFFF0F2F1),
  );

  /// The sets in the order the concept cycles through them.
  static const List<GoogleSeaGlass> values = [original, beach, jewel, frost];

  /// The sea glass appearance of [color], a color of this set.
  static LiquidGlassAppearance appearance(Color color) =>
      LiquidGlassAppearance.seaGlass(color: color);

  @override
  List<Object?> get props => [
    name,
    primary,
    secondary,
    tertiary,
    tertiaryLight,
    positive,
    positiveLight,
    error,
    accent,
    avatar,
    clear,
  ];
}
