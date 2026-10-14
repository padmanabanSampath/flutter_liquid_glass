import 'package:equatable/equatable.dart';
import 'package:flutter/painting.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_appearance.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_settings.dart';

/// A mineral tint for mineral sea glass (see
/// [LiquidGlassSettings.mineralSeaGlass]): the color the minerals in the
/// glass give it.
///
/// [swatch] is the tint as a swatch shows it; [body] the glass's body as it
/// shows over a light surface, paler and greyer, as frosted glass of that
/// tint looks. Give [body] to [LiquidGlassAppearance.seaGlass], or use
/// [appearance].
class MineralTint with Equatable {
  /// Creates a mineral tint.
  const MineralTint({
    required this.name,
    required this.swatch,
    required this.body,
  });

  /// The name of the tint, such as "Seafoam Jade".
  final String name;

  /// The tint as a swatch shows it.
  final Color swatch;

  /// The glass's body color.
  final Color body;

  /// Green glass from iron and chromium.
  static const seafoamJade = MineralTint(
    name: 'Seafoam Jade',
    swatch: Color(0xFF62EAB5),
    body: Color(0xFF98E2CC),
  );

  /// Pale blue glass from copper.
  static const ceruleanAqua = MineralTint(
    name: 'Cerulean Aqua',
    swatch: Color(0xFF4AB6D4),
    body: Color(0xFF8ED6EE),
  );

  /// Honey glass from iron and sulfur.
  static const amber = MineralTint(
    name: 'Amber',
    swatch: Color(0xFFD3A745),
    body: Color(0xFFD6C08F),
  );

  /// Nearly colorless glass, frosted white.
  static const frostedQuartz = MineralTint(
    name: 'Frosted Quartz',
    swatch: Color(0xFFD1D5D7),
    body: Color(0xFFDCDCE0),
  );

  /// Deep blue glass from cobalt.
  static const cobalt = MineralTint(
    name: 'Cobalt',
    swatch: Color(0xFF4365D0),
    body: Color(0xFF90A1D8),
  );

  /// Brown glass from iron and manganese, like bottle glass.
  static const driftwoodBrown = MineralTint(
    name: 'Driftwood Brown',
    swatch: Color(0xFF996545),
    body: Color(0xFFB9A69A),
  );

  /// Every tint, in swatch order.
  static const List<MineralTint> values = [
    seafoamJade,
    ceruleanAqua,
    amber,
    frostedQuartz,
    cobalt,
    driftwoodBrown,
  ];

  /// The sea glass appearance of glass of this tint.
  LiquidGlassAppearance get appearance =>
      LiquidGlassAppearance.seaGlass(color: body);

  @override
  List<Object?> get props => [name, swatch, body];
}
