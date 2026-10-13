import 'dart:math' as math;

import 'package:equatable/equatable.dart';
import 'package:flutter/painting.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_settings.dart';
import 'package:liquid_glass_renderer/src/sea_glass.dart';

/// The light that shades sea glass slabs and casts their shadows (see
/// [SeaGlassFinish.shading] and [SeaGlassFinish.shadow]).
///
/// Without a [position] the light is far away and shines along
/// [LiquidGlassSettings.lightAngle], like daylight: every shape is lit from
/// the same side. With one it hangs at that point, [height] above the
/// glass, like a lamp or a flashlight: each shape is lit from its own
/// direction, and shadows fan out away from the light and lengthen as it
/// comes down. Moving the light only changes uniforms; it never re-renders
/// the geometry.
///
/// [color] times [intensity] is the light; [ambient] reaches every shape
/// wherever the light is, as the room's light does. The defaults are white
/// light and no ambient, which leaves the glass colors as given.
class GlassLight with Equatable {
  /// Creates a light.
  const GlassLight({
    this.position,
    this.height = 600,
    this.color = const Color(0xFFFFFFFF),
    this.intensity = 1,
    this.ambient = const Color(0xFF000000),
    this.reach,
  });

  /// Restores a light produced by [toJson].
  factory GlassLight.fromJson(Object? json) {
    if (json is! Map<String, Object?>) return const GlassLight();
    const defaults = GlassLight();
    double number(String key, double fallback) =>
        (json[key] as num?)?.toDouble() ?? fallback;
    final position = json['position'];
    return GlassLight(
      position: position is List && position.length == 2
          ? Offset(
              (position[0] as num).toDouble(),
              (position[1] as num).toDouble(),
            )
          : null,
      height: number('height', defaults.height),
      color: Color((json['color'] as num?)?.toInt() ?? 0xFFFFFFFF),
      intensity: number('intensity', defaults.intensity),
      ambient: Color((json['ambient'] as num?)?.toInt() ?? 0xFF000000),
      reach: (json['reach'] as num?)?.toDouble(),
    );
  }

  /// Where the light hangs, in global logical pixels (the screen), or `null`
  /// for a far light along [LiquidGlassSettings.lightAngle].
  ///
  /// Global, because a light belongs to the room: glass that scrolls past
  /// it is lit differently as it moves. Feed it a pointer's
  /// `globalPosition` to aim it.
  final Offset? position;

  /// How high above the glass a positioned light hangs, in logical pixels.
  ///
  /// Lower lights cast longer shadows and shade a shape more from the side;
  /// a shape right below the light is lit evenly.
  final double height;

  /// The color of the light.
  final Color color;

  /// How bright the light is; `1` leaves the glass colors as given.
  final double intensity;

  /// Light that reaches every shape regardless of [position], such as a dim
  /// room's. Its alpha is ignored; darker colors are dimmer. Black, the
  /// default, adds none.
  final Color ambient;

  /// The radius, in logical pixels, over which a positioned light falls to
  /// half its brightness, as a flashlight's beam does; `null` lights
  /// everything equally.
  final double? reach;

  /// [height], never below one logical pixel.
  double get effectiveHeight => math.max(1, height);

  /// [intensity]; never negative.
  double get effectiveIntensity => math.max(0, intensity);

  /// Returns a copy with the supplied fields replaced. Pass
  /// [clearPosition] to make it a far light again, or [clearReach] to light
  /// everything equally.
  GlassLight copyWith({
    Offset? position,
    bool clearPosition = false,
    double? height,
    Color? color,
    double? intensity,
    Color? ambient,
    double? reach,
    bool clearReach = false,
  }) => GlassLight(
    position: clearPosition ? null : position ?? this.position,
    height: height ?? this.height,
    color: color ?? this.color,
    intensity: intensity ?? this.intensity,
    ambient: ambient ?? this.ambient,
    reach: clearReach ? null : reach ?? this.reach,
  );

  /// Serializes the light for presets and tooling.
  Map<String, Object> toJson() => {
    if (position case final position?) 'position': [position.dx, position.dy],
    'height': height,
    'color': color.toARGB32(),
    'intensity': intensity,
    'ambient': ambient.toARGB32(),
    if (reach case final reach?) 'reach': reach,
  };

  @override
  List<Object?> get props => [
    position,
    height,
    color,
    intensity,
    ambient,
    reach,
  ];
}
