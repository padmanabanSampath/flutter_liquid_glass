import 'dart:math' as math;

import 'package:equatable/equatable.dart';
import 'package:flutter/material.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:meta/meta.dart';

/// Represents a shape that can be used by a [LiquidGlass] widget.
sealed class LiquidShape extends OutlinedBorder with Equatable {
  const LiquidShape({super.side = BorderSide.none});

  @protected
  OutlinedBorder get _equivalentOutlinedBorder;

  @override
  Path getInnerPath(Rect rect, {TextDirection? textDirection}) {
    return _equivalentOutlinedBorder.getInnerPath(
      rect,
      textDirection: textDirection,
    );
  }

  @override
  Path getOuterPath(Rect rect, {TextDirection? textDirection}) {
    return _equivalentOutlinedBorder.getOuterPath(
      rect,
      textDirection: textDirection,
    );
  }

  @override
  void paint(Canvas canvas, Rect rect, {TextDirection? textDirection}) {
    _equivalentOutlinedBorder.paint(canvas, rect, textDirection: textDirection);
  }

  @override
  List<Object?> get props => [side];
}

/// Represents a squircle shape that can be used by a [LiquidGlass] widget.
///
/// Works like a [RoundedSuperellipseBorder].
class LiquidRoundedSuperellipse extends LiquidShape {
  /// Creates a new [LiquidRoundedSuperellipse] with the given [borderRadius].
  const LiquidRoundedSuperellipse({
    required this.borderRadius,
    super.side = BorderSide.none,
  });

  /// The radius of the squircle.
  ///
  /// This is the radius of the corners of the squircle.
  final double borderRadius;

  @override
  OutlinedBorder get _equivalentOutlinedBorder => RoundedSuperellipseBorder(
    borderRadius: BorderRadius.all(Radius.circular(borderRadius)),
    side: side,
  );

  @override
  LiquidRoundedSuperellipse copyWith({
    BorderSide? side,
    double? borderRadius,
  }) {
    return LiquidRoundedSuperellipse(
      side: side ?? this.side,
      borderRadius: borderRadius ?? this.borderRadius,
    );
  }

  @override
  ShapeBorder scale(double t) {
    return LiquidRoundedSuperellipse(
      borderRadius: borderRadius * t,
      side: side.scale(t),
    );
  }

  @override
  List<Object?> get props => [...super.props, borderRadius];
}

/// Represents an ellipse shape that can be used by a [LiquidGlass] widget.
///
/// Works like an [OvalBorder].
class LiquidOval extends LiquidShape {
  /// Creates a new [LiquidOval] with the given [side].
  const LiquidOval({super.side = BorderSide.none});

  @override
  OutlinedBorder get _equivalentOutlinedBorder => const OvalBorder();

  @override
  OutlinedBorder copyWith({BorderSide? side}) {
    return LiquidOval(
      side: side ?? this.side,
    );
  }

  @override
  ShapeBorder scale(double t) {
    return LiquidOval(
      side: side.scale(t),
    );
  }
}

/// Represents a rounded rectangle shape that can be used by a [LiquidGlass]
/// widget.
///
/// Works like a [RoundedRectangleBorder].
class LiquidRoundedRectangle extends LiquidShape {
  /// Creates a new [LiquidRoundedRectangle] with the given [borderRadius].
  const LiquidRoundedRectangle({
    required this.borderRadius,
    super.side = BorderSide.none,
  });

  /// The radius of the rounded rectangle.
  ///
  /// This is the radius of the corners of the rounded rectangle.
  final double borderRadius;

  @override
  OutlinedBorder get _equivalentOutlinedBorder => RoundedRectangleBorder(
    borderRadius: BorderRadius.all(Radius.circular(borderRadius)),
    side: side,
  );

  @override
  LiquidRoundedRectangle copyWith({
    BorderSide? side,
    double? borderRadius,
  }) {
    return LiquidRoundedRectangle(
      side: side ?? this.side,
      borderRadius: borderRadius ?? this.borderRadius,
    );
  }

  @override
  ShapeBorder scale(double t) {
    return LiquidRoundedRectangle(
      borderRadius: borderRadius * t,
      side: side.scale(t),
    );
  }

  @override
  List<Object?> get props => [...super.props, borderRadius];
}

/// A wave running around the outline of a [LiquidBlob].
@immutable
class LiquidWave with Equatable {
  /// Creates a wave of [count] crests, each [amplitude] high, placed by
  /// [phase].
  const LiquidWave({
    required this.count,
    required this.amplitude,
    this.phase = 0,
  });

  /// How many crests run around the outline. `2` stretches the outline into
  /// an oval, `3` to `5` give a pebble its irregular, lumpy outline, and
  /// larger counts scallop it.
  final int count;

  /// How far the crests reach, as a share of the radius. Around `0.02` to
  /// `0.1` reads as a natural pebble; the waves of one blob are kept below
  /// `0.6` in all.
  final double amplitude;

  /// Where the crests sit, in radians. Turning it makes the outline flow
  /// around the shape.
  final double phase;

  @override
  List<Object?> get props => [count, amplitude, phase];
}

/// An organic outline that can be used by a [LiquidGlass] widget: the
/// ellipse that fills the box, its radius rippled by [waves].
///
/// Sea glass is rarely a perfect oval. A few low waves give a tumbled
/// pebble ([LiquidBlob.pebble]); many even ones scallop it
/// ([LiquidBlob.cookie]). Change a wave's [LiquidWave.phase] or
/// [LiquidWave.amplitude] every frame and the glass lives: its outline flows
/// and breathes, and refraction, lighting and blending follow it. Only the
/// geometry around a changing shape is drawn again.
///
/// The outline stays within its box. At most [maxWaves] waves are drawn.
class LiquidBlob extends LiquidShape {
  /// Creates a blob rippled by [waves].
  const LiquidBlob({this.waves = const [], super.side = BorderSide.none});

  /// A scalloped, cookie-like outline of [lobes] round crests, [depth] deep,
  /// turned by [phase].
  factory LiquidBlob.cookie({
    int lobes = 10,
    double depth = 0.05,
    double phase = 0,
  }) => LiquidBlob(
    waves: [LiquidWave(count: lobes, amplitude: depth, phase: phase)],
  );

  /// A tumbled pebble: an irregular, lumpy oval, different for every
  /// [seed]. [irregularity] scales how far it strays from the oval.
  factory LiquidBlob.pebble({int seed = 0, double irregularity = 1}) {
    final random = math.Random(seed);
    double phase() => random.nextDouble() * 2 * math.pi;
    return LiquidBlob(
      waves: [
        LiquidWave(count: 2, amplitude: 0.05 * irregularity, phase: phase()),
        LiquidWave(count: 3, amplitude: 0.04 * irregularity, phase: phase()),
        LiquidWave(count: 5, amplitude: 0.012 * irregularity, phase: phase()),
      ],
    );
  }

  /// A living pebble at [time], in seconds: its waves turn at different
  /// speeds and swell and ebb, so the outline flows and breathes. Drive
  /// [time] from a ticker; the same [seed] keeps the same pebble.
  factory LiquidBlob.alive({required double time, int seed = 0}) {
    final random = math.Random(seed);
    double phase() => random.nextDouble() * 2 * math.pi;
    return LiquidBlob(
      waves: [
        LiquidWave(
          count: 2,
          amplitude: 0.06 + 0.02 * math.sin(0.9 * time),
          phase: phase() + 0.35 * time,
        ),
        LiquidWave(
          count: 3,
          amplitude: 0.05 + 0.02 * math.sin(1.3 * time + 1),
          phase: phase() - 0.5 * time,
        ),
        LiquidWave(
          count: 5,
          amplitude: 0.02 + 0.01 * math.sin(1.7 * time + 2),
          phase: phase() + 0.8 * time,
        ),
      ],
    );
  }

  /// Most waves a blob is drawn with.
  static const int maxWaves = 3;

  /// The waves rippling the outline; only the first [maxWaves] are drawn.
  final List<LiquidWave> waves;

  /// [waves] as drawn: at most [maxWaves], their amplitudes scaled down to
  /// `0.6` in all so the outline never folds over.
  List<LiquidWave> get effectiveWaves {
    final drawn = waves.take(maxWaves).toList();
    final total = drawn.fold<double>(
      0,
      (sum, wave) => sum + wave.amplitude.abs(),
    );
    if (total <= 0.6) return drawn;
    return [
      for (final wave in drawn)
        LiquidWave(
          count: wave.count,
          amplitude: wave.amplitude * 0.6 / total,
          phase: wave.phase,
        ),
    ];
  }

  /// The radius of the outline at [angle] (radians, clockwise from the
  /// x-axis) for a box of [halfSize], and its derivative with respect to
  /// the angle.
  ///
  /// The waves are normalized so the outline reaches the box at most.
  /// Mirrors `blobRadius` in sdf.glsl and fake_glass_shape.glsl.
  @internal
  (double, double) radiusAt(double angle, Size halfSize) {
    final a = math.max(halfSize.width, 1e-4);
    final b = math.max(halfSize.height, 1e-4);
    final c = math.cos(angle);
    final s = math.sin(angle);
    final denominator = math.max(b * b * c * c + a * a * s * s, 1e-12);
    final ellipse = a * b / math.sqrt(denominator);
    final ellipseSlope =
        -ellipse *
        ellipse *
        ellipse *
        (a * a - b * b) *
        s *
        c /
        (a * a * b * b);
    var ripple = 1.0;
    var rippleSlope = 0.0;
    var norm = 1.0;
    for (final wave in effectiveWaves) {
      final x = wave.count * angle + wave.phase;
      ripple += wave.amplitude * math.cos(x);
      rippleSlope -= wave.amplitude * wave.count * math.sin(x);
      norm += wave.amplitude.abs();
    }
    return (
      ellipse * ripple / norm,
      (ellipseSlope * ripple + ellipse * rippleSlope) / norm,
    );
  }

  /// The waves as the shaders read them, twelve floats: count, amplitude and
  /// phase of each of [maxWaves] waves, then the normalization.
  @internal
  List<double> get shaderParameters {
    final drawn = effectiveWaves;
    final values = <double>[];
    var norm = 1.0;
    for (var i = 0; i < maxWaves; i++) {
      if (i < drawn.length) {
        values.addAll([
          drawn[i].count.toDouble(),
          drawn[i].amplitude,
          drawn[i].phase,
        ]);
        norm += drawn[i].amplitude.abs();
      } else {
        values.addAll([0, 0, 0]);
      }
    }
    return [...values, norm, 0, 0];
  }

  @override
  OutlinedBorder get _equivalentOutlinedBorder => const OvalBorder();

  @override
  Path getOuterPath(Rect rect, {TextDirection? textDirection}) {
    final halfSize = Size(rect.width / 2, rect.height / 2);
    final center = rect.center;
    // Enough points that each crest stays round.
    final crests = effectiveWaves.fold(
      0,
      (most, wave) => math.max(most, wave.count),
    );
    final steps = math.max(96, crests * 16);
    final points = <Offset>[];
    for (var i = 0; i < steps; i++) {
      final angle = 2 * math.pi * i / steps;
      final (radius, _) = radiusAt(angle, halfSize);
      points.add(
        center + Offset(math.cos(angle) * radius, math.sin(angle) * radius),
      );
    }
    return Path()..addPolygon(points, true);
  }

  @override
  Path getInnerPath(Rect rect, {TextDirection? textDirection}) =>
      getOuterPath(rect.deflate(side.strokeInset));

  @override
  void paint(Canvas canvas, Rect rect, {TextDirection? textDirection}) {
    if (side.style == BorderStyle.none || side.width == 0) return;
    canvas.drawPath(getOuterPath(rect), side.toPaint());
  }

  @override
  LiquidBlob copyWith({BorderSide? side, List<LiquidWave>? waves}) =>
      LiquidBlob(side: side ?? this.side, waves: waves ?? this.waves);

  @override
  ShapeBorder scale(double t) => LiquidBlob(side: side.scale(t), waves: waves);

  @override
  List<Object?> get props => [...super.props, waves];
}
