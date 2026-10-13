import 'dart:math' as math;

import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:meta/meta.dart';

/// Conservative pixel support of Flutter's Gaussian shadow mask.
///
/// [BoxShadow.blurRadius] is converted to sigma before rasterization. Reserving
/// only the radius clips the low-energy tail, especially when the shadow is
/// painted into a bounded saveLayer before translucent glass.
@internal
double glassShadowBlurSupport(double blurRadius) =>
    Shadow.convertRadiusToSigma(blurRadius) * 3;

/// The shadows [settings]' light makes a sea glass shape cast (see
/// [SeaGlassFinish.shadow]), their offsets in global logical pixels.
///
/// The shadow is a crisp copy of the shape, tinted by the glass, cast away
/// from the light: farther as the light comes down or moves aside. Small,
/// rounded pieces focus the light they let through into a glow of the
/// body's color at the near edge of their shadow; broad slabs do not.
/// [center] is the shape's center on screen, [size] its size and [body]
/// its body color. Empty without a cast shadow.
@internal
List<BoxShadow> seaGlassCastShadows(
  LiquidGlassSettings settings,
  Color body,
  Offset center,
  Size size,
) {
  final shortSide = size.shortestSide;
  final longSide = size.longestSide;
  final finish = settings.seaGlassFinish;
  final strength = finish.effectiveShadow;
  final elevation = finish.effectiveElevation;
  if (strength <= 0 || elevation <= 0) return const [];
  final light = settings.light;
  // Round pebbles are domes: they stand taller than flat slabs, so they
  // cast longer shadows, and they focus the light they let through into a
  // glow at the near edge of their shadow.
  final pebble =
      _smoothstep(0.6, 0.9, shortSide / math.max(longSide, 1e-3)) *
      (1 - _smoothstep(110, 180, shortSide));
  final rise = elevation * (1 + 0.45 * pebble);
  final Offset offset;
  if (light.position case final position?) {
    final height = light.effectiveHeight;
    final lift = math.min(rise, height * 0.5);
    offset = (center - position) * (lift / (height - lift));
  } else {
    // A far light about 32 degrees above the surface.
    offset = settings.lightDirection * (rise * 1.6);
  }
  // A dim light against bright ambient light casts a fainter shadow.
  final lightLevel = light.effectiveIntensity;
  final ambientLevel = light.ambient.computeLuminance();
  final presence = strength * lightLevel / (lightLevel + ambientLevel + 1e-3);
  if (presence <= 0) return const [];
  double deep(double channel) => math.pow(channel, 1.5).toDouble();
  final shade = Color.from(
    alpha: 0.5 * presence,
    red: deep(body.r) * 0.25,
    green: deep(body.g) * 0.25,
    blue: deep(body.b) * 0.25,
  );
  final reach = offset.distance;
  final shadows = [
    BoxShadow(
      color: shade,
      offset: offset,
      blurRadius: 3 + reach * 0.1,
    ),
  ];
  if (pebble > 0 && reach > 0) {
    shadows.add(
      BoxShadow(
        color: Color.from(
          alpha: 0.9 * pebble * presence,
          red: math.min(body.r * 1.05, 1),
          green: math.min(body.g * 1.05, 1),
          blue: math.min(body.b * 1.05, 1),
        ),
        offset: offset * 0.45,
        blurRadius: 3 + reach * 0.12,
        spreadRadius: -reach * 0.05,
      ),
    );
  }
  return shadows;
}

double _smoothstep(double edge0, double edge1, double x) {
  final t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
  return t * t * (3 - 2 * t);
}

/// [shadows] whose offsets are global vectors, mapped into the frame whose
/// transform to the screen is [toGlobal].
@internal
List<BoxShadow> localizeShadowOffsets(
  List<BoxShadow> shadows,
  Matrix4 toGlobal,
) {
  if (shadows.isEmpty) return shadows;
  final inverse = Matrix4.tryInvert(toGlobal);
  if (inverse == null) return const [];
  final origin = MatrixUtils.transformPoint(inverse, Offset.zero);
  return [
    for (final shadow in shadows)
      shadow.copyWith(
        offset: MatrixUtils.transformPoint(inverse, shadow.offset) - origin,
      ),
  ];
}

/// Paints [BoxShadow]s for a [LiquidShape] using canvas primitives
/// (drawRRect, drawCircle, drawRSuperellipse, etc.) instead of drawPath.
///
/// This avoids the cost of rasterizing an arbitrary [Path] with a blur
/// [MaskFilter], which is significantly slower than the dedicated GPU-
/// accelerated primitives that Impeller/Skia provide for simple shapes.
@internal
class GlassShadow extends SingleChildRenderObjectWidget {
  /// Creates a new [GlassShadow] widget with the given [shape], [shadows], and
  /// optional [child].
  const GlassShadow({
    required this.shape,
    required this.shadows,
    required this.settings,
    this.appearanceVisibility = 1,
    this.body = const Color(0x00000000),
    this.castsShadow = true,
    super.child,
    super.key,
  });

  /// The shape to paint shadows for.
  final LiquidShape shape;

  final LiquidGlassSettings settings;

  /// Per-shape materialization progress.
  final double appearanceVisibility;

  /// The sea glass body color that tints the shadow the light casts (see
  /// [seaGlassCastShadows]).
  final Color body;

  /// Whether to draw the shadow the light casts, as well as [shadows].
  final bool castsShadow;

  /// The list of shadows to paint.
  ///
  /// Only outer-equivalent shadows are supported; [BoxShadow.blurStyle] is
  /// ignored. When any shadow has a non-zero [BoxShadow.offset], the glass
  /// shape is cut out of the composed shadow stack so the shadow does not
  /// bleed through the translucent glass body.
  final List<BoxShadow> shadows;

  @override
  RenderObject createRenderObject(BuildContext context) {
    return _RenderGlassShadow(
      shape: shape,
      shadows: shadows,
      visibility: appearanceVisibility,
      settings: settings,
      body: body,
      castsShadow: castsShadow,
    );
  }

  @override
  void updateRenderObject(
    BuildContext context,
    // ignore: library_private_types_in_public_api
    _RenderGlassShadow renderObject,
  ) {
    renderObject
      ..shape = shape
      ..shadows = shadows
      ..visibility = appearanceVisibility
      ..settings = settings
      ..body = body
      ..castsShadow = castsShadow;
  }
}

class _RenderGlassShadow extends RenderProxyBox {
  _RenderGlassShadow({
    required this._shape,
    required this._shadows,
    required double visibility,
    required this._settings,
    required this._body,
    required this._castsShadow,
  }) : _visibility = visibility.clamp(0, 1);

  bool get castsShadow => _castsShadow;
  bool _castsShadow;
  set castsShadow(bool value) {
    if (_castsShadow == value) return;
    _castsShadow = value;
    markNeedsPaint();
  }

  LiquidGlassSettings get settings => _settings;
  LiquidGlassSettings _settings;
  set settings(LiquidGlassSettings value) {
    if (_settings == value) return;
    _settings = value;
    markNeedsPaint();
  }

  Color get body => _body;
  Color _body;
  set body(Color value) {
    if (_body == value) return;
    _body = value;
    markNeedsPaint();
  }

  /// The shadows the light casts, followed by the given ones.
  List<BoxShadow> get _allShadows {
    if (!castsShadow ||
        settings.seaGlassFinish.effectiveShadow <= 0 ||
        !attached ||
        !hasSize) {
      return shadows;
    }
    final toGlobal = getTransformTo(null);
    final center = MatrixUtils.transformPoint(
      toGlobal,
      size.center(Offset.zero),
    );
    return [
      ...localizeShadowOffsets(
        seaGlassCastShadows(settings, body, center, size),
        toGlobal,
      ),
      ...shadows,
    ];
  }

  LiquidShape get shape => _shape;
  LiquidShape _shape;
  set shape(LiquidShape value) {
    if (_shape == value) return;
    _shape = value;
    markNeedsPaint();
  }

  List<BoxShadow> get shadows => _shadows;
  List<BoxShadow> _shadows;
  set shadows(List<BoxShadow> value) {
    if (_shadows == value) return;
    _shadows = value;
    markNeedsPaint();
  }

  double get visibility => _visibility;
  double _visibility = 1;
  set visibility(double value) {
    if (_visibility == value) return;
    _visibility = value.clamp(0, 1);
    markNeedsPaint();
  }

  @override
  Rect get paintBounds {
    var bounds = super.paintBounds;
    final shadows = _allShadows;
    if (visibility <= 0 || shadows.isEmpty) return bounds;

    final shapeBounds = Offset.zero & size;
    for (final shadow in shadows) {
      // Report the same conservative Gaussian support used by paint()'s
      // saveLayer. Without this, Flutter culls the blurred pixels outside the
      // render box and different blur radii collapse to the same hard ring.
      final extent = math
          .max(
            shadow.spreadRadius +
                glassShadowBlurSupport(
                  shadow.blurRadius * visibility,
                ),
            0,
          )
          .toDouble();
      bounds = bounds.expandToInclude(
        shapeBounds.shift(shadow.offset).inflate(extent),
      );
    }
    return bounds;
  }

  @override
  void paint(PaintingContext context, Offset offset) {
    // Hidden glass casts nothing; skip the layer its cutout would need.
    final shadows = visibility > 0 ? _allShadows : const <BoxShadow>[];
    if (shadows.isNotEmpty) {
      final rect = offset & size;
      final canvas = context.canvas;

      final needsCutout = shadows.any((s) => s.offset != Offset.zero);

      if (needsCutout) {
        var layerBounds = rect;
        for (final shadow in shadows) {
          layerBounds = layerBounds.expandToInclude(
            rect
                .shift(shadow.offset)
                .inflate(
                  shadow.spreadRadius +
                      glassShadowBlurSupport(
                        shadow.blurRadius * visibility,
                      ),
                ),
          );
        }
        canvas.saveLayer(layerBounds, Paint());
      }

      for (final shadow in shadows) {
        final shadowRect = rect
            .shift(shadow.offset)
            .inflate(shadow.spreadRadius);
        final paint = shadow
            .copyWith(
              blurRadius: shadow.blurRadius * visibility,
              blurStyle: needsCutout ? BlurStyle.normal : BlurStyle.outer,
              color: shadow.color.withValues(
                alpha: shadow.color.a * visibility,
              ),
            )
            .toPaint();

        _drawShape(canvas, shadowRect, paint);
      }

      if (needsCutout) {
        _drawShape(
          canvas,
          // A wobbling outline sinks inward; the shadow reaches under it.
          rect.deflate(.5 + settings.seaGlassFinish.effectiveWobble),
          Paint()..blendMode = BlendMode.dstOut,
        );
        canvas.restore();
      }
    }

    super.paint(context, offset);
  }

  void _drawShape(Canvas canvas, Rect rect, Paint paint) {
    switch (shape) {
      case LiquidRoundedSuperellipse(:final borderRadius):
        canvas.drawRSuperellipse(
          RSuperellipse.fromRectAndRadius(
            rect,
            Radius.circular(borderRadius),
          ),
          paint,
        );
      case LiquidOval():
        canvas.drawOval(rect, paint);
      case LiquidBlob():
        canvas.drawPath(shape.getOuterPath(rect), paint);
      case LiquidRoundedRectangle(:final borderRadius):
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            rect,
            Radius.circular(borderRadius),
          ),
          paint,
        );
    }
  }
}
