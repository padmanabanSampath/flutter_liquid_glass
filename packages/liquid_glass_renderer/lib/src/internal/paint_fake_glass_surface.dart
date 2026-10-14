import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_shaders/flutter_shaders.dart';
import 'package:liquid_glass_renderer/liquid_glass_renderer.dart';
import 'package:liquid_glass_renderer/src/internal/fake_glass_blend.dart';
import 'package:liquid_glass_renderer/src/internal/rounded_superellipse_parameters.dart';
import 'package:liquid_glass_renderer/src/liquid_glass_settings.dart';
import 'package:meta/meta.dart';

/// Extra logical pixels required for the portion of the contour that lies
/// outside the shape's nominal bounds.
double fakeGlassSurfaceOutset(LiquidGlassSettings settings) {
  return math
      .max(
        settings.contourWidth + 1,
        0,
      )
      .toDouble();
}

const _noSuperellipse = <double>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// WORKAROUND for an Impeller/Metal coverage bug on macOS, pending an
/// upstream fix (repro project: `impeller_pixel_aligned_coverage`, kept
/// outside this repo for the Flutter issue): a
/// shader-filled draw whose quad edge lands on a half-integer device-pixel
/// boundary renders its alpha ramp quantized to whole pixels near that
/// edge — the analytic silhouette stair-steps. Integer-aligned quad edges
/// render smoothly.
///
/// The drawn rect only bounds where the shader runs — the shape is placed
/// by `FlutterFragCoord`, so growing the quad changes nothing visible.
/// Expanding each edge out to the nearest whole device pixel keeps the
/// raster phase off the bad half-pixel grid at any position.
///
/// [transform] is the draw's local-to-device transform
/// (`canvas.getTransform()`); only axis-aligned transforms are snapped —
/// anything else passes through unchanged.
///
/// The snap is computed at paint time. Compositor-only motion (a retained
/// `OffsetLayer` moving this subtree without a repaint) can land the baked
/// quad back on the bad phase; such moves are transient and the surface
/// realigns on the next repaint.
Rect fakeGlassSurfaceQuad(Rect localQuad, Float64List transform) {
  final s = transform;
  // Anything but an axis-aligned scale+translate: nothing sane to snap.
  if (s[1] != 0 ||
      s[2] != 0 ||
      s[3] != 0 ||
      s[4] != 0 ||
      s[6] != 0 ||
      s[7] != 0 ||
      s[11] != 0 ||
      s[14] != 0 ||
      s[0] == 0 ||
      s[5] == 0) {
    return localQuad;
  }
  double axis(
    double localMin,
    double localMax,
    double scale,
    double shift,
  ) {
    final a = scale * localMin + shift;
    final b = scale * localMax + shift;
    return (math.min(a, b).floor() - shift) / scale;
  }

  double axisMax(
    double localMin,
    double localMax,
    double scale,
    double shift,
  ) {
    final a = scale * localMin + shift;
    final b = scale * localMax + shift;
    return (math.max(a, b).ceil() - shift) / scale;
  }

  return Rect.fromLTRB(
    axis(localQuad.left, localQuad.right, s[0], s[12]),
    axis(localQuad.top, localQuad.bottom, s[5], s[13]),
    axisMax(localQuad.left, localQuad.right, s[0], s[12]),
    axisMax(localQuad.top, localQuad.bottom, s[5], s[13]),
  );
}

/// The surface tint FakeGlass washes over [appearance]'s face.
Color _surfaceTint(
  LiquidGlassAppearance appearance,
  LiquidGlassSettings settings,
) {
  final surfaceTint = appearance.colorModel.approximateSurfaceTint(
    appearance.tint,
  );
  // Under sea glass the tint is the body, which the backdrop filter applies,
  // so the wash fades out with the finish's strength.
  return surfaceTint.withValues(
    alpha:
        surfaceTint.a *
        appearance.visibility.clamp(0.0, 1.0) *
        (1 - settings.seaGlassFinish.effectiveStrength),
  );
}

/// Where [settings]' light hangs in [renderObject]'s own logical
/// coordinates, for [paintFakeGlassSurface]; `null` for a far light or when
/// the slab shading is off.
@internal
Offset? fakeGlassLightPosition(
  LiquidGlassSettings settings,
  RenderBox renderObject,
) {
  final position = settings.light.position;
  if (position == null ||
      settings.seaGlassFinish.effectiveShading <= 0 ||
      !renderObject.attached) {
    return null;
  }
  return renderObject.globalToLocal(position);
}

/// The slab shading's light (see uSeaGlassLight in fake_glass_surface.frag):
/// a positioned light relative to the surface's center.
List<double> _lightUniforms(
  GlassLight light,
  Offset? position,
  Size size, {
  required double colorClarity,
  required double glitter,
}) {
  final intensity = light.effectiveIntensity;
  final center = size.center(Offset.zero);
  return [
    if (position != null) ...[
      position.dx - center.dx,
      position.dy - center.dy,
      light.effectiveHeight,
      light.reach ?? 0,
    ] else ...[
      0,
      0,
      0,
      0,
    ],
    light.color.r * intensity,
    light.color.g * intensity,
    light.color.b * intensity,
    // The light color's fourth float: SeaGlassFinish.colorClarity.
    colorClarity,
    light.ambient.r,
    light.ambient.g,
    light.ambient.b,
    // The ambient light's fourth float: SeaGlassFinish.glitter.
    glitter,
  ];
}

/// The members of a blend group one FakeGlass surface draws the union of.
@internal
typedef FakeGlassSurfaceGroup = ({
  /// Blend radius in layer pixels.
  double blend,

  /// The member this surface belongs to.
  FakeGlassBlendMember self,

  /// [self] and its nearest neighbors, in widget order (see
  /// fakeGlassNeighbors).
  List<FakeGlassBlendMember> members,
});

/// Paints one analytic FakeGlass surface with the same logical-pixel setting
/// contract used by RealGlass.
///
/// With a [group], the surface draws the group's smooth union where its
/// shape is the nearest member, so bridges to its neighbors appear and the
/// merged glass is drawn once.
///
/// [lightPosition] is where a positioned [GlassLight] hangs, in the shape's
/// own logical coordinates (see [fakeGlassLightPosition]).
void paintFakeGlassSurface(
  Canvas canvas, {
  required ui.FragmentShader shader,
  required Size size,
  required LiquidShape shape,
  required LiquidGlassSettings settings,
  required LiquidGlassAppearance appearance,
  required double devicePixelRatio,
  bool exteriorOnly = false,
  FakeGlassSurfaceGroup? group,
  Offset? lightPosition,
}) {
  final appearanceVisibility = appearance.visibility.clamp(0.0, 1.0);
  final seaGlass = settings.seaGlassFinish;
  final tint = _surfaceTint(appearance, settings);
  final faceEmission =
      appearance.colorModel
          .faceTransfer(
            size.shortestSide,
            tintAmount: settings.effectiveTintAmount,
          )
          ?.emission ??
      const Color(0x00000000);
  final shapeType = switch (shape) {
    LiquidOval() => 0.0,
    LiquidRoundedRectangle() => 1.0,
    LiquidRoundedSuperellipse() => 2.0,
    LiquidBlob() => 3.0,
  };
  final cornerRadius = switch (shape) {
    LiquidOval() => 0.0,
    LiquidRoundedRectangle(:final borderRadius) => borderRadius,
    LiquidRoundedSuperellipse(:final borderRadius) => borderRadius,
    LiquidBlob() => 0.0,
  };
  final opticalThickness = settings.effectiveEdgeDistanceRange;
  // The engine copies a fragment shader's uniforms when a draw records it,
  // so one shader serves both passes.
  void writeUniforms({required bool thicknessPass}) =>
      shader.setFloatUniforms((uniforms) {
        uniforms
          ..setSize(size)
          ..setFloats([shapeType, cornerRadius])
          ..setColor(tint)
          ..setFloats([
            settings.highlight * appearanceVisibility,
            GlassRim.highlightWidth,
            opticalThickness,
            GlassRim.highlightWrap,
            GlassRim.highlightOppositeStrength * appearanceVisibility,
            settings.contourStrength *
                appearance.colorModel.contourScale(
                  size.shortestSide,
                  settings.effectiveTintAmount,
                ) *
                appearanceVisibility,
            settings.contourWidth,
            0, // contour transmittance
            0, // contour offset
            settings.contourDirectionality,
            settings.bevelShadowStrength * appearanceVisibility,
            GlassRim.bevelShadowDepth,
            GlassRim.bevelShadowOffset,
            GlassRim.bevelShadowDirectionality,
            0, // bevel shadow size response
          ])
          ..setOffset(settings.lightDirection)
          ..setFloat(1 / math.max(devicePixelRatio, 0.01))
          ..setFloat(exteriorOnly ? 1 : 0)
          ..setFloat(appearance.colorModel.fakeGlintLuminance)
          ..setFloats([
            faceEmission.r * appearanceVisibility,
            faceEmission.g * appearanceVisibility,
            faceEmission.b * appearanceVisibility,
          ])
          ..setFloats(
            switch (shape) {
              LiquidRoundedSuperellipse() => roundedSuperellipseParameters(
                size,
                cornerRadius,
              ),
              // A blob's waves ride in the superellipse's slots.
              LiquidBlob() => shape.shaderParameters,
              _ => _noSuperellipse,
            },
          )
          ..setFloat(appearance.colorModel.fakeGlintFaceGain)
          ..setFloat(appearance.colorModel.fakeGlintVibrancy)
          ..setFloats([
            seaGlass.effectiveStrength * appearanceVisibility,
            seaGlass.effectiveDepth,
            seaGlass.effectiveScatter,
            seaGlass.effectiveRimGlow,
            seaGlass.effectiveGrain,
            seaGlass.effectiveGrainSize,
            seaGlass.effectiveSparkle,
            if (thicknessPass) 1 else 0,
            appearance.tint.r,
            appearance.tint.g,
            appearance.tint.b,
            appearance.tint.a,
          ])
          ..setFloats(_groupUniforms(group, settings))
          ..setFloats([
            seaGlass.effectiveGloss,
            seaGlass.effectiveShading,
            seaGlass.effectiveGlow,
            seaGlass.effectiveFlecks,
          ])
          ..setFloats(
            _lightUniforms(
              settings.light,
              lightPosition,
              size,
              colorClarity: settings.seaGlassFinish.effectiveColorClarity,
              glitter: settings.seaGlassFinish.effectiveGlitter,
            ),
          );
      });
  final contourOutset = fakeGlassSurfaceOutset(settings);
  // A member also draws its half of the bridges, which reach up to half the
  // blend radius beyond its box, in its own (possibly scaled) frame.
  final bridgeOutset = group == null
      ? 0.0
      : group.blend / math.max(group.self.distanceScale, 1e-3);
  final quad = fakeGlassSurfaceQuad(
    (Offset.zero & size).inflate(contourOutset + bridgeOutset),
    canvas.getTransform(),
  );
  if (seaGlass.isEnabled && !exteriorOnly && appearanceVisibility > 0) {
    // The backdrop filter applied the face of the thin rim; multiplying by
    // the thicker face's share deepens the body toward the center.
    writeUniforms(thicknessPass: true);
    canvas.drawRect(
      quad,
      Paint()
        ..shader = shader
        ..blendMode = BlendMode.multiply,
    );
  }
  writeUniforms(thicknessPass: false);
  canvas.drawRect(quad, Paint()..shader = shader);
}

/// The blend group uniforms of fake_glass_surface.frag: uGroup, then four
/// slots each of uMemberFrame, uMemberPlace, uMemberShape, uMemberRse (three
/// per slot), uMemberTint and uMemberBody. Zero without a group.
List<double> _groupUniforms(
  FakeGlassSurfaceGroup? group,
  LiquidGlassSettings settings,
) {
  const slots = fakeGlassGroupMembers;
  if (group == null || group.members.length < 2) {
    return List<double>.filled(4 + slots * (4 + 4 + 4 + 12 + 4 + 4), 0);
  }
  final members = group.members.take(slots).toList();
  final self = group.self;
  final frames = [for (final member in members) self.frameTo(member)];
  // Every slot of one uniform array, unused slots zero.
  List<double> each(List<double> Function(int slot) values, int width) => [
    for (var slot = 0; slot < slots; slot++)
      ...(slot < members.length ? values(slot) : List<double>.filled(width, 0)),
  ];
  return [
    group.blend,
    members.length.toDouble(),
    members.indexOf(self).toDouble(),
    self.distanceScale,
    ...each((slot) => frames[slot].sublist(0, 4), 4),
    ...each(
      (slot) => [
        frames[slot][4],
        frames[slot][5],
        members[slot].distanceScale,
        0,
      ],
      4,
    ),
    ...each(
      (slot) => [
        members[slot].type,
        members[slot].size.width / 2,
        members[slot].size.height / 2,
        members[slot].radius,
      ],
      4,
    ),
    ...each((slot) => members[slot].superellipse, 12),
    ...each((slot) {
      final tint = _surfaceTint(members[slot].appearance, settings);
      return [tint.r, tint.g, tint.b, tint.a];
    }, 4),
    ...each((slot) {
      final body = members[slot].appearance.tint;
      return [body.r, body.g, body.b, body.a];
    }, 4),
  ];
}
