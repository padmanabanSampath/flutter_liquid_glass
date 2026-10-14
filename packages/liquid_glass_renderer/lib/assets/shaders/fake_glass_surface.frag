// Copyright 2025, Tim Lehmann for whynotmake.it

#version 460 core
precision mediump float;

#include <flutter/runtime_effect.glsl>

uniform vec2 uSize;
uniform float uShapeType;
uniform float uCornerRadius;
uniform vec4 uTint;
uniform float uHighlight;
uniform float uHighlightWidth;
uniform float uThickness;
uniform float uHighlightWrap;
uniform float uOppositeHighlight;
uniform float uContourStrength;
uniform float uContourWidth;
uniform float uContourTransmittance;
uniform float uContourOffset;
uniform float uContourDirectionality;
uniform float uBevelStrength;
uniform float uBevelDepth;
uniform float uBevelOffset;
uniform float uBevelDirectionality;
uniform float uBevelSizeResponse;
uniform vec2 uLightDirection;
// Logical size of one physical pixel.
uniform float uPixelSize;
// 1 when drawing only the border ring outside the shape clip. The clip path is
// the silhouette there; the analytic SDF may approximate it by a pixel.
uniform float uExteriorOnly;
// Luminance of the neutral glint target: FakeGlass cannot scale it by the
// face it cannot see, so clear glass uses its best constant.
uniform float uGlintLuminance;
// Premultiplied emission of the face the backdrop filter produced, so the
// inner shadow can shade only the transmitted part of that face.
uniform vec3 uFaceEmission;
// Rounded superellipse parameters (see roundedSuperellipseParameters).
uniform vec4 uRseDegreeAndSpans;
uniform vec4 uRseCircleCenters;
uniform vec4 uRseSemiAxisAndRadii;
// Face contribution to the glint target (see the color model's
// glintFaceGain and glintVibrancy). FakeGlass approximates the face with
// the surface tint plus emission; clear glass brightens that proxy
// strongly while regular glass brightens mostly the transmitted chroma.
uniform float uGlintFaceGain;
uniform float uGlintVibrancy;
// Sea glass finish (SeaGlassFinish), in logical pixels. The backdrop
// filter applies the face of the thin rim. A thickness pass, drawn with a
// multiply blend before the surface, deepens it toward the center as the
// final render shader does; the surface then adds the rim glow, grain and
// sparkles.
//   uSeaGlassBody:    strength, depth, scatter, rimGlow
//   uSeaGlassSurface: grain, grainSize, sparkle, 1 for the thickness pass
//   uSeaGlassColor:   body color, density
uniform vec4 uSeaGlassBody;
uniform vec4 uSeaGlassSurface;
uniform vec4 uSeaGlassColor;
// A blend group (see fake_glass_blend.dart): the members this surface
// blends with, itself included, in widget order. The surface draws their
// smooth union where it is the nearest member, so the group is drawn once.
//   uGroup:          blend radius in layer pixels, member count (0 without a
//                    group), this surface's slot, its distance scale
//   uMemberFrame[i]: map from this surface's centered frame to the
//                    member's, row by row
//   uMemberPlace[i]: offset of that map, the member's distance scale to
//                    layer pixels
//   uMemberShape[i]: type, half size, corner radius
//   uMemberRse:      the members' rounded superellipse parameters, three
//                    each
//   uMemberTint[i]:  surface tint
//   uMemberBody[i]:  sea glass body color and density
uniform vec4 uGroup;
uniform vec4 uMemberFrame[4];
uniform vec4 uMemberPlace[4];
uniform vec4 uMemberShape[4];
uniform vec4 uMemberRse[12];
uniform vec4 uMemberTint[4];
uniform vec4 uMemberBody[4];
// Sea glass gloss, slab shading, glow and flecks (SeaGlassFinish.gloss,
// shading, glow, flecks).
uniform vec4 uSeaGlassGloss;
// The light that shades sea glass slabs (GlassLight), in this surface's
// centered frame and logical pixels:
//   uSeaGlassLight:      position (xy) and height (z) of a positioned light,
//                        and the reach of its falloff (w, 0 for none); a
//                        zero height is a far light along uLightDirection
//   uSeaGlassLightColor: the light's color times its intensity
//   uSeaGlassAmbient:    the ambient light (rgb), and the glitter of a deep
//                        etch (w, SeaGlassFinish.glitter)
uniform vec4 uSeaGlassLight;
uniform vec4 uSeaGlassLightColor;
uniform vec4 uSeaGlassAmbient;

// Tint and body color of this pixel: the surface's own, or blended toward
// the next member's across a group's seam.
vec4 gTint;
vec4 gSeaGlassColor;

const vec3 LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);

layout(location = 0) out vec4 fragColor;

#include "fake_glass_shape.glsl"
#include "sea_glass.glsl"

// The sea glass face over white at a thickness, as the final render shader
// computes it: the body filters the light and the frost scatters part of
// it in the body's color, more at the grazing rim.
vec3 seaGlassWhiteFace(
  float thickness,
  vec3 bodyColor,
  float density,
  vec2 sizeResponse
) {
  vec3 transmittance = pow(
      bodyColor,
      vec3(density * sizeResponse.x * mix(0.3, 1.0, thickness))
  );
  float scatter = clamp(
      uSeaGlassBody.z * sizeResponse.y * mix(1.6, 1.0, thickness),
      0.0,
      1.0
  );
  // Colored glass clears (SeaGlassFinish.colorClarity).
  scatter *= 1.0 - 0.7 * seaGlassColorClear(
      gSeaGlassColor,
      clamp(uSeaGlassLightColor.w, 0.0, 1.0)
  );
  vec3 glow = mix(vec3(1.0), transmittance, 0.6);
  return transmittance * (1.0 - scatter) + glow * scatter;
}

// How much the face at this thickness keeps of the rim face the backdrop
// filter applied: below one toward the thicker center, per channel, for a
// multiply blend.
vec3 seaGlassThicknessRatio(float inward) {
  float thickness = seaGlassThickness(inward, uSeaGlassBody.y);
  vec2 sizeResponse = seaGlassSizeResponse(
      0.5 * min(uSize.x, uSize.y),
      uSeaGlassBody.y
  );
  float density = clamp(gSeaGlassColor.a, 0.0, 1.0);
  vec3 bodyColor = clamp(gSeaGlassColor.rgb, vec3(0.02), vec3(1.0));
  vec3 ratio = seaGlassWhiteFace(thickness, bodyColor, density, sizeResponse) /
      max(
        seaGlassWhiteFace(0.0, bodyColor, density, sizeResponse),
        vec3(1e-3)
      );
  return mix(vec3(1.0), min(ratio, vec3(1.0)), clamp(uSeaGlassBody.x, 0.0, 1.0));
}

// How brightly the pebble's dome catches the light, as the final render
// shader's seaGlassGloss computes it, in logical pixels.
float seaGlassGlossLobe(vec3 surface, vec3 toward, float sharpness) {
  vec3 halfway = normalize(toward + vec3(0.0, 0.0, 1.0));
  float level = pow(halfway.z, sharpness);
  float lobe = pow(max(dot(surface, halfway), 0.0), sharpness);
  return max(lobe - level, 0.0) / (1.0 - level);
}

float seaGlassGloss(float inward, vec2 normal) {
  float x = clamp(inward / max(uSeaGlassBody.y, 0.001), 0.0, 1.0);
  vec3 surface = normalize(vec3(normal * 1.2 * (1.0 - x), 1.0));
  vec2 lightAxis = normalize(uLightDirection + vec2(0.00001)) * 0.75;
  vec3 toLight = normalize(vec3(-lightAxis, 0.66));
  vec3 reflected = normalize(vec3(lightAxis, 0.66));
  float glint = seaGlassGlossLobe(surface, toLight, 48.0);
  float caustic = 0.35 * seaGlassGlossLobe(surface, reflected, 16.0);
  float outline = 0.3 * (1.0 - smoothstep(0.0, 1.5 * uPixelSize, inward));
  return glint + caustic + outline;
}

// The thick milky slab's own face, premultiplied, which covers the face the
// backdrop filter produced as much as the glass scatters, as the final
// render shader's slab shading does.
vec4 seaGlassSlab(vec2 position, float inward, vec2 normal, float scatter) {
  // From the surface's center, as the final render shader lights a shape.
  // Explicitly highp: this file is mediump, where the squared length of a
  // light more than about 256 px away overflows half precision (65504).
  highp vec2 toward =
      uSeaGlassLight.z > 0.0 ? uSeaGlassLight.xy : -uLightDirection;
  highp float distance = max(length(toward), 1e-6);
  toward /= distance;
  // A positioned light right above the surface lights it evenly.
  float tilt = uSeaGlassLight.z > 0.0
      ? clamp(distance / (0.5 * uSeaGlassLight.z), 0.0, 1.0)
      : 1.0;
  vec2 shade = slabShadeDirection(toward, uSeaGlassLight.z > 0.0);
  float depth = max(uSeaGlassBody.y, 0.001);
  vec2 shading = slabShading(
      position / max(uSize * 0.5, vec2(1.0)),
      shade,
      dot(normal, shade),
      inward / depth,
      tilt,
      clamp(uSeaGlassGloss.z, 0.0, 2.0)
  );
  float density = clamp(gSeaGlassColor.a, 0.0, 1.0);
  vec3 body = mix(
      vec3(1.0),
      clamp(gSeaGlassColor.rgb, vec3(0.02), vec3(1.0)),
      density
  );
  vec3 lit = slabColor(body, shading);
  float falloff = 1.0;
  if (uSeaGlassLight.z > 0.0 && uSeaGlassLight.w > 0.0) {
    vec2 d = (position - uSeaGlassLight.xy) / uSeaGlassLight.w;
    falloff = 1.0 / (1.0 + dot(d, d));
  }
  lit *= uSeaGlassAmbient.rgb + uSeaGlassLightColor.rgb * falloff;
  float strength = clamp(uSeaGlassGloss.y, 0.0, 1.0);
  float cover = scatter * strength *
      mix(1.0, slabEdgeScatter(inward, depth), strength);
  // FakeGlass stays SDR.
  return vec4(min(lit, vec3(1.0)) * cover, cover);
}

// The varying part of the sea glass face, premultiplied, to composite over
// the face the backdrop filter and the thickness pass produced.
vec4 seaGlassOverlay(vec2 position, float inward, float facing, vec2 normal) {
  float strength = clamp(uSeaGlassBody.x, 0.0, 1.0);
  float thickness = seaGlassThickness(inward, uSeaGlassBody.y);
  vec2 sizeResponse = seaGlassSizeResponse(
      0.5 * min(uSize.x, uSize.y),
      uSeaGlassBody.y
  );
  float density = clamp(gSeaGlassColor.a, 0.0, 1.0) * sizeResponse.x;
  vec3 bodyColor = clamp(gSeaGlassColor.rgb, vec3(0.02), vec3(1.0));
  vec3 pastel = mix(vec3(1.0), pow(bodyColor, vec3(0.4 * density)), 0.85);
  float rimLight = mix(0.4, 1.0, smoothstep(-1.0, 1.0, facing));
  float rimGlow = clamp(uSeaGlassBody.w, 0.0, 1.0) * (1.0 - thickness) *
      rimLight * 0.6;
  vec4 overlay = vec4(pastel * rimGlow, rimGlow);
  if (uSeaGlassGloss.y > 0.0) {
    float scatter = clamp(
        uSeaGlassBody.z * sizeResponse.y * mix(1.6, 1.0, thickness),
        0.0,
        1.0
    ) * (1.0 - 0.7 * seaGlassColorClear(
        gSeaGlassColor,
        clamp(uSeaGlassLightColor.w, 0.0, 1.0)
    ));
    overlay = overlay + seaGlassSlab(position, inward, normal, scatter) *
        (1.0 - overlay.a);
  }

  highp float grainSize = max(uSeaGlassSurface.y, 0.25);
  float grain = seaGlassPits(position, grainSize).z *
      clamp(uSeaGlassSurface.x, 0.0, 1.0) * 0.36;
  // Positive grain lightens and negative grain darkens the face.
  vec4 speck = grain > 0.0
      ? vec4(vec3(grain), grain)
      : vec4(vec3(0.0), -grain);
  overlay = speck + overlay * (1.0 - speck.a);

  float fleckAmount = clamp(uSeaGlassGloss.w, 0.0, 1.0);
  if (fleckAmount > 0.0) {
    // The overlay cannot read the face below it, so the dark chips take the
    // slab's deeper tone of the body, as the final render shader's do.
    // Inside the glass, as the final render shader's: soft edges, veiled
    // by the milky body and fading toward the thin rim.
    vec2 flecks = seaGlassFlecks(
        position,
        grainSize,
        max(uPixelSize, grainSize * 0.35)
    ) *
        fleckAmount *
        seaGlassFleckDepth(thickness, uSeaGlassBody.z);
    vec3 body = mix(
        vec3(1.0),
        bodyColor,
        clamp(gSeaGlassColor.a, 0.0, 1.0)
    );
    vec3 dark = seaGlassFlecked(body, vec2(1.0, 0.0));
    overlay = vec4(dark * flecks.x, flecks.x) + overlay * (1.0 - flecks.x);
    float light = 0.28 * flecks.y;
    overlay = vec4(vec3(light), light) + overlay * (1.0 - light);
  }

  float glitterAmount = clamp(uSeaGlassAmbient.w, 0.0, 1.0);
  if (glitterAmount > 0.0) {
    vec2 glitter = seaGlassGlitter(position, grainSize * 1.6, uPixelSize) *
        glitterAmount;
    vec3 body = mix(
        vec3(1.0),
        bodyColor,
        clamp(gSeaGlassColor.a, 0.0, 1.0)
    );
    overlay = vec4(body * 0.88 * glitter.x, glitter.x) +
        overlay * (1.0 - glitter.x);
    float light = 0.36 * glitter.y;
    overlay = vec4(vec3(light), light) + overlay * (1.0 - light);
  }

  float sparkle = seaGlassSparkle(
      position,
      grainSize,
      uPixelSize,
      normalize(uLightDirection + vec2(0.00001))
  ) *
      clamp(uSeaGlassSurface.z, 0.0, 1.0);
  overlay = vec4(vec3(sparkle), sparkle) + overlay * (1.0 - sparkle);
  float gloss = clamp(
    seaGlassGloss(inward, normal) * clamp(uSeaGlassGloss.x, 0.0, 1.0),
    0.0,
    1.0
  );
  overlay = vec4(vec3(gloss), gloss) + overlay * (1.0 - gloss);
  return overlay * strength;
}

float shapeDistance(vec2 p) {
  return sdFakeGlassShape(
    uShapeType,
    p,
    uSize * 0.5,
    uCornerRadius,
    uRseDegreeAndSpans,
    uRseCircleCenters,
    uRseSemiAxisAndRadii
  );
}

vec2 fakeShapeNormal(
  float type,
  vec2 p,
  vec2 halfSize,
  float cornerRadius,
  vec4 waves0,
  vec4 waves1,
  vec4 waves2
) {
  if (type > 2.5) {
    return fakeBlobNormal(p, halfSize, waves0, waves1, waves2);
  }
  if (type < 0.5) {
    vec2 radius = max(halfSize, vec2(0.001));
    return normalize(p / (radius * radius) + vec2(0.00001));
  }
  float radius = clamp(cornerRadius, 0.0, min(halfSize.x, halfSize.y));
  vec2 q = abs(p) - halfSize + vec2(radius);
  vec2 corner = max(q, 0.0);
  if (dot(corner, corner) > 0.0001) {
    return normalize(corner) * sign(p);
  }
  return q.x > q.y ? vec2(sign(p.x), 0.0) : vec2(0.0, sign(p.y));
}

vec2 shapeNormal(vec2 p) {
  return fakeShapeNormal(
    uShapeType,
    p,
    uSize * 0.5,
    uCornerRadius,
    uRseDegreeAndSpans,
    uRseCircleCenters,
    uRseSemiAxisAndRadii
  );
}

// Signed distance in layer pixels and outward normal, in this surface's
// frame, of a group member at p in this surface's centered frame. The
// member's uniforms are passed in: SkSL indexes uniform arrays only with
// loop indices.
float groupMemberSample(
  vec4 frame,
  vec4 place,
  vec4 shape,
  vec4 rse0,
  vec4 rse1,
  vec4 rse2,
  vec2 p,
  out vec2 normal
) {
  vec2 q = vec2(
    frame.x * p.x + frame.y * p.y,
    frame.z * p.x + frame.w * p.y
  ) + place.xy;
  float distance = sdFakeGlassShape(shape.x, q, shape.yz, shape.w, rse0, rse1, rse2);
  vec2 n = fakeShapeNormal(shape.x, q, shape.yz, shape.w, rse0, rse1, rse2);
  // Gradients map back with the transpose.
  vec2 back = vec2(
    frame.x * n.x + frame.z * n.y,
    frame.y * n.x + frame.w * n.y
  );
  float length = max(sqrt(dot(back, back)), 1e-6);
  normal = back / length;
  return distance * place.z;
}

// Integral of the border ramp from the silhouette to outward distance t, all
// in physical pixels. See contourIntegral in the final render shader.
float contourIntegral(float t, float offset, float width) {
  float u = clamp(t - offset, 0.0, width);
  return u - u * u / (2.0 * width);
}

void main() {
  vec2 position = FlutterFragCoord().xy - uSize * 0.5;
  float distance = shapeDistance(position);
  gTint = uTint;
  gSeaGlassColor = uSeaGlassColor;
  vec2 groupNormal = vec2(0.0);
  if (uGroup.y > 0.5) {
    // The smooth union of the group, in widget order and layer pixels, as
    // RealGlass blends a group (smoothUnionSample in sdf.glsl). The
    // surface draws only where it is the nearest member.
    float blend = uGroup.x;
    float unionDistance = 0.0;
    vec2 unionNormal = vec2(0.0);
    float nearest = 1e9;
    float second = 1e9;
    float nearestSlot = -1.0;
    vec4 secondTint = uTint;
    vec4 secondBody = uSeaGlassColor;
    for (int i = 0; i < 4; i++) {
      if (float(i) >= uGroup.y) break;
      vec2 normal;
      float memberDistance = groupMemberSample(
        uMemberFrame[i],
        uMemberPlace[i],
        uMemberShape[i],
        uMemberRse[3 * i],
        uMemberRse[3 * i + 1],
        uMemberRse[3 * i + 2],
        position,
        normal
      );
      if (i == 0) {
        unionDistance = memberDistance;
        unionNormal = normal;
      } else {
        float radius = blend * min(0.5 * length(unionNormal - normal), 1.0);
        float e = max(radius - abs(unionDistance - memberDistance), 0.0);
        float weight = clamp(
          0.5 + (memberDistance - unionDistance) / (2.0 * max(radius, 1e-4)),
          0.0,
          1.0
        );
        unionDistance = min(unionDistance, memberDistance) -
            e * e * 0.25 / max(radius, 1e-4);
        unionNormal = mix(normal, unionNormal, weight);
      }
      if (memberDistance < nearest) {
        if (nearestSlot >= 0.0) {
          second = nearest;
          secondTint = gTint;
          secondBody = gSeaGlassColor;
        }
        nearest = memberDistance;
        nearestSlot = float(i);
        gTint = uMemberTint[i];
        gSeaGlassColor = uMemberBody[i];
      } else if (memberDistance < second) {
        second = memberDistance;
        secondTint = uMemberTint[i];
        secondBody = uMemberBody[i];
      }
    }
    if (abs(nearestSlot - uGroup.z) > 0.5) {
      fragColor = vec4(0.0);
      return;
    }
    // Across the seam the colors meet halfway, from either side.
    float share = clamp(
      0.5 + (second - nearest) / (2.0 * max(blend, 1e-4)),
      0.5,
      1.0
    );
    gTint = mix(secondTint, gTint, share);
    gSeaGlassColor = mix(secondBody, gSeaGlassColor, share);
    distance = unionDistance / max(uGroup.w, 1e-4);
    groupNormal = unionNormal;
  }
  float inward = max(-distance, 0.0);
  if (uSeaGlassSurface.w > 0.5) {
    // The thickness pass: drawn with a multiply blend, so outside the
    // shape it is transparent and leaves the backdrop alone.
    float coverage = clamp(0.5 - distance / uPixelSize, 0.0, 1.0);
    fragColor = vec4(seaGlassThicknessRatio(inward) * coverage, coverage);
    return;
  }

  vec2 normal = uGroup.y > 0.5
      ? normalize(groupNormal + vec2(0.00001))
      : shapeNormal(position);
  vec2 lightDirection = normalize(uLightDirection + vec2(0.00001));
  float facing = dot(normal, -lightDirection);

  // Same SDF-space lighting contract as RealGlass (see
  // liquid_glass_final_render_core.glsl): a one-sided exterior border and a
  // glint line anchored at the silhouette.
  float tangency = abs(dot(normal, vec2(-lightDirection.y, lightDirection.x)));
  // The border is box-filtered over one physical pixel, split at the
  // silhouette, so a sub-pixel border does not alias into a dotted line.
  float outwardPixels = distance / uPixelSize;
  float contourOffsetPixels = uContourOffset / uPixelSize;
  float contourWidthPixels = uContourWidth / uPixelSize;
  vec2 contourBand = vec2(0.0);
  if (uContourWidth > 0.0) {
    float lower = outwardPixels - 0.5;
    float upper = outwardPixels + 0.5;
    contourBand = vec2(
      contourIntegral(max(upper, 0.0), contourOffsetPixels, contourWidthPixels) -
          contourIntegral(max(lower, 0.0), contourOffsetPixels, contourWidthPixels),
      contourIntegral(min(upper, 0.0), contourOffsetPixels, contourWidthPixels) -
          contourIntegral(min(lower, 0.0), contourOffsetPixels, contourWidthPixels)
    );
  }
  float contourStrength = clamp(uContourStrength, 0.0, 1.0) *
      mix(1.0, tangency, clamp(uContourDirectionality, 0.0, 1.0));
  float silhouetteCoverage = clamp(0.5 - outwardPixels, 0.0, 1.0);
  // In-material share of the border, relative to the material's coverage.
  float contourAbsorption = clamp(
    contourBand.y / max(silhouetteCoverage, 0.001) * contourStrength,
    0.0,
    1.0
  );
  float backdropContourAbsorption = contourAbsorption *
      (1.0 - clamp(uContourTransmittance, 0.0, 1.0));
  float bevelShadow = 0.0;

  float bevelDepth = max(uBevelDepth, 0.001);
  float sizeProgress = smoothstep(
    bevelDepth * 3.5,
    bevelDepth * 5.0,
    min(uSize.x, uSize.y) * 0.5
  );
  float bevelEnergy = mix(
    1.0,
    1.875,
    sizeProgress * clamp(uBevelSizeResponse, 0.0, 1.0)
  );
  // The rim's band is displaced along the light, as in RealGlass.
  float bevelShift = max(uBevelOffset, 0.0) * facing;
  float bevelPenumbra = 2.0 * bevelShift;
  float bevelLeading = bevelPenumbra > 0.001
      ? smoothstep(0.0, bevelPenumbra, inward)
      : 1.0;
  float bevelFalloff = 1.0 - smoothstep(0.0, bevelDepth, inward - bevelShift);
  float wrappedFacing = smoothstep(0.0, 1.0, facing * 0.5 + 0.5);
  float directionalShadow = mix(
    1.0,
    wrappedFacing,
    clamp(uBevelDirectionality, 0.0, 1.0)
  );
  bevelShadow = clamp(
    bevelLeading * bevelFalloff * directionalShadow *
        uBevelStrength * bevelEnergy,
    0.0,
    1.0
  );

  float glintWidth = max(
    uHighlightWidth > 0.0 ? uHighlightWidth : uContourWidth,
    0.001
  );
  float glintProfile =
      clamp(1.0 - inward / glintWidth, 0.0, 1.0) +
      0.21 * clamp(1.0 - inward / (glintWidth * 4.0), 0.0, 1.0);
  float wrapExponent = exp2(2.0 - 4.0 * clamp(uHighlightWrap, 0.0, 1.0));
  float lobe = pow(max(1.0 - tangency, 0.0), wrapExponent);
  float returnWeight = facing >= 0.0
      ? 1.0
      : clamp(uOppositeHighlight, 0.0, 1.0);
  float glint = clamp(
    max(uHighlight, 0.0) * 0.252 * lobe * returnWeight * glintProfile,
    0.0,
    1.0
  );

  float tintAlpha = gTint.a;
  // The glint pulls the lit face toward a bright target. Without backdrop
  // access FakeGlass approximates the face with the surface tint plus
  // emission and pulls toward it with source-over of an emissive target;
  // only RealGlass amplifies the true backdrop chroma under the glint and
  // keeps the headroom above white.
  float materialCoverage = uExteriorOnly > 0.5
      ? 0.0
      : clamp(0.5 - distance / uPixelSize, 0.0, 1.0);
  float backdropAbsorption = 1.0 -
      (1.0 - backdropContourAbsorption) * (1.0 - bevelShadow);
  float materialAlpha = 1.0 - (1.0 - tintAlpha) *
      (1.0 - backdropAbsorption);
  float exteriorContourAlpha = clamp(contourBand.x * contourStrength, 0.0, 1.0);
  // The inner shadow absorbs the filtered face, which includes the face's
  // own emission; adding that share back leaves only the transmitted light
  // shaded, as in RealGlass.
  vec3 litPremultiplied =
      gTint.rgb * tintAlpha * (1.0 - contourAbsorption) +
      uFaceEmission * bevelShadow * (1.0 - tintAlpha) *
          (1.0 - backdropContourAbsorption);
  float litAlpha = materialAlpha;
  if (uSeaGlassBody.x > 0.0) {
    vec4 seaGlass = seaGlassOverlay(position, inward, facing, normal);
    litPremultiplied = seaGlass.rgb + litPremultiplied * (1.0 - seaGlass.a);
    litAlpha = seaGlass.a + litAlpha * (1.0 - seaGlass.a);
  }
  vec3 faceProxy = gTint.rgb * tintAlpha + uFaceEmission;
  vec3 glintTarget = vec3(uGlintLuminance) +
      uGlintFaceGain * faceProxy +
      uGlintVibrancy * (faceProxy - vec3(dot(faceProxy, LUMA_WEIGHTS)));
  litPremultiplied =
      litPremultiplied * (1.0 - glint) + glintTarget * glint;
  litAlpha = 1.0 - (1.0 - litAlpha) * (1.0 - glint);
  // FakeGlass stays SDR, as Skia needs: premultiplied color never exceeds
  // alpha. Where the glint's emission would, the glass covers that much more
  // of the backdrop, so the composite reaches SDR white but never exceeds it.
  litPremultiplied = min(litPremultiplied, vec3(1.0));
  litAlpha = max(
    litAlpha,
    max(max(litPremultiplied.r, litPremultiplied.g), litPremultiplied.b)
  );
  float alpha = litAlpha * materialCoverage + exteriorContourAlpha;
  fragColor = vec4(
    max(litPremultiplied * materialCoverage, vec3(0.0)),
    clamp(alpha, 0.0, 1.0)
  );
}
