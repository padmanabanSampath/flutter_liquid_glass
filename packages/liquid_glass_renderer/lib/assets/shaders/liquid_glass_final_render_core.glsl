// Copyright 2025, Tim Lehmann for whynotmake.it
//
// Final rendering pass for liquid glass with pre-computed geometry
// This shader reads displacement data from a pre-computed texture and applies
// the liquid glass effect efficiently

// Refraction is evaluated in global filter coordinates. Keep the affine
// mapping and sub-pixel displacement in high precision on GLES; mediump turns
// the coordinate subtraction into visible shimmer on large layers.
precision highp float;

#include <flutter/runtime_effect.glsl>
#include "gpu/displacement_encoding.glsl"
#include "render.glsl"

uniform vec2 uSize;
uniform vec2 uGeometryOffset;
uniform vec2 uGeometrySize;

uniform vec4 uTint;
uniform vec3 uOpticalProps;
uniform vec3 uLightConfig;
uniform vec2 uLightDirection;
uniform vec4 uHighlightColor;
uniform vec4 uContourColor;
uniform vec4 uLightingShapeConfig;
uniform vec3 uContourConfig;
uniform vec4 uProfileConfig;
uniform vec3 uMaterialConfig;
uniform vec3 uBevelShadowConfig;
uniform vec4 uAppearanceConfig;
uniform vec4 uFilterToMatteBasis;
uniform vec2 uFilterToMatteOffset;
// 1.0 lets material alpha cross-fade the frost away, and the rest of the
// material with it; 0.0 keeps unfrosted glass opaque and cross-fades the
// material to the refracted backdrop instead, so visibility 0 is an exact
// identity of the backdrop.
uniform float uBlurFade;
// 1.0 folds sub-pixel frost into this pass: the refracted texel plus its two
// diagonal neighbours, weighted 1/2, 1/4, 1/4 (sigma ~0.7 device px), instead
// of a blur pass. Whole-texel offsets, because the filter input is sampled
// nearest.
uniform float uSoften;
// Matte-space rect (LTRB, device px) the filter captured backdrop for.
// Outside it the filter input is transparent or clamped, so displaced samples
// are mirrored back in at its edge.
uniform vec4 uBackdropBounds;
// The matte fills the top-left of a texture that only grows: sub-rect size
// over texture size. uGeometrySize is the sub-rect in device px.
uniform vec2 uGeometryUVScale;
// Texel size of the material map's texture; the map fills its top-left.
uniform vec2 uMaterialTextureSize;

// Per-model parameter blocks for the three adaptive color-model slots the
// layer assigned this frame: slot code 1..3 selects
// uColorModelParams[(code - 1) * 8 .. code * 8 - 1]. Packing, mirrored from
// GlassColorParameters.toShaderParameters:
//   +0: emissionColor.rgb,           emissionAlpha.clear
//   +1: emissionAlpha.tinted, -,     transmittanceSizeStart/End
//   +2: transmittanceSmall c/t,      transmittanceLarge c/t
//   +3: luminanceLift c/t,           chromaGain c/t
//   +4: toneLow   (base, amplitude, pivot, exponent)
//   +5: toneHigh  (base, amplitude, pivot, exponent)
//   +6: toneGammaLuminance, glintLuminance, glintFaceGain, glintVibrancy
//   +7: contourResponse
uniform vec4 uColorModelParams[24];

// Sea glass finish (SeaGlassFinish), distances in device pixels:
//   uSeaGlassBody:    strength, depth, scatter, rimGlow
//   uSeaGlassSurface: grain, grainSize, sparkle, pitting
// Strength 0 disables the stage; every branch on it is uniform, so regular
// glass renders exactly as without it.
uniform vec4 uSeaGlassBody;
uniform vec4 uSeaGlassSurface;
// The layer's shapes, two vec4 each, that the sea glass frost belongs to,
// so it moves, turns and scales with each shape: the matte-space center
// (xy) and the half size in the shape's own device pixels (zw), then the
// inverse affine basis from matte to shape space, packed like the geometry
// pass's. A negative half size ends the list.
uniform vec4 uSeaGlassShapes[32];
// Radius in device pixels of the disc the sea glass frost scatters its
// backdrop samples over, in place of a blur pass; 0 without.
uniform float uSeaGlassFrostRadius;
// Sea glass gloss, slab shading, glow and flecks (SeaGlassFinish.gloss,
// shading, glow, flecks).
uniform vec4 uSeaGlassGloss;
// The light that shades sea glass slabs (GlassLight), in matte device
// pixels:
//   uSeaGlassLight:      position (xy) and height (z) of a positioned light,
//                        and the reach of its falloff (w, 0 for none); a
//                        zero height is a far light along uLightDirection
//   uSeaGlassLightColor: the light's color times its intensity
//   uSeaGlassAmbient:    the ambient light
uniform vec4 uSeaGlassLight;
uniform vec4 uSeaGlassLightColor;
uniform vec4 uSeaGlassAmbient;

vec4 colorModelSlot(int slot, int field) {
    return uColorModelParams[slot * 8 + field];
}

float uDisplacementScale = uOpticalProps.x;
float uDispersion = uOpticalProps.y;
// Rim lighting depth; the matte encodes edge distance up to 4x this.
float uThickness = uOpticalProps.z;
float uLightIntensity = uLightConfig.x;
float uBackdropScale = uLightConfig.y;
float uSaturation = uLightConfig.z;
float uEdgeWidth = uContourConfig.x;
float uContourTransmittance = uContourConfig.y;
float uContourDirectionality = uContourConfig.z;
float uContourOffset = uProfileConfig.x;
vec2 uMaterialCenter = uProfileConfig.yz;
float uSpecularWrap = uProfileConfig.w;
float uTransmissionGamma = uMaterialConfig.x;
float uVibrancy = uMaterialConfig.y;
// iOS 27 Liquid Glass slider: 0 = Clear, 1 = Tinted.
float uTintAmount = clamp(uMaterialConfig.z, 0.0, 1.0);
float uBevelShadowStrength = uBevelShadowConfig.x;
float uBevelShadowDepth = uBevelShadowConfig.y;
float uBevelShadowOffset = uBevelShadowConfig.z;
float uBevelShadowDirectionality = uLightingShapeConfig.x;
float uBevelShadowSizeResponse = uLightingShapeConfig.y;
float uHighlightWidth = uLightingShapeConfig.z;
float uHighlightOppositeStrength = uLightingShapeConfig.w;
float uSeaGlassStrength = clamp(uSeaGlassBody.x, 0.0, 1.0);
float uSeaGlassDepth = max(uSeaGlassBody.y, 0.001);
float uSeaGlassScatter = clamp(uSeaGlassBody.z, 0.0, 1.0);
float uSeaGlassRimGlow = clamp(uSeaGlassBody.w, 0.0, 1.0);
float uSeaGlassGrain = clamp(uSeaGlassSurface.x, 0.0, 1.0);
float uSeaGlassGrainSize = max(uSeaGlassSurface.y, 0.25);
float uSeaGlassSparkle = clamp(uSeaGlassSurface.z, 0.0, 1.0);
float uSeaGlassPitting = max(uSeaGlassSurface.w, 0.0);
// SeaGlassFinish.colorClarity, in the light color's fourth component.
float uSeaGlassColorClarity = clamp(uSeaGlassLightColor.w, 0.0, 1.0);

// The fitted/default CA range is below the pixel response of the backdrop
// sampler: the pinned three-scene scan found no score or decoded-image gain
// through |CA| = .01. Bound the fast path by the maximum encoded displacement,
// rather than by CA alone, so a large-refraction surface does not silently lose
// visible dispersion. The threshold is a conservative quarter-pixel total
// red-to-blue spread (an eighth pixel on either side of green).
const float kDispersionSubpixelThreshold = 0.25;
// Border strength; the dark iOS 27 face raises it with the slider.
float gContourAlpha = uContourColor.a;
// iOS 27 glint recolor, measured on the pinned solid-palette probes in both
// appearances: the glint mixes the lit face toward a bright target whose
// luminance sits above SDR white and whose chroma is the face chroma
// amplified. The simulator captures give a mix weight of 0.14; an iPhone
// renders the same glass 1.8x stronger, so highlight = 1 matches the device.
const float kGlintLuminance = 1.6;
const float kGlintVibrancy = 2.85;
const float kGlintPeak = 0.252;
// A faint inward bleed four glint widths deep carries a fifth of the line's
// weight.
const float kGlintBleed = 0.21;
const float kGlintBleedReach = 4.0;
// The glint target is Lh + faceGain * face + vibrancy * face chroma. Regular
// glass pulls toward a fixed bright target; clear glass instead brightens
// its own face. Set per color model in main().
float gGlintLuminance = kGlintLuminance;
float gGlintFaceGain = 0.0;
float gGlintVibrancy = kGlintVibrancy;
// The contour is reconstructed from a sampled SDF. Test a wider coverage
// transition independently from the encoded exterior range so distance
// decoding and geometry placement remain bit-for-bit unchanged.
const float kContourCoverageFeather = 1.0;

uniform sampler2D uBackgroundTexture;
uniform sampler2D uGeometryTexture;
#if SHAPE_APPEARANCE || SHAPE_TINT
uniform sampler2D uMaterialTexture;
#endif
#if SHAPE_APPEARANCE
uniform sampler2D uMaterialLinearTexture;
#endif

layout(location = 0) out vec4 fragColor;

#if SHAPE_APPEARANCE
vec4 shapeLookup(
    int index,
    vec2 materialTextureSize,
    float rowCenter
) {
    return texture(
        uMaterialTexture,
        vec2(
            (float(index) + 0.5) / materialTextureSize.x,
            rowCenter / materialTextureSize.y
        )
    );
}
#endif

// Transmittance of one model's face. The small-side and large-side
// clear/tinted endpoints blend over a smoothstep of the shape's short side;
// an empty size range makes the model size-independent. Dark iOS 27 glass
// keeps its emission but becomes denser with size: controls up to 75 pt
// transmit like light glass and surfaces from 105 pt are denser from the
// start.
float parametricTransmittance(
    int slot,
    float shortSide,
    float tintAmount
) {
    vec4 sizeRange = colorModelSlot(slot, 1);
    float sizeMix = sizeRange.w > sizeRange.z
        ? smoothstep(sizeRange.z, sizeRange.w, shortSide)
        : 0.0;
    vec4 transmittance = colorModelSlot(slot, 2);
    return mix(
        mix(transmittance.x, transmittance.z, sizeMix),
        mix(transmittance.y, transmittance.w, sizeMix),
        tintAmount
    );
}

// Neutral wash of one model's untinted face: premultiplied emission over
// the wash's opacity, the complement of its transmittance.
vec4 parametricNeutralTint(int slot, float shortSide, float tintAmount) {
    float transmittance =
        parametricTransmittance(slot, shortSide, tintAmount);
    vec4 emission = colorModelSlot(slot, 0);
    vec4 sizeRange = colorModelSlot(slot, 1);
    float alpha = 1.0 - transmittance;
    float emissionAlpha = mix(emission.w, sizeRange.x, tintAmount);
    return vec4(
        emission.rgb * emissionAlpha / max(alpha, 1e-4),
        alpha
    );
}

// Luminance lift and chroma gain of one model's untinted face.
vec2 parametricFaceTransfer(int slot, float tintAmount) {
    vec4 liftChroma = colorModelSlot(slot, 3);
    return vec2(
        mix(liftChroma.x, liftChroma.y, tintAmount),
        mix(liftChroma.z, liftChroma.w, tintAmount)
    );
}

// The brightness-mapped range of tint tones of one model, evaluated per
// channel as clamp(lo + (hi - lo) * tint^gamma) where lo and hi are
// luminance ramps base + amplitude * (min(1, Y / pivot))^exponent and the
// gamma is 1 + toneGammaLuminance * (1 - Y).
vec3 parametricTintTone(int slot, vec3 tint, float backdropLuminance) {
    float luminance = clamp(backdropLuminance, 0.0, 1.0);
    vec4 low = colorModelSlot(slot, 4);
    vec4 high = colorModelSlot(slot, 5);
    vec4 misc = colorModelSlot(slot, 6);
    float floor_ = low.x + low.y * pow(
        min(1.0, luminance / max(low.z, 1e-6)),
        low.w
    );
    float ceiling = high.x + high.y * pow(
        min(1.0, luminance / max(high.z, 1e-6)),
        high.w
    );
    return clamp(
        vec3(floor_) + (vec3(ceiling) - floor_) * pow(
            max(tint, vec3(0.0)),
            vec3(1.0 + misc.x * (1.0 - luminance))
        ),
        0.0,
        1.0
    );
}

// Glint target parameters of one model: luminance, face gain, vibrancy.
vec3 parametricGlint(int slot) {
    return colorModelSlot(slot, 6).yzw;
}

// How strongly a model's border follows the opacity the slider adds.
float parametricContourResponse(int slot) {
    return colorModelSlot(slot, 7).x;
}

// Shares of a color model code: direct (0) plus the layer's three model
// slots (1..3). Merged shapes blend these shares rather than the codes, so
// a merge between two adaptive models does not pass through direct.
vec4 colorModelSharesOf(float code) {
    return vec4(
        code < 0.5 ? 1.0 : 0.0,
        abs(code - 1.0) < 0.5 ? 1.0 : 0.0,
        abs(code - 2.0) < 0.5 ? 1.0 : 0.0,
        code > 2.5 ? 1.0 : 0.0
    );
}

float contourExtent() {
    return max(
        kContourCoverageFeather,
        uContourOffset + uEdgeWidth + kContourCoverageFeather
    );
}

// The dark border starts at the silhouette (shifted outward by
// contourOffset) and fades linearly outward over contourWidth. It is often
// narrower than a pixel, so it is box-filtered over the same one-pixel
// footprint as the silhouette instead of being point-sampled, which would
// alias into a dotted line along curves. This is the ramp's integral from
// the silhouette to outward distance t.
float contourIntegral(float t) {
    float u = clamp(t - uContourOffset, 0.0, uEdgeWidth);
    return u - u * u / (2.0 * uEdgeWidth);
}

// Border coverage of the pixel, split at the silhouette: x lies outside the
// material and is composited over the backdrop, y lies inside it.
vec2 contourCoverage(float signedEdgeDistance) {
    if (uEdgeWidth <= 0.0) {
        return vec2(0.0);
    }
    float outward = -signedEdgeDistance;
    float lower = outward - 0.5;
    float upper = outward + 0.5;
    return vec2(
        contourIntegral(max(upper, 0.0)) - contourIntegral(max(lower, 0.0)),
        contourIntegral(min(upper, 0.0)) - contourIntegral(min(lower, 0.0))
    );
}

// The fraction of the normal perpendicular to the light axis. The glint
// lives where it is zero; the dark border concentrates where it is one.
float lightAxisTangency(vec2 surfaceNormal) {
    return abs(dot(surfaceNormal, vec2(-uLightDirection.y, uLightDirection.x)));
}

float contourDirection(vec2 surfaceNormal) {
    return mix(
        1.0,
        lightAxisTangency(surfaceNormal),
        clamp(uContourDirectionality, 0.0, 1.0)
    );
}

vec2 mirrorBackgroundUV(vec2 uv, vec2 inverseTextureSize) {
    // Image-filter sampler edge behavior differs between Impeller backends.
    // Preserve every coordinate inside the input texture exactly, and mirror
    // only displaced samples that genuinely leave it. This avoids GLES decal
    // black without clamping Metal samples into stretched edge pixels.
    vec2 mirrored = vec2(1.0) - abs(mod(uv, vec2(2.0)) - vec2(1.0));
    vec2 halfTexel = inverseTextureSize * 0.5;
    return clamp(mirrored, halfTexel, vec2(1.0) - halfTexel);
}

vec2 filterDeltaFromMatteDelta(vec2 matteDelta, vec4 basis) {
    // Invert the live filter->matte affine basis so material-centered source
    // mapping remains stable under ancestor transforms.
    float determinant = basis.x * basis.w - basis.y * basis.z;
    if (abs(determinant) < 1e-6) {
        return vec2(0.0);
    }
    return vec2(
        basis.w * matteDelta.x - basis.y * matteDelta.y,
        -basis.z * matteDelta.x + basis.x * matteDelta.y
    ) / determinant;
}

vec2 mirrorIntoBackdrop(vec2 sourceOffset, vec2 matteCoord) {
    vec2 sampleMatte = matteCoord + vec2(
        dot(uFilterToMatteBasis.xy, sourceOffset),
        dot(uFilterToMatteBasis.zw, sourceOffset)
    );
    // Keep bilinear footprints, and the softening taps one texel out, off
    // the uncaptured side of the edge.
    float margin = 0.5 + uSoften;
    vec2 lo = uBackdropBounds.xy + margin;
    vec2 hi = uBackdropBounds.zw - margin;
    if (all(greaterThanEqual(sampleMatte, lo)) &&
        all(lessThanEqual(sampleMatte, hi))) {
        return sourceOffset;
    }
    vec2 extent = max(hi - lo, vec2(0.0));
    vec2 mirrored = lo + extent - abs(extent - abs(sampleMatte - lo));
    mirrored = clamp(mirrored, lo, max(hi, lo));
    return sourceOffset + filterDeltaFromMatteDelta(
        mirrored - sampleMatte,
        uFilterToMatteBasis
    );
}

vec3 applySpecularHighlights(
    vec3 baseColor,
    vec3 transmittedColor,
    float signedEdgeDistance,
    vec2 surfaceNormal
) {
    if (
        uLightIntensity < 0.01 &&
        gContourAlpha < 0.01 &&
        uBevelShadowStrength < 0.001
    ) {
        return baseColor;
    }

    float inwardDistance = max(signedEdgeDistance, 0.0);
    float glintWidth = max(
        uHighlightWidth > 0.0 ? uHighlightWidth : uEdgeWidth,
        0.001
    );
    // In-material share of the border, relative to the material's coverage.
    float outlineCoverage = contourCoverage(signedEdgeDistance).y /
        max(clamp(signedEdgeDistance + 0.5, 0.0, 1.0), 0.001);
    // The glint is a thin line anchored at the silhouette with a faint
    // inward bleed. Both are linear ramps in logical distance, so the line
    // stays crisp at every size and scale factor.
    float glintProfile =
        clamp(1.0 - inwardDistance / glintWidth, 0.0, 1.0) +
        kGlintBleed * clamp(
            1.0 - inwardDistance / (glintWidth * kGlintBleedReach),
            0.0,
            1.0
        );

    if (
        outlineCoverage < 0.01 &&
        glintProfile < 0.001 &&
        uBevelShadowStrength < 0.001
    ) {
        return baseColor;
    }

    vec2 normalXY = surfaceNormal;

    // Both walls along the light axis catch the glint; it fades linearly
    // with the normal's tangential component. highlightWrap = 0.5 is the
    // linear falloff measured on iOS 27; lower values narrow the lobes and
    // higher values carry them further around corners.
    float wrapExponent = exp2(2.0 - 4.0 * clamp(uSpecularWrap, 0.0, 1.0));
    float axisAlignment = 1.0 - lightAxisTangency(normalXY);
    float lobe = pow(max(axisAlignment, 0.0), wrapExponent);
    float returnWeight = dot(normalXY, -uLightDirection) >= 0.0
        ? 1.0
        : clamp(uHighlightOppositeStrength, 0.0, 1.0);
    float glint = clamp(
        max(uLightIntensity, 0.0) * kGlintPeak * lobe * returnWeight *
            glintProfile,
        0.0,
        1.0
    );
    // Both branches are uniform across the draw. Disabled lighting layers skip
    // their ALU without introducing fragment divergence, texture samples, or
    // another compositor pass.
    float bevelShadow = 0.0;
    if (uBevelShadowStrength >= 0.001) {
        float configuredBevelDepth = max(uBevelShadowDepth, 0.001);
        float sizeAwareBevelDepth = configuredBevelDepth;
        float surfaceHalfMinor = min(uGeometrySize.x, uGeometrySize.y) * 0.5;
        float sizeProgress = smoothstep(
            configuredBevelDepth * 3.5,
            configuredBevelDepth * 5.0,
            surfaceHalfMinor
        );
        float sizeEnergy = mix(
            1.0,
            1.875,
            sizeProgress * clamp(uBevelShadowSizeResponse, 0.0, 1.0)
        );
        // A bevel occupies proportionally less of a large surface, so its wall
        // contributes more integrated shadow without extending farther into
        // the face. Small controls retain the configured strength; larger
        // surfaces grow smoothly up to 2x. This avoids the broad matte wash
        // produced by globally increasing strength or band depth.
        // The raised rim shades the face like a wall lit along the light
        // axis: its band is displaced by the offset along the light. Below
        // the lit wall it falls inside the face, along the sides it starts
        // at the rim, and below the far wall it is pushed out past the rim.
        // The penumbra is as wide as the displacement.
        float lightFacing = dot(normalXY, -uLightDirection);
        float shadowShift = max(uBevelShadowOffset, 0.0) * lightFacing;
        float penumbra = 2.0 * shadowShift;
        float bevelLeadingEdge = penumbra > 0.001
            ? smoothstep(0.0, penumbra, inwardDistance)
            : 1.0;
        float bevelFalloff = 1.0 - smoothstep(
            0.0,
            sizeAwareBevelDepth,
            inwardDistance - shadowShift
        );
        float bevelBand = bevelLeadingEdge * bevelFalloff;
        // Remap the signed SDF-normal response across the full contour before
        // applying directionality. Clamping dot() at zero creates a visible
        // half-plane seam whose endpoints project as wedges into circles and
        // blended shapes. The cubic ramp remains strongest on the source-facing
        // wall, reaches zero only opposite the source, and has zero slope at
        // both ends. The highlight is added after this shadow, allowing the
        // bright rim to eclipse the dark wall where they overlap.
        float wrappedLightFacing = smoothstep(
            0.0,
            1.0,
            lightFacing * 0.5 + 0.5
        );
        float bevelDirection = mix(
            1.0,
            wrappedLightFacing,
            clamp(uBevelShadowDirectionality, 0.0, 1.0)
        );
        bevelShadow = clamp(
            bevelBand *
                bevelDirection *
                uBevelShadowStrength *
                sizeEnergy,
            0.0,
            1.0
        );
    }
    // The border absorbs the transmitted backdrop only where the material
    // still overlaps it; its exterior part is composited in main().
    float edgeAbsorption = clamp(
        outlineCoverage * gContourAlpha * contourDirection(normalXY),
        0.0,
        1.0
    );
    vec3 result = baseColor * (1.0 - edgeAbsorption) +
        uContourColor.rgb * edgeAbsorption +
        transmittedColor * edgeAbsorption *
            clamp(uContourTransmittance, 0.0, 1.0);
    // The shadow absorbs only light that came through the glass: the wash
    // emits the same over any backdrop, so over black the face is unshaded.
    result = max(
        result - transmittedColor * (1.0 - edgeAbsorption) * bevelShadow,
        vec3(0.0)
    );
    // The glint recolors the lit face rather than adding white: it pulls
    // luminance toward a target above SDR white and amplifies the face's own
    // chroma, so glass over color glints in that color.
    float resultLuminance = dot(result, LUMA_WEIGHTS);
    vec3 glintTarget = uHighlightColor.rgb * gGlintLuminance +
        result * gGlintFaceGain +
        (result - vec3(resultLuminance)) * gGlintVibrancy;
    // Only the lower bound is clamped: amplified chroma must not produce
    // negative (out-of-gamut) channels, while the upper side keeps its
    // headroom above SDR white for extended-range surfaces.
    result = max(mix(result, glintTarget, glint), vec3(0.0));

    return result;
}

// MARK: Sea glass

#include "sea_glass.glsl"

// Sparkles pull toward this luminance; above SDR white like the glint, so
// extended-range surfaces keep their headroom.
const float kSeaGlassSparkleLuminance = 1.3;
// Share of the full body color the paper-thin rim still carries.
const float kSeaGlassRimPath = 0.3;
// Grains over which the frost of two merged shapes cross-fades in their
// bridge.
const float kSeaGlassAnchorFade = 4.0;
// Backdrop samples of the scattered frost, the refracted one included.
const int kSeaGlassFrostTaps = 12;
// The golden angle, which spreads the taps evenly over the disc.
const float kGoldenAngle = 2.39996323;

// Where p lies in a shape's own frame, through its inverse affine basis.
vec2 seaGlassLocal(vec2 p, vec4 box, vec4 frame) {
    vec2 delta = p - box.xy;
    return vec2(
        frame.x * delta.x + frame.y * delta.y,
        frame.z * delta.x + frame.w * delta.y
    );
}

// The light's direction in a shape's own frame.
vec2 seaGlassLocalLight(vec4 frame) {
    vec2 light = vec2(
        frame.x * uLightDirection.x + frame.y * uLightDirection.y,
        frame.z * uLightDirection.x + frame.w * uLightDirection.y
    );
    return light / max(length(light), 1e-6);
}

// The frost belongs to the shape whose box, in its own frame, p lies
// deepest in. Where a second box is nearly as close, as in the bridge of
// two merged shapes, both frosts cross-fade instead of meeting at a seam.
// Returns the second shape's weight; the boxes and frames receive both
// shapes, and halfMinor their half short side, blended the same way.
float seaGlassOwners(
    vec2 p,
    out vec4 firstBox,
    out vec4 firstFrame,
    out vec4 secondBox,
    out vec4 secondFrame,
    out float halfMinor
) {
    float firstDistance = 1e9;
    float secondDistance = 1e9;
    firstBox = vec4(uMaterialCenter, 0.0, 0.0);
    firstFrame = vec4(1.0, 0.0, 0.0, 1.0);
    secondBox = firstBox;
    secondFrame = firstFrame;
    for (int i = 0; i < 16; i++) {
        vec4 box = uSeaGlassShapes[2 * i];
        if (box.z < 0.0) {
            break;
        }
        vec4 frame = uSeaGlassShapes[2 * i + 1];
        vec2 q = abs(seaGlassLocal(p, box, frame)) - box.zw;
        float d = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
        if (d < firstDistance) {
            secondDistance = firstDistance;
            secondBox = firstBox;
            secondFrame = firstFrame;
            firstDistance = d;
            firstBox = box;
            firstFrame = frame;
        } else if (d < secondDistance) {
            secondDistance = d;
            secondBox = box;
            secondFrame = frame;
        }
    }
    float weight = 0.5 * (1.0 - smoothstep(
        0.0,
        kSeaGlassAnchorFade * uSeaGlassGrainSize,
        secondDistance - firstDistance
    ));
    halfMinor = mix(
        min(firstBox.z, firstBox.w),
        min(secondBox.z, secondBox.w),
        weight
    );
    return weight;
}

// The pits of the owning shapes at their local positions, cross-faded with
// their variance kept so the grain does not flatten in a bridge.
vec3 seaGlassOwnedPits(vec2 first, vec2 second, float weight) {
    vec3 pits = seaGlassPits(first, uSeaGlassGrainSize);
    if (weight > 0.001) {
        vec3 other = seaGlassPits(second, uSeaGlassGrainSize);
        pits = (pits * (1.0 - weight) + other * weight) /
            sqrt((1.0 - weight) * (1.0 - weight) + weight * weight);
    }
    return pits;
}

// Direction toward the light, in matte space, from the shape whose box is
// box (not normalized).
vec2 seaGlassTowardLight(vec4 box) {
    return uSeaGlassLight.z > 0.0
        ? uSeaGlassLight.xy - box.xy
        : -uLightDirection;
}

// The direction, in matte space, the slab shading of the shape whose box is
// box comes from (slabShadeDirection).
vec2 seaGlassShadeDirection(vec4 box) {
    vec2 toward = seaGlassTowardLight(box);
    return slabShadeDirection(
        toward / max(length(toward), 1e-6),
        uSeaGlassLight.z > 0.0
    );
}

// The slab shading (slabShading) of the shape whose box and frame are
// given, at local, its own frame's position of this pixel, inward device
// pixels inside the outline, whose outward normal is normal.
vec2 seaGlassSlabShading(
    vec2 local,
    vec4 box,
    vec4 frame,
    vec2 normal,
    float inward
) {
    float distance = max(length(seaGlassTowardLight(box)), 1e-6);
    vec2 shade = seaGlassShadeDirection(box);
    vec2 towardLocal = vec2(
        frame.x * shade.x + frame.y * shade.y,
        frame.z * shade.x + frame.w * shade.y
    );
    towardLocal /= max(length(towardLocal), 1e-6);
    // A positioned light right above a shape lights it evenly.
    float tilt = uSeaGlassLight.z > 0.0
        ? clamp(distance / (0.5 * uSeaGlassLight.z), 0.0, 1.0)
        : 1.0;
    return slabShading(
        local / max(box.zw, vec2(1.0)),
        towardLocal,
        dot(normal, shade),
        inward / uSeaGlassDepth,
        tilt,
        clamp(uSeaGlassGloss.z, 0.0, 1.0)
    );
}

// The light reaching p: the ambient light plus the light itself, which a
// positioned light with a reach halves at that distance.
vec3 seaGlassIllumination(vec2 p) {
    float falloff = 1.0;
    if (uSeaGlassLight.z > 0.0 && uSeaGlassLight.w > 0.0) {
        vec2 d = (p - uSeaGlassLight.xy) / uSeaGlassLight.w;
        falloff = 1.0 / (1.0 + dot(d, d));
    }
    return uSeaGlassAmbient.rgb + uSeaGlassLightColor.rgb * falloff;
}

// The sea glass face over the refracted backdrop.
//
// The body filters the backdrop like colored glass, more where the rounded
// pebble is thicker (Beer-Lambert with a parabolic thickness profile over
// uSeaGlassDepth), so the center is deepest in color and the rim pales. The
// frost scatters part of the light: front light scattered inside the body
// leaves it in the body's color, so the glass glows even over dark content,
// more at the grazing rim and brighter over light backdrops. The rim glows
// in a pastel of the body where it faces the light. Returns the face;
// transmitted receives the share of it that came through the glass.
vec3 seaGlassFace(
    vec3 backdrop,
    float backdropLuminance,
    vec4 body,
    float inwardDistance,
    vec2 surfaceNormal,
    float grain,
    float halfMinor,
    vec2 slab,
    vec3 illumination,
    out vec3 transmitted
) {
    float thickness = seaGlassThickness(inwardDistance, uSeaGlassDepth);
    vec2 sizeResponse = seaGlassSizeResponse(halfMinor, uSeaGlassDepth);
    float density = clamp(body.a, 0.0, 1.0);
    vec3 bodyColor = clamp(body.rgb, vec3(0.02), vec3(1.0));
    vec3 transmittance = pow(
        bodyColor,
        vec3(
            density * sizeResponse.x *
                mix(kSeaGlassRimPath, 1.0, thickness)
        )
    );
    float scatter = clamp(
        uSeaGlassScatter * sizeResponse.y * mix(1.6, 1.0, thickness),
        0.0,
        1.0
    );
    float shading = clamp(uSeaGlassGloss.y, 0.0, 1.0);
    scatter *= mix(
        1.0,
        slabEdgeScatter(inwardDistance, uSeaGlassDepth),
        shading
    );
    // Colored glass clears (SeaGlassFinish.colorClarity).
    scatter *= 1.0 - 0.7 * seaGlassColorClear(body, uSeaGlassColorClarity);
    transmitted = backdrop * transmittance * (1.0 - scatter);
    float ambient = mix(0.7, 1.0, clamp(backdropLuminance, 0.0, 1.0));
    vec3 glow = mix(vec3(1.0), transmittance, 0.6) * ambient;
    if (shading > 0.0) {
        // A thick milky slab shows its own color, brightened by the light.
        vec3 slabBody = mix(vec3(1.0), bodyColor, density);
        glow = mix(glow, slabColor(slabBody, slab) * illumination, shading);
    }
    vec3 face = transmitted + glow * scatter;
    vec3 pastel = mix(vec3(1.0), pow(bodyColor, vec3(0.4 * density)), 0.85);
    // The rim band fades quadratically over the rounded shoulder; it is
    // strongest on the wall facing the light and never vanishes opposite it.
    float rimBand = 1.0 - thickness;
    float lightFacing = dot(surfaceNormal, -uLightDirection);
    float rimLight = mix(0.4, 1.0, smoothstep(-1.0, 1.0, lightFacing));
    face = mix(
        face,
        max(face, pastel * mix(0.8, 1.0, ambient)),
        clamp(uSeaGlassRimGlow * rimBand * rimLight * 0.6, 0.0, 1.0)
    );
    face += vec3(grain * uSeaGlassGrain * 0.18);
    return max(face, vec3(0.0));
}


// How brightly the pebble's dome catches the light at a pixel inwardDistance
// inside its outline, whose outward normal is surfaceNormal.
//
// The dome is the thickness profile (seaGlassThickness): its surface tilts
// from flat at full thickness to 50 degrees at the outline. A light 41
// degrees above the surface, along uLightDirection, glints off the shoulder
// facing it; its reflection inside the glass lights the far wall more
// softly; and the outline, where the surface turns away, catches a line of
// light about a pixel and a half wide. A flat face reflects neither lobe,
// so the body color at the center stays untouched.
float seaGlassGlossLobe(vec3 normal, vec3 toward, float sharpness) {
    vec3 halfway = normalize(toward + vec3(0.0, 0.0, 1.0));
    float level = pow(halfway.z, sharpness);
    float lobe = pow(max(dot(normal, halfway), 0.0), sharpness);
    return max(lobe - level, 0.0) / (1.0 - level);
}

float seaGlassGloss(float inwardDistance, vec2 surfaceNormal) {
    float x = clamp(inwardDistance / uSeaGlassDepth, 0.0, 1.0);
    vec3 normal = normalize(vec3(surfaceNormal * 1.2 * (1.0 - x), 1.0));
    vec2 lightAxis = uLightDirection * 0.75;
    vec3 toLight = normalize(vec3(-lightAxis, 0.66));
    vec3 reflected = normalize(vec3(lightAxis, 0.66));
    float glint = seaGlassGlossLobe(normal, toLight, 48.0);
    float caustic = 0.35 * seaGlassGlossLobe(normal, reflected, 16.0);
    float outline = 0.3 * (1.0 - smoothstep(0.0, 1.5, inwardDistance));
    return glint + caustic + outline;
}

void main() {
    // Map image-filter fragment coordinates back into the layer-local geometry
    // matte. Apple Metal surfaces expose global filter coordinates, while
    // other backends may expose clip-local coordinates. These uniforms are
    // snapshotted with the frame, never overwritten through a shared texture.
    vec2 fragCoord = FlutterFragCoord().xy;
    vec2 screenUV = fragCoord / uSize;

    vec2 matteCoord = vec2(
        dot(uFilterToMatteBasis.xy, fragCoord),
        dot(uFilterToMatteBasis.zw, fragCoord)
    ) + uFilterToMatteOffset;
    vec2 geometryUV = (matteCoord - uGeometryOffset) / uGeometrySize;

    if (
        any(lessThan(geometryUV, vec2(0.0))) ||
        any(greaterThan(geometryUV, vec2(1.0)))
    ) {
        fragColor = vec4(0.0);
        return;
    }

    vec4 geometryData = texture(
        uGeometryTexture,
        geometryUV * uGeometryUVScale
    );
    vec4 materialTint = uTint;
    float appearanceVisibility = clamp(uAppearanceConfig.y, 0.0, 1.0);
    // Weight of the material over the refracted backdrop. Frosted glass
    // renders the material in full and its alpha fades it with the frost;
    // opaque unfrosted glass fades it here. Either way it fades linearly,
    // never twice.
    float materialVisibility = mix(appearanceVisibility, 1.0, uBlurFade);
    vec4 colorModelShares = colorModelSharesOf(uAppearanceConfig.x);
    #if SHAPE_TINT
    {
        float materialRasterScale = max(uAppearanceConfig.z, 1.0);
        vec2 materialSize = max(
            ceil(uGeometrySize / materialRasterScale),
            vec2(1.0)
        );
        vec2 materialTextureSize = uMaterialTextureSize;
        vec2 materialLinearUV =
            (geometryUV * (materialSize - vec2(1.0)) + vec2(0.5)) /
            materialTextureSize;
        materialTint = texture(uMaterialTexture, materialLinearUV);
    }
    #elif SHAPE_APPEARANCE
    {
        float materialRasterScale = max(uAppearanceConfig.z, 1.0);
        vec2 materialSize = max(
            ceil(uGeometrySize / materialRasterScale),
            vec2(1.0)
        );
        vec2 materialTextureSize = uMaterialTextureSize;
        vec2 materialNearestUV =
            (floor(geometryUV * materialSize) + vec2(0.5)) /
            materialTextureSize;
        vec2 materialLinearUV =
            (geometryUV * (materialSize - vec2(1.0)) + vec2(0.5)) /
            materialTextureSize;
        // Keep contributor IDs discrete, while bilinearly upsampling only the
        // transition weight. Pair changes are allowed to be approximate: this
        // map is a transient tint gradient, never the geometry matte.
        vec4 contributors = texture(uMaterialTexture, materialNearestUV);
        vec4 filtered = texture(uMaterialLinearTexture, materialLinearUV);
        float primaryWeight = clamp(filtered.b, 0.0, 1.0);
        // Where the nearest shape swaps, the filtered footprint holds one
        // pair in both orders: the IDs differ but their sum does not. The
        // nearest-shape weight never reaches 0.5 there, so use the weight
        // stored for the pair's lower-indexed shape instead.
        const float kIdTolerance = 0.5 / 255.0;
        if (
            abs(filtered.r - contributors.r) > kIdTolerance &&
            abs(
                filtered.r + filtered.g - contributors.r - contributors.g
            ) < kIdTolerance
        ) {
            float lowerWeight = clamp(filtered.a, 0.0, 1.0);
            primaryWeight = contributors.r <= contributors.g
                ? lowerWeight
                : 1.0 - lowerWeight;
        }
        int primary = int(clamp(
            floor(contributors.r * 16.0),
            0.0,
            15.0
        ));
        int secondary = int(clamp(
            floor(contributors.g * 16.0),
            0.0,
            15.0
        ));
        float secondaryWeight = 1.0 - primaryWeight;
        vec4 primaryTint = shapeLookup(
            primary,
            materialTextureSize,
            materialSize.y + 0.5
        );
        vec4 secondaryTint = shapeLookup(
            secondary,
            materialTextureSize,
            materialSize.y + 0.5
        );
        vec4 primaryResponse = shapeLookup(
            primary,
            materialTextureSize,
            materialSize.y + 1.5
        );
        vec4 secondaryResponse = shapeLookup(
            secondary,
            materialTextureSize,
            materialSize.y + 1.5
        );
        primaryResponse.xyz *= 4.0;
        secondaryResponse.xyz *= 4.0;
        float primaryPackedResponse = primaryResponse.w * 7.0;
        float secondaryPackedResponse = secondaryResponse.w * 7.0;
        float primaryColorModel = floor(primaryPackedResponse * 0.5);
        float secondaryColorModel = floor(secondaryPackedResponse * 0.5);
        primaryResponse.w = primaryPackedResponse - primaryColorModel * 2.0;
        secondaryResponse.w =
            secondaryPackedResponse - secondaryColorModel * 2.0;
        float blendedTintAlpha =
            secondaryTint.a * secondaryWeight +
            primaryTint.a * primaryWeight;
        vec3 blendedTintPremultiplied =
            secondaryTint.rgb * secondaryTint.a * secondaryWeight +
            primaryTint.rgb * primaryTint.a * primaryWeight;
        vec3 blendedTintColor = blendedTintAlpha > 0.0001
            ? blendedTintPremultiplied / blendedTintAlpha
            : mix(secondaryTint.rgb, primaryTint.rgb, primaryWeight);
        materialTint = vec4(blendedTintColor, blendedTintAlpha);
        vec4 appearance = mix(
            secondaryResponse,
            primaryResponse,
            primaryWeight
        );
        appearanceVisibility = clamp(appearance.w, 0.0, 1.0);
        colorModelShares = mix(
            colorModelSharesOf(secondaryColorModel),
            colorModelSharesOf(primaryColorModel),
            primaryWeight
        );
        materialVisibility = mix(appearanceVisibility, 1.0, uBlurFade);
        uSaturation = appearance.x;
        uTransmissionGamma = appearance.y;
        uVibrancy = appearance.z;
    }
    #endif

    float maxDisplacement = max(uDisplacementScale, 0.001);
    float signedEdgeDistance = decodeSignedEdgeDistance(
        geometryData,
        4.0 * max(uThickness, 1.0),
        contourExtent()
    );
    // Box-filtered coverage of one physical pixel, as Core Animation
    // rasterizes the silhouette: a pixel-aligned edge stays hard, so the
    // glint's first row is not diluted by a wider feather.
    float materialAlpha = clamp(signedEdgeDistance + 0.5, 0.0, 1.0);
    // The geometry pass writes an all-zero texel wherever no shape or
    // contour reaches, which decodes to no coverage. Testing the raw texel
    // keeps those pixels out even where the decode goes wrong: the
    // Mali-G615 Vulkan driver (r44p1) decoded them as lying on an edge,
    // washing the whole layer rectangle in glass.
    if (
        geometryData.b <= 0.5 / 255.0 ||
        (
            materialAlpha < 0.01 &&
            contourCoverage(signedEdgeDistance).x * gContourAlpha < 0.01
        )
    ) {
        fragColor = vec4(0.0);
        return;
    }
    vec2 displacement =
        decodeDisplacement(geometryData, maxDisplacement) *
        appearanceVisibility;
    vec2 surfaceNormal = decodeSurfaceNormal(geometryData);
    // Sea glass pits break up the backdrop seen through the face. Like the
    // bevel, they settle with visibility.
    vec3 seaGlassPitSample = vec3(0.0);
    // Positions and frames of the shapes the frost at this pixel belongs
    // to; the frost is evaluated in their own frames.
    vec2 seaGlassFirst = matteCoord - uMaterialCenter;
    vec2 seaGlassSecond = seaGlassFirst;
    vec4 seaGlassFirstFrame = vec4(1.0, 0.0, 0.0, 1.0);
    vec4 seaGlassSecondFrame = seaGlassFirstFrame;
    float seaGlassSecondWeight = 0.0;
    float seaGlassHalfMinor = 0.0;
    vec4 firstBox = vec4(0.0);
    vec4 secondBox = vec4(0.0);
    if (uSeaGlassStrength > 0.0) {
        seaGlassSecondWeight = seaGlassOwners(
            matteCoord,
            firstBox,
            seaGlassFirstFrame,
            secondBox,
            seaGlassSecondFrame,
            seaGlassHalfMinor
        );
        seaGlassFirst = seaGlassLocal(matteCoord, firstBox, seaGlassFirstFrame);
        seaGlassSecond = seaGlassLocal(
            matteCoord,
            secondBox,
            seaGlassSecondFrame
        );
        seaGlassPitSample = seaGlassOwnedPits(
            seaGlassFirst,
            seaGlassSecond,
            seaGlassSecondWeight
        );
        displacement += seaGlassPitSample.xy * uSeaGlassPitting *
            uSeaGlassStrength * appearanceVisibility *
            (1.0 - 0.45 * seaGlassColorClear(
                materialTint,
                uSeaGlassColorClarity
            ));
    }

    vec2 invUSize = 1.0 / uSize;
    vec2 backdropScaleOffset = vec2(0.0);
    if (abs(uBackdropScale - 1.0) > 0.0001) {
        // backdropShrink is one lens over the whole face, about the material
        // center of the layer, uniform up to the silhouette. The bevel
        // displacement adds on top of it. It never enlarges: magnifiers
        // re-render their content instead (see the example's loupe).
        vec2 filterDeltaFromCenter = filterDeltaFromMatteDelta(
            matteCoord - uMaterialCenter,
            uFilterToMatteBasis
        );
        float magnification = clamp(uBackdropScale, 0.25, 1.0);
        backdropScaleOffset =
            filterDeltaFromCenter *
            (1.0 / magnification - 1.0) *
            appearanceVisibility;
    }
    vec4 refractColor;
    // Skip two texture reads only when the maximum channel separation is
    // subpixel. The uniform predicate stays coherent across the layer and the
    // displacement bound keeps this optimization valid for either CA sign.
    if (
        abs(uDispersion) * maxDisplacement <=
        kDispersionSubpixelThreshold
    ) {
        vec2 sourceOffset = backdropScaleOffset + displacement;
        vec2 refractedUV;
        if (sourceOffset.x == 0.0 && sourceOffset.y == 0.0) {
            // Undisplaced glass fetches its own texel, bypassing the sampler,
            // so it reproduces the backdrop exactly even when the sampler is
            // bilinear (whose fixed-point sub-texel weights are never exactly
            // zero at texel centres).
            refractedUV = (floor(fragCoord) + 0.5) * invUSize;
            #ifdef IMPELLER_TARGET_OPENGLES
            // The GLES runtime stages also emit GLSL ES 1.00, which has no
            // texelFetch. The texel centre is exact under nearest sampling;
            // bilinear can differ by a few LSB at hard edges.
            refractColor = texture(uBackgroundTexture, refractedUV);
            #else
            refractColor = texelFetch(
                uBackgroundTexture,
                ivec2(floor(fragCoord)),
                0
            );
            #endif
        } else {
            refractedUV = screenUV +
                mirrorIntoBackdrop(sourceOffset, matteCoord) * invUSize;
            refractColor = texture(
                uBackgroundTexture,
                mirrorBackgroundUV(refractedUV, invUSize)
            );
        }
        if (uSeaGlassFrostRadius > 0.0) {
            // The rough surface sends each ray a random way: the frost is
            // the mean of samples spread over a disc (a Vogel spiral, at a
            // random turn per pixel, anchored to the glass so the grain
            // moves with it). Like the bevel, it settles with visibility.
            float radius = uSeaGlassFrostRadius * appearanceVisibility *
                (1.0 - 0.55 * seaGlassColorClear(
                    materialTint,
                    uSeaGlassColorClarity
                ));
            float turn = 6.2831853 * seaGlassHash(
                floor(seaGlassFirst)
            );
            vec2 direction = vec2(cos(turn), sin(turn));
            const vec2 kGoldenTurn = vec2(
                cos(kGoldenAngle),
                sin(kGoldenAngle)
            );
            vec4 frost = refractColor;
            for (int i = 1; i < kSeaGlassFrostTaps; i++) {
                direction = vec2(
                    direction.x * kGoldenTurn.x - direction.y * kGoldenTurn.y,
                    direction.x * kGoldenTurn.y + direction.y * kGoldenTurn.x
                );
                vec2 offset = direction * radius *
                    sqrt(float(i) / float(kSeaGlassFrostTaps));
                frost += texture(
                    uBackgroundTexture,
                    mirrorBackgroundUV(
                        screenUV + mirrorIntoBackdrop(
                            sourceOffset + offset,
                            matteCoord
                        ) * invUSize,
                        invUSize
                    )
                );
            }
            refractColor = frost / float(kSeaGlassFrostTaps);
        }
        if (uSoften > 0.5) {
            // The offset scales with visibility, so hidden glass collapses
            // the kernel onto the unfiltered backdrop.
            vec2 softenTap = vec2(appearanceVisibility) * invUSize;
            vec2 tapA = mirrorBackgroundUV(refractedUV + softenTap, invUSize);
            vec2 tapB = mirrorBackgroundUV(refractedUV - softenTap, invUSize);
            refractColor = 0.5 * refractColor + 0.25 * (
                texture(uBackgroundTexture, tapA) +
                texture(uBackgroundTexture, tapB)
            );
        }
    } else {
        float dispersionStrength = uDispersion * 0.5;
        vec2 redOffset = displacement * (1.0 + dispersionStrength);
        vec2 blueOffset = displacement * (1.0 - dispersionStrength);
        
        vec2 redUV = mirrorBackgroundUV(
            screenUV + mirrorIntoBackdrop(
                backdropScaleOffset + redOffset,
                matteCoord
            ) * invUSize,
            invUSize
        );
        vec2 greenUV = mirrorBackgroundUV(
            screenUV + mirrorIntoBackdrop(
                backdropScaleOffset + displacement,
                matteCoord
            ) * invUSize,
            invUSize
        );
        vec2 blueUV = mirrorBackgroundUV(
            screenUV + mirrorIntoBackdrop(
                backdropScaleOffset + blueOffset,
                matteCoord
            ) * invUSize,
            invUSize
        );
        
        float red = texture(uBackgroundTexture, redUV).r;
        vec4 greenSample = texture(uBackgroundTexture, greenUV);
        float blue = texture(uBackgroundTexture, blueUV).b;
        
        refractColor = vec4(red, greenSample.g, blue, greenSample.a);
    }
    
    vec3 transmittedColor = vec3(0.0);
    vec3 baseColor = vec3(0.0);
    float directShare = colorModelShares.x;
    if (directShare > 0.0) {
        transmittedColor = pow(
            max(refractColor.rgb, vec3(0.0)),
            vec3(max(uTransmissionGamma, 0.01))
        );
        vec3 materialColor = materialTint.rgb * materialTint.a;
        transmittedColor *= 1.0 - materialTint.a;
        baseColor = materialColor + transmittedColor;
        baseColor = applySaturation(baseColor, uSaturation);
        float chroma = max(max(baseColor.r, baseColor.g), baseColor.b) -
            min(min(baseColor.r, baseColor.g), baseColor.b);
        baseColor = clamp(
            baseColor + vec3(chroma * max(uVibrancy, 0.0)),
            0.0,
            1.0
        );
    }
    if (directShare < 1.0) {
        // Apple's public tint is not a flat source-over wash. Its documented
        // "range of tones" is selected from backdrop brightness, while tint
        // opacity linearly mixes that opaque tonal result with the untinted
        // material. The untinted material itself treats luminance and
        // chroma separately (see parametricFaceTransfer). Saturation and
        // gamma stay available as relative adjustments where 1 is Apple's
        // face.
        float adaptiveShare = 1.0 - directShare;
        float backdropLuminance = dot(refractColor.rgb, LUMA_WEIGHTS);
        // Slot-weighted blend of each model's outputs. Wash, face transfer,
        // tint tone, border response and glint target all mix linearly, so
        // merged models cross-fade without passing through a third. The
        // direct share contributes the shared default glint target.
        vec4 washSum = vec4(0.0);
        vec2 transferSum = vec2(0.0);
        vec3 toneSum = vec3(0.0);
        float contourBoost = 0.0;
        vec3 glintSum = vec3(
            directShare * kGlintLuminance,
            0.0,
            directShare * kGlintVibrancy
        );
        for (int slot = 0; slot < 3; slot++) {
            float share = colorModelShares[slot + 1];
            if (share <= 0.0) {
                continue;
            }
            vec4 wash = parametricNeutralTint(
                slot,
                uAppearanceConfig.w,
                uTintAmount
            );
            washSum += share * vec4(wash.rgb * wash.a, wash.a);
            transferSum +=
                share * parametricFaceTransfer(slot, uTintAmount);
            // A model's border strengthens with the opacity the slider adds.
            contourBoost += share * parametricContourResponse(slot) * (
                parametricTransmittance(slot, uAppearanceConfig.w, 0.0) -
                parametricTransmittance(
                    slot,
                    uAppearanceConfig.w,
                    uTintAmount
                )
            );
            if (materialTint.a >= 0.001) {
                toneSum += share * parametricTintTone(
                    slot,
                    materialTint.rgb,
                    backdropLuminance
                );
            }
            glintSum += share * parametricGlint(slot);
        }
        gContourAlpha *= 1.0 + contourBoost;
        gGlintLuminance = glintSum.x;
        gGlintFaceGain = glintSum.y;
        gGlintVibrancy = glintSum.z;
        vec4 neutralTint = vec4(
            washSum.rgb / max(washSum.a, 1e-4),
            washSum.a / adaptiveShare
        );
        vec2 faceTransfer = transferSum / adaptiveShare;
        float transmittedLuminance = pow(
            clamp(
                backdropLuminance *
                    (1.0 + faceTransfer.x * (1.0 - backdropLuminance)),
                0.0,
                1.0
            ),
            max(uTransmissionGamma, 0.01)
        );
        vec3 neutralTransmission =
            vec3(transmittedLuminance * (1.0 - neutralTint.a)) +
            (refractColor.rgb - vec3(backdropLuminance)) *
                (faceTransfer.y * uSaturation);
        vec3 neutralBase = clamp(
            neutralTint.rgb * neutralTint.a + neutralTransmission,
            0.0,
            1.0
        );
        float neutralChroma =
            max(max(neutralBase.r, neutralBase.g), neutralBase.b) -
            min(min(neutralBase.r, neutralBase.g), neutralBase.b);
        neutralBase = clamp(
            neutralBase + vec3(neutralChroma * max(uVibrancy, 0.0)),
            0.0,
            1.0
        );
        vec3 ios27Base = neutralBase;
        if (materialTint.a >= 0.001) {
            ios27Base = mix(
                neutralBase,
                toneSum / vec3(adaptiveShare),
                materialTint.a
            );
        }
        baseColor = mix(ios27Base, baseColor, directShare);
        transmittedColor = mix(
            neutralTransmission * (1.0 - materialTint.a),
            transmittedColor,
            directShare
        );
    }

    if (uSeaGlassStrength > 0.0) {
        // The tint is the body color here, not a wash; the appearance's
        // gamma and saturation still shape the light that reaches it.
        vec3 seaGlassBackdrop = pow(
            max(refractColor.rgb, vec3(0.0)),
            vec3(max(uTransmissionGamma, 0.01))
        );
        float seaGlassLuminance = dot(seaGlassBackdrop, LUMA_WEIGHTS);
        seaGlassBackdrop = max(
            mix(vec3(seaGlassLuminance), seaGlassBackdrop, uSaturation),
            vec3(0.0)
        );
        vec2 slab = vec2(0.0);
        vec3 illumination = vec3(1.0);
        if (uSeaGlassGloss.y > 0.0) {
            float inward = max(signedEdgeDistance, 0.0);
            slab = seaGlassSlabShading(
                seaGlassFirst,
                firstBox,
                seaGlassFirstFrame,
                surfaceNormal,
                inward
            );
            if (seaGlassSecondWeight > 0.001) {
                slab = mix(
                    slab,
                    seaGlassSlabShading(
                        seaGlassSecond,
                        secondBox,
                        seaGlassSecondFrame,
                        surfaceNormal,
                        inward
                    ),
                    seaGlassSecondWeight
                );
            }
            illumination = seaGlassIllumination(matteCoord);
        }
        vec3 seaGlassTransmitted;
        vec3 seaGlassBase = seaGlassFace(
            seaGlassBackdrop,
            seaGlassLuminance,
            materialTint,
            max(signedEdgeDistance, 0.0),
            surfaceNormal,
            seaGlassPitSample.z,
            seaGlassHalfMinor,
            slab,
            illumination,
            seaGlassTransmitted
        );
        baseColor = mix(baseColor, seaGlassBase, uSeaGlassStrength);
        transmittedColor = mix(
            transmittedColor,
            seaGlassTransmitted,
            uSeaGlassStrength
        );
    }

    // Reconstruct the original material silhouette from the signed SDF. The
    // geometry alpha is only an expanded support mask, allowing the attached
    // contour to sit outside without turning those pixels into glass.
    vec3 litColor = applySpecularHighlights(
        baseColor,
        transmittedColor,
        signedEdgeDistance,
        surfaceNormal
    );
    if (uSeaGlassStrength > 0.0 && uSeaGlassGloss.w > 0.0) {
        // The chips sit inside the glass, not on it: their edges soften
        // with their size, the milky body veils them and they fade toward
        // the thin rim (seaGlassFleckDepth), in tones of the body.
        float fleckSoftness = max(0.75, uSeaGlassGrainSize * 0.35);
        vec2 flecks = seaGlassFlecks(
            seaGlassFirst,
            uSeaGlassGrainSize,
            fleckSoftness
        );
        if (seaGlassSecondWeight > 0.001) {
            flecks = mix(
                flecks,
                seaGlassFlecks(
                    seaGlassSecond,
                    uSeaGlassGrainSize,
                    fleckSoftness
                ),
                seaGlassSecondWeight
            );
        }
        float fleckDepth = seaGlassFleckDepth(
            seaGlassThickness(max(signedEdgeDistance, 0.0), uSeaGlassDepth),
            uSeaGlassScatter
        );
        litColor = seaGlassFleckedInside(
            litColor,
            materialTint,
            flecks * clamp(uSeaGlassGloss.w, 0.0, 1.0) * uSeaGlassStrength *
                fleckDepth
        );
    }
    if (uSeaGlassStrength > 0.0 && uSeaGlassSparkle > 0.0) {
        float sparkle = seaGlassSparkle(
            seaGlassFirst,
            uSeaGlassGrainSize,
            0.75,
            seaGlassLocalLight(seaGlassFirstFrame)
        );
        if (seaGlassSecondWeight > 0.001) {
            sparkle = mix(
                sparkle,
                seaGlassSparkle(
                    seaGlassSecond,
                    uSeaGlassGrainSize,
                    0.75,
                    seaGlassLocalLight(seaGlassSecondFrame)
                ),
                seaGlassSecondWeight
            );
        }
        sparkle *= uSeaGlassSparkle * uSeaGlassStrength;
        litColor = mix(
            litColor,
            vec3(kSeaGlassSparkleLuminance),
            clamp(sparkle, 0.0, 1.0)
        );
    }
    if (uSeaGlassStrength > 0.0 && uSeaGlassGloss.x > 0.0) {
        float gloss = seaGlassGloss(
            max(signedEdgeDistance, 0.0),
            surfaceNormal
        ) * uSeaGlassGloss.x * uSeaGlassStrength;
        litColor = mix(
            litColor,
            vec3(kSeaGlassSparkleLuminance),
            clamp(gloss, 0.0, 1.0)
        );
    }
    // The lit material (face, tint, saturation, lighting and in-material
    // contour) cross-fades to the refracted backdrop, whose refraction
    // scales with visibility. Fading the output rather than each parameter
    // keeps coupled factors, such as the dark face's wash and lift, linear
    // and monotonic.
    vec3 finalColor = mix(refractColor.rgb, litColor, materialVisibility);
    // Inside the material, contour absorption is handled before highlights so
    // specular light can eclipse it. Only the part outside the material is
    // composited as a translucent attached boundary. It lies outside the
    // alpha fade, so it always scales with visibility.
    float fadeAlpha = mix(1.0, appearanceVisibility, uBlurFade);
    float visibleMaterialAlpha = materialAlpha * fadeAlpha;
    float externalContourAlpha =
        contourCoverage(signedEdgeDistance).x *
        gContourAlpha *
        contourDirection(surfaceNormal) *
        appearanceVisibility;
    float alpha = visibleMaterialAlpha + externalContourAlpha;
    vec3 premultipliedColor = finalColor * visibleMaterialAlpha +
        uContourColor.rgb * externalContourAlpha;

    fragColor = vec4(premultipliedColor, alpha);
}
