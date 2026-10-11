// Copyright 2025, Tim Lehmann for whynotmake.it
//
// The tumbled sea glass surface shared by the final render pass and
// FakeGlass: the pits the sea left in the glass and the glints where single
// pits catch the light. Positions are relative to the glass so the frost
// moves with it, in the caller's pixel unit. Everything is explicitly highp:
// FakeGlass is mediump, where the hash loses its randomness.

// Share of sparkle cells holding a pit facet; each cell is four grains
// wide. Only the facets turned toward the light glint.
const float kSeaGlassSparkleDensity = 0.45;

// "Hash without Sine" (Dave Hoskins, MIT): stable in highp on every
// backend, unlike sin()-based hashes whose precision varies between GPUs.
highp float seaGlassHash(highp vec2 p) {
    highp vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

highp float seaGlassNoise(highp vec2 p) {
    highp vec2 cell = floor(p);
    highp vec2 f = p - cell;
    highp vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(
            seaGlassHash(cell),
            seaGlassHash(cell + vec2(1.0, 0.0)),
            u.x
        ),
        mix(
            seaGlassHash(cell + vec2(0.0, 1.0)),
            seaGlassHash(cell + vec2(1.0, 1.0)),
            u.x
        ),
        u.y
    );
}

// The pits at position p for grains grainSize across. xy is the direction,
// in -1..1, the pits scatter the backdrop along; z is the grain's
// brightness, centered on zero. The pits are about two grains across; the
// grain adds a finer octave.
highp vec3 seaGlassPits(highp vec2 p, highp float grainSize) {
    highp vec2 q = p / (2.0 * grainSize);
    highp float a = seaGlassNoise(q);
    highp float b = seaGlassNoise(q + vec2(37.2, 17.9));
    highp float fine = seaGlassNoise(q * 2.3 + vec2(-11.3, 5.7));
    return vec3(
        2.0 * a - 1.0,
        2.0 * b - 1.0,
        0.2 * (a + b) + 0.75 * fine - 0.575
    );
}

// Tiny glints where single pits face the light: at most one facet per
// cell, in a fraction of the cells, each tilted a random way. A facet
// glints as it turns toward lightDirection (the way the light travels), so
// moving the light makes the frost twinkle. minRadius keeps a glint at
// least that many units wide (about one physical pixel).
highp float seaGlassSparkle(
    highp vec2 p,
    highp float grainSize,
    highp float minRadius,
    highp vec2 lightDirection
) {
    highp float cellSize = grainSize * 4.0;
    highp vec2 cell = floor(p / cellSize);
    highp float pick = seaGlassHash(cell + vec2(71.3, 19.1));
    if (pick >= kSeaGlassSparkleDensity) {
        return 0.0;
    }
    // Kept off the cell border so a glint is never cut by its neighbour.
    highp vec2 center = (cell + vec2(0.25) + 0.5 * vec2(
        seaGlassHash(cell + vec2(3.7, 91.1)),
        seaGlassHash(cell + vec2(57.9, 8.3))
    )) * cellSize;
    highp float radius = max(grainSize * 0.6, minRadius);
    highp float falloff = max(1.0 - length(p - center) / radius, 0.0);
    highp float energy = pick / kSeaGlassSparkleDensity;
    highp float tilt = 6.2831853 * seaGlassHash(cell + vec2(13.1, 47.7));
    highp float facing = max(
        dot(vec2(cos(tilt), sin(tilt)), -lightDirection),
        0.0
    );
    return min(
        falloff * falloff * energy * facing * facing * facing * 2.5,
        1.0
    );
}

// Larger glass behaves like thicker material, as Apple's does: it absorbs
// more of its body color and scatters light more softly. x scales the
// body's path and y the scatter, by the shape's half short side relative to
// the pebble's depth: a 64 pt tile with the default 18 pt depth is near 1,
// a large card thicker. Mirrored by seaGlassSizeResponse in Dart.
vec2 seaGlassSizeResponse(float halfMinor, float depth) {
    float size = smoothstep(1.0, 4.0, halfMinor / max(depth, 0.001));
    return vec2(mix(0.85, 1.2, size), mix(0.9, 1.3, size));
}

// Parabolic thickness of a rounded pebble: 0 at the silhouette, 1 from
// depth inward.
float seaGlassThickness(float inwardDistance, float depth) {
    float x = clamp(inwardDistance / max(depth, 0.001), 0.0, 1.0);
    return x * (2.0 - x);
}
