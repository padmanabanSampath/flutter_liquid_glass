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

// MARK: Slab shading (SeaGlassFinish.shading, glow)

// How the light shades a thick slab of milky glass. The body color shows
// where the slab is thickest, just inside its center toward the light. The
// light that goes through brightens the slab where it is thin, along the
// outline except on the wall facing the light, and most where it gathers,
// on the far side (glow). The half facing the light keeps the body color,
// so the pale glow is on the far side only. x is the brightening, a factor
// on the body; y the share of white.
//   q:      position in the shape's box, -1..1 per axis
//   toward: unit direction toward the light in the shape's frame
//   rim:    how much the outline here faces the light, -1..1
//   edge:   inward distance from the outline over the slab's depth
//   tilt:   how much the light's direction shades the slab, 0..1; a light
//           right above a shape lights it evenly
vec2 slabShading(
    vec2 q,
    vec2 toward,
    float rim,
    float edge,
    float tilt,
    float glow
) {
    float across = clamp(dot(q, toward), -1.0, 1.0) * tilt;
    float thin = (1.0 - smoothstep(0.0, 1.4, edge)) *
        mix(1.0, 0.1, smoothstep(0.0, 1.0, rim * tilt));
    float far = smoothstep(-0.05, 1.0, -across);
    return vec2(max(0.1 * thin, 0.17 * glow * far), 0.0);
}

// The slab's color under shading (slabShading). Light added to the body
// saturates its strongest channels first, so a blue slab glows toward cyan
// and white.
vec3 slabColor(vec3 body, vec2 shading) {
    vec3 lit = min(body * (1.0 + shading.x), vec3(1.0));
    return mix(lit, vec3(1.0), shading.y);
}

// The share of its scatter a slab keeps at inward distance from its
// outline, for a slab of the given depth: the thin edge is clear glass, so
// the backdrop shows through it, tinted and bent by the lens, and the milk
// sets in a little inside.
float slabEdgeScatter(float inward, float depth) {
    return mix(0.25, 1.0, smoothstep(0.0, 0.35 * depth, inward));
}

// The direction a slab's shading comes from, given the unit direction
// toward the light: the face also reflects the sky above the screen, so
// under daylight it is shaded from a little higher up than the light alone.
vec2 slabShadeDirection(vec2 towardLight, bool positioned) {
    vec2 shade = towardLight + vec2(0.0, positioned ? 0.0 : -0.3);
    return shade / max(length(shade), 1e-6);
}

// MARK: Flecks (SeaGlassFinish.flecks)

// Signed distance from uv to the edge from a to b of a shard wound
// counterclockwise, positive inside.
highp float seaGlassShardEdge(highp vec2 uv, highp vec2 a, highp vec2 b) {
    highp vec2 e = b - a;
    return dot(uv - a, vec2(-e.y, e.x)) / max(length(e), 1e-6);
}

// The corner of a shard about a quarter turn k past its first, turned a
// little further at random, at a random distance up to size from its
// middle. The turn is small, so its sine and cosine are approximated.
highp vec2 seaGlassShardCorner(
    highp vec2 cell,
    highp float k,
    highp float size
) {
    highp float jitter =
        0.55 * (2.0 * seaGlassHash(cell + vec2(17.3 * k, 5.1)) - 1.0);
    highp vec2 turned = vec2(1.0 - 0.5 * jitter * jitter, jitter);
    highp vec2 corner = k < 0.5
        ? turned
        : k < 1.5
        ? vec2(-turned.y, turned.x)
        : k < 2.5
        ? -turned
        : vec2(turned.y, -turned.x);
    return corner * size *
        mix(0.45, 1.0, seaGlassHash(cell + vec2(3.9, 23.7 * k + 1.0)));
}

// The shard of cell home of a terrazzo layer at p, if the cell holds one
// (see seaGlassChip): its coverage, antialiased over aa units.
highp float seaGlassShard(
    highp vec2 p,
    highp vec2 home,
    highp float cellSize,
    highp vec2 seed,
    highp float density,
    highp float minSize,
    highp float maxSize,
    highp float aa
) {
    highp vec2 cell = home + seed;
    if (seaGlassHash(cell) >= density) {
        return 0.0;
    }
    highp vec2 center = (home + vec2(0.15) + 0.7 * vec2(
        seaGlassHash(cell + vec2(3.7, 91.1)),
        seaGlassHash(cell + vec2(57.9, 8.3))
    )) * cellSize;
    highp vec2 axis = normalize(vec2(
        seaGlassHash(cell + vec2(13.1, 47.7)) - 0.5,
        seaGlassHash(cell + vec2(47.7, 13.1)) - 0.5
    ) + vec2(1e-3, 0.0));
    highp vec2 d = p - center;
    highp float stretch = mix(1.0, 2.2, seaGlassHash(cell + vec2(7.7, 63.1)));
    highp vec2 uv = vec2(dot(d, axis), dot(d, vec2(-axis.y, axis.x)) * stretch);
    highp float size = cellSize *
        mix(minSize, maxSize, seaGlassHash(cell + vec2(29.3, 5.9)));
    highp vec2 c0 = seaGlassShardCorner(cell, 0.0, size);
    highp vec2 c1 = seaGlassShardCorner(cell, 1.0, size);
    highp vec2 c2 = seaGlassShardCorner(cell, 2.0, size);
    highp vec2 c3 = seaGlassShardCorner(cell, 3.0, size);
    highp float inside = min(
        min(seaGlassShardEdge(uv, c0, c1), seaGlassShardEdge(uv, c1, c2)),
        min(seaGlassShardEdge(uv, c2, c3), seaGlassShardEdge(uv, c3, c0))
    );
    // The stretch scales distances across the shard by up to its factor.
    return clamp(inside / (max(aa, 1e-4) * stretch) + 0.5, 0.0, 1.0);
}

// One layer of terrazzo chips: in a share density of the cells, an angular
// shard (four random corners, some nearly a triangle) of random size,
// stretch and turn, anywhere in its cell. A shard reaches at most 0.15 of a
// cell past its border, so p only looks up its own cell and the three
// nearest neighbors. Sizes run from minSize to maxSize times the cell size,
// at most 0.3. Returns the coverage at p, antialiased over aa units.
highp float seaGlassChip(
    highp vec2 p,
    highp float cellSize,
    highp vec2 seed,
    highp float density,
    highp float minSize,
    highp float maxSize,
    highp float aa
) {
    highp vec2 q = p / cellSize;
    highp vec2 home = floor(q);
    highp vec2 side = sign(q - home - vec2(0.5));
    return max(
        max(
            seaGlassShard(p, home, cellSize, seed, density, minSize, maxSize, aa),
            seaGlassShard(
                p,
                home + vec2(side.x, 0.0),
                cellSize,
                seed,
                density,
                minSize,
                maxSize,
                aa
            )
        ),
        max(
            seaGlassShard(
                p,
                home + vec2(0.0, side.y),
                cellSize,
                seed,
                density,
                minSize,
                maxSize,
                aa
            ),
            seaGlassShard(
                p,
                home + side,
                cellSize,
                seed,
                density,
                minSize,
                maxSize,
                aa
            )
        )
    );
}

// The chips at p for grains grainSize across: x the dark ones, deeper bits
// of the body, and y the light ones, whitish crumbs.
highp vec2 seaGlassFlecks(
    highp vec2 p,
    highp float grainSize,
    highp float aa
) {
    return vec2(
        seaGlassChip(p, grainSize * 3.1, vec2(0.0), 0.7, 0.18, 0.3, aa),
        seaGlassChip(
            p + vec2(0.37, 0.71) * grainSize,
            grainSize * 3.6,
            vec2(113.0, 271.0),
            0.7,
            0.15,
            0.3,
            aa
        )
    );
}

// The face with flecks of coverage flecks (seaGlassFlecks) over it.
vec3 seaGlassFlecked(vec3 face, vec2 flecks) {
    float luminance = dot(face, vec3(0.2126, 0.7152, 0.0722));
    vec3 dark = mix(vec3(luminance), face, 0.8) * 0.9;
    face = mix(face, dark, clamp(flecks.x, 0.0, 1.0));
    return mix(face, mix(face, vec3(1.0), 0.35), clamp(flecks.y, 0.0, 1.0));
}

// The face with chips embedded in glass of tint (premultiplied body color,
// alpha its density): dark chips a deeper tone of the body, light ones a
// pale tint of it, rather than grey and white specks on the surface.
vec3 seaGlassFleckedInside(vec3 face, vec4 tint, vec2 flecks) {
    vec3 body = mix(
        vec3(1.0),
        clamp(tint.rgb, vec3(0.02), vec3(1.0)),
        clamp(tint.a, 0.0, 1.0)
    );
    vec3 dark = face * mix(vec3(1.0), body, 0.55) * 0.84;
    vec3 light = mix(face, mix(body, vec3(1.0), 0.65), 0.45);
    face = mix(face, dark, clamp(flecks.x, 0.0, 1.0));
    return mix(face, light, clamp(flecks.y, 0.0, 1.0));
}

// How deep inside the glass its chips show, from 0 to 1: the milky body
// veils them as much as it scatters, and they fade where the glass thins
// toward its rim, so they read as suspended inside the stone.
float seaGlassFleckDepth(float thickness, float scatter) {
    return mix(0.3, 1.0, clamp(thickness, 0.0, 1.0)) *
        (1.0 - 0.45 * clamp(scatter, 0.0, 1.0));
}

// How much SeaGlassFinish.colorClarity clears glass of body tint, from 0
// to clarity: glass of a color clears as much as it holds (its alpha), so
// frosted white and colorless glass keep their frost, and a blended bridge
// clears gradually with its color.
float seaGlassColorClear(vec4 tint, float clarity) {
    if (clarity <= 0.0) {
        return 0.0;
    }
    vec3 c = clamp(tint.rgb, vec3(0.0), vec3(1.0));
    float high = max(max(c.r, c.g), c.b);
    float chroma = (high - min(min(c.r, c.g), c.b)) / max(high, 0.001);
    return clarity * smoothstep(0.04, 0.3, chroma) * clamp(tint.a, 0.0, 1.0);
}
