// Copyright 2025, Tim Lehmann for whynotmake.it
//
// sdRoundedBox and sdEllipse follow Inigo Quilez's 2D distance functions
// (https://iquilezles.org/articles/distfunctions2d/), MIT License,
// Copyright © 2015 Inigo Quilez; see third_party/inigo_quilez/LICENSE.
// sdRoundedSuperellipse is adapted from Flutter's Impeller
// (impeller/entity/shaders/uber_sdf.frag, Flutter 3.47.1):
//   Copyright 2013 The Flutter Authors. All rights reserved.
//   Use of this source code is governed by a BSD-style license that can be
//   found in third_party/flutter/LICENSE.

// Signed distances of FakeGlass shapes, in logical pixels of the shape's
// centered local space. Shared by the analytic surface and the backdrop edge
// pass so both put the silhouette in the same place.

float sdRoundedBox(vec2 p, vec2 halfSize, float radius) {
    radius = clamp(radius, 0.0, min(halfSize.x, halfSize.y));
    vec2 q = abs(p) - halfSize + vec2(radius);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

float sdEllipse(vec2 p, vec2 radius) {
    radius = max(radius, vec2(0.001));
    float k0 = length(p / radius);
    float k1 = length(p / (radius * radius));
    return k0 * (k0 - 1.0) / max(k1, 0.001);
}

// First-order distance F / |grad F| to the unit superellipse
// |x|^n + |y|^n = 1. Near the silhouette it is within 0.03 pt of the
// six-step bisection real glass uses (0.12 pt within 6 pt), without its loop
// of sin, cos and pow.
float sdSuperellipseArc(vec2 p, float n) {
    p = abs(p) + vec2(1e-6);
    float s = pow(p.x, n) + pow(p.y, n);
    float f = pow(s, 1.0 / n) - 1.0;
    vec2 gradient = vec2(pow(p.x, n - 1.0), pow(p.y, n - 1.0)) *
        pow(s, 1.0 / n - 1.0);
    return f / max(length(gradient), 1e-3);
}

// Flutter 3.47's rounded superellipse (Impeller's symmetric UberSDF): per
// octant, a circular arc around the diagonal and a superellipse arc towards
// the edge midpoint. The parameters come from roundedSuperellipseParameters.
float sdRoundedSuperellipse(
    vec2 p,
    vec2 halfSize,
    float radius,
    vec4 degreeAndSpans,
    vec4 circleCenters,
    vec4 semiAxisAndRadii
) {
    if (radius <= 1e-4) {
        return max(abs(p).x - halfSize.x, abs(p).y - halfSize.y);
    }
    vec2 local = abs(p);
    float c = semiAxisAndRadii.x - semiAxisAndRadii.y;
    vec2 octant;
    float degree;
    float span;
    float axis;
    float circleRadius;
    vec2 circleCenter;
    if (local.y + c > local.x) {
        octant = local + vec2(0.0, c);
        degree = degreeAndSpans.x;
        axis = semiAxisAndRadii.x;
        span = degreeAndSpans.z;
        circleCenter = circleCenters.xy;
        circleRadius = semiAxisAndRadii.z;
    } else {
        octant = local.yx - vec2(0.0, c);
        degree = degreeAndSpans.y;
        axis = semiAxisAndRadii.y;
        span = degreeAndSpans.w;
        circleCenter = circleCenters.zw;
        circleRadius = semiAxisAndRadii.w;
    }
    vec2 relative = octant - circleCenter;
    float deltaTheta = atan(relative.y, relative.x) - 0.78539816;
    deltaTheta = mod(deltaTheta + 3.14159265, 6.28318531) - 3.14159265;
    if (abs(deltaTheta) < abs(span)) {
        return length(relative) - circleRadius;
    }
    if (degree < 2.0) {
        return max(abs(octant).x - axis, abs(octant).y - axis);
    }
    return sdSuperellipseArc(octant / max(axis, 1e-4), degree) * axis;
}

// A blob (LiquidBlob): the ellipse of half size h, its radius rippled by up
// to three waves, packed as in the geometry pass's sdf.glsl (where it is
// documented): (count1, amplitude1, phase1, count2), (amplitude2, phase2,
// count3, amplitude3), (phase3, norm, -, -). Returns the radius at angle
// theta and its derivative.
//
// Explicitly highp: FakeGlass is mediump, where a^2 b^2 and ellipse^3
// overflow half precision (65504) once a half axis passes about 16 px.
highp vec2 fakeBlobRadius(
    highp float theta,
    highp vec2 h,
    highp vec4 waves0,
    highp vec4 waves1,
    highp vec4 waves2
) {
    highp float c = cos(theta);
    highp float s = sin(theta);
    highp vec2 axes = max(h, vec2(1e-4));
    highp float a = axes.x;
    highp float b = axes.y;
    highp float ellipse =
        a * b * inversesqrt(max(b * b * c * c + a * a * s * s, 1e-12));
    highp float ellipseSlope =
        -ellipse * ellipse * ellipse * (a * a - b * b) * s * c /
        (a * a * b * b);
    highp float x1 = waves0.x * theta + waves0.z;
    highp float x2 = waves0.w * theta + waves1.y;
    highp float x3 = waves1.z * theta + waves2.x;
    highp float ripple = 1.0 + waves0.y * cos(x1) + waves1.x * cos(x2) +
        waves1.w * cos(x3);
    highp float rippleSlope = -waves0.y * waves0.x * sin(x1) -
        waves1.x * waves0.w * sin(x2) - waves1.w * waves1.z * sin(x3);
    highp float norm = max(waves2.y, 1e-4);
    return vec2(
        ellipse * ripple / norm,
        (ellipseSlope * ripple + ellipse * rippleSlope) / norm
    );
}

// The angle of p around the blob's center. atan(0, 0) is undefined (SPIR-V
// leaves it so), so the center itself takes angle 0.
highp float fakeBlobAngle(highp vec2 p) {
    return dot(p, p) > 0.0 ? atan(p.y, p.x) : 0.0;
}

// The radial gap to the blob's outline, shortened by the outline's slope
// where the ray meets it.
float sdFakeBlob(
    highp vec2 p,
    highp vec2 h,
    highp vec4 waves0,
    highp vec4 waves1,
    highp vec4 waves2
) {
    highp vec2 radius =
        fakeBlobRadius(fakeBlobAngle(p), h, waves0, waves1, waves2);
    highp float slope = radius.y / max(radius.x, 1e-4);
    return (length(p) - radius.x) * inversesqrt(1.0 + slope * slope);
}

// The blob's outward normal where the ray through p meets its outline.
vec2 fakeBlobNormal(
    highp vec2 p,
    highp vec2 h,
    highp vec4 waves0,
    highp vec4 waves1,
    highp vec4 waves2
) {
    highp float theta = fakeBlobAngle(p);
    highp vec2 radius = fakeBlobRadius(theta, h, waves0, waves1, waves2);
    vec2 radial = vec2(cos(theta), sin(theta));
    vec2 tangent = vec2(-radial.y, radial.x);
    return normalize(radial - tangent * (radius.y / max(radius.x, 1e-4)));
}

// type: 0 oval, 1 rounded rectangle, 2 rounded superellipse, 3 blob, whose
// waves arrive in the superellipse's parameters. Deeper than
// 6 pt inside, where only the soft bevel reads the distance, a superellipse
// uses its rounded box (they agree to about half a point there).
float sdFakeGlassShape(
    float type,
    vec2 p,
    vec2 halfSize,
    float radius,
    vec4 degreeAndSpans,
    vec4 circleCenters,
    vec4 semiAxisAndRadii
) {
    if (type < 0.5) {
        return sdEllipse(p, halfSize);
    }
    if (type > 2.5) {
        return sdFakeBlob(
            p,
            halfSize,
            degreeAndSpans,
            circleCenters,
            semiAxisAndRadii
        );
    }
    float box = sdRoundedBox(p, halfSize, radius);
    if (type < 1.5 || box < -6.0) {
        return box;
    }
    float superellipse = sdRoundedSuperellipse(
        p,
        halfSize,
        min(radius, min(halfSize.x, halfSize.y)),
        degreeAndSpans,
        circleCenters,
        semiAxisAndRadii
    );
    return mix(superellipse, box, smoothstep(4.0, 6.0, -box));
}
