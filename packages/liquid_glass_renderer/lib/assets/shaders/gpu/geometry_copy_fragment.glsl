// Copyright 2025, Tim Lehmann for whynotmake.it
//
// Copies the previous geometry matte into a new one, shifted by whole pixels,
// before the geometry pass redraws only the regions around shapes that
// changed (see FlutterGpuGeometryRenderer). Sampled nearest at texel
// centers, so the copy is exact.

precision highp float;

uniform sampler2D uPreviousMatte;

layout(std140) uniform CopyUniforms {
    // xy: offset from a pixel of the new matte to the same point in the
    // previous one, in pixels. zw: size of the previous matte's texture.
    vec4 uCopySource;
} copyUniforms;

out vec4 fragColor;

void main() {
    vec2 source = gl_FragCoord.xy + copyUniforms.uCopySource.xy;
    fragColor = texture(uPreviousMatte, source / copyUniforms.uCopySource.zw);
}
