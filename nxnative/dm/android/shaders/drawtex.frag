#version 450
// Android port: nvnCommandBufferDrawTexture (gfx_vk_exec.c draw_tex): the source region of the texture, through the handle's sampler, into every bound colour target
// (dm/gfx/msl/draw_tex.metal's mapping): d = the destination's origin (pixels) and source texels per destination pixel, s = the source's origin (texels) and its level size
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(push_constant) uniform P { vec4 d; vec4 s; } p;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
layout(location = 5) out vec4 o5;
layout(location = 6) out vec4 o6;
layout(location = 7) out vec4 o7;
void main() {
    vec2 st = p.s.xy + (gl_FragCoord.xy - p.d.xy) * p.d.zw;
    vec4 c = texture(tex, st / p.s.zw);
    o0 = c; o1 = c; o2 = c; o3 = c; o4 = c; o5 = c; o6 = c; o7 = c;
}
