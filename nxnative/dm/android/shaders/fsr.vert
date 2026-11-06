#version 450
// Android port: the present's upscaling passes (gfx_vk_fsr.c): one triangle over the whole target
void main() {
    vec2 p = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1) * 4.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
}
