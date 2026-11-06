#version 450
// Android port: nvnCommandBufferDrawTexture (gfx_vk_exec.c draw_tex): one triangle over the whole pass, cut to the destination rectangle by the scissor
void main() {
    vec2 p = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1) * 4.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
}
