#version 450
// Android port: the present's bilinear scale (DM_VK_UPSCALE=bilinear, or a picture that is not made larger): fsr/fsr1.glsl, drawn by gfx_vk_fsr.c
#extension GL_GOOGLE_include_directive : require
#define FSR_H 0
#include "fsr/fsr1.glsl"
void main() { fsr_blit_main(); }
