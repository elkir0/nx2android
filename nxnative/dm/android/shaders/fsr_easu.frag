#version 450
// Android port: FSR 1 EASU at present time, 32-bit: fsr/fsr1.glsl, drawn by gfx_vk_fsr.c
#extension GL_GOOGLE_include_directive : require
#define FSR_H 0
#include "fsr/fsr1.glsl"
void main() { fsr_easu_main(); }
