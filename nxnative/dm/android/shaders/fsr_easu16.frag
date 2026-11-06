#version 450
// Android port: FSR 1 EASU at present time, float16_t arithmetic (shaderFloat16): fsr/fsr1.glsl, drawn by gfx_vk_fsr.c
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define FSR_H 1
#include "fsr/fsr1.glsl"
void main() { fsr_easu_main(); }
