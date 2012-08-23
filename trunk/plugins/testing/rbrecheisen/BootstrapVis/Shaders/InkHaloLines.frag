/**
 * Fragment Shader: Inks the fins with lines
 * by Ron Otten
 *
 * 2009-01-08
 *
 */

#version 120
#extension GL_EXT_gpu_shader4 	   : enable

// ================================================

uniform vec3 finColor;
uniform vec3 lineColor;

uniform float minLuminosity;
uniform float maxLuminosity;

// ================================================

#define luminosity			gl_Color.b

void main(void)
{
	float luminosityThreshold = mix(minLuminosity, maxLuminosity, luminosity);
	gl_FragColor = vec4(mix(lineColor, finColor, luminosityThreshold), 1);
}

