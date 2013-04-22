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

uniform bool inkLines;

// ================================================

#define lineWidth			gl_Color.g
#define distanceFromLine	gl_Color.r
#define luminosity			gl_Color.b

void main(void)
{
	if (inkLines)
	{
		float lineThreshold = smoothstep(lineWidth - 0.01, lineWidth, distanceFromLine);
		float luminosityThreshold = mix(minLuminosity, maxLuminosity, luminosity);

		vec3 litLineColor = mix(lineColor, finColor, luminosityThreshold);	
		gl_FragColor = vec4(mix(litLineColor, finColor, lineThreshold), 1);
	}
	else
	{
		gl_FragColor = vec4(finColor, 1);
	}
}

