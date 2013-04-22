/**
 * Fragment Shader:
 *	Builds a silhouette out of the bound depth texture
 *  and overlays the bound color texture on top of it.
 *
 * by Ron Otten
 *
 * 2009-01-08
 *
 */

#version 120
#extension GL_EXT_gpu_shader4 	   : enable

// ================================================

uniform sampler2D depthSampler;
uniform sampler2D colorSampler;

uniform vec2 pixelRatio;

uniform int fillDilation;
uniform vec3 fillColor;

uniform int outlineWidth;
uniform vec3 lineColor;

uniform float depthTreshold;

uniform float nearPlane;
uniform float farPlane;

uniform bool orthographic;

varying vec2 minScreenExtent;
varying vec2 maxScreenExtent;

// ================================================

#define kernelRadius fillDilation


vec2 kernelToTexCoord(in ivec2 coord)
{
	vec2 offset;
	offset.x = coord.x * pixelRatio.x;
	offset.y = coord.y * pixelRatio.y;	
	
	return clamp(gl_TexCoord[0].st + offset, vec2(0,0), vec2(1,1));	
}

float linearizeDepth(in float depth)
{	
	if (orthographic)
		return mix(nearPlane, farPlane, depth);
	else
		return (nearPlane * farPlane) /( farPlane - depth * (farPlane - nearPlane));
}

void main(void)
{
	float pointDepth;
	float silDepth;
	float lineDepth;

	// Perform an "early out"-test based on the screen space extents.
	if (clamp(gl_TexCoord[0].st, minScreenExtent, maxScreenExtent) != gl_TexCoord[0].st)
	{
		discard;		
	}
	else
	{		
		// Sample the depth at the point itself.
		pointDepth = texture2D(depthSampler, gl_TexCoord[0].st).r;
			
		// Sample the depth for the silhouette.
		silDepth = 1;
		for(int i = -kernelRadius; i < kernelRadius; ++i)		
			for(int j = -kernelRadius; j < kernelRadius; ++j)
			{
				// Don't bother with the point itself again.
				if ((i == 0) && (j == 0)) continue;

				// Don't bother with any kernel cells with a distance from the kernel's center that is larger than the kernel's radius.
				if (kernelRadius < length(vec2(i,j))) continue;

				float depthSample = texture2D(depthSampler, kernelToTexCoord(ivec2(i,j))).r;
				silDepth = min(silDepth, depthSample);
			}
		
		// Sample the depth for the outlines.
		lineDepth = 1;
		for(int i = -(kernelRadius + outlineWidth); i < (kernelRadius + outlineWidth); ++i)
			for(int j = -(kernelRadius + outlineWidth); j < (kernelRadius + outlineWidth); ++j)
			{
				// Don't bother with any kernel cells we will already have processed, i.e. |(i,j)| <= kernelRadius
				if (kernelRadius >= length(vec2(i,j))) continue;

				// Don't bother with any kernel cells with a too large radius again...
				if ((kernelRadius + outlineWidth) < length(vec2(i,j))) continue;

				float depthSample = texture2D(depthSampler, kernelToTexCoord(ivec2(i,j))).r;
				lineDepth = min(lineDepth, depthSample);
			}
		
		//	Linearize the depth buffer for a fair comparison with the depthTreshold
		float linPointDepth = linearizeDepth(pointDepth);
		float linSilDepth = linearizeDepth(silDepth);
		float linLineDepth = linearizeDepth(lineDepth);
		
		if ((pointDepth > 0) && (pointDepth < 1) && (linPointDepth < (linSilDepth + depthTreshold)) && (linPointDepth < (linLineDepth + depthTreshold)))
		{
			gl_FragColor = texture2D(colorSampler, gl_TexCoord[0].st);
			gl_FragDepth = pointDepth;
		}
		else if ((silDepth > 0) && (silDepth < 1) && (linSilDepth < (linLineDepth + depthTreshold)))
		{
			gl_FragColor = vec4(fillColor, 1);
			gl_FragDepth = silDepth;
		}
		else if ((lineDepth > 0) && (lineDepth < 1))
		{
			gl_FragColor = vec4(lineColor, 1);
			gl_FragDepth = lineDepth;
		}
		else
		{
			discard;
		}
	}
}

