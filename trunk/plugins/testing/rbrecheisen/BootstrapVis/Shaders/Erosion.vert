#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform int  fillDilation;
uniform int  outlineWidth;
uniform vec2 pixelRatio;
uniform vec3 minExtent;
uniform vec3 maxExtent;

varying vec2 minScreenExtent;
varying vec2 maxScreenExtent;

void main( void )
{
	gl_Position = gl_Vertex;
	gl_FrontColor = gl_Color;
	gl_TexCoord[0] = gl_MultiTexCoord0;

	vec4 corners[8];
	corners[0] = vec4(minExtent.x, minExtent.y, minExtent.z, 1);
	corners[1] = vec4(minExtent.x, minExtent.y, maxExtent.z, 1);
	corners[2] = vec4(minExtent.x, maxExtent.y, minExtent.z, 1);
	corners[3] = vec4(minExtent.x, maxExtent.y, maxExtent.z, 1);
	corners[4] = vec4(maxExtent.x, minExtent.y, minExtent.z, 1);
	corners[5] = vec4(maxExtent.x, minExtent.y, maxExtent.z, 1);
	corners[6] = vec4(maxExtent.x, maxExtent.y, minExtent.z, 1);
	corners[7] = vec4(maxExtent.x, maxExtent.y, maxExtent.z, 1);

	minScreenExtent = vec2(1,1);
	maxScreenExtent = vec2(0,0);

	for(int i = 0; i < 8; ++i)
	{
		vec4 projectedCorner = gl_ModelViewProjectionMatrix * corners[i];
		vec2 screenCorner    = 0.5 * (projectedCorner.xy / projectedCorner.w) + 0.5;
		minScreenExtent = min(minScreenExtent, screenCorner);
		maxScreenExtent = max(maxScreenExtent, screenCorner);
	}

	vec2 expansion = pixelRatio * (2 + fillDilation + outlineWidth);
	minScreenExtent -= expansion;
	maxScreenExtent += expansion;	
}
