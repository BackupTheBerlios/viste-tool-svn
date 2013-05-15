#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D densitySampler;
uniform sampler2D densityDepthSampler;
uniform float maxDensity;
uniform int kernelRadius;
uniform vec2 pixelRatio;

#define SQRT_2PI 2.506628274635
#define SIGMA (2.0 * kernelRadius + 1.0) / 6.0

vec2 kernelToTexCoord( in ivec2 coord )
{
	vec2 offset;
	offset.x = coord.x * pixelRatio.x;
	offset.y = coord.y * pixelRatio.y;	
	
	return clamp( gl_TexCoord[0].st + offset, vec2(0,0), vec2(1,1) );
}

float gauss( float radius )
{
	float tmp = -(radius * radius) / (2.0 * SIGMA * SIGMA);
	tmp = exp( tmp );
	tmp = tmp * (1.0 / (SQRT_2PI * SIGMA));
	return tmp;
}

void main( void )
{
	float density = 0.0;
	float sampleDepth = texture2D( densityDepthSampler, gl_TexCoord[0].st ).r;
	
	for( int i = -kernelRadius; i < kernelRadius; ++i )
	{
		for( int j = -kernelRadius; j < kernelRadius; ++j )
		{
			float radius = length( vec2(i,j) );
			float weight = gauss( radius );
			float tmp = texture2D( densitySampler, kernelToTexCoord( ivec2(i,j) ) ).r;
			density += (weight * tmp);

			float depth = texture2D( densityDepthSampler, kernelToTexCoord( ivec2(i,j) ) ).r;
			sampleDepth = min( sampleDepth, depth );
		}
	}
	
	gl_FragColor = vec4( vec3( density ), 1.0 );
	gl_FragDepth = sampleDepth;
}
