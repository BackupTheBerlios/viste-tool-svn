#version 120
#extension GL_EXT_gpu_shader4 : enable

//--------------------------------------------------------------

uniform sampler2D erosionSampler;
uniform sampler2D erosionDepthSampler;
uniform sampler2D densitySampler;
uniform sampler2D silhouetteSampler;
uniform sampler2D silhouetteDepthSampler;

uniform vec2 pixelRatio;
uniform vec4 lineColor;
uniform vec4 densityColor;

uniform int fillErosion;
uniform int lineWidth;

uniform float fillOpacity;
uniform float maxDensity;
uniform float depthThreshold;

uniform bool firstPass;
uniform bool densityColoring;

varying vec2 minScreenExtent;
varying vec2 maxScreenExtent;

//--------------------------------------------------------------

#define kernelRadius fillErosion

//--------------------------------------------------------------

vec2 kernelToTexCoord( in ivec2 coord )
{
	vec2 offset;
	offset.x = coord.x * pixelRatio.x;
	offset.y = coord.y * pixelRatio.y;	
	
	return clamp( gl_TexCoord[0].st + offset, vec2(0,0), vec2(1,1) );	
}

//--------------------------------------------------------------

void main( void )
{
	if( clamp( gl_TexCoord[0].st, minScreenExtent, maxScreenExtent ) != gl_TexCoord[0].st )
	{
		discard;		
	}
	else
    {
		vec4 prevColor = vec4( 0.0 );
		float prevDepth = 0.0;

        if( ! firstPass )
        {            
            prevColor = texture2D( erosionSampler, gl_TexCoord[0].st );
            prevDepth = texture2D( erosionDepthSampler, gl_TexCoord[0].st ).r;
            
            if( prevColor.rgb != vec3(0) )
            {
                gl_FragColor = prevColor;
                gl_FragDepth = prevDepth;
                return;
            }
        }

		// Check the current pixel color. If it is (0,0,0) we assume it's background
		// and skip further processing. If it's a foreground pixel we start to look
		// in its immediate neighborhood

        vec4 pointColor = texture2D( silhouetteSampler, gl_TexCoord[0].st );
        
		if( pointColor.rgb != vec3( 0 ) )
		{		
			vec4 sampleColor = pointColor;
			float sampleDepth = texture2D( silhouetteDepthSampler, gl_TexCoord[0].st ).r;

			bool isOutline = false;

			for( int i = -kernelRadius; i < kernelRadius; i++ )
			{
			    for( int j = -kernelRadius; j < kernelRadius; j++ )
			    {
			        if( i == 0 && j == 0 )
			            continue;
			        if( length( vec2(i,j) ) > kernelRadius )
			            continue;

			        vec4 color = texture2D( silhouetteSampler, kernelToTexCoord( ivec2( i, j ) ) );
					if( color.rgb == vec3( 0 ) )
					{
						sampleColor = vec4( 0 );
						sampleDepth = 1.0;

						j = kernelRadius;
						i = kernelRadius;
					}
			    }
			}

			// If the sample color was set to (0,0,0,0) the pixel should be set to the
			// background color, thereby performing the erosion. If not, the pixel is
			// sufficiently inside the interior of the silhouette. However, we need to
			// check whether it is an outline pixel by extending the kernel radius.

			if( sampleColor != vec4( 0 ) )
			{
				for( int i = -(kernelRadius + lineWidth); i < (kernelRadius + lineWidth); i++ )
				{
					for( int j = -(kernelRadius + lineWidth); j < (kernelRadius + lineWidth); j++ )
					{
						if( i == 0 && j == 0 )
						    continue;
						if( length( vec2(i,j) ) <= kernelRadius )
							continue;
						if( length( vec2(i,j) ) > (kernelRadius + lineWidth) )
						    continue;

						vec4 color = texture2D( silhouetteSampler, kernelToTexCoord( ivec2( i, j ) ) );
						if( color.rgb == vec3( 0 ) )
						{
							sampleColor = lineColor;
							sampleDepth = sampleDepth;
							isOutline = true;

							j = (kernelRadius + lineWidth);
							i = (kernelRadius + lineWidth);
						}
					}
				}
			}

   			gl_FragColor = sampleColor;
   			gl_FragDepth = sampleDepth;
		}
		else
		{
			discard;
		}
	}
}
