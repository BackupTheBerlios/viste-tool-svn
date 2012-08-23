#version 120
#extension GL_EXT_gpu_shader4 	   : enable

//----------------------------------------------------------------

uniform sampler2D depthSampler;
uniform sampler2D colorSampler;
uniform sampler2D densitySampler;
uniform sampler2D previousSampler;
uniform sampler2D previousDepthSampler;

uniform vec2 pixelRatio;
uniform vec4 fillColor;
uniform vec4 lineColor;
uniform vec4 densityColor;

uniform int fillDilation;
uniform int outlineWidth;

uniform float depthTreshold;
uniform float nearPlane;
uniform float farPlane;
uniform float maxDensity;

uniform bool orthographic;
uniform bool firstPass;
uniform bool densityColoring;
uniform bool densityWeighting;
uniform bool overwriteEnabled;

varying vec2 minScreenExtent;
varying vec2 maxScreenExtent;

//----------------------------------------------------------------

vec2 kernelToTexCoord(in ivec2 coord)
{
	vec2 offset;
	offset.x = coord.x * pixelRatio.x;
	offset.y = coord.y * pixelRatio.y;	
	
	return clamp(gl_TexCoord[0].st + offset, vec2(0,0), vec2(1,1));	
}

//----------------------------------------------------------------

float linearizeDepth(in float depth)
{	
	if (orthographic)
		return mix(nearPlane, farPlane, depth);
	else
		return (nearPlane * farPlane) /( farPlane - depth * (farPlane - nearPlane));
}

//----------------------------------------------------------------

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
		vec4 prevColor = vec4( 0.0 );
		float prevDepth = 0.0;
        bool contourOnly = false;
        
        if( firstPass == false )
	    {
		    prevColor = texture2D( previousSampler, gl_TexCoord[0].st );
		    prevDepth = texture2D( previousDepthSampler, gl_TexCoord[0].st ).r;
		    
            if( prevColor.rgb != vec3( 0 ) )
            {
                if( overwriteEnabled )
                {
                    gl_FragColor = prevColor;
                    gl_FragDepth = prevDepth;
                    return;
                }
                else
                {                
                    contourOnly = true;
                }
            }
        }

		// Sample the depth at the point itself.
		pointDepth = texture2D(depthSampler, gl_TexCoord[0].st).r;
		
	    float density = 255.0 * texture2D( densitySampler, gl_TexCoord[0].st ).r;
		density = clamp( density / maxDensity, 0.0, maxDensity );
	    
		float weight = 0.0;
		int kernelRadius = fillDilation;

		if( densityWeighting )
		{
			for( int i = -(kernelRadius + outlineWidth); i < (kernelRadius + outlineWidth); ++i )
			{
				for( int j = -(kernelRadius + outlineWidth); j < (kernelRadius + outlineWidth); ++j )
				{
					vec2 coord = kernelToTexCoord( ivec2(i,j) );
					weight = max( weight, 255.0 * texture2D( densitySampler, coord ).r );
				}
			}

			weight = clamp( weight / maxDensity, 0.0, maxDensity );
			kernelRadius = int(ceil( weight * float(fillDilation )));
			kernelRadius = clamp( kernelRadius, 0, fillDilation );
		}

		// Sample the depth for the silhouette.
		silDepth = 1;		
		for(int i = -(kernelRadius + outlineWidth); i < (kernelRadius + outlineWidth); ++i)
        {		
			for(int j = -(kernelRadius + outlineWidth); j < (kernelRadius + outlineWidth); ++j)
			{
				// Don't bother with the point itself again.
				if( (i == 0) && (j == 0) ) continue;

				// Don't bother with any kernel cells with a distance from the kernel's center that is larger than the kernel's radius.
				if( length( vec2(i,j) ) > kernelRadius ) continue;

				float depthSample = texture2D(depthSampler, kernelToTexCoord(ivec2(i,j))).r;
				silDepth = min(silDepth, depthSample);
			}
		}

		lineDepth = 1;

		// Sample the depth for the outlines.
		for(int i = -(kernelRadius + outlineWidth); i < (kernelRadius + outlineWidth); ++i)
		{
			for(int j = -(kernelRadius + outlineWidth); j < (kernelRadius + outlineWidth); ++j)
			{
				// Don't bother with any kernel cells we will already have processed, i.e. |(i,j)| <= kernelRadius
				if (kernelRadius >= length(vec2(i,j))) continue;

				// Don't bother with any kernel cells with a too large radius again...
				if ( length(vec2(i,j)) > (kernelRadius + outlineWidth) ) continue;

				float depthSample = texture2D(depthSampler, kernelToTexCoord(ivec2(i,j))).r;
				lineDepth = min(lineDepth, depthSample);
			}
	    }

		//	Linearize the depth buffer for a fair comparison with the depthTreshold
		float linPointDepth = linearizeDepth(pointDepth);
		float linSilDepth = linearizeDepth(silDepth);
		float linLineDepth = linearizeDepth(lineDepth);
		
		vec4 finalColor = fillColor;
		
		if( densityColoring )
		{
			vec4 color = vec4(densityColor.rgb, density);
			color.rgb *= color.a;
			finalColor = finalColor * (1.0 - color.a) + color;
		}

		if( (pointDepth > 0) && (pointDepth < 1) && (linPointDepth < linSilDepth + depthTreshold) && (linPointDepth < linLineDepth + depthTreshold) )
		{
            gl_FragColor = contourOnly ? prevColor : finalColor;
            gl_FragDepth = contourOnly ? prevDepth : pointDepth;
		}
		else 
		if( (silDepth > 0) && (silDepth < 1) && (linSilDepth < linLineDepth + depthTreshold) )
		{
  			gl_FragColor = contourOnly ? prevColor : finalColor;
   			gl_FragDepth = contourOnly ? prevDepth : silDepth;
		}
    	else 
		if( (lineDepth > 0) && (lineDepth < 1) )
		{
      		gl_FragColor = lineColor;
			gl_FragDepth = lineDepth;
		}
		else
		{
			discard;
		}
	}
}

