#ifndef __vtkFiberConfidenceMapperShaders_h
#define __vtkFiberConfidenceMapperShaders_h

const char * vtkFiberConfidenceMapperShaders_VertexShader =

		"void main(void)\n" \
		"{\n" \
		"	gl_Position = gl_Vertex;\n" \
		"	gl_FrontColor = gl_Color;\n" \
		"	gl_TexCoord[0] = gl_MultiTexCoord0;\n" \
		"}\n";

const char * vtkFiberConfidenceMapperShaders_SilhouetteFragShader =

		"#version 120\n" \
		"#extension GL_EXT_gpu_shader4 : enable\n" \

		"uniform sampler2D depthBuffer;\n" \
		"uniform float depthThreshold;\n" \
		"uniform float depthNear;\n" \
		"uniform float depthFar;\n" \
		"uniform bool  ortho;\n" \
		"uniform vec4  color;\n" \
		"uniform vec2  pixelRatio;\n" \
		"uniform vec4  outlineColor;\n" \
		"uniform int   outlineThickness;\n" \
		"uniform int   dilation;\n" \

		"vec2 kernelToTexCoord( in ivec2 coord )\n" \
		"{\n" \
		"	vec2 offset;\n" \
		"	offset.x = coord.x * pixelRatio.x;\n" \
		"	offset.y = coord.y * pixelRatio.y;\n" \
		"	return clamp( gl_TexCoord[0].st + offset,\n" \
		"		vec2( 0, 0 ), vec2( 1, 1 ) );\n" \
		"}\n" \

		"float linearize( in float depth )\n" \
		"{\n" \
		"	if( ortho )\n" \
		"		return mix( depthNear, depthFar, depth );\n" \
		"	else\n" \
		"		return (depthNear * depthFar) / (depthFar - depth * (depthFar - depthNear));\n" \
		"}\n" \

		"void main( void )\n" \
		"{\n" \
		"	float pointDepth = texture2D( depthBuffer, gl_TexCoord[0].st ).r;\n" \
		"	float silhouetteDepth = 1.0;\n" \

		"	int radius = dilation;\n" \
		"	for( int i = -radius; i < radius; ++i )\n" \
		"	{\n" \
		"		for( int j = -radius; j < radius; ++j )\n" \
		"		{\n" \
		"			if( i == 0.0 && j == 0.0 ) continue;\n" \
		"			if( dilation < length( vec2( i, j ) ) ) continue;\n" \
		"			float depth = texture2D( depthBuffer, kernelToTexCoord( ivec2( i, j ) ) ).r;\n" \
		"			silhouetteDepth = min( silhouetteDepth, depth );\n" \
		"		}\n" \
		"	}\n" \

		"	float outlineDepth = 1.0;\n" \
		"	radius = dilation + outlineThickness;\n" \
		"	for( int i = -radius; i < radius; ++i )\n" \
		"	{\n" \
		"		for( int j = -radius; j < radius; ++j )\n" \
		"		{\n" \
		"			if( dilation >= length( vec2( i, j ) ) ) continue;\n" \
		"			if( dilation + outlineThickness < length( vec2( i, j ) ) ) continue;\n" \
		"			float depth = texture2D( depthBuffer, kernelToTexCoord( ivec2( i, j ) ) ).r;\n" \
		"			outlineDepth = min( outlineDepth, depth );\n" \
		"		}\n" \
		"	}\n" \

		"	float linearSilhouetteDepth = linearize( silhouetteDepth );\n" \
		"	float linearOutlineDepth = linearize( outlineDepth );\n" \
		"	float linearPointDepth = linearize( pointDepth );\n" \

		"	//if( pointDepth > 0.0 && \n" \
		"	//	pointDepth < 1.0 && \n" \
		"	//	linearPointDepth < (linearSilhouetteDepth + depthThreshold) && \n" \
		"	//	linearPointDepth < (linearOutlineDepth + depthThreshold) )\n" \
		"	//{\n" \
		"	//	gl_FragColor = color;\n" \
		"	//	gl_FragDepth = pointDepth;\n" \
		"	//}\n" \
		"	/*else*/ if( silhouetteDepth > 0.0 && \n" \
		"			 silhouetteDepth < 1.0 && \n" \
		"			 linearSilhouetteDepth < (linearOutlineDepth + depthThreshold) )\n" \
		"	{\n" \
		"		gl_FragColor = color;\n" \
		"		gl_FragDepth = silhouetteDepth;\n" \
		"	}\n" \
		"	else if( outlineDepth > 0.0 && outlineDepth < 1.0 )\n" \
		"	{\n" \
		"		gl_FragColor = outlineColor;\n" \
		"		gl_FragDepth = outlineDepth;\n" \
		"	}\n" \
		"	else\n" \
		"	{\n" \
		"		discard;\n" \
		"	}\n" \
		"}";

const char * vtkFiberConfidenceMapperShaders_CheckerBoardFragShader =

		"#version 120\n" \
		"#extension GL_EXT_gpu_shader4 : enable\n" \

		"uniform sampler2D depthBuffer;\n" \
		"uniform float depthThreshold;\n" \
		"uniform float depthNear;\n" \
		"uniform float depthFar;\n" \
		"uniform bool  ortho;\n" \
		"uniform vec4  color;\n" \
		"uniform vec2  pixelRatio;\n" \
		"uniform vec4  outlineColor;\n" \
		"uniform int   outlineThickness;\n" \
		"uniform int   dilation;\n" \
		"uniform int   checkerSize;\n" \

		"vec2 kernelToTexCoord( in ivec2 coord )\n" \
		"{\n" \
		"	vec2 offset;\n" \
		"	offset.x = coord.x * pixelRatio.x;\n" \
		"	offset.y = coord.y * pixelRatio.y;\n" \
		"	return clamp( gl_TexCoord[0].st + offset,\n" \
		"		vec2( 0, 0 ), vec2( 1, 1 ) );\n" \
		"}\n" \

		"float linearize( in float depth )\n" \
		"{\n" \
		"	if( ortho )\n" \
		"		return mix( depthNear, depthFar, depth );\n" \
		"	else\n" \
		"		return (depthNear * depthFar) / (depthFar - depth * (depthFar - depthNear));\n" \
		"}\n" \

		"void main( void )\n" \
		"{\n" \
		"	int x = int( gl_FragCoord.x / checkerSize );\n" \
		"	int y = int( gl_FragCoord.y / checkerSize );\n" \

		"	bool render = true;\n" \
		"	if( mod( x, 2 ) == 0.0 && mod( y, 2 ) == 0.0 )\n" \
		"		render = false;\n" \
		"	if( mod( x, 2 ) != 0.0 && mod( y, 2 ) != 0.0 )\n" \
		"		render = false;\n" \

		"	float pointDepth = texture2D( depthBuffer, gl_TexCoord[0].st ).r;\n" \
		"	float silhouetteDepth = 1.0;\n" \

		"	int radius = dilation;\n" \
		"	for( int i = -radius; i < radius; ++i )\n" \
		"	{\n" \
		"		for( int j = -radius; j < radius; ++j )\n" \
		"		{\n" \
		"			if( i == 0.0 && j == 0.0 ) continue;\n" \
		"			if( radius < length( vec2( i, j ) ) ) continue;\n" \
		"			float depth = texture2D( depthBuffer, kernelToTexCoord( ivec2( i, j ) ) ).r;\n" \
		"			silhouetteDepth = min( silhouetteDepth, depth );\n" \
		"		}\n" \
		"	}\n" \

		"	float outlineDepth = 1.0;\n" \
		"	radius = dilation + outlineThickness;\n" \
		"	for( int i = -radius; i < radius; ++i )\n" \
		"	{\n" \
		"		for( int j = -radius; j < radius; ++j )\n" \
		"		{\n" \
		"			if( i == 0.0 && j == 0.0 ) continue;\n" \
		"			if( radius < length( vec2( i, j ) ) ) continue;\n" \
		"			float depth = texture2D( depthBuffer, kernelToTexCoord( ivec2( i, j ) ) ).r;\n" \
		"			outlineDepth = min( outlineDepth, depth );\n" \
		"		}\n" \
		"	}\n" \

		"	float linearSilhouetteDepth = linearize( silhouetteDepth );\n" \
		"	float linearOutlineDepth = linearize( outlineDepth );\n" \
		"	float linearPointDepth = linearize( pointDepth );\n" \

		"	//if( pointDepth > 0.0 && \n" \
		"	//	pointDepth < 1.0 && \n" \
		"	//	linearPointDepth < (linearSilhouetteDepth + depthThreshold) && \n" \
		"	//	linearPointDepth < (linearOutlineDepth + depthThreshold) )\n" \
		"	//{\n" \
		"	//	gl_FragColor = color;\n" \
		"	//	gl_FragDepth = pointDepth;\n" \
		"	//}\n" \
		"	/*else*/ if( silhouetteDepth > 0.0 && \n" \
		"			 silhouetteDepth < 1.0 && \n" \
		"			 linearSilhouetteDepth < (linearOutlineDepth + depthThreshold) )\n" \
		"	{\n" \
		"		if( render )\n" \
		"			gl_FragColor = color;\n" \
		"		else\n" \
		"			gl_FragColor = vec4( 0 );\n" \
		"		gl_FragDepth = silhouetteDepth;\n" \
		"	}\n" \
		"	else if( outlineDepth > 0.0 && outlineDepth < 1.0 )\n" \
		"	{\n" \
		"		gl_FragColor = outlineColor;\n" \
		"		gl_FragDepth = outlineDepth;\n" \
		"	}\n" \
		"	else\n" \
		"	{\n" \
		"		discard;\n" \
		"	}\n" \
		"}\n";

const char * vtkFiberConfidenceMapperShaders_HolesFragShader =

		"#version 120\n" \
		"#extension GL_EXT_gpu_shader4 : enable\n" \

		"uniform sampler2D depthBuffer;\n" \
		"uniform float depthThreshold;\n" \
		"uniform float depthNear;\n" \
		"uniform float depthFar;\n" \
		"uniform bool  ortho;\n" \
		"uniform vec4  color;\n" \
		"uniform vec2  pixelRatio;\n" \
		"uniform vec4  outlineColor;\n" \
		"uniform int   outlineThickness;\n" \
		"uniform int   dilation;\n" \
		"uniform int   checkerSize;\n" \
		"uniform int   holeSize;\n" \

		"vec2 kernelToTexCoord( in ivec2 coord )\n" \
		"{\n" \
		"	vec2 offset;\n" \
		"	offset.x = coord.x * pixelRatio.x;\n" \
		"	offset.y = coord.y * pixelRatio.y;\n" \
		"	return clamp( gl_TexCoord[0].st + offset,\n" \
		"		vec2( 0, 0 ), vec2( 1, 1 ) );\n" \
		"}\n" \

		"float linearize( in float depth )\n" \
		"{\n" \
		"	if( ortho )\n" \
		"		return mix( depthNear, depthFar, depth );\n" \
		"	else\n" \
		"		return (depthNear * depthFar) / (depthFar - depth * (depthFar - depthNear));\n" \
		"}\n" \

		"void main( void )\n" \
		"{\n" \
		"	vec2 p;\n" \
		"	p.x = floor( gl_FragCoord.x / checkerSize );\n" \
		"	p.y = floor( gl_FragCoord.y / checkerSize );\n" \

		"	vec2 corner[4];\n" \
		"	corner[0] = vec2( p.x * checkerSize, p.y * checkerSize );\n" \
		"	corner[1] = vec2( p.x * checkerSize + checkerSize, p.y * checkerSize );\n" \
		"	corner[2] = vec2( p.x * checkerSize + checkerSize, p.y * checkerSize + checkerSize );\n" \
		"	corner[3] = vec2( p.x * checkerSize, p.y * checkerSize + checkerSize );\n" \
		
		"	vec2 position = gl_FragCoord.xy;\n" \
		"	bool render = true;\n" \
		"	for( int i = 0; i < 4; ++i )\n" \
		"	{\n" \
		"		if( distance( position, corner[i] ) < (holeSize / 2.0) )\n" \
		"		{\n" \
		"			render = false;\n" \
		"			break;\n" \
		"		}\n" \
		"	}\n" \

		"	float pointDepth = texture2D( depthBuffer, gl_TexCoord[0].st ).r;\n" \
		"	float silhouetteDepth = 1.0;\n" \

		"	int radius = dilation;\n" \
		"	for( int i = -radius; i < radius; ++i )\n" \
		"	{\n" \
		"		for( int j = -radius; j < radius; ++j )\n" \
		"		{\n" \
		"			if( i == 0.0 && j == 0.0 ) continue;\n" \
		"			if( radius < length( vec2( i, j ) ) ) continue;\n" \
		"			float depth = texture2D( depthBuffer, kernelToTexCoord( ivec2( i, j ) ) ).r;\n" \
		"			silhouetteDepth = min( silhouetteDepth, depth );\n" \
		"		}\n" \
		"	}\n" \

		"	float outlineDepth = 1.0;\n" \
		"	radius = dilation + outlineThickness;\n" \
		"	for( int i = -radius; i < radius; ++i )\n" \
		"	{\n" \
		"		for( int j = -radius; j < radius; ++j )\n" \
		"		{\n" \
		"			if( i == 0.0 && j == 0.0 ) continue;\n" \
		"			if( radius < length( vec2( i, j ) ) ) continue;\n" \
		"			float depth = texture2D( depthBuffer, kernelToTexCoord( ivec2( i, j ) ) ).r;\n" \
		"			outlineDepth = min( outlineDepth, depth );\n" \
		"		}\n" \
		"	}\n" \

		"	float linearSilhouetteDepth = linearize( silhouetteDepth );\n" \
		"	float linearOutlineDepth = linearize( outlineDepth );\n" \
		"	float linearPointDepth = linearize( pointDepth );\n" \

		"	//if( pointDepth > 0.0 && \n" \
		"	//	pointDepth < 1.0 && \n" \
		"	//	linearPointDepth < (linearSilhouetteDepth + depthThreshold) && \n" \
		"	//	linearPointDepth < (linearOutlineDepth + depthThreshold) )\n" \
		"	//{\n" \
		"	//	gl_FragColor = color;\n" \
		"	//	gl_FragDepth = pointDepth;\n" \
		"	//}\n" \
		"	/*else*/ if( silhouetteDepth > 0.0 && \n" \
		"			 silhouetteDepth < 1.0 && \n" \
		"			 linearSilhouetteDepth < (linearOutlineDepth + depthThreshold) )\n" \
		"	{\n" \
		"		if( render )\n" \
		"			gl_FragColor = color;\n" \
		"		else\n" \
		"			gl_FragColor = vec4( 0 );\n" \
		"		gl_FragDepth = silhouetteDepth;\n" \
		"	}\n" \
		"	else if( outlineDepth > 0.0 && outlineDepth < 1.0 )\n" \
		"	{\n" \
		"		gl_FragColor = outlineColor;\n" \
		"		gl_FragDepth = outlineDepth;\n" \
		"	}\n" \
		"	else\n" \
		"	{\n" \
		"		discard;\n" \
		"	}\n" \
		"}\n";

const char * vtkFiberConfidenceMapperShaders_DensityFragShader =

		"void main(void)\n" \
		"{\n" \
		"	gl_FragColor = vec4( 1.0 / 255.0 );\n" \
		"}";

const char * vtkFiberConfidenceMapperShaders_BlurringFragShader =

		"#version 120\n" \
		"#extension GL_EXT_gpu_shader4 : enable\n" \

		"#define PI 3.1415926535\n" \

		"uniform sampler2D colorBuffer;\n" \
		"uniform sampler2D depthBuffer;\n" \
		"uniform float blurringBrightness;\n" \
		"uniform int   blurringRadius;\n" \
		"uniform vec2  pixelRatio;\n" \

		"vec2 kernelToTexCoord( in ivec2 coord )\n" \
		"{\n" \
		"	vec2 offset;\n" \
		"	offset.x = coord.x * pixelRatio.x;\n" \
		"	offset.y = coord.y * pixelRatio.y;\n" \
		"	return clamp( gl_TexCoord[0].st + offset,\n" \
		"		vec2( 0, 0 ), vec2( 1, 1 ) );\n" \
		"}\n" \

		"float gauss( float x, float y, float sigma )\n" \
		"{\n" \
		"	float power  = (x * x + y * y) / (2.0 * sigma * sigma);\n" \
		"	float prefix = 1.0 / (2.0 * PI * sigma * sigma);\n" \
		"	float result = prefix * exp( -power );\n" \
		"	return result;\n" \
		"}\n" \

		"float mean()\n" \
		"{\n" \
		"	float result = 1.0 / (4.0 * blurringRadius * blurringRadius);\n" \
		"	return result;\n" \
		"}\n" \

		"void main(void)\n" \
		"{\n" \
		"	vec4 outColor = texture2D( colorBuffer, gl_TexCoord[0].st );\n" \
		"	float depth   = texture2D( depthBuffer, gl_TexCoord[0].st ).r;\n" \
		"	if( depth == 1.0 )\n" \
		"		discard;\n" \
		"	if( blurringRadius > 0 )\n" \
		"	{\n" \
		"		outColor = vec4( 0 );\n" \
		"		float sigma = blurringRadius / 2.0;\n" \
		"		for( int i = -blurringRadius; i < blurringRadius; ++i )\n" \
		"		{\n" \
		"			for( int j = -blurringRadius; j < blurringRadius; ++j )\n" \
		"			{\n" \
		"				vec2 coord = kernelToTexCoord( ivec2( i, j ) );\n" \
		"				vec4 color = texture2D( colorBuffer, coord );\n" \
		"				float factor = blurringBrightness * gauss( j, i, sigma );\n" \
		"				color *= factor;\n" \
		"				outColor += color;\n" \
		"			}\n" \
		"		}\n" \
		"	}\n" \
		"	gl_FragColor = outColor;\n" \
		"	gl_FragDepth = depth;\n" \
		"}\n";

const char * vtkFiberConfidenceMapperShaders_OutputFragShader =

		"#version 120\n" \
		"#extension GL_EXT_gpu_shader4 : enable\n" \

		"uniform sampler2D colorBuffer;\n" \
		"uniform sampler2D depthBuffer;\n" \

		"void main(void)\n" \
		"{\n" \
		"	vec4 color   = texture2D( colorBuffer, gl_TexCoord[0].st );\n" \
		"	float depth  = texture2D( depthBuffer, gl_TexCoord[0].st ).r;\n" \
		"	gl_FragColor = color;\n" \
		"	gl_FragDepth = depth;\n" \
		"}\n";

#endif