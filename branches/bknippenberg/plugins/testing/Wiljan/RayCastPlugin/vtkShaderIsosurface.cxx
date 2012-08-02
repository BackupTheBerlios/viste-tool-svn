#include "vtkShaderIsosurface.h"
#include "vtkObjectFactory.h"

#include <iostream>
#include <sstream>

vtkStandardNewMacro( vtkShaderIsosurface );

///////////////////////////////////////////////////////////////////
vtkShaderIsosurface::vtkShaderIsosurface()
{
}

///////////////////////////////////////////////////////////////////
vtkShaderIsosurface::~vtkShaderIsosurface()
{
}

///////////////////////////////////////////////////////////////////
std::string vtkShaderIsosurface::GetVertexShader()
{
	std::ostringstream str;

	str << "void main( void )\n";
	str << "{\n";
	str << "	gl_Position = ftransform();\n";
	str << "	gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n";
	str << "	gl_TexCoord[0] = \n";
	str << "		gl_TextureMatrix[0] * gl_MultiTexCoord0;\n";
	str << "}\n";

	return str.str();
}

///////////////////////////////////////////////////////////////////
std::string vtkShaderIsosurface::GetFragShader()
{
	std::ostringstream str;

	str << "uniform sampler3D volumeBuffer;\n";
	str << "uniform sampler2D frontBuffer;\n";
	str << "uniform sampler2D rayBuffer;\n";
        str << "uniform sampler2D depthBuffer;\n";
	str << "uniform float diagonal;\n";
	str << "uniform float dx;\n";
	str << "uniform float dy;\n";

	// Viewport, formatted as [Xmin, Ymin, XSize, YSize], all in number of pixels.
	// These should be the viewport values of the 3D subcanvas.

	str << "uniform vec4 viewport;\n";

	str << "uniform float stepSize;\n";
	str << "uniform float isoValue;\n";
	str << "uniform float isoValueOpacity;\n";
        str << "uniform vec3 isoValueColor;\n";
        str << "uniform float clippingX1;\n";
        str << "uniform float clippingX2;\n";
        str << "uniform float clippingY1;\n";
        str << "uniform float clippingY2;\n";
        str << "uniform float clippingZ1;\n";
        str << "uniform float clippingZ2;\n";
        str << "uniform float clippingMinThreshold;\n";
        str << "uniform float clippingMaxThreshold;\n";
        str << "uniform vec3 worldDimensions;\n";
        str << "uniform vec3  cameraposition;\n";
	str << "\n";
	str << "const float SQRT2 = 1.4142135623730950;\n";
	str << "const float DELTA = 0.005;\n";
	str << "\n";
	str << "vec3 computeNormal( sampler3D data, vec3 pos )\n";
	str << "{\n";
	str << "	vec3 sample1, sample2;\n";
	str << "	sample1.x = texture3D( data, pos - vec3( DELTA, 0.0, 0.0 ) ).a;\n";
	str << "	sample2.x = texture3D( data, pos + vec3( DELTA, 0.0, 0.0 ) ).a;\n";
	str << "	sample1.y = texture3D( data, pos - vec3( 0.0, DELTA, 0.0 ) ).a;\n";
	str << "	sample2.y = texture3D( data, pos + vec3( 0.0, DELTA, 0.0 ) ).a;\n";
	str << "	sample1.z = texture3D( data, pos - vec3( 0.0, 0.0, DELTA ) ).a;\n";
	str << "	sample2.z = texture3D( data, pos + vec3( 0.0, 0.0, DELTA ) ).a;\n";
	str << "	vec3 N = normalize( sample2 - sample1 );\n";
	str << "	return N;\n";
	str << "}\n";
	str << "\n";
	str << "vec3 applyLighting( vec3 color, vec3 N, vec3 V, vec3 L )\n";
	str << "{\n";
	str << "	vec3 H = normalize( L + V );\n";
	str << "	float Sa = 0.3;\n";
	str << "	float Sd = max( dot( L, N ), 0.0 );\n";
	str << "	float Ss = pow( max( dot( H, N ), 0.0 ), 8.0 );\n";
	str << "	vec3 ambient  = Sa * vec3( 0.2, 0.2, 0.2 );\n";
	str << "	vec3 diffuse  = Sd * vec3( 0.6, 0.6, 0.6 );\n";
        str << "	vec3 specular = Ss * vec3( 0.4, 0.4, 0.4 );\n";
	str << "	if( Sd <= 0.0 )\n";
	str << "		specular = vec3( 0 );\n";
	str << "	return color * (ambient + diffuse) + specular;\n";
	str << "}\n";
	str << "\n";
	str << "void main( void )\n";
	str << "{\n";

	str << "	vec2 lookUp = vec2(dx * gl_FragCoord.x, dy * gl_FragCoord.y);\n";

	str << "	vec4 tmp = texture2D( rayBuffer, lookUp );\n";
	str << "	vec3 rayDir = (tmp.xyz * 2.0) - vec3( 1 );\n";
	str << "	float rayLength = tmp.w * SQRT2;\n";
	str << "	vec3 step = stepSize * rayDir / diagonal;\n";
	str << "	vec3 position = texture2D( frontBuffer, lookUp ).xyz;\n";
        str << "	float distance1 = 0.0;\n";
	str << "	vec4 final = vec4( 0.0 );\n";

	str << "        float depth = texture2D(depthBuffer, lookUp).x;\n";

	// Translate the screen coordinates plus the depth to world coordinates.
	// We need to include the viewport, otherwise rendering will be incorrect
	// for viewports with an offset.

	str << "        vec4 hvolpos = vec4(2.0 * (gl_FragCoord.x - viewport.x) / viewport.z - 1.0,\n";
	str << "                            2.0 * (gl_FragCoord.y - viewport.y) / viewport.w - 1.0,\n";
	str << "                            2.0 * texture2D(depthBuffer, lookUp).x - 1.0, \n";
	str << "                            1.0 );\n";
	str << "        hvolpos = gl_ModelViewProjectionMatrixInverse * hvolpos;\n";
	str << "        hvolpos /= hvolpos.w;\n";

	str << "        float distance2;\n";
	str << "        hvolpos.x = hvolpos.x / worldDimensions.x;\n";
	str << "        hvolpos.y = hvolpos.y / worldDimensions.y;\n";
	str << "        hvolpos.z = hvolpos.z / worldDimensions.z;\n";
	str << "        cameraposition.x / worldDimensions.x;\n";
	str << "        cameraposition.y / worldDimensions.y;\n";
	str << "        cameraposition.z / worldDimensions.z;\n";
	str << "        vec3 pos = hvolpos.xyz;\n";
	str << "        float depthDistance = distance( cameraposition , pos );\n";

        // check if intersects or has intersected with opaque geometry

        str << "    distance2 = distance( cameraposition , position );\n";
        str << "    if ( distance2 > depthDistance )\n";
        str << "    {\n";
        str << "        gl_FragColor = vec4( 0.0 );\n";
        str << "        return;\n";
        str << "    }\n";

        // part for grayscale values on clipping planes.

        str << "        if (clippingX1 > 0.0 && clippingX1 < 1.0 && (position.x <= clippingX1 + 0.0001) && (position.x >= clippingX1 - 0.0001))\n";
        str << "        {\n";
        str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
        str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
        str << "                {\n";
        str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
        str << "                        return;\n";
        str << "                }\n";
        str << "        }\n";

        str << "        if (clippingX2 > 0.0 && clippingX2 < 1.0 && (position.x >= clippingX2 - 0.0001) && (position.x <= clippingX2 + 0.0001) )\n";
        str << "        {\n";
        str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
        str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
        str << "                {\n";
        str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
        str << "                        return;\n";
        str << "                }\n";
        str << "        }\n";

        str << "        if (clippingY1 > 0.0 && clippingY1 < 1.0 && (position.y <= clippingY1 + 0.0001) && (position.y >= clippingY1 - 0.0001))\n";
        str << "        {\n";
        str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
        str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
        str << "                {\n";
        str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
        str << "                        return;\n";
        str << "                }\n";
        str << "        }\n";

        str << "        if (clippingY2 > 0.0 && clippingY2 < 1.0 && (position.y >= clippingY2 - 0.0001) && (position.y <= clippingY2 + 0.0001)  )\n";
        str << "        {\n";
        str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
        str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
        str << "                {\n";
        str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
        str << "                        return;\n";
        str << "                }\n";
        str << "        }\n";

        str << "        if (clippingZ1 > 0.0 && clippingZ1 < 1.0 && (position.z <= clippingZ1 + 0.0001) && (position.z >= clippingZ1 - 0.0001) )\n";
        str << "        {\n";
        str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
        str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
        str << "                {\n";
        str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
        str << "                        return;\n";
        str << "                }\n";
        str << "        }\n";

        str << "        if (clippingZ2 > 0.0 && clippingZ2 < 1.0 && (position.z >= clippingZ2 - 0.0001) && (position.z <= clippingZ2 + 0.0001)  )\n";
        str << "        {\n";
        str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
        str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
        str << "                {\n";
        str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
        str << "                        return;\n";
        str << "                }\n";
        str << "        }\n";

        // end of part for grayscale values on clipping planes.

	str << "	for( int i = 0; i < 2048; i++ )\n";
	str << "	{\n";

        // check if intersects with opaque geometry

        str << "    distance2 = distance( cameraposition , position );\n";
        str << "    if ( distance2 > depthDistance )\n";
        str << "    {\n";
        str << "        break;\n";
        str << "    }\n";

        // end of check if intersects with opaque geometry


	str << "		float sample = texture3D( volumeBuffer, position ).a;\n";
        str << "		if( sample >= isoValue )\n";
	str << "		{\n";
	str << "			float factor = length( step );\n";
	str << "			for( int i = 0; i < 6; i++ )\n";
	str << "			{\n";
	str << "				factor = 0.5 * sign( isoValue - sample ) * abs( factor );\n";
	str << "				position += rayDir * factor;\n";
	str << "				sample = texture3D( volumeBuffer, position ).a;\n";
	str << "			}\n";
	str << "			vec3 N = computeNormal( volumeBuffer, position );\n";
	str << "			vec3 V = rayDir;\n";
	str << "			vec3 L = V;\n";
        str << "			final = vec4( isoValueColor, 1 );\n";
	str << "			final.rgb = applyLighting( final.rgb, N, V, L );\n";
        str << "                        final.a = isoValueOpacity;\n";
	str << "			break;\n";
	str << "		}\n";
        str << "		if( distance1 > rayLength ) break;\n";
	str << "		position += step;\n";
        str << "		distance1 += length( step );\n";
	str << "	}\n";
	str << "\n";
	str << "	gl_FragColor = final;\n";
	str << "}\n";

	return str.str();
}
