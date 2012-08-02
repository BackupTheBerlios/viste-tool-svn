#include "vtkShaderMIP.h"

#include "vtkObjectFactory.h"

#include <iostream>
#include <sstream>

vtkStandardNewMacro( vtkShaderMIP );

///////////////////////////////////////////////////////////////////
vtkShaderMIP::vtkShaderMIP()
{
}

///////////////////////////////////////////////////////////////////
vtkShaderMIP::~vtkShaderMIP()
{
}

///////////////////////////////////////////////////////////////////
std::string vtkShaderMIP::GetVertexShader()
{
  std::ostringstream str;

  str << "void main( void )\n";
  str << "{\n";
  str << "	gl_Position = ftransform();\n";
  str << "	gl_TexCoord[0] = \n";
  str << "		gl_TextureMatrix[0] * gl_MultiTexCoord0;\n";
  str << "}\n";

  return str.str();
}

///////////////////////////////////////////////////////////////////
std::string vtkShaderMIP::GetFragShader()
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

  str << "uniform float clippingX1;\n";
  str << "uniform float clippingX2;\n";
  str << "uniform float clippingY1;\n";
  str << "uniform float clippingY2;\n";
  str << "uniform float clippingZ1;\n";
  str << "uniform float clippingZ2;\n";
  str << "uniform float clippingMinThreshold;\n";
  str << "uniform float clippingMaxThreshold;\n";
  str << "uniform float stepSize;\n";
  str << "uniform vec3 worldDimensions;\n";
  str << "uniform vec3  cameraposition;\n";
  str << "\n";
  str << "const float SQRT2 = 1.4142135623730950;\n";
  str << "\n";
  str << "void main( void )\n";
  str << "{\n";
  str << "	vec2 lookUp = vec2( dx*gl_FragCoord.x, dy*gl_FragCoord.y );\n";
  str << "	vec4 tmp = texture2D( rayBuffer, lookUp );\n";
  str << "	vec3 rayDir = (tmp.xyz * 2.0) - vec3( 1 );\n";
  str << "	float rayLength = tmp.w * SQRT2;\n";
  str << "	vec3 step = (rayDir * stepSize / diagonal);\n";
  str << "	vec3 position = texture2D( frontBuffer, lookUp ).xyz;\n";
  str << "  vec3 startPosition = position;\n";
  str << "	float max = 0.0;\n";
  str << "	float distance1 = 0.0;\n";
  str << "        float depth = texture2D( depthBuffer, lookUp ).x;\n";

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
  str << "    if ( distance2 >= depthDistance )\n";
  str << "    {\n";
  str << "        gl_FragColor = vec4( 0.0 );\n";
  str << "        return;\n";
  str << "    }\n";

  // part for grayscale values on clipping planes. Here we check whether we should
  // draw gray-scale values. In addition is checked whether we are not on the
  // wrong side of any clipping plane since otherwise some artifacts appear
  // that look especially fugly on any opaque geometry.


  str << "        if (clippingX2 > 0.0 && clippingX2 < 1.0 && (position.x >= clippingX2 - 0.0001) && (position.x <= clippingX2 + 0.0001) )\n";
  str << "        {\n";
  str << "                float sample1 = texture3D( volumeBuffer, position ).a;\n";
  str << "                if ( (sample1 < clippingMinThreshold || sample1 >= clippingMaxThreshold) )\n";
  str << "                {\n";
  str << "                        gl_FragColor = vec4( sample1,sample1,sample1,1 );\n";
  str << "                        return;\n";
  str << "                }\n";
  str << "        }\n";

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

  str << "\n";
  str << "	for( int i = 0; i < 2048; i++ )\n";
  str << "	{\n";

  // check if intersects with opaque geometry

  str << "    distance2 = distance( cameraposition , position );\n";
  str << "    if ( distance2 >= depthDistance )\n";
  str << "    {\n";
  str << "        break;\n";
  str << "    }\n";

  // end of check if intersects with opaque geometry

  str << "		float sample = texture3D( volumeBuffer, position ).a;\n";
  str << "		if( sample > max )\n";
  str << "			max = sample;\n";
  str << "		distance1 += length( step );\n";
  str << "		if( distance1 > rayLength )\n";
  str << "			break;\n";
  str << "		position += step;\n";
  str << "	}\n";
  str << "\n";
  str << "	gl_FragColor = vec4( max );\n";
  str << "}\n";

  return str.str();
}
