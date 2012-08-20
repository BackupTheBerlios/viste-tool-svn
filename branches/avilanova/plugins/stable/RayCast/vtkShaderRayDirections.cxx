#include "vtkShaderRayDirections.h"

#include "vtkObjectFactory.h"

#include <iostream>
#include <sstream>

vtkStandardNewMacro( vtkShaderRayDirections );

///////////////////////////////////////////////////////////////////
vtkShaderRayDirections::vtkShaderRayDirections()
{
}

///////////////////////////////////////////////////////////////////
vtkShaderRayDirections::~vtkShaderRayDirections()
{
}

///////////////////////////////////////////////////////////////////
std::string vtkShaderRayDirections::GetVertexShader()
{
  std::ostringstream str;

  str << "void main( void )\n";
  str << "{\n";
  str << "	gl_Position = ftransform();\n";
  str << "	gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;\n";
  str << "}\n";

  return str.str();
}

///////////////////////////////////////////////////////////////////
std::string vtkShaderRayDirections::GetFragShader()
{
  std::ostringstream str;

  str << "uniform sampler2D frontBuffer;\n";
  str << "uniform sampler2D backBuffer;\n";
  str << "uniform float dx;\n";
  str << "uniform float dy;\n";
  str << "\n";
  str << "const float SQRT2 = 1.4142135623730950;\n";
  str << "\n";
  str << "void main( void )\n";
  str << "{\n";
  str << "	vec2 lookUp = vec2( dx*gl_FragCoord.x, dy*gl_FragCoord.y );\n";
  str << "	vec3 frontPos = texture2D( frontBuffer, lookUp ).xyz;\n";
  str << "	vec3 backPos = texture2D( backBuffer, lookUp ).xyz;\n";
  str << "	vec3 rayDir = backPos - frontPos;\n";
  str << "	float rayLength = length( rayDir );\n";
  str << "	rayDir = (rayDir + vec3( 1 )) * 0.5;\n";
  str << "\n";
  str << "gl_FragColor = vec4( rayDir, rayLength / SQRT2 );\n";
  str << "}\n";

  return str.str();
}
