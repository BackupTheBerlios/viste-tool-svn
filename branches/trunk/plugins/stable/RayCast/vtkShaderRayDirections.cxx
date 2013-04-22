/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
