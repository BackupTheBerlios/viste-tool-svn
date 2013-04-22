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

/**
 * vtkVertexShader.cxx
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 * - Renamed glCreateObject() to CreateGlShader()
 *
 * 2006-01-30	Tim Peeters
 * - Switch from glew to VTK OpenGL extension manager.
 *   Use vtkgl::CreateShader(vtkgl::VERTEX_SHADER) instead of
 *   glCreateShader(GL_VERTEX_SHADER).
 */

#include "vtkVertexShader.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkVertexShader);

vtkVertexShader::vtkVertexShader()
{
  // Nothing to do. Everything was already done in vtkShaderObject.
}

vtkVertexShader::~vtkVertexShader()
{
  // Nothing.
}

bool vtkVertexShader::CreateGlShader()
{
  if (this->GetHandleValid())
    {
    // no need to create a new handle.
    return true;
    }

  vtkDebugMacro("Calling glCreateShader(GL_VERTEX_SHADER)");
  GLuint handle = vtkgl::CreateShader(vtkgl::VERTEX_SHADER);
  vtkDebugMacro("glCreateShader() returned handle "<<handle<<".");
  this->SetHandle(handle);
  return true;
}

} // namespace bmia
