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
 * vtkShaderUniform.h
 * by Tim Peeters
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Switch to OpenGL 2.0
 * - Renamed glUniform() to SetGlUniform()
 * - Renamed glSpecificUniform() to SetGlUniformSpecific()
 */

#ifndef bmia_vtkShaderUniform_h
#define bmia_vtkShaderUniform_h

#include "vtkShaderBaseHandle.h"
#include <vtkgl.h> // for vtkgl::GLchar

namespace bmia {

/**
 * Representation of a uniform variable for GLSL.
 * Subclass of vtkShaderBaseHandle for easy setting of the handle of the
 * associated shader program. This means that a vtkShaderUniform object
 * should only be associated to ONE shader program and not multiple to
 * avoid confusion!
 * Setting of the handle is done by the shader program using this uniform.
 */
class vtkShaderUniform : public vtkShaderBaseHandle
{
public:
  /**
   * Set/Get the name of the uniform. This must be a null-terminated string.
   * The array element operator ``[]'' and the structure field operator ``.''
   * may be used in the name to select elements of an array or fields of a
   * structure. White spaces are not allowed.
   */
  vtkSetStringMacro(Name);
  vtkGetStringMacro(Name);

  /**
   * Pass this uniform to the shader.
   * Always call this->glGetLocation() first to initialize this->Location.
   * Returns false if there was an error.
   */
  virtual bool SetGlUniform();

protected:
  vtkShaderUniform();
  ~vtkShaderUniform();

  //vtkGetMacro(Location, GLint);

  /**
   * Type-specific setting of uniform. Must be implemented in subclasses.
   * Can assume that this->Location is valid (not -1).
   */
  virtual void SetGlUniformSpecific() = 0;

  /**
   * The location of this uniform variable.
   */
  GLint Location;

private:
  /**
   * The name associated with this uniform.
   */
	vtkgl::GLchar* Name;

};

} // namespace bmia

#endif // bmia_vtkShaderUniform_h
