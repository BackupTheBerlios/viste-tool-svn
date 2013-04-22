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
 * vtkShaderBaseHandle.h
 * by Tim Peeters
 *
 * 2005-05-17	Tim Peeters
 * - First version, based on the (old) vtkShaderBase.h
 *
 * 2006-06-06	Tim Peeters
 * - Switched to OpenGL 2.0 (removed ARB in function calls and types)
 * - Removed glDelete() function. Now implemented in vtkShaderObject
 *   and vtkShaderProgram as DeleteGlShader() and DeleteGlProgram().
 */

#ifndef bmia_vtkShaderBaseHandle_h
#define bmia_vtkShaderBaseHandle_h

#include "vtkShaderBase.h"

namespace bmia {

/**
 * Base class for GLSL shader related subclasses with GLhandles.
 */
class vtkShaderBaseHandle : public vtkShaderBase
{
public:
  /**
   * Gets/Sets the handle for the current shader object or program.
   */
  GLuint GetHandle();
  void SetHandle(GLuint newhandle);
  void UnsetHandle();

  /**
   * True if a handle was given a value using SetHandle() and UnsetHandle()
   * was not called afterwards.
   */
  bool GetHandleValid();

protected:
  vtkShaderBaseHandle();
  ~vtkShaderBaseHandle();

  /**
   * Deletes the shader program or object associated with this->Handle
   * if this->Handle is valid (GetHandleValid()).
   */
  //void glDelete();

private:

  /**
   * Handle for shader or vertex object. Initialize in subclasses!
   */
  GLuint Handle;

  /**
   * True if this->Handle was given a value using SetHandle and UnsetHandle
   * was not called afterwards.
   */
  bool HandleValid;

};

} // namespace bmia

#endif // __vtkShaderBaseHandle_h
