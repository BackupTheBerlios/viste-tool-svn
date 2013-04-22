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
 * vtkShaderBase.h
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-05-17	Tim Peeters
 * - Removed functionality dealing with GLhandles. This was moved to  a new
 *   vtkShaderBaseHandle subclass.
 *
 * 2005-06-06	Tim Peeters
 * - Switched to OpenGL 2.0
 *
 * 2005-07-04	Tim Peeters
 * - Instead of using glew, set GL_GLEXT_PROTOTYPES and then include gl.h
 *
 * 2005-07-14	Tim Peeters
 * - Use glew again. On Windows it seems to be needed.
 * - Add SupportsOpenGLVersion() function.
 *
 * 2006-01-30	Tim Peeters
 * - Removed #include <GL/glew.h>
 */

#ifndef bmia_vtkShaderBase_h
#define bmia_vtkShaderBase_h

//#include <GL/glew.h> // for OpenGL types and some functions
		     // TODO: can this be done without glew?
#include <vtkgl.h>

#include <vtkObject.h>

//#define GL_GLEXT_PROTOTYPES 1
//#include "GL/gl.h"

namespace bmia {

/**
 * Base class for all GLSL shader related subclasses. Implements
 * printing of info logs, etc.
 * NOTE: include this header file before including any rendering header
 * files because glew.h must be included before gl.h.
 */
class vtkShaderBase : public vtkObject
{
public:

  /**
   * Returns true if the specified version of OpenGL is supported, and
   * false otherwise.
   */
  static bool SupportsOpenGLVersion(int atLeastMajor, int atLeastMinor);

protected:
  vtkShaderBase();
  ~vtkShaderBase();

private:

};

} // namespace bmia

#endif // bmia_vtkShaderBase_h
