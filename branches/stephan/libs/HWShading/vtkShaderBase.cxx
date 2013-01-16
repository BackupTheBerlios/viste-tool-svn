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
 * vtkShaderBase.cxx
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-05-17	Tim Peeters
 * - Removed most implementations because they were moved to
 *   vktShaderBaseHandle subclass.
 *
 * 2005-07-14	Tim Peeters
 * - Added SupportsOpenGLVersion() function.
 */

#include "vtkShaderBase.h"

namespace bmia {

vtkShaderBase::vtkShaderBase()
{
  // nothing
}

vtkShaderBase::~vtkShaderBase()
{
  // nothing was created by this class, so don't destroy anything either.
  // do that in the subclasses that were creating things.
}

// from http://developer.nvidia.com/object/nv_ogl2_support.html
bool vtkShaderBase::SupportsOpenGLVersion(int atLeastMajor, int atLeastMinor)
{
  const char* version;
  int major, minor;

  //glewInit();
  version = (const char *) glGetString(GL_VERSION);
  cout<<"OpenGL version is "<<version<<endl;
  //vtkDebugMacro(<<"OpenGL version is "<<version);

  if (sscanf(version, "%d.%d", &major, &minor) == 2) {
  if (major > atLeastMajor)
    return true;
  if (major == atLeastMajor && minor >= atLeastMinor)
    return true;
  } else {
    /* OpenGL version string malformed! */
  }
  return false;
}

} // namespaced bmia
