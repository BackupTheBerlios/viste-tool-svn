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
 * vtkShaderBaseHandle.cxx
 * by Tim Peeters
 *
 * 2005-05-17	Tim Peeters
 * - First version, based on the (old) vtkShaderBase.cxx
 *
 * 2005-06-06	Tim Peeters
 * - Switch to OpenGL 2.0
 * - Removed glDelete()
 */

#include "vtkShaderBaseHandle.h"

namespace bmia {

vtkShaderBaseHandle::vtkShaderBaseHandle()
{
  this->HandleValid = false;
  // this->Handle is not initialized here
}

vtkShaderBaseHandle::~vtkShaderBaseHandle()
{
  // nothing was created by this class, so don't destroy anything either.
  // do that in the subclasses that were creating things.
}

GLuint vtkShaderBaseHandle::GetHandle()
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"Calling GetHandle() without a valid handle!");
    }
  return this->Handle;
}

void vtkShaderBaseHandle::SetHandle(GLuint newhandle)
{
  this->Handle = newhandle;
  this->HandleValid = true;
}

void vtkShaderBaseHandle::UnsetHandle()
{
  this->HandleValid = false;
}

bool vtkShaderBaseHandle::GetHandleValid()
{
  return this->HandleValid;
}

} // namespace bmia
