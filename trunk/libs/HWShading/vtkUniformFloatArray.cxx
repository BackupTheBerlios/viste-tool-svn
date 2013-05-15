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
 * vtkUniformFloatArray.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version, not finished/working yet!
 */

#include "vtkUniformFloatArray.h"
#include <vtkObjectFactory.h>
#include <vtkFloatArray.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformFloatArray);
vtkCxxSetObjectMacro(vtkUniformFloatArray, Value, vtkFloatArray);

vtkUniformFloatArray::vtkUniformFloatArray()
{
  //this->Count = 0;
  //this->Value = NULL;
  this->Value = NULL;
}

vtkUniformFloatArray::~vtkUniformFloatArray()
{
  if (this->Value)
    {
    this->Value->UnRegister(this);
    this->Value = NULL;
    }
}

void vtkUniformFloatArray::SetGlUniformSpecific()
{
  //glUniform1fvARB(this->Location, this->Count, this->Value);

  if (!this->Value)
    {
    vtkWarningMacro(<<"Not passing uniform float array with value NULL.");
    return;
    }

  int components = this->Value->GetNumberOfComponents();
  int tuples = this->Value->GetNumberOfTuples();

  float* array = this->Value->....//HIER VERDER
  switch(tuples)
    {
    case 1:
      {
      glUniform1fv
      break;
      }
    case 2:
      {
      glUniform2fv
      break;
      }
    case 3:
      {
      glUniform3fv
      break;
      }
    case 4:
      {
      glUniform4fv
      break;
      }
    default:
      {
      glUniform1fv
      break;
      }
}

} // namespace bmia
