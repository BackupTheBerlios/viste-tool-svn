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
 * vtkShaderUniformCollection.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkShaderUniformCollection_h
#define bmia_vtkShaderUniformCollection_h

#include <vtkCollection.h>
#include "vtkShaderUniform.h" // for inline methods/static casts

namespace bmia {

class vtkShaderUniformCollection : public vtkCollection
{
public:
  static vtkShaderUniformCollection* New();

  /**
   * Add a ShaderObject to the list
   */
  void AddItem(vtkShaderUniform* su)
    {
    this->vtkCollection::AddItem((vtkObject*)su);
    };

  /**
   * Get the next ShaderUniform in the list. Return NULL when at the end of the 
   * list.
   */
  vtkShaderUniform *GetNextItem()
    {
    return static_cast<vtkShaderUniform *>(this->GetNextItemAsObject());
    };

  /**
   * Get the ith shader object from the list.
   */
  vtkShaderUniform* GetItem(int i)
    {
    return static_cast<vtkShaderUniform *>(this->GetItemAsObject(i));
    };

protected:
  vtkShaderUniformCollection() {};
  ~vtkShaderUniformCollection() {};

private:
  // hide the standard AddItem from the user and the compiler.
  void AddItem(vtkObject *o) { this->vtkCollection::AddItem(o); };

};

} // namespace bmia

#endif // bmia_vtkShaderObjectCollection_h
