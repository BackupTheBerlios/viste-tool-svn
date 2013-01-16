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
 * vtkEigensystemToGPU.h
 * by Tim Peeters
 *
 * 2007-01-10	Tim Peeters
 * - First version. Copy of vtkTensorToEigensystemFilter.h
 */

#ifndef bmia_vtkEigensystemToGPU_h
#define bmia_vtkEigensystemToGPU_h

#include <vtkSimpleImageToImageFilter.h>

namespace bmia {

/**
 * Input is tensor data.
 *
 * Computes eigensystem from tensor data, and shift/scales the components of
 * the eigenvalues to 0..1 instead of -1..1 so that they can be put in a texture.
 * Also, Cl, Cp, Cs is added as a fourth component
 */
class vtkEigensystemToGPU : public vtkSimpleImageToImageFilter
{
public:
  static vtkEigensystemToGPU *New();

protected:

  vtkEigensystemToGPU() {};
  ~vtkEigensystemToGPU() {};

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

private:

}; // class vtkEigensystemToGPU

} // namespace bmia

#endif // bmia_vtkEigensystemToGPU_h
