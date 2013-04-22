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
 * vtkDoubleToUCharFilter.h
 * by Tim Peeters
 *
 * 2005-03-29	Tim Peeters
 * - First version
 *
 * 2005-06-03	Tim Peeters
 * - Use bmia namespace
 */

#ifndef bmia_vtkDoubleToUCharFilter_h
#define bmia_vtkDoubleToUCharFilter_h

#include <vtkSimpleImageToImageFilter.h>

namespace bmia {

/**
 * Datasets containing double data in the range <0..1> are converted
 * to unsigned char datasets with values from 0 to 255.
 *
 * vtkImageShiftScale is not used because that always assumes scalar
 * data with one component per tuple, while this is not always the case.
 * I didn't try to patch that one because it already has changes in
 * CVS version and upgrading to CVS is not practical for me.
 */
class vtkDoubleToUCharFilter : public vtkSimpleImageToImageFilter
{
public:
  static vtkDoubleToUCharFilter* New();

protected:
  vtkDoubleToUCharFilter();
  ~vtkDoubleToUCharFilter();

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

private:
  vtkDoubleToUCharFilter(const vtkDoubleToUCharFilter&); // Not implemented
  void operator=(const vtkDoubleToUCharFilter&); // Not implemented
};

} // namespace bmia

#endif
