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
 * vtkThresholdMask.h
 * by Tim Peeters
 *
 * 2009-03-21	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkThresholdMask_h
#define bmia_vtkThresholdMask_h

#include <vtkSimpleImageToImageFilter.h>

namespace bmia {

/**
 * Class for masking a ImageData using a threshold on a
 * second ImageData with the same dimensions.
 * The output data will be a copy of input data, but with all voxels where
 * in the ThresholdInput the value is smaller than Threshold, the output data
 * will have all zeroes.
 */
class vtkThresholdMask : public vtkSimpleImageToImageFilter
{
public:
  static vtkThresholdMask* New();

  /**
   * Set the scalar volume that is used for masking
   * It must have a scalar array filled with double values.
   * If ThresholdInput is NULL then the output data will be an unmasked
   * copy of the input data.
   */
  void SetThresholdInput(vtkImageData* w);
  vtkGetObjectMacro(ThresholdInput, vtkImageData);

  vtkSetMacro(Threshold, double);
  vtkGetMacro(Threshold, double);

protected:

  vtkThresholdMask();
  ~vtkThresholdMask();

  vtkImageData* ThresholdInput;
  double Threshold;

  virtual void SimpleExecute(vtkImageData* input, vtkImageData* output);

private:

}; // class


} // namespace bmia
#endif // bmia_vtkThresholdMask_h
