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
 * vtkTensorToScalarFilter.h
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 */

#ifndef bmia_vtkTensorToScalarFilter_h
#define bmia_vtkTensorToScalarFilter_h

// Includes VTK

#include <vtkSimpleImageToImageFilter.h>
#include <vtkImageData.h>

namespace bmia {

/** Abstract class that takes a tensor volume and produces a scalar volume
    as output. Subclasses must implement the 'ComputeScalar' function, which
    computes a scalar value for each tensor value.
*/

class vtkTensorToScalarFilter : public vtkSimpleImageToImageFilter
{
public:

protected:

    /** Constructor. */

	vtkTensorToScalarFilter()
	{
	};

    /** Destructor. */

	~vtkTensorToScalarFilter()
	{
	};

    /** Execute the filter.
        @param input    Input tensor data.
        @param output   Output scalar data. */

	virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

    /** Compute a scalar value from the given tensor value. Implemented
        in subclasses.
        @param tensor   The tensor value. */

	virtual double ComputeScalar(double tensor[6]) = 0;

private:

}; // class vtkTensorToScalarFilter

} // namespace bmia

#endif // bmia_vtkTensorToScalarFilter_h
