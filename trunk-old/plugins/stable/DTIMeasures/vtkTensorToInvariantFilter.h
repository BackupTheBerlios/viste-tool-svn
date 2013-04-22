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
 * vtkTensorToInvariantFilter.h
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 * 2012-03-19	Ralph Brecheisen
 * - Explicitly indicate 6-valuedness of tensor in ComputeScalar().
 */

#ifndef bmia_vtkTensorToInvariantFilter_h
#define bmia_vtkTensorToInvariantFilter_h

// Includes

#include "vtkTensorToScalarFilter.h"

// Includes tensor math

#include "TensorMath/Invariants.h"

namespace bmia {

/** This class computes scalar invariants based on input tensor data. Looping
    through all voxels is done in the parent class 'vtkTensorToScalarFilter'.
    The actual invariant computations are done in the 'Invariants' class. */

class vtkTensorToInvariantFilter : public vtkTensorToScalarFilter
{
public:

    /** Constructor call. */

    static vtkTensorToInvariantFilter * New();

    /** Specify the invariant measure for the output. */

    vtkSetClampMacro(Invariant, int, 0, Invariants::numberOfMeasures);

    /** Get the current invariant measure. */

    vtkGetMacro(Invariant, int);

protected:

    /** Constructor. */

    vtkTensorToInvariantFilter();

    /** Destructor. */

    ~vtkTensorToInvariantFilter();

    /** Compute the scalar invariant based on the given tensor value.
        @param tensor   The tensor value. */

    virtual double ComputeScalar(double * tensor6);

private:

    /** Current invariant measure. */

    int Invariant;

}; // class vtkTensorToInvariantFilter

} // namespace bmia

#endif // bmia_vtkTensorToInvariantFilter_h
