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
 * vtkTensorToInvariantFilter.cxx
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 * 2012-03-19	Ralph Brecheisen
 * - Fixed bug (ticket #55) where 6-valued tensors are intermixed
 *   with 9-valued tensors. In the ComputeScalar() method I convert
 *   between these two.
 */

// Includes

#include "vtkTensorToInvariantFilter.h"

// Includes VTK

#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkTensorToInvariantFilter);

//-----------------------------[ Constructor ]-----------------------------\\

vtkTensorToInvariantFilter::vtkTensorToInvariantFilter()
{
    // Set default invariant measure
    this->Invariant = Invariants::K1;
}

//-----------------------------[ Destructor ]------------------------------\\

vtkTensorToInvariantFilter::~vtkTensorToInvariantFilter()
{
}

double vtkTensorToInvariantFilter::ComputeScalar(double * tensor6)
{
	// Convert 6-valued tensor to 9-valued tensor.
	double tensor9[9];
	vtkTensorMath::Tensor6To9(tensor6, tensor9);

	// Use library functions to compute the actual invariant measure.
    return Invariants::computeInvariant(this->Invariant, tensor9);
}

} // namespace bmia
