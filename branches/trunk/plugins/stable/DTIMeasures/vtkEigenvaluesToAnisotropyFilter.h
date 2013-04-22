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

/*
 * vtkEigenvaluesToAnisotropyFilter.h
 *
 * 2006-12-26	Tim Peeters
 * - First version.
 *
 * 2011-03-11	Evert van Aart
 * - Added additional comments.
 *
 */


#ifndef bmia_vtkEigenvaluesToAnisotropyFilter_h
#define bmia_vtkEigenvaluesToAnisotropyFilter_h


/** Includes - Custom Files */

#include "vtkEigenvaluesToScalarFilter.h"
#include "TensorMath/AnisotropyMeasures.h"

/** Includes - VTK */

#include <vtkObjectFactory.h>


namespace bmia {


/** This class computes an anisotropy measure value using the three eigenvalues
	of a tensor as input. Looping through all input voxels is done in the parent
	class, "vtkEigenvaluesToScalarFilter". The actual measure computations are
	done in the "AnisotropyMeasures" class. 
*/

class vtkEigenvaluesToAnisotropyFilter: public vtkEigenvaluesToScalarFilter
{
	public:
  
		/** Constructor Call */

		static vtkEigenvaluesToAnisotropyFilter * New();

		/** Specify the anisotropy measure for the output. */

		vtkSetClampMacro(Measure, int, 0, AnisotropyMeasures::numberOfMeasures);
		
		/** Get the current measure. */

		vtkGetMacro(Measure, int);

	protected:
	
		/** Constructor */

		vtkEigenvaluesToAnisotropyFilter();

		/** Destructor */

		~vtkEigenvaluesToAnisotropyFilter();

		/** Compute a scalar value, based on the three eigenvalues of a tensor. 
			@param eVals	Eigenvalues. */
		
		virtual double ComputeScalar(double eVals[3]);

		/** Current anisotropy measure. */
  
		int Measure;

}; // class vtkEigenvaluesToAnisotropyFilter


} // namespace bmia


#endif // bmia_vtkEigenvaluesToAnisotropyFilter_h
