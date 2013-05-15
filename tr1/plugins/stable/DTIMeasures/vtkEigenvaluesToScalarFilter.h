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
 * vtkEigenvaluesToScalarFilter.h
 *
 * 2006-02-22	Tim Peeters
 * - First version.
 *
 * 2006-05-12	Tim Peeters
 * - Add progress updates
 *
 * 2011-03-10	Evert van Aart
 * - Added additional comments.
 *
 */


#ifndef bmia_vtkEigenvaluesToScalarFilter_h
#define bmia_vtkEigenvaluesToScalarFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>


namespace bmia {

/** Abstract class that takes a volume of eigenvalues as input and has a scalar 
	volume as output. Subclasses must implement the "ComputeScalar" function,
	which computes a scalar value from the three eigenvalues. 
*/

class vtkEigenvaluesToScalarFilter : public vtkSimpleImageToImageFilter
{
	public:

	protected:

		/** Constructor */
		
		vtkEigenvaluesToScalarFilter() 
		{

		};

		/** Destructor */
	
		~vtkEigenvaluesToScalarFilter() 
		{

		};

		/** Execute the filter.
			@param input	Input eigensystem data.
			@param output	Output scalar data. */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

		/** Compute a scalar value from three eigenvalues, ordered in descending 
			order of magnitude. Implemented in subclasses.
			@param eVals	Three eigenvalues. */
  
		virtual double ComputeScalar(double eVals[3]) = 0;

	private:


}; // class vtkEigenvaluesToScalarFilter


} // namespace bmia


#endif // bmia_vtkEigenvaluesToScalarFilter_h
