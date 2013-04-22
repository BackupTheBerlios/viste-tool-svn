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
 * vtkTensorToEigensystemFilter.h
 *
 * 2006-02-22	Tim Peeters
 * - First version, to replace vtkTensorPropertiesFilter which no longer
 *   works correctly with the new VTK 5.0 pipeline.
 *
 * 2006-05-12	Tim Peeters
 * - Add progress updates.
 *
 * 2007-02-15	Tim Peeters
 * - Clean up a bit by deleting some old obsolete comments.
 * - Use new (faster) "EigenSystem" function.
 *
 * 2008-09-09	Tim Peeters
 * - Use "EigenSystemSorted" instead of "EigenSystem". Now my hardware glyphs
 *   are oriented correctly :s
 *
 * 2010-09-03	Tim Peeters
 * - Input now has "Scalar" array with 6-component tensors instead of
 *   "Tensor" array with 9-component tensors.
 *
 * 2011-03-11	Evert van Aart
 * - Added additional comments.
 *
 */


#ifndef bmia_vtkTensorToEigensystemFilter_h
#define bmia_vtkTensorToEigensystemFilter_h


/** Includes - Custom Files */

#include "TensorMath/vtkTensorMath.h"

/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>


namespace bmia {


/** Class for creating a volume of eigensystems from a tensor volume. The 
	eigensystem can be used more directly for getting e.g. the main diffusion 
	direction (the direction of the first eigenvalue) or computing indices such 
	as fractional anisotropy.
	
	The output data contains three vector arrays named "Eigenvector 1",
	"Eigenvectors 2" and "Eigenvectors 3", which contain the eigenvectors,
	and three scalar arrays named "Eigenvalues 1", "Eigenvalues 2" and
	"Eigenvalues 3" containing the eigenvalues. These arrays can be acquieed 
	using the "GetScalars(<name>)" function of the point data of the output.
 */


class vtkTensorToEigensystemFilter : public vtkSimpleImageToImageFilter
{
	public:
  
		/** Constructor Call */

		static vtkTensorToEigensystemFilter * New();

	protected:

		/** Constructor */

		vtkTensorToEigensystemFilter() 
		{

		};

		/** Destructor */

		~vtkTensorToEigensystemFilter() 
		{

		};

		/** Executes the filter.
			@param input	Input image, containing the DTI tensors.
			@param output	Output image, containing the eigensystem data. */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

	private:


}; // class vtkTensorToEigensystemFilter


} // namespace bmia


#endif // bmia_vtkTensorToEigensystemFilter_h
