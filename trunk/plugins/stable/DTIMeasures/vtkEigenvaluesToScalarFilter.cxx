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
 * vtkEigenvaluesToScalarFilter.cxx
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


/** Includes */

#include "vtkEigenvaluesToScalarFilter.h"


namespace bmia {


//----------------------------[ SimpleExecute ]----------------------------\\

void vtkEigenvaluesToScalarFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
{
	// Start reporting the progress of this filter
	this->UpdateProgress(0.0);

	if (!input)
	{
		vtkErrorMacro(<<"No input has been set!");
		return;
	}

	if (!output)
	{
		vtkErrorMacro(<<"No output has been set!");
		return;
	}

	// Get the point data of the input
	vtkPointData * inPD = input->GetPointData();
  
	if (!inPD)
	{
		vtkErrorMacro(<<"Input does not contain point data!");
		return;
	}

	// Get the point data of the output
	vtkPointData * outPD = output->GetPointData();
  
	if (!outPD)
	{
		vtkErrorMacro(<<"Output does not contain point data!");
		return;
	}

	// Get the number of points in the input image
	int numberOfPoints = input->GetNumberOfPoints();
  
	if (numberOfPoints < 1)
	{
		vtkWarningMacro(<<"Number of points in the input is not positive!");
		return;
	}

	// Initialize the output
	output->CopyStructure(input);
	output->SetScalarTypeToDouble();

	// Get the three eigenvalue arrays from the input
	vtkDataArray * Eigenvalues[3];
	Eigenvalues[0] = inPD->GetScalars("Eigenvalue 1");
	Eigenvalues[1] = inPD->GetScalars("Eigenvalue 2");
	Eigenvalues[2] = inPD->GetScalars("Eigenvalue 3");

	// Check if the arrays are correct
	for (int i = 0; i < 3; ++i)
	{
		if (!Eigenvalues[i])
		{
			vtkErrorMacro(<<"Input point data does not have Eigenvalue "
							<< (i + 1) << " array!");
			return;
		}

		if (numberOfPoints != Eigenvalues[i]->GetNumberOfTuples())
		{
			vtkErrorMacro(<<"Number of tuples for Eigenvalue " << (i + 1)
							<< " array does not match number of points!");
		}
	}

	// Create the output array
	vtkDoubleArray * outArray = vtkDoubleArray::New();
	outArray->SetNumberOfComponents(1);
	outArray->SetNumberOfTuples(numberOfPoints);

	// ID of the current point
	vtkIdType ptId;

	// Input eigenvalues
	double inEigenValues[3];

	// Output scalar
	double outScalar;

	// Loop through all points in the image
	for (ptId = 0; ptId < numberOfPoints; ptId++)
	{
		// Get the eigenvalues
		inEigenValues[0] = Eigenvalues[0]->GetTuple1(ptId);
		inEigenValues[1] = Eigenvalues[1]->GetTuple1(ptId);
		inEigenValues[2] = Eigenvalues[2]->GetTuple1(ptId);

		// Compute the output scalar
		outScalar = this->ComputeScalar(inEigenValues);

		// Add the scalar to the output image
		outArray->SetTuple1(ptId, outScalar);

		// Update the progress bar
		if (ptId % 50000 == 0) 
		{
			this->UpdateProgress(((float) ptId) / ((float) numberOfPoints));
		}

	} // for [ptId]

	// Add the scalars to the output
	outPD->SetScalars(outArray);

	outArray->Delete(); 
	outArray = NULL;

	// Done!
	this->UpdateProgress(1.0);

}


} // namespace bmia
