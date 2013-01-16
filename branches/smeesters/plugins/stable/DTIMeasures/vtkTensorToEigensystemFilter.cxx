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
 * vtkTensorToEigensystemFilter.cxx
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


/** Includes */

#include "vtkTensorToEigensystemFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkTensorToEigensystemFilter);


void vtkTensorToEigensystemFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
{
	vtkDebugMacro(<<"Computing eigen systems from tensors...");

	// Initialize the progress bar
	this->SetProgressText("Computing eigenvalues and eigenvectors from tensors...");
	this->UpdateProgress(0.0);

	if (!input)
	{
		vtkErrorMacro(<<"Input has not been set!");
		return;
	}

	if (!output)
	{
		vtkErrorMacro(<<"Output has not been set!");
		return;
	}

	// Get the input point data
	vtkPointData * inPD = input->GetPointData();
  
	if (!inPD)
	{
		vtkErrorMacro(<<"Input does not contain point data!");
		return;
	}

	// Get the input tensor array
	vtkDataArray * inTensors = inPD->GetArray("Tensors");
 
	if (!inTensors)
	{
		vtkWarningMacro(<<"Input data has no tensors!");
		return;
	}

	// Get the output point data
	vtkPointData * outPD = output->GetPointData();
  
	if (!outPD)
	{
		vtkErrorMacro(<<"Output does not contain point data!");
		return;
	}

	int numberOfPoints = input->GetNumberOfPoints();
  
	if (numberOfPoints != inTensors->GetNumberOfTuples())
	{
		vtkErrorMacro(<<"Size mismatch between point data and tensor array!");
		return;
	}

	if (numberOfPoints < 1)
	{
		vtkWarningMacro(<<"Number of points in input image is non-positive!");
		return;
	}

	// Output scalar arrays
	vtkFloatArray * Arrays[6];

	for (int i = 0; i < 6; ++i) 
	{
		Arrays[i] = vtkFloatArray::New();
	}

	// Set the names of the arrays
	Arrays[0]->SetName("Eigenvector 1");
	Arrays[1]->SetName("Eigenvector 2");
	Arrays[2]->SetName("Eigenvector 3");
	Arrays[3]->SetName("Eigenvalue 1");
	Arrays[4]->SetName("Eigenvalue 2");
	Arrays[5]->SetName("Eigenvalue 3");

	// Set the number of components (three for eigenvectors, one for -values)
	Arrays[0]->SetNumberOfComponents(3);
	Arrays[1]->SetNumberOfComponents(3);
	Arrays[2]->SetNumberOfComponents(3);
	Arrays[3]->SetNumberOfComponents(1);
	Arrays[4]->SetNumberOfComponents(1);
	Arrays[5]->SetNumberOfComponents(1);

	// Set the number of tuples (equal to the number of input points)
	Arrays[0]->SetNumberOfTuples(numberOfPoints);
	Arrays[1]->SetNumberOfTuples(numberOfPoints);
	Arrays[2]->SetNumberOfTuples(numberOfPoints);
	Arrays[3]->SetNumberOfTuples(numberOfPoints);
	Arrays[4]->SetNumberOfTuples(numberOfPoints);
	Arrays[5]->SetNumberOfTuples(numberOfPoints);

	// ID of the current point	
	vtkIdType ptId;
  
	// Current tensor
	double tensor6[6];

	// Allocate arrays for the eigenvalues and -vectors
	double * eigenValues  = new double[3];
	double * eigenVectors = new double[9];

	// Loop through all points in the image
	for (ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		// Get the current tensor
		inTensors->GetTuple(ptId, tensor6);

		// The "IsNullTensor" check is not required for correctness, but it 
		// will save computation time on sparse datasets.

		if (vtkTensorMath::IsNullTensor(tensor6))
		{
			// Set the outputs to zero
			Arrays[0]->SetTuple3(ptId, 0.0, 0.0, 0.0);
			Arrays[1]->SetTuple3(ptId, 0.0, 0.0, 0.0);
			Arrays[2]->SetTuple3(ptId, 0.0, 0.0, 0.0);
			Arrays[3]->SetTuple1(ptId, 0.0);
			Arrays[4]->SetTuple1(ptId, 0.0);
			Arrays[5]->SetTuple1(ptId, 0.0);
		}

		// Try to compute the eigensystem. If this fails, it may be because of 
		// negative eigenvalues. To avoid holes in the eigensystem image, we set
		// the eigensystem of these tensors to all zeros. 
		else if (!(vtkTensorMath::EigenSystemSorted(tensor6, eigenVectors, eigenValues)))
		{
			vtkDebugMacro(<<"Could not compute eigensystem for tensor (("
							<< tensor6[0] << ", " << tensor6[1] << ", " << tensor6[2] << "), ("
							<< tensor6[1] << ", " << tensor6[3] << ", " << tensor6[4] << "), ("
							<< tensor6[2] << ", " << tensor6[4] << ", " << tensor6[5] << "))");
		} 
	
		// Valid eigensystem
		else
		{ 
			// Add the vectors and values to the output arrays
			for (int i = 0; i < 3; ++i)
			{
				Arrays[i    ]->SetTuple3(ptId,	eigenVectors[3 * i], 
												eigenVectors[3 * i + 1], 
												eigenVectors[3 * i + 2]);
				Arrays[3 + i]->SetTuple1(ptId, eigenValues[i]);
			} 
		}

		// Update the progress bar
		if (ptId % 50000 == 0) 
			this->UpdateProgress(((float) ptId) / ((float) numberOfPoints));

	} // for [ptId]

	// Delete the temporary arrays
	delete[] eigenValues; 
	delete[] eigenVectors; 

	// Copy the structure of the input image to the output
	output->CopyStructure(input);

	// Add the arrays to the output and select active scalars and vectors
	outPD->AddArray(Arrays[0]);
	outPD->AddArray(Arrays[1]);
	outPD->AddArray(Arrays[2]);
	outPD->AddArray(Arrays[3]);
	outPD->AddArray(Arrays[4]);
	outPD->AddArray(Arrays[5]);

	outPD->SetActiveScalars(Arrays[3]->GetName());
	outPD->SetActiveVectors(Arrays[0]->GetName());

	// Delete the arrays
	Arrays[0]->Delete();
	Arrays[1]->Delete();
	Arrays[2]->Delete();
	Arrays[3]->Delete();
	Arrays[4]->Delete();
	Arrays[5]->Delete();

	// Done!
	this->UpdateProgress(1.0);
	vtkDebugMacro(<<"Eigensystem computation finished.");

}


} // namespace bmia
