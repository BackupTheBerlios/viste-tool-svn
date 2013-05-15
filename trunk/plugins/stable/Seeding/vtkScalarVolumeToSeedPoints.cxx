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
 * vtkScalarVolumeToSeedPoints.cxx
 *
 * 2011-05-10	Evert van Aart
 * - First version
 *
 */


/** Includes */

#include "vtkScalarVolumeToSeedPoints.h"


namespace bmia {


vtkStandardNewMacro(vtkScalarVolumeToSeedPoints);


//-----------------------------[ Constructor ]-----------------------------\\

vtkScalarVolumeToSeedPoints::vtkScalarVolumeToSeedPoints()
{
	// Initialize variables
	this->minThreshold = 0.0;
	this->maxThreshold = 1.0;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkScalarVolumeToSeedPoints::~vtkScalarVolumeToSeedPoints()
{

}


//-------------------------------[ Execute ]-------------------------------\\

void vtkScalarVolumeToSeedPoints::Execute()
{
	// Get the input data set and cast it to an image
	vtkImageData * image = vtkImageData::SafeDownCast(this->GetInput());

	// If the input has not been set, OR if it's not an image, we throw an error
	if (!image)
	{
		vtkErrorMacro(<< "Input image has not been set!");
		return;
	}

	// Get the point data of the input
	vtkPointData * imagePD = image->GetPointData();

	// Check if the input has been set
	if (!imagePD)
	{
		vtkErrorMacro(<< "Input image does not contain point data!");
		return;
	}

	// Get the scalar array
	vtkDataArray * imageScalars = imagePD->GetScalars();

	// Check if the input has been set
	if (!imageScalars)
	{
		vtkErrorMacro(<< "Input image does not contain a scalar array!");
		return;
	}

	// Get the number of seed points
	vtkIdType numberOfPoints = image->GetNumberOfPoints();

	// Get the output seed points
	vtkUnstructuredGrid * output = this->GetOutput();

	// Create a new point set
	vtkPoints * newPoints = vtkPoints::New();
	newPoints->SetDataTypeToDouble();

	// Add the points to the output data set
	output->SetPoints(newPoints);
	newPoints->Delete();

	// Point coordinates
	double p[3];

	// Current scalar value
	double scalar;

	// Loop through all voxels
	for (vtkIdType i = 0; i < numberOfPoints; ++i)
	{
		// Get the voxel scalar value
		scalar = imageScalars->GetTuple1(i);

		// If the scalar value lies between the two thresholds...
		if (scalar >= this->minThreshold && scalar <= this->maxThreshold)
		{
			// ...get the coordinates of its voxel...
			image->GetPoint(i, p);

			// ...and add those coordinates to the output.
			newPoints->InsertNextPoint(p);
		}

	} // for [every voxel]
}


} // namespace bmia
