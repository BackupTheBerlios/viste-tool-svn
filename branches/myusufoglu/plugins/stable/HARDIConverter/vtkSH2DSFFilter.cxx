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
 * vtkSH2DSFFilter.cxx
 *
 * 2011-08-05	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "vtkSH2DSFFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkSH2DSFFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkSH2DSFFilter::vtkSH2DSFFilter()
{
	// Third-order tessellation by default
	this->tessOrder = 3;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkSH2DSFFilter::~vtkSH2DSFFilter()
{

}


//----------------------------[ SimpleExecute ]----------------------------\\

void vtkSH2DSFFilter::SimpleExecute (vtkImageData * input, vtkImageData * output)
{
	this->SetProgressText("Converting Spherical Harmonics to Discrete Sphere Function...");

	// Check if both input and output have been set
	if (input == NULL || output == NULL)
	{
		vtkErrorMacro(<< "Input and/or output not set!");
		return;
	}

	// Get the point data of the input
	vtkPointData * inPD = input->GetPointData();

	if (!inPD)
	{
		vtkErrorMacro(<< "Input does not contain point data!");
		return;
	}

	// Get the SH array
	vtkDataArray * shCoefficients = inPD->GetScalars();

	if (!shCoefficients)
	{
		vtkErrorMacro(<< "Input volume does not contain a scalar array!");
		return;
	}

	// Get the point data of the output
	vtkPointData * outPD = output->GetPointData();

	if (!outPD)
	{
		vtkErrorMacro(<< "Output does not contain point data!");
		return;
	}

	// Copy the structure of the input image to the output
	output->CopyStructure(input);

	int shOrder;

	// Get the SH order, based on the number of coefficients
	switch(shCoefficients->GetNumberOfComponents())
	{
		case 1:		shOrder = 0;	break;
		case 6:		shOrder = 2;	break;
		case 15:	shOrder = 4;	break;
		case 28:	shOrder = 6;	break;
		case 45:	shOrder = 8;	break;

		default:
			vtkErrorMacro(<< "Number of SH coefficients is not supported!");
			return;
	}

	// Create a sphere tessellator, using an icosahedron as the starting sphere
	Visualization::sphereTesselator<double> * tess = new Visualization::sphereTesselator<double>(Visualization::icosahedron);
	vtkPolyData * tessellatedSphere = vtkPolyData::New();
	tess->tesselate(this->tessOrder);
	tess->getvtkTesselation(true, tessellatedSphere);
	delete tess;

	// Get the tessellation points
	vtkPoints * tessellationPoints = tessellatedSphere->GetPoints();
	int numberOfTessPoints = tessellationPoints->GetNumberOfPoints();

	// Allocate new array for the unit vectors
	double ** unitVectors = new double*[numberOfTessPoints];

	double p[3];

	// Loop through all tessellation points
	for (int i = 0; i < numberOfTessPoints; ++i)
	{
		unitVectors[i] = new double[3];

		// Get the current point
		tessellationPoints->GetPoint(i, p);

		// Normalize the point, and store it in the unit vector array
		vtkMath::Normalize(p);

		unitVectors[i][0] = p[0];
		unitVectors[i][1] = p[1];
		unitVectors[i][2] = p[2];
	}

	// Triangulate the tessellated sphere
	vtkIntArray * trianglesArray = vtkIntArray::New();
	trianglesArray->SetName("Triangles");
	SphereTriangulator * triangulator = new SphereTriangulator;
	triangulator->triangulateFromUnitVectors(tessellationPoints, trianglesArray);
	delete triangulator;
	tessellatedSphere->Delete();

	// Store the number of tessellation points (angles)
	int numberOfAngles = numberOfTessPoints;

	// Create an array for the angles
	vtkDoubleArray * anglesArray = vtkDoubleArray::New();
	anglesArray->SetName("Spherical Directions");
	anglesArray->SetNumberOfComponents(2);
	anglesArray->SetNumberOfTuples(numberOfAngles);

	// We'll also store the angle pairs in a vector
	std::vector<double *> anglesVector;

	// Convert each point to spherical coordinates, and add it to the array and to the vector
	for (int i = 0; i < numberOfTessPoints; ++i)
	{
		double * sc = new double[2];
		sc[0] = atan2((sqrt(pow(unitVectors[i][0], 2) + pow(unitVectors[i][1], 2))), unitVectors[i][2]);
		sc[1] = atan2(unitVectors[i][1], unitVectors[i][0]);
		anglesArray->SetTuple2(i, sc[0], sc[1]);
		anglesVector.push_back(sc);
	}

	// Delete the unit vector array, since we no longer need it
	for (int i = 0; i < numberOfAngles; ++i)
	{
		delete[] (unitVectors[i]);
	}

	delete[] unitVectors;

	// Create an array for the output values (radii)
	vtkDoubleArray * outValues = vtkDoubleArray::New();
	outValues->SetName("Vectors");
	outValues->SetNumberOfComponents(numberOfTessPoints);
	outValues->SetNumberOfTuples(shCoefficients->GetNumberOfTuples());

	// Add the three arrays to the output
	outPD->AddArray(trianglesArray);
	outPD->AddArray(anglesArray);
	outPD->AddArray(outValues);

	trianglesArray->Delete();
	anglesArray->Delete();
	outValues->Delete();

	// Create a temporary array for a single output tuple
	double * outValueArray = new double[numberOfTessPoints];

	// Initialize the progress bar
	int progressStepSize = (int) ((double) shCoefficients->GetNumberOfTuples() / 25.0);
	progressStepSize += (progressStepSize == 0) ? 1 : 0;
	this->UpdateProgress(0.0);

	// Loop through all input points
	for (vtkIdType pointId = 0; pointId < shCoefficients->GetNumberOfTuples(); ++pointId)
	{
		// Update the progress bar
		if ((pointId % progressStepSize) == 0)
			this->UpdateProgress((double) pointId / (double) shCoefficients->GetNumberOfTuples());

		// Convert the SH coefficients to a deformator
		double * coeffs = shCoefficients->GetTuple(pointId);
		std::vector<double> radii = HARDITransformationManager::CalculateDeformator(coeffs, &anglesVector, shOrder);

		// Check if we succeeded
		if (radii.size() != numberOfTessPoints)
			continue;

		// Copy the deformator values to the temporary array
		int j = 0;
		for (std::vector<double>::iterator i = radii.begin(); i != radii.end(); ++i, ++j)
		{
			outValueArray[j] = (*i);
		}

		// Store the output tuple
		outValues->SetTuple(pointId, outValueArray);
	}

	// Done, delete temporary arrays
	for (std::vector<double *>::iterator i = anglesVector.begin(); i != anglesVector.end(); ++i)
	{
		delete [] (*i);
	}

	anglesVector.clear();

	delete [] outValueArray;
}


} // namespace bmia
