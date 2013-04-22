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
 * vtkMEVToStreamlineColor.cxx
 *
 * 2010-10-05	Evert van Aart
 * - First Version.
 * 
 * 2010-10-11	Evert van Aart
 * - Changed "abs" to "fabs"; now works under Linux.
 * 
 */


/** Includes */

#include "vtkDirectionToStreamlineColor.h"


/** Used to compute the "tolerance" variable */
#define TOLERANCE_DENOMINATOR 1000000


namespace bmia {


vtkStandardNewMacro(vtkDirectionToStreamlineColor);


//-----------------------------[ Constructor ]-----------------------------\\

vtkDirectionToStreamlineColor::vtkDirectionToStreamlineColor()
{
	// Set pointers to NULL
	this->aiImageData		= NULL;
	this->aiScalars			= NULL;
	this->cellAI			= NULL;

	// Set processing variables to default values
	this->ShiftValues		= false;
	this->UseAIWeighting	= false;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkDirectionToStreamlineColor::~vtkDirectionToStreamlineColor()
{

}


//-------------------------------[ Execute ]-------------------------------\\

void vtkDirectionToStreamlineColor::Execute()
{
	// Get the input of the filter
	vtkPolyData * input = this->GetInput();

	// Check if the input exists
	if (!input)
		return;

	// Get the fiber lines array
	vtkCellArray * oldFiberLines = input->GetLines();

	// Check if the lines array exists
	if (!oldFiberLines)
		return;

	// Get the points of the existing fibers
	vtkPoints * oldPoints = input->GetPoints();

	// Check if the points exist
	if (!oldPoints)
		return;

	// Reset pointers
	this->aiScalars	= NULL;
	this->cellAI	= NULL;

	// If we want to use AI weighting, we need to get the scalar array
	if (this->UseAIWeighting)
	{
		// Check if the AI data has been set
		if (!this->aiImageData)
			return;

		// Check if the AI data contains point data
		if (!this->aiImageData->GetPointData())
			return;

		// Get the AI scalar array
		this->aiScalars = this->aiImageData->GetPointData()->GetScalars();

		// Check if the AI scalar array exists
		if (!this->aiScalars)
			return;

		// The AI array should contain one component
		if (this->aiScalars->GetNumberOfComponents() != 1)
			return;

		// Create the cell array for the AI values
		this->cellAI = vtkDataArray::CreateDataArray(aiScalars->GetDataType());
		this->cellAI->SetNumberOfComponents(1);
		this->cellAI->SetNumberOfTuples(8);
	}

	// Create the output scalar array
	vtkUnsignedCharArray * newScalars = vtkUnsignedCharArray::New();
	newScalars->SetNumberOfComponents(3);
	newScalars->SetNumberOfTuples(input->GetNumberOfPoints());

	// Tolerance variable, used in the "FindCell" functions
	double tolerance;

	// Compute the tolerance
	if (this->UseAIWeighting)
	{
		tolerance = this->aiImageData->GetLength() / TOLERANCE_DENOMINATOR;
	}

	double			currentPoint[3];	// Current fiber point coordinates
	double			prevPoint[3];		// Previous fiber point coordinates
	double			currentSegment[3];	// Current fiber segment
	double			iAI;				// Interpolated AI value
	unsigned char	rgbOut[3];			// Output RGB values
	vtkIdType		numberOfPoints;		// Number of points in a fiber
	vtkIdType *		pointList;			// List of point IDs of a fiber

	// Initialize traversal of the fiber lines
	oldFiberLines->InitTraversal();

	// Loop through all input fibers
	for (int fiberId = 0; fiberId < oldFiberLines->GetNumberOfCells(); ++fiberId)
	{
		// Get point list and number of points of the next fiber
		oldFiberLines->GetNextCell(numberOfPoints, pointList);

		// We need at least two points
		if (numberOfPoints < 2)
			continue;

		// Get the first fiber point
		oldPoints->GetPoint(pointList[0], prevPoint);

		// This is the first point
		bool firstPoint = true;

		// Loop through all points in the fiber (except for the first one)
		for (int listId = 1; listId < numberOfPoints; ++listId)
		{
			// Get the point ID of the current point
			vtkIdType pointId = pointList[listId];

			// Get the point coordinates
			oldPoints->GetPoint(pointId, currentPoint);

			// Compute the current segment
			currentSegment[0] = currentPoint[0] - prevPoint[0];
			currentSegment[1] = currentPoint[1] - prevPoint[1];
			currentSegment[2] = currentPoint[2] - prevPoint[2];

			// Normalize the line segment
			vtkMath::Normalize(currentSegment);

			// Either shift the values to the range [0, 1]...
			if (this->ShiftValues)
			{
				currentSegment[0] = (currentSegment[0] / 2.0) + 0.5;
				currentSegment[1] = (currentSegment[1] / 2.0) + 0.5;
				currentSegment[2] = (currentSegment[2] / 2.0) + 0.5;
			}
			// ...or use the absolute values.
			else
			{
				currentSegment[0] = fabs(currentSegment[0]);
				currentSegment[1] = fabs(currentSegment[1]);
				currentSegment[2] = fabs(currentSegment[2]);
			}

			// Weight RGB values with AI value if needed
			if (this->UseAIWeighting)
			{
				vtkIdType	currentCellId	= 0;		// ID of current cell
				vtkCell *	currentCell		= NULL;		// Cell containing current point
				int			subId;						// Sub-ID, used in "FindCell" function
				double		pCoords[3];					// Parametric coordinates, used in "FindCell" function
				double		weights[8];					// Interpolation weights

				// Find the current cell, and compute the interpolation weights
				currentCellId = this->aiImageData->FindCell(currentPoint, currentCell, currentCellId, tolerance, subId, pCoords, weights);

				// Skip point if the cell was not found
				if (currentCellId < 0)
					continue;

				// Get the pointer to the current cell
				currentCell = this->aiImageData->GetCell(currentCellId);

				// Copy cell AI values to AI cell array
				this->aiScalars->GetTuples(currentCell->GetPointIds(), this->cellAI);

				// Interpolate AI value
				this->interpolateAI(&iAI, weights);

				// Clamp to the range [0, 1]
				if (iAI < 0.0)
					iAI = 0.0;
				if (iAI > 1.0)
					iAI = 1.0;
			}
			else
			{
				iAI = 1.0;
			}

			// Compute output RGB values
			rgbOut[0] = (unsigned char) (currentSegment[0] * iAI * 255.0);
			rgbOut[1] = (unsigned char) (currentSegment[1] * iAI * 255.0);
			rgbOut[2] = (unsigned char) (currentSegment[2] * iAI * 255.0);

			// Save RGB values in output
			newScalars->SetTupleValue(pointId, rgbOut);

			// If this was the first point (actually the second point in the fiber), we also
			// store the computed RGB values for the first fiber point (because we cannot
			// compute a segment for the very first point).

			if (firstPoint)
			{
				newScalars->SetTupleValue(pointList[0], rgbOut);
				firstPoint = false;
			}

			// Update the previous point

			prevPoint[0] = currentPoint[0];
			prevPoint[1] = currentPoint[1];
			prevPoint[2] = currentPoint[2];
		}
	}

	// Get the output of the filter
	vtkPolyData * output = this->GetOutput();
	
	// Set the scalars, points, and lines of the output	
	output->GetPointData()->SetScalars(newScalars);
	output->SetPoints(oldPoints);
	output->SetLines(oldFiberLines);

	// Delete the new scalars
	newScalars->Delete();

	// If necessary, delete the AI cell array
	if (this->cellAI)
	{
		this->cellAI->Delete();
		this->cellAI = NULL;
	}
}


//----------------------------[ interpolateAI ]----------------------------\\

void vtkDirectionToStreamlineColor::interpolateAI(double * output, double * weights)
{
	// Set output to zero
	(*output) = 0.0;

	// Temporary AI value
	double tempAI;

	// Loop through all points in the cell
	for (int i = 0; i < 8; ++i)
	{
		// Get current AI value
		this->cellAI->GetTuple(i, &tempAI);

		// Add current AI value to output
		(*output) += weights[i] * tempAI;
	}
}


} // namespace bmia