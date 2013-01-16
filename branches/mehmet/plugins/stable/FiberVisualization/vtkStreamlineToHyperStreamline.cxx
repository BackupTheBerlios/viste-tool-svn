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
 * vtkStreamlineToHyperStreamline.cxx
 *
 * 2005-10-18	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToStreamGeneralCylinder of the DTITool.
 *
 * 2010-09-24	Evert van Aart
 * - First version for the DTITool3. Added some minor code revisions, extended comments. 
 * - Class now uses pre-computed eigensystem data (eigenvectors and eigenvalues), rather than computing
 *	 them on-the-fly using the original DTI tensors.
 *
 */


/** Includes */

#include "vtkStreamlineToHyperStreamline.h"


/** Definitions */

#define CX 0	// X Direction
#define CY 1	// Y Direction
#define CZ 2	// Z Direction

#define ERROR_PRECISION 1.0e-20f	// We consider this to be zero

#define TOLERANCE_DENOMINATOR 1000	// Used in the "FindCell" functions


namespace bmia {


vtkStandardNewMacro(vtkStreamlineToHyperStreamline);


//-----------------------------[ Constructor ]-----------------------------\\

vtkStreamlineToHyperStreamline::vtkStreamlineToHyperStreamline()
{
	// Set hyper scale to default value [1.0]
	this->HyperScale = 1.0;

	// Set tolerance to default value [1.0]
	this->tolerance = 1.0;

	// Set all image/array pointers to NULL
	this->eigenData		= NULL;
	this->eigenVectors0 = NULL;
	this->eigenVectors1 = NULL;
	this->eigenVectors2 = NULL;
	this->eigenValues0	= NULL;
	this->eigenValues1	= NULL;
	this->eigenValues2	= NULL;
}


//---------------------------[ fixFirstVectors ]---------------------------\\

void  vtkStreamlineToHyperStreamline::fixFirstVectors(double newFrame[3][3], double * lineSegment)
{
	// Auxiliary vector
	double aux[3];

	// Align the Z-vector of the frame with the line segment
	if (vtkMath::Dot(newFrame[CZ], lineSegment) < 0.0)
	{
		newFrame[CZ][0] = -newFrame[CZ][0];
		newFrame[CZ][1] = -newFrame[CZ][1];
		newFrame[CZ][2] = -newFrame[CZ][2];
	}

    // Check if the system is right-handed
	vtkMath::Cross(newFrame[CX], newFrame[CY], aux);

	// If not, flip the Y-vector of the frame
	if (vtkMath::Dot(aux, newFrame[CZ]) < 0.0)
	{
		newFrame[CY][0] = -newFrame[CY][0];
		newFrame[CY][1] = -newFrame[CY][1];
		newFrame[CY][2] = -newFrame[CY][2];
	}
}


//------------------------------[ fixVectors ]-----------------------------\\

void  vtkStreamlineToHyperStreamline::fixVectors(double prevFrame[3][3], double newFrame[3][3], double * lineSegment)
{
	// First execute the "fixFirstVectors" function.
    this->fixFirstVectors(newFrame, lineSegment);

	// Check if the X-vectors of the frames are aligned (i.e., their dot product
	// is positive). If not, fix this by flipping the X-vector of the new frame.
	// Since we want to keep the frame right-handed, we also flip the Y-vector.

	if (vtkMath::Dot(prevFrame[CX], newFrame[CX]) < 0.0)
	{
			newFrame[CY][0] = -newFrame[CY][0];
			newFrame[CY][1] = -newFrame[CY][1];
			newFrame[CY][2] = -newFrame[CY][2];

			newFrame[CX][0] = -newFrame[CX][0];
			newFrame[CX][1] = -newFrame[CX][1];
			newFrame[CX][2] = -newFrame[CX][2];
	}
}


//----------------------------[ getEigenSystem ]---------------------------\\

bool vtkStreamlineToHyperStreamline::getEigenSystem(double * point)
{
	vtkCell *	cell	= NULL;		// Cell containing current point
	vtkIdType	cellId	= -1;		// ID of current cell
	int			subId	= -1;		// Used in "FindCell" function
	double		P[3];				// Parametric Coordinates
	double		w[8];				// Interpolation weights

	// Becomes "true" if the point is inside the volume
	bool calculated = false;

	// Find the cell containing the current point
	cellId = this->eigenData->FindCell(point, cell, cellId, this->tolerance, subId, P, w);
	
	// Checks if the point is located inside the volume
	if (cellId >= 0)
	{
		// Get the cell pointer
		cell = this->eigenData->GetCell(cellId);

		// Interpolate the eigensystem data
		interpolateEigenSystem(cell, w);

		// Calculation succesful
		calculated = true;
	}

	// Return "false" if point was outside of volume, "true" otherwise.
	return calculated;
}


//--------------------[ calculateCoordinateFramePlane ]--------------------\\

void vtkStreamlineToHyperStreamline::calculateCoordinateFramePlane(	double prevFrame[3][3], 
																	double newFrame[3][3], 
																	double * currentPoint,
																	double * prevPoint, 
																	double * lineSegment, 
																	vtkIdType pointId			)
{
	// Create temporary line segment
	double tempSegment[3];
	tempSegment[0] = lineSegment[0];
	tempSegment[1] = lineSegment[1];
	tempSegment[2] = lineSegment[2];

	// If the line segment is all zero (i.e., the current point is the same
	// as the previous point), we use the previous frame; this should never happen!

	if (vtkMath::Norm(tempSegment) < ERROR_PRECISION)
	{
		this->copyFrames(newFrame, prevFrame);
		return;
	}

	// Get the eigensystem at the current point. If this function returns "false"
	// (because the point is located outside the volume), we simply copy the 
	// previous frame.

	if (this->getEigenSystem(currentPoint))
	{
		// Compute the radii of the hyper streamline
		this->Radius1 = (this->interpolatedEigenSystem.eigenValues[1] / this->maxEigenValue1) * this->HyperScale;
		this->Radius2 = (this->interpolatedEigenSystem.eigenValues[2] / this->maxEigenValue1) * this->HyperScale;
		
		// Copy the eigenvectors to the new frame
		newFrame[CX][0] = this->interpolatedEigenSystem.eigenVector1[0];
		newFrame[CX][1] = this->interpolatedEigenSystem.eigenVector1[1];
		newFrame[CX][2] = this->interpolatedEigenSystem.eigenVector1[2];

		newFrame[CY][0] = this->interpolatedEigenSystem.eigenVector2[0];
		newFrame[CY][1] = this->interpolatedEigenSystem.eigenVector2[1];
		newFrame[CY][2] = this->interpolatedEigenSystem.eigenVector2[2];

		newFrame[CZ][0] = this->interpolatedEigenSystem.eigenVector0[0];
		newFrame[CZ][1] = this->interpolatedEigenSystem.eigenVector0[1];
		newFrame[CZ][2] = this->interpolatedEigenSystem.eigenVector0[2];

		// Fix the vectors of the frames to ensure consistent orientation between
		// subsequent frames, and to make sure that the frames are right-handed.

		this->fixVectors(prevFrame, newFrame, lineSegment);
	}
	else
	{
		// Copy the old frame to the new one.
		this->copyFrames(newFrame, prevFrame);
	}
}


//------------------[ calculateFirstCoordinateFramePlane ]-----------------\\

void vtkStreamlineToHyperStreamline::calculateFirstCoordinateFramePlane(double prevFrame[3][3], 
																		double newFrame[3][3], 
																		double * currentPoint,
																		double * prevPoint, 
																		double * lineSegment, 
																		vtkIdType pointId			)
{
	// Create temporary line segment
	double tempSegment[3];
	tempSegment[0] = lineSegment[0];
	tempSegment[1] = lineSegment[1];
	tempSegment[2] = lineSegment[2];

	// If the line segment is all zero (i.e., the current point is the same
	// as the previous point), we use the previous frame; this should never happen!

	if (vtkMath::Norm(tempSegment) < ERROR_PRECISION)
	{
		this->copyFrames(newFrame, prevFrame);
		return;
	}

	// Get the eigensystem at the current point. If this function returns "false"
	// (because the point is located outside the volume), we simply copy the 
	// previous frame.

	if (this->getEigenSystem(prevPoint))
	{
		// Compute the radii of the hyper streamline
		this->Radius1 = (this->interpolatedEigenSystem.eigenValues[1] / this->maxEigenValue1) * this->HyperScale;
		this->Radius2 = (this->interpolatedEigenSystem.eigenValues[2] / this->maxEigenValue1) * this->HyperScale;
		
		// Copy the eigenvectors to the new frame
		newFrame[CX][0] = this->interpolatedEigenSystem.eigenVector1[0];
		newFrame[CX][1] = this->interpolatedEigenSystem.eigenVector1[1];
		newFrame[CX][2] = this->interpolatedEigenSystem.eigenVector1[2];

		newFrame[CY][0] = this->interpolatedEigenSystem.eigenVector2[0];
		newFrame[CY][1] = this->interpolatedEigenSystem.eigenVector2[1];
		newFrame[CY][2] = this->interpolatedEigenSystem.eigenVector2[2];

		newFrame[CZ][0] = this->interpolatedEigenSystem.eigenVector0[0];
		newFrame[CZ][1] = this->interpolatedEigenSystem.eigenVector0[1];
		newFrame[CZ][2] = this->interpolatedEigenSystem.eigenVector0[2];

		// Fix the vectors of the frames to ensure consistent orientation between
		// subsequent frames, and to make sure that the frames are right-handed.

		this->fixFirstVectors(newFrame, lineSegment);
	}
	else
	{
		// Copy the old frame to the new one. Since this function is only called during
		// the very first step, the old frame will be a 3x3 identity matrix.

		this->copyFrames(newFrame, prevFrame);
	}
}


//------------------------[ interpolateEigenSystem ]-----------------------\\

void vtkStreamlineToHyperStreamline::interpolateEigenSystem(vtkCell * cell, double * w)
{
	// Used as output for the "GetTuple" functions
	double tempVector[3];

	// Set all eigensystem components to zero
	interpolatedEigenSystem.eigenVector0[0] = 0.0;
	interpolatedEigenSystem.eigenVector0[1] = 0.0;
	interpolatedEigenSystem.eigenVector0[2] = 0.0;
	interpolatedEigenSystem.eigenVector1[0] = 0.0;
	interpolatedEigenSystem.eigenVector1[1] = 0.0;
	interpolatedEigenSystem.eigenVector1[2] = 0.0;
	interpolatedEigenSystem.eigenVector2[0] = 0.0;
	interpolatedEigenSystem.eigenVector2[1] = 0.0;
	interpolatedEigenSystem.eigenVector2[2] = 0.0;
	interpolatedEigenSystem.eigenValues[0]  = 0.0;
	interpolatedEigenSystem.eigenValues[1]  = 0.0;
	interpolatedEigenSystem.eigenValues[2]  = 0.0;

	// Loop through all eight points in the cell
	for (int k = 0; k < 8; k++)
	{
		// Get the point ID of the current point
		vtkIdType pointId = cell->PointIds->GetId(k);

		// First eigenvector
		eigenVectors0->GetTuple(pointId, tempVector);
		interpolatedEigenSystem.eigenVector0[0] += w[k] * tempVector[0];
		interpolatedEigenSystem.eigenVector0[1] += w[k] * tempVector[1];
		interpolatedEigenSystem.eigenVector0[2] += w[k] * tempVector[2];

		// Second eigenvector
		eigenVectors1->GetTuple(pointId, tempVector);
		interpolatedEigenSystem.eigenVector1[0] += w[k] * tempVector[0];
		interpolatedEigenSystem.eigenVector1[1] += w[k] * tempVector[1];
		interpolatedEigenSystem.eigenVector1[2] += w[k] * tempVector[2];

		// Third eigenvector
		eigenVectors2->GetTuple(pointId, tempVector);
		interpolatedEigenSystem.eigenVector2[0] += w[k] * tempVector[0];
		interpolatedEigenSystem.eigenVector2[1] += w[k] * tempVector[1];
		interpolatedEigenSystem.eigenVector2[2] += w[k] * tempVector[2];

		// Eigenvalues
		eigenValues0->GetTuple(pointId, &(tempVector[0]));
		eigenValues1->GetTuple(pointId, &(tempVector[1]));
		eigenValues2->GetTuple(pointId, &(tempVector[2]));
		interpolatedEigenSystem.eigenValues[0] += w[k] * tempVector[0];
		interpolatedEigenSystem.eigenValues[1] += w[k] * tempVector[1];
		interpolatedEigenSystem.eigenValues[2] += w[k] * tempVector[2];
	}
}


//-------------------------------[ Execute ]-------------------------------\\

void vtkStreamlineToHyperStreamline::Execute()
{
	// Check if the tensor data exists
	if (!(this->eigenData))
	{
		vtkErrorMacro(<<"No eigensystem data defined!");
		return;
	}

	// Check if the point data exists
	if (!(this->eigenData->GetPointData()))
	{
		vtkErrorMacro(<<"No point data defined!");
		return;
	}

	// Get the data arrays containing the eigenvectors and -values
	this->eigenVectors0 = (vtkFloatArray *) this->eigenData->GetPointData()->GetArray("Eigenvector 1");
	this->eigenVectors1 = (vtkFloatArray *) this->eigenData->GetPointData()->GetArray("Eigenvector 2");
	this->eigenVectors2 = (vtkFloatArray *) this->eigenData->GetPointData()->GetArray("Eigenvector 3");
	this->eigenValues0  = (vtkFloatArray *) this->eigenData->GetPointData()->GetArray("Eigenvalue 1");
	this->eigenValues1  = (vtkFloatArray *) this->eigenData->GetPointData()->GetArray("Eigenvalue 2");
	this->eigenValues2  = (vtkFloatArray *) this->eigenData->GetPointData()->GetArray("Eigenvalue 3");

	// Check if the arrays all exist
	if (!(this->eigenVectors0) || !(this->eigenVectors1) || !(this->eigenVectors2) ||
		!(this->eigenValues0 ) || !(this->eigenValues1 ) || !(this->eigenValues2 )  )
	{
		vtkErrorMacro(<<"One or more eigensystem arrays not found!");
		return;
	}

	// Check if the number of components is correct for each array
	if (	(this->eigenVectors0->GetNumberOfComponents() != 3) || 
			(this->eigenVectors1->GetNumberOfComponents() != 3) ||
			(this->eigenVectors2->GetNumberOfComponents() != 3)	||
			(this->eigenValues0->GetNumberOfComponents()  != 1) ||
			(this->eigenValues1->GetNumberOfComponents()  != 1) ||
			(this->eigenValues2->GetNumberOfComponents()  != 1)	 )
	{
		vtkErrorMacro(<<"Wrong number of components in one or more eigensystem arrays!");
		return;
	}

	// Pre-compute the tolerance value
	this->tolerance = this->eigenData->GetLength() / TOLERANCE_DENOMINATOR;

	// Reset the maximum second eigenvalue
	this->maxEigenValue1 = 0.0;

	// Used to store the eigenvalue
	double tempEV = 0.0;

	// Loop through all points
	for (vtkIdType ptId = 0; ptId < this->eigenValues1->GetNumberOfTuples(); ++ptId)
	{
		// Get the second eigenvalue
		this->eigenValues1->GetTuple(ptId, &tempEV);

		// Update maximum second eigenvalue
		if (tempEV > this->maxEigenValue1)
			this->maxEigenValue1 = tempEV;
	}

	// Run the default execution function
	vtkStreamlineToStreamGeneralCylinder::Execute();
}


} // namespace bmia


/** Undefine Temporary Definitions */

#undef CX
#undef CY
#undef CZ

#undef ERROR_PRECISION

#undef TOLERANCE_DENOMINATOR
