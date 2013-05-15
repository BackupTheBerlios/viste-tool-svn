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
 * vtkStreamlineToStreamTube.cxx
 *
 * 2005-10-17	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToStreamTube of the DTITool.
 *
 * 2010-09-23	Evert van Aart
 * - First version for the DTITool3. Identical to old versions, save for some code revisions
 *	 and extra comments.
 *
 */


/** Includes */

#include "vtkStreamlineToStreamTube.h"


/** Definitions */

#define CX 0	// X Direction
#define CY 1	// Y Direction
#define CZ 2	// Z Direction

#define ERROR_PRECISION 1.0e-20f	// We consider this to be zero


namespace bmia {


vtkStandardNewMacro(vtkStreamlineToStreamTube);


//-----------------------------[ Constructor ]-----------------------------\\

vtkStreamlineToStreamTube::vtkStreamlineToStreamTube()
{
  /** Number of sides of the tube */

  this->NumberOfSides = 6;

  /** Scale factor radius of the streamtube */

  this->Radius  = 1.0;
  this->Radius1 = 1.0;
  this->Radius2 = 1.0;
}


//--------------------[ calculateCoordinateFramePlane ]--------------------\\

void vtkStreamlineToStreamTube::calculateCoordinateFramePlane(	double prevFrame[3][3], 
																double newFrame[3][3], 
																double * currentPoint,
																double * prevPoint, 
																double * lineSegment, 
																vtkIdType pointId			)
{
	// Call the default function of the parent class
	this->vtkStreamlineToStreamGeneralCylinder::calculateCoordinateFramePlane(	prevFrame, 
																				newFrame, 
																				currentPoint, 
																				prevPoint, 
																				lineSegment, 
																				pointId			);
	
	// Radius scale factor
	double scalar = 1.0;

	// If needed, get the scalar value from the input
	if (this->oldScalars && this->useScalarsForThicknessFlag)
	{
		// We always use the first component for the radius scaling
		scalar = this->oldScalars->GetComponent(pointId, 0);
	}


	// Multiply the radius by the scalar value of the current point
	this->Radius1 = this->Radius * scalar;
	this->Radius2 = this->Radius * scalar;
}
	

//------------------[ calculateFirstCoordinateFramePlane ]-----------------\\

void vtkStreamlineToStreamTube::calculateFirstCoordinateFramePlane(	double prevFrame[3][3], 
																	double newFrame[3][3], 
																	double * currentPoint,
																	double * prevPoint, 
																	double * lineSegment, 
																	vtkIdType pointId			)
{
	// Call the default function of the parent class
	this->vtkStreamlineToStreamGeneralCylinder::calculateFirstCoordinateFramePlane(	prevFrame, 
																					newFrame, 
																					currentPoint, 
																					prevPoint, 
																					lineSegment, 
																					pointId			);

	// Radius scale factor
	double scalar = 1.0;

	// If needed, get the scalar value from the input
	if (this->oldScalars && this->useScalarsForThicknessFlag)
	{
		// We always use the first component for the radius scaling
		scalar = this->oldScalars->GetComponent(pointId, 0);
	}

	// Multiply the radius by the scalar value of the current point
	this->Radius1 = this->Radius * scalar;
	this->Radius2 = this->Radius * scalar;
}


//------------------------------[ SetRadius ]------------------------------\\

void vtkStreamlineToStreamTube::SetRadius(float newRadius)
{
	// Store the radius
	this->Radius	= newRadius;
	this->Radius1	= newRadius;
	this->Radius2	= newRadius;
}


} // namespace bmia


/** Undefine Temporary Definitions */

#undef CX
#undef CY
#undef CZ

#undef ERROR_PRECISION
