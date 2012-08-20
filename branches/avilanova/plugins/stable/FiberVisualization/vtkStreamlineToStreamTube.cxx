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
