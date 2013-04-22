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
 * vtkStreamlineToStreamGeneralCylinder.h
 *
 * 2005-10-17	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToStreamGeneralCylinder of the DTITool.
 *
 * 2010-09-24	Evert van Aart
 * - First version for the DTITool3. Identical to old versions, save for some code revisions
 *	 and extra comments.
 *
 * 2011-01-19	Evert van Aart
 * - Improved ERROR checking and -handling.
 * - Added Qt progress bar.
 *
 * 2011-04-16	Evert van Aart
 * - Changed the Qt Progress bar to "updateProgress" calls.
 *
 */

/** Includes */
#include "vtkStreamlineToStreamGeneralCylinder.h"


/** Definitions */

#define CX 0	// X Direction
#define CY 1	// Y Direction
#define CZ 2	// Z Direction

#define ERROR_PRECISION 1.0e-20f	// We consider this to be zero


namespace bmia {


vtkStandardNewMacro(vtkStreamlineToStreamGeneralCylinder);


//-----------------------------[ Constructor ]-----------------------------\\

vtkStreamlineToStreamGeneralCylinder::vtkStreamlineToStreamGeneralCylinder()
{
	/** Number of sides of the tube. */
  
	this->NumberOfSides = 6;

	/** Scale factor radius of the streamtube */
  
	this->Radius1 = 1.0;
	this->Radius2 = 1.0;
  
	/** Do not use the scalars by default. */
  
	useScalarsForThicknessFlag	= false;
	useScalarsForColorFlag		= true;
}


//------------------------[ useScalarsForThickness ]-----------------------\\

void vtkStreamlineToStreamGeneralCylinder::useScalarsForThickness(bool flag)
{
	// Store setting
	this->useScalarsForThicknessFlag = flag;
}


//--------------------------[ useScalarsForColor ]-------------------------\\

void vtkStreamlineToStreamGeneralCylinder::useScalarsForColor(bool flag)
{
	// Store setting
	this->useScalarsForColorFlag = flag;
}


//-----------------------------[ createPoints ]----------------------------\\

void vtkStreamlineToStreamGeneralCylinder::createPoints(double * pt, double * v1, double * v2, double r1, double r2, vtkIdType pointId)
{
	// Compute the angle step size.
	double theta = (2.0 * vtkMath::Pi()) / this->NumberOfSides;

	double xT[3];		// Point coordinates
	double normal[3];	// Normal at point
	vtkIdType id;		// Point ID
	
	// Loop counter-closewise through the circle

	for (int k = 0; k < this->NumberOfSides; k++)
	{
		for (int j = 0; j < 3; j++)
		{
			// Compute the normal
			normal[j] = ( r1 * v1[j] * cos((double) k * theta) +
						  r2 * v2[j] * sin((double) k * theta));

			// Compute the actual point coordinates
				xT[j] = pt[j] + normal[j];
		}

		// Insert the coordinates and save the new point ID
		id = this->newPts->InsertNextPoint(xT);

		// If needed, store the scalar value
		if (this->useScalarsForColorFlag && this->newScalars && this->oldScalars && pointId >= 0)
		{
			// If we're not dealing with RGB color values (which are store as unsigned characters),
			// simply copy the old scalar tuple.

			if (this->newScalars->GetDataType() != VTK_UNSIGNED_CHAR)
			{
				this->newScalars->InsertTuple(id, this->oldScalars->GetTuple(pointId));
			}

			// If we are dealing with RGB values, we first need to get the values from the array,
			// store them in a temporary vector, and then save them in the new array. We only do this
			// if the number of componenets is equal to three.

			else if (this->numberOfScalarComponents == 3)
			{
				unsigned char rgb[3];

				((vtkUnsignedCharArray *) this->oldScalars)->GetTupleValue(pointId, rgb);

				((vtkUnsignedCharArray *) this->newScalars)->SetTupleValue(id, rgb);
			}
		}

		// Normalize the normal to make it even more normal
		vtkMath::Normalize(normal); 

		// Insert the normal into the data array
		this->newNormals->InsertTuple(id, normal);
	}
}


//----------------------------[ createPolygons ]---------------------------\\

void vtkStreamlineToStreamGeneralCylinder::createPolygons(int initialId, vtkIdType idLine)
{
	// Insert the next cell, set the number of points
	vtkIdType newCell = this->newStrips->InsertNextCell((this->NumberOfSides + 1) * 2);

	// Insert all new cell points into the cell, as well as the points
	// of the crosssection of the previous fiber point
	for (int k = 0; k < this->NumberOfSides; k++)
	{
		this->newStrips->InsertCellPoint(initialId + k + this->NumberOfSides);	
		this->newStrips->InsertCellPoint(initialId + k);
	}

	// Connect both sides of the tube
	this->newStrips->InsertCellPoint(initialId + this->NumberOfSides);
	this->newStrips->InsertCellPoint(initialId);

	// Copy cell data (used for single fiber colors) from input to output
	if (this->inCD && this->outCD)
	{
		this->outCD->CopyData(this->inCD, idLine, newCell);
	}
}


//------------------------------[ copyFrames ]-----------------------------\\

void vtkStreamlineToStreamGeneralCylinder::copyFrames(double prevFrame[3][3], double newFrame[3][3])
{
	// Copy all frame elements from the current frame to the previous frame
	prevFrame[CX][CX] = newFrame[CX][CX];
	prevFrame[CX][CY] = newFrame[CX][CY];
	prevFrame[CX][CZ] = newFrame[CX][CZ];

	prevFrame[CY][CX] = newFrame[CY][CX];
	prevFrame[CY][CY] = newFrame[CY][CY];
	prevFrame[CY][CZ] = newFrame[CY][CZ];

	prevFrame[CZ][CX] = newFrame[CZ][CX];
	prevFrame[CZ][CY] = newFrame[CZ][CY];
	prevFrame[CZ][CZ] = newFrame[CZ][CZ];
}	


//--------------------[ calculateCoordinateFramePlane ]--------------------\\

void vtkStreamlineToStreamGeneralCylinder::calculateCoordinateFramePlane(	double prevFrame[3][3], 
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
	// as the previous point), we use the vector from the previous frame.

	if (vtkMath::Norm(tempSegment) < ERROR_PRECISION)
	{
		newFrame[CZ][0] = prevFrame[CZ][0];
		newFrame[CZ][1] = prevFrame[CZ][1];
		newFrame[CZ][2] = prevFrame[CZ][2];
	}
	else
	{
		// Normalize the current line segment
		vtkMath::Normalize(tempSegment);

		// Set the Z-direction of the frame equal to the normalized line segment
		newFrame[CZ][CX] = tempSegment[CX];
		newFrame[CZ][CY] = tempSegment[CY];
		newFrame[CZ][CZ] = tempSegment[CZ];
	}
	
	// Compute dot product between the X-vector of the previous frame and the
	// Z-vector of the new frame.

	float dot = fabs(vtkMath::Dot(prevFrame[CX], newFrame[CZ]));

	// Check if the two vectors are parallel

	if (dot < (1.0 - ERROR_PRECISION))
	{
		// If they're not parallel, check if the angle between them is more than 90 degrees.
		if (vtkMath::Dot(prevFrame[CZ], newFrame[CZ]) < 0)
		{
			// Recompute X-vector of previous frame
			prevFrame[CX][0] =- prevFrame[CX][0];
			prevFrame[CX][1] =- prevFrame[CX][1];
			prevFrame[CX][2] =- prevFrame[CX][2];
		}

		// Calculate the other two vectors of the new frame.
		vtkMath::Cross(newFrame[CZ], prevFrame[CX], newFrame[CY]);
		vtkMath::Normalize(newFrame[CY]);
		vtkMath::Cross(newFrame[CY], newFrame[CZ], newFrame[CX]);
	}
	else
	{
		// Check if the angle between two vectors is more than 90 degrees.
		if (vtkMath::Dot(prevFrame[CZ], newFrame[CZ]) < 0)
		{
			// Recompute Y-vector of previous frame
			prevFrame[CY][0] =- prevFrame[CY][0];
			prevFrame[CY][1] =- prevFrame[CY][1];
			prevFrame[CY][2] =- prevFrame[CY][2];
		}

		// Calculate the other two vectors of the new frame.
		vtkMath::Cross(prevFrame[CY], newFrame[CZ], newFrame[CX]);
		vtkMath::Normalize(newFrame[CX]);
		vtkMath::Cross(newFrame[CZ], newFrame[CX], newFrame[CY]);
	}
}


//------------------[ calculateFirstCoordinateFramePlane ]-----------------\\

void vtkStreamlineToStreamGeneralCylinder::calculateFirstCoordinateFramePlane(	double prevFrame[3][3], 
																				double newFrame[3][3], 
																				double * currentPoint,
																				double * prevPoint, 
																				double * lineSegment, 
																				vtkIdType pointId			)
{
	// In this class, we simply use the general frame calculation function.
	this->calculateCoordinateFramePlane(prevFrame, newFrame, currentPoint, prevPoint, lineSegment, pointId);
}


//-------------------------------[ Execute ]-------------------------------\\

void vtkStreamlineToStreamGeneralCylinder::Execute()
{
	// Get the input of the filter
    vtkPolyData * input = this->GetInput();

	// Check if the input exists
	if (!input)
	{
		vtkErrorMacro(<< "No input defined!");
		return;
	}

	// Get the output of the filter
    vtkPolyData * output = this->GetOutput();

	// Check if the output exists
	if (!output)
	{
		vtkErrorMacro(<< "No output defined!");
		return;
	}

	vtkPoints    * points;			// Input points
	vtkIdType    * idLinePoints;	// List of point IDs
	vtkCellArray * lines;			// Input lines

	// Get the data arrays of the input
	points	= input->GetPoints();

	if (!points)
	{
		vtkErrorMacro(<< "Input does not contain points!");
		return;
	}

	lines	= input->GetLines();

	if (!lines)
	{
		vtkErrorMacro(<< "Input does not contain fibers!");
		return;
	}

	if (!input->GetPointData())
	{
		vtkErrorMacro(<< "Input does not contain point data!");
		return;
	}

	// Create the output point array
	this->newPts = vtkPoints::New();
	this->newPts ->Allocate(2500); 
	output->SetPoints(this->newPts);
	this->newPts->Delete();

	// Create the output normals array
	this->newNormals = vtkFloatArray::New();
	this->newNormals->SetNumberOfComponents(3);
	this->newNormals->Allocate(7500);
	output->GetPointData()->SetNormals(this->newNormals);
	this->newNormals->Delete();

	// Get the cell data of the in- and output (used to store single fiber colors)
	this->outCD = output->GetCellData();
	this->inCD  = input->GetCellData();

	// Allocate the output cell data using the input cell data as size reference
	outCD->CopyAllocate(inCD);

	// Copy all fields from the input cell data
	inCD->CopyAllOn();

	// Get the input scalars
	this->oldScalars = input->GetPointData()->GetScalars();

	// Reset the new scalars pointer
	this->newScalars = NULL;

	// Input scalar value
	double * scalar = NULL;

	// If we are using the scalars for the cylinder colors, create the output scalar array
	if (this->oldScalars && useScalarsForColorFlag)
	{
		// Create a general Data Array if we're not dealing with unsigned characters (RGB)
		if (this->oldScalars->GetDataType() != VTK_UNSIGNED_CHAR)
		{
			this->newScalars = vtkDataArray::CreateDataArray(this->oldScalars->GetDataType());
		}
		// If the type is "unsigned char", we need to create a special Unsigned Character array
		else
		{
			this->newScalars = (vtkDataArray *) vtkUnsignedCharArray::New();
		}

		// Set properties of the scalar array
		this->newScalars->SetNumberOfComponents(this->oldScalars->GetNumberOfComponents());
		this->newScalars->SetNumberOfTuples(this->oldScalars->GetNumberOfTuples() * (this->NumberOfSides + 1));

		// Set scalar array to output
		output->GetPointData()->SetScalars(this->newScalars);
		this->newScalars->Delete();

		// Store the number of scalar components
		this->numberOfScalarComponents = this->oldScalars->GetNumberOfComponents();
	}

	// Create the output lines array
	this->newStrips = vtkCellArray::New();
	this->newStrips->Allocate(1000);
	output->SetStrips(this->newStrips);
	this->newStrips->Delete();

	// Setup the progress bar
	this->SetProgressText("Constructing 3D fiber geometry...");
	this->UpdateProgress(0.0);

	// Get the number of fibers
	int numberOfLines = lines->GetNumberOfCells();

	// Initialize the traversal of the fibers
	lines->InitTraversal();

	// Compute the step size for the progress bar
	int progressStepSize = (int) ((float) numberOfLines / 25.0f);
	progressStepSize += (progressStepSize == 0) ? 1 : 0;

	// Loop through all fibers
	for (int idLine = 0; idLine < numberOfLines; idLine++)
	{
		// Update progress bar
		if ((idLine % progressStepSize) == 0)
		{
			this->UpdateProgress((double) idLine / (double) numberOfLines);
		}

		// Number of fiber points
		vtkIdType numberOfPoints;
		
		// Get the point IDs and the number of points for the current fiber
		lines->GetNextCell(numberOfPoints, idLinePoints);

		// At least two fiber points are needed to create a cylinder
		if (numberOfPoints < 2)
		{
			continue;
		}

		double Pt[3];		// Current fiber point
		double prevPt[3];	// Previous fiber point
		double v[3];		// Vector between two consecutive points		

		// Coordinate frames at current and previous points
		double currentFrame[3][3]	= {{1,0,0}, {0,1,0}, {0,0,1}};
		double prevFrame[3][3]		= {{1,0,0}, {0,1,0}, {0,0,1}};

		// Get the number of points in the new points array. The ID of
		// the next point will be equal to this number.

		int initialId = this->newPts->GetNumberOfPoints();

		// Get the first two fiber points
		points->GetPoint(idLinePoints[0], prevPt);
		points->GetPoint(idLinePoints[1], Pt);
		
		// Compute the first line segment
		v[CX] = Pt[CX] - prevPt[CX];
		v[CY] = Pt[CY] - prevPt[CY];
		v[CZ] = Pt[CZ] - prevPt[CZ];

		// Compute the first coordinate frame
		calculateFirstCoordinateFramePlane(prevFrame, currentFrame, Pt, prevPt, v, idLinePoints[0]);
		
		// Add the cross-section for the first and second points
		createPoints(prevPt, currentFrame[CX], currentFrame[CY], this->Radius1, this->Radius2, idLinePoints[0]);			
		createPoints(Pt,     currentFrame[CX], currentFrame[CY], this->Radius1, this->Radius2, idLinePoints[0]);	

		// Create the polygons between the first two cross-sections.		
		createPolygons(initialId, idLine); 

		// Increment the initial ID. Next time, when constructing the polygons, we
		// start at the first point of the second cross-section.

		initialId += this->NumberOfSides;

		// Update the previous point
		prevPt[0] = Pt[0];
		prevPt[1] = Pt[1];
		prevPt[2] = Pt[2];

		// Update the previous frame
		copyFrames(prevFrame, currentFrame);

		// Loop through the remainder of the fiber points
		for (int linePoint = 2; linePoint < numberOfPoints; linePoint++)
		{
			// Get the current point
		    points->GetPoint(idLinePoints[linePoint], Pt);

			// Compute the current line segment
			v[CX] = Pt[CX] - prevPt[CX];
			v[CY] = Pt[CY] - prevPt[CY];
			v[CZ] = Pt[CZ] - prevPt[CZ];

			// Compute the new coordinate frame
			calculateCoordinateFramePlane(prevFrame, currentFrame, Pt, prevPt, v, idLinePoints[linePoint]);
			
			// Create the points of the cross-section at the current point
			createPoints(Pt, currentFrame[CX], currentFrame[CY], this->Radius1, this->Radius2, idLinePoints[linePoint]);	
			
			// Create the polygons between the cross-sections of the current and previous point.
			createPolygons(initialId, idLine); 

			// Update the initial ID for the points
			initialId += NumberOfSides;

			// Update the previous point
			prevPt[0] = Pt[0];
			prevPt[1] = Pt[1];
			prevPt[2] = Pt[2];

			// Update the previous frame
			copyFrames(prevFrame, currentFrame);
		}
	}

	// Finalize the progress bar
	this->UpdateProgress(1.0);

	// Squeeze the output to regain over-allocated memory
	output->Squeeze();
}


} // namespace bmia


/** Undefine Temporary Definitions */

#undef CX
#undef CY
#undef CZ
#undef ERROR_PRECISION
