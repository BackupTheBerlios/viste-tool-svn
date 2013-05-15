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
 * vtkStreamlineToSimplifiedStreamline.cxx
 *
 * 2005-10-17	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToSimplifiedStreamline of the DTITool.
 *
 * 2010-09-24	Evert van Aart
 * - First version for the DTITool3. Identical to old versions, save for some code revisions
 *	 and extra comments.
 *
 * 2010-10-19	Evert van Aart
 * - Added support for copying "vtkCellData" from input to output. This is needed because of a new coloring
 *   technique, which applies a single color to one whole fiber. Instead of storing the same color values
 *   for each fiber point, we store the color in the "vtkCellData" field, which is then applied to the
 *   whole fiber.
 *
 * 2011-02-01	Evert van Aart
 * - Added support for bypassing this filter.
 *
 */


/** Includes */

#include "vtkStreamlineToSimplifiedStreamline.h"


/** Definitions */

#define CX 0	// X Direction
#define CY 1	// Y Direction
#define CZ 2	// Z Direction

#define ERROR_PRECISION 1.0e-20f	// We consider this to be zero


namespace bmia {


vtkStandardNewMacro(vtkStreamlineToSimplifiedStreamline);


//-----------------------------[ Constructor ]-----------------------------\\

vtkStreamlineToSimplifiedStreamline::vtkStreamlineToSimplifiedStreamline()
{
	// The length of the segments composing the fibers. A value of 1.0 corresponds
	// one time the cross diameter of a cell (i.e., the length of the "spacing" vector).

	this->StepLength = 1.0;

	// By default, do not bypass this filter

	this->doBypass = false;
}


//-----------------------------[ createPoint ]-----------------------------\\

void vtkStreamlineToSimplifiedStreamline::createPoint(double * pt)
{
	// ID of the new point
	vtkIdType id;

	// Insert the coordinates into the points array
	id = this->newPts->InsertNextPoint(pt);

	// Insert the ID in the list of output points
	this->idFiberPoints->InsertNextId(id);
}


//-------------------------------[ Execute ]-------------------------------\\

void vtkStreamlineToSimplifiedStreamline::Execute()
{
	// Get the input of the filter
    vtkPolyData * input = this->GetInput();

	// Get the output of the filter
    vtkPolyData * output = this->GetOutput();

	// Check if the input exists
	if (!input)
	{
		vtkErrorMacro(<<"No input defined!");
		return;	
	}

	// Check if the output exists
	if (!output)
	{
		vtkErrorMacro(<<"No output defined!");
		return;	
	}

	// Bypass the filter if necessary
	if (this->doBypass)
	{
		output->DeepCopy(input);
		return;
	}

	vtkPoints    * points;			// Input points
	vtkIdType    * idLinePoints;	// List of point IDs
	vtkDataArray * scalars;			// Input scalars
	vtkCellArray * lines;			// Input lines

	// Number of components in the scalars array
	int scalarsNumberOfComponents = 1;

	// Data type of the scalars
	int scalarsDataType;

	// Get the data arrays of the input
	points = input->GetPoints();
	lines  = input->GetLines();

	// Create the output point array
    this->newPts = vtkPoints::New();
	this->newPts ->Allocate(2500);
	output->SetPoints(this->newPts);
	this->newPts->Delete();

	// Create the output lines array	
	this->newFiberLine = vtkCellArray::New();
	this->newFiberLine->Allocate(this->newFiberLine->EstimateSize(2, VTK_CELL_SIZE));
	output->SetLines(this->newFiberLine);
	this->newFiberLine->Delete();

	// Get the cell data of the in- and output (used to store single fiber colors)
	vtkCellData * outCD = output->GetCellData();
	vtkCellData * inCD  = input->GetCellData();

	// Allocate the output cell data using the input cell data as size reference
	outCD->CopyAllocate(inCD);

	// Copy all fields from the input cell data
	inCD->CopyAllOn();

	// Create the ID point list
	idFiberPoints = vtkIdList::New();
	idFiberPoints->Allocate(2500);

	// Get the scalars array
	scalars = input->GetPointData()->GetScalars();

	double * scalarTuplePt		= NULL;		// Scalars of current point
	double * scalarTuplePrevPt	= NULL;		// Scalars of previous point
	double * scalarTuple		= NULL;		// Auxilary scalar array

	if (scalars != NULL) // if scalaras where generated this are also transfered
	{
		// Get the number of components of the scalar array
		scalarsNumberOfComponents = scalars->GetNumberOfComponents();

		// If the scalar array doesn't have any components, do nothing
		if (scalarsNumberOfComponents <= 0)
		{
			scalars = NULL;
		}
		else
		{
			// Create the output scalar array
			scalarsDataType = scalars->GetDataType();
			this->newScalars = vtkDataArray::CreateDataArray(scalarsDataType);
			this->newScalars->Allocate(75000);
			this->newScalars->SetNumberOfComponents(scalarsNumberOfComponents);
			output->GetPointData()->SetScalars(newScalars);
			newScalars->Delete();

			// Create the tuple arrays
			scalarTuple			= new double[scalarsNumberOfComponents];
			scalarTuplePt		= new double[scalarsNumberOfComponents];
			scalarTuplePrevPt	= new double[scalarsNumberOfComponents];
		}
	}
	
	// Get the number of fibers	
	int numberOfLines = lines->GetNumberOfCells();

	// Initialize traversal of the fibers
	lines->InitTraversal();

	// Loop through all fibers
	for (int idLine = 0; idLine < numberOfLines; idLine++)
	{
		// Number of points in the fiber
		vtkIdType numberOfPoints;
		
		// Get the point IDs and the number of points for the current fiber
		lines->GetNextCell(numberOfPoints, idLinePoints);

		// Process variables
		float distance		= 0;
		float maxDistance	= StepLength;

		// At least two fiber points are needed to create a cylinder
		if (numberOfPoints > 1)
		{ 
			double Pt[3];		// Current point
			double prevPt[3];	// Previous point
			double newPt[3];	// New point
			double v[3];		// Vector between two consecutive points
			double length;		// Length of "v"

			// Get the first two fiber points
			points->GetPoint(idLinePoints[0], prevPt);
			points->GetPoint(idLinePoints[1], Pt);

			// Compute the first line segment
			v[CX] = Pt[CX] - prevPt[CX];
			v[CY] = Pt[CY] - prevPt[CY];
			v[CZ] = Pt[CZ] - prevPt[CZ];

			// Normalize the line segment if its length is not zero
			if (vtkMath::Norm(v) > ERROR_PRECISION)
			{
				vtkMath::Normalize(v);
			}

			// Add the first point to the output
			this->createPoint(prevPt);

			// If we need to copy the scalars...
			if (scalars)
			{
				// ...get the scalars of the first two points...
				scalars->GetTuple(idLinePoints[0],scalarTuplePrevPt);
				scalars->GetTuple(idLinePoints[1],scalarTuplePt);

				// ...and write the scalar of the first point to the ouput
				this->newScalars->InsertNextTuple(scalarTuplePrevPt);
			}
			
			// Loop through the rest of the fiber points
			for (int linePoint = 1; linePoint < numberOfPoints; linePoint++)
			{
				// Get the point coordinates
			    points->GetPoint(idLinePoints[linePoint], Pt);

				// Get the scalar value(s)
				if (scalars)
				{
					scalars->GetTuple(idLinePoints[linePoint], scalarTuplePt);
				}
		
				// Compute the line segment
				v[CX] = Pt[CX] - prevPt[CX];
				v[CY] = Pt[CY] - prevPt[CY];
				v[CZ] = Pt[CZ] - prevPt[CZ];

				// Compute the length of the line segment
				length = (float) vtkMath::Norm(v);

				// Check if the line segment length is not not
				if (length > ERROR_PRECISION)
				{
					// Normalize the line segment
					vtkMath::Normalize(v);

					// Initialize auxiliary length variable
					float auxLength = 0.0f;
					
					// Repeat while the line segment is longer than the maximum step size.
					while (length > maxDistance)
					{
						// Auxilary length is a multiple of "maxDistance"
						auxLength	+= maxDistance;

						// Decrement line segment length
						length		-= maxDistance;

						// Create a new point. The point is located a multiple of the maximum
						// step size away from the previous point along the direction of "v".

						newPt[CX] = prevPt[CX] + auxLength * v[CX];
						newPt[CY] = prevPt[CY] + auxLength * v[CY];
						newPt[CZ] = prevPt[CZ] + auxLength * v[CZ];
						
						// Add the point to the output
						createPoint(newPt);	

						// If we need to copy the scalars...
						if (scalars)
						{
							// Repeat for all scalar components
							for (int i = 0; i < scalarsNumberOfComponents; i++)
							{
								// Interpolate the scalar component
								scalarTuple[i] = scalarTuplePrevPt[i] + auxLength * (scalarTuplePt[i] - scalarTuplePrevPt[i]);
							}

							// Add the tuple to the output
							newScalars->InsertNextTuple(scalarTuple);
						}

						// Reset the distance
						distance = 0.0f;
					}

					// Add the remaining length (which is less than the maximum step length)
					// to the distance variable.

					distance += length;
					
					// If the total distance (accumulated over several short steps) exceeds
					// the maximum step length, we add a new point to the output.

					if (distance >= maxDistance)
					{
						// "m" is the length between the last point and the limit
						// imposed by the maximum step length.

						float m = length - (distance - maxDistance);
				
						// Compute a new point, which is located "maxDistance" away from
						// the last added point.

						newPt[CX] = prevPt[CX] + m * v[CX];
						newPt[CY] = prevPt[CY] + m * v[CY];
						newPt[CZ] = prevPt[CZ] + m * v[CZ];
							
						// Add the point to the output
						createPoint(newPt);

						// If we need to copy the scalars...
						if (scalars)
						{
							// Repeat for all scalar components
							for (int i = 0; i < scalarsNumberOfComponents; i++)
							{
								// Interpolate the scalar component
								scalarTuple[i] = scalarTuplePrevPt[i] + m * (scalarTuplePt[i] - scalarTuplePrevPt[i]);
							}

							// Add the tuple to the output
							newScalars->InsertNextTuple(scalarTuple);
						}

						// Update the distance
						distance = distance - maxDistance;
					}

					// Update the previous point
					prevPt[0] = Pt[0];
					prevPt[1] = Pt[1];
					prevPt[2] = Pt[2];

					// Update the previous scalar tuple
					if (scalars)
					{
						for (int i = 0; i < scalarsNumberOfComponents; i++)
						{
							scalarTuplePrevPt[i] = scalarTuplePt[i];
						}
					}
				}
			}

			// Add the last point to the output
			createPoint(Pt);

			// Add the last scalar value to the output
			if (scalars)
			{
				newScalars->InsertNextTuple(scalarTuplePt);
			}

			// Insert the list point ID into the output fibers
			vtkIdType newCell = newFiberLine->InsertNextCell(idFiberPoints);

			// Copy cell data (used for single fiber colors) from input to output
			if (inCD && outCD)
			{
				outCD->CopyData(inCD, idLine, newCell);
			}

			// Reset the point list
			idFiberPoints->Reset();
		}
	}

	// Squeeze the output to regain over-allocated memory.
	output->Squeeze();

	// Delete the point list
	idFiberPoints->Delete();

	// Delete the scalar tuples arrays
	if (scalars)
	{
		delete [] scalarTuple;
		delete [] scalarTuplePrevPt;
		delete [] scalarTuplePt;
	}
};


} // namespace bmia


/** Undefine Temporary Definitions */

#undef CX
#undef CY
#undef CZ
#undef ERROR_PRECISION
