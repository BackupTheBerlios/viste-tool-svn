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
 * vtkStreamlineToHyperStreamPrisma.cxx
 *
 * 2005-11-07	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToHyperStreamPrisma of the DTITool.
 *
 * 2010-09-29	Evert van Aart
 * - First version for the DTITool3. Added some minor code revisions, extended comments. 
 *
 * 2010-10-25	Evert van Aart
 * - Added support for copying "vtkCellData" from input to output. This is needed because of a new coloring
 *   technique, which applies a single color to one whole fiber. Instead of storing the same color values
 *   for each fiber point, we store the color in the "vtkCellData" field, which is then applied to the
 *   whole fiber.
 *
 */


/** Includes */

#include "vtkStreamlineToHyperStreamPrisma.h"

/** Definitions */

#define CX 0	// X Direction
#define CY 1	// Y Direction
#define CZ 2	// Z Direction


namespace bmia {


vtkStandardNewMacro(vtkStreamlineToHyperStreamPrisma);


//-----------------------------[ Constructor ]-----------------------------\\

vtkStreamlineToHyperStreamPrisma::vtkStreamlineToHyperStreamPrisma()
{
	// Set the number of sides to four (constant)
	this->NumberOfSides = 4;

	// By default, we render ribbons instead of tubes
	this->TubeNotRibbons = true;
}


//----------------------------[ createPoints ]-----------------------------\\

void vtkStreamlineToHyperStreamPrisma::createPoints(double * pt, double * v1, double * v2, double r1, double r2, vtkIdType pointId)
{
	// If we want to render tubes, we simply use the function of the parent class.
	if (this->TubeNotRibbons)
	{
		this->vtkStreamlineToHyperStreamline::createPoints(pt, v1, v2, r1, r2, pointId);
	}
	// Code for creating ribbons
	else
	{
		double xT[3];		// Temporary point coordinates
		double radial[3];	// Input vector times input radius
		vtkIdType id;		// Point ID of new points

		// First point: Input point + Vector "v1" * Radius "r1"
		for (int j = 0; j < 3; j++)
		{
			radial[j] = r1 * v1[j];
			xT[j] = pt[j] + radial[j];
		}

		// Add new point to output
		id = this->newPts->InsertNextPoint(xT);

		// If needed, store the scalar value
		this->copyScalarValues(pointId, id);

		// Insert normal for the new point
		this->newNormals->InsertTuple(id, v2);

		// Second point: Input point + Vector "v2" * Radius "r2"
		for (int j = 0; j < 3; j++)
		{
			radial[j] = r2 * v2[j];
			xT[j] = pt[j] + radial[j];
		}

		id = this->newPts->InsertNextPoint(xT);

		this->copyScalarValues(pointId, id);

		this->newNormals->InsertTuple(id, v1);
		
		// Third point: Input point - Vector "v1" * Radius "r1"
		for (int j = 0; j < 3; j++)
		{
			radial[j] = -r1 * v1[j];
			xT[j] = pt[j] + radial[j];
		}

		id = this->newPts->InsertNextPoint(xT);

		this->copyScalarValues(pointId, id);

		this->newNormals->InsertTuple(id, v2);
		
		// Fourth point: Input point + Vector "v2" * Radius "r2"
		for (int j = 0; j < 3; j++)
		{
			radial[j] = -r2 * v2[j];
			xT[j] = pt[j] + radial[j];
		}

		id = this->newPts->InsertNextPoint(xT);

		this->copyScalarValues(pointId, id);

		this->newNormals->InsertTuple(id, v1);
	}
}


//---------------------------[ copyScalarValues ]--------------------------\\

void vtkStreamlineToHyperStreamPrisma::copyScalarValues(vtkIdType pointIdIn, vtkIdType pointIdOut)
{
	// Return if we can not or do not need to store the scalar value
	if (!(this->useScalarsForColorFlag && this->newScalars && this->oldScalars && pointIdIn >= 0))
		return;

	// If we're not dealing with RGB color values (which are store as unsigned characters),
	// simply copy the old scalar tuple.

	if (this->newScalars->GetDataType() != VTK_UNSIGNED_CHAR)
	{
		this->newScalars->InsertTuple(pointIdOut, this->oldScalars->GetTuple(pointIdIn));
	}

	// If we are dealing with RGB values, we first need to get the values from the array,
	// store them in a temporary vector, and then save them in the new array. We only do this
	// if the number of componenets is equal to three.

	else if (this->numberOfScalarComponents == 3)
	{
		unsigned char rgb[3];

		((vtkUnsignedCharArray *) this->oldScalars)->GetTupleValue(pointIdIn,  rgb);
		((vtkUnsignedCharArray *) this->newScalars)->SetTupleValue(pointIdOut, rgb);
	}
}


//----------------------------[ createPolygons ]---------------------------\\

void  vtkStreamlineToHyperStreamPrisma::createPolygons(int initialId, vtkIdType idLine)
{
	// If we want to render tubes, we simply use the function of the parent class.
	if (this->TubeNotRibbons)
	{
		this->vtkStreamlineToHyperStreamline::createPolygons(initialId, idLine);
	}
	// Code for creating ribbons
	else
	{
		// Create new polygon with five points
 		newStrips->InsertNextCell(5);
		
		// Insert polygon points
		newStrips->InsertCellPoint(initialId+0);	
		newStrips->InsertCellPoint(initialId+4);
		newStrips->InsertCellPoint(initialId+6);	
		newStrips->InsertCellPoint(initialId+0);
		newStrips->InsertCellPoint(initialId+2);	
		
		// Create new polygon with five points
		newStrips->InsertNextCell(5);

		// Insert polygon points
		newStrips->InsertCellPoint(initialId+5);	
		newStrips->InsertCellPoint(initialId+1);
		newStrips->InsertCellPoint(initialId+3);	
		newStrips->InsertCellPoint(initialId+5);
		newStrips->InsertCellPoint(initialId+7);	
		
	}
}


} // namespace bmia


/** Undefine Temporary Definitions */

#undef CX
#undef CY
#undef CZ


