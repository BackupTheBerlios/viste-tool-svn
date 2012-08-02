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
 * vtk2DROIFiberFilter.cxx
 *
 * 2010-11-02	Evert van Aart
 * - First version
 *
 * 2010-11-22	Evert van Aart
 * - Fixed an error where the output would contain no fibers if all
 *   ROIs were "NOT" ROIs.
 *
 */


/** Includes */

#include "vtk2DROIFiberFilter.h"


namespace bmia {


vtkStandardNewMacro(vtk2DROIFiberFilter);



//-----------------------------[ Constructor ]-----------------------------\\

vtk2DROIFiberFilter::vtk2DROIFiberFilter()
{
	// Set default values of class variables
	this->input				= NULL;
	this->output			= NULL;
	this->inPD				= NULL;
	this->outPD				= NULL;
	this->inCD				= NULL;
	this->outCD				= NULL;
	this->outputPoints		= NULL;
	this->outputLines		= NULL;
	this->currentROI		= NULL;
	this->firstX[0]			= 0.0;
	this->firstX[1]			= 0.0;
	this->firstX[2]			= 0.0;
	this->lastX[0]			= 0.0;
	this->lastX[1]			= 0.0;
	this->lastX[2]			= 0.0;
	this->firstID			= -1;
	this->lastID			= -1;
	this->CutFibersAtROI	= false;
	this->numberOfNOTs		= 0;
}


//------------------------------[ Destructor ]-----------------------------\\

vtk2DROIFiberFilter::~vtk2DROIFiberFilter()
{
	// Iterator for the list of ROIs
	std::list<ROISettings>::iterator ROIIter;

	// Loop through all added ROIs
	for (ROIIter = this->ROIList.begin(); ROIIter != this->ROIList.end(); ++ROIIter)
	{
		// Do nothing if the ROI does not have any polygons
		if ((*ROIIter).ROIPolygons.empty())
			continue;

		// Iterator for the list of ROI polygons
		std::list<vtkPolygon *>::iterator PolygonIter;

		// Delete all polygons
		for (PolygonIter = (*ROIIter).ROIPolygons.begin(); PolygonIter != (*ROIIter).ROIPolygons.end(); ++PolygonIter)
		{
			(*PolygonIter)->Delete();
		}

		// Clear the list of polygons
		(*ROIIter).ROIPolygons.clear();
	}

	// Clear the list of ROIs
	this->ROIList.clear();
}


//--------------------------------[ addROI ]-------------------------------\\

void vtk2DROIFiberFilter::addROI(vtkPolyData * rROI, bool rNOT)
{
	// Do nothing if no pointer has been set
	if (!rROI)
		return;

	// Copy the settings to a new struct
	ROISettings newROI;
	newROI.ROI = rROI;
	newROI.bNOT = rNOT;
	
	// Attempt to create polygons from the input poly data object
	if (!this->createPolygons(&newROI))
		return;

	// Keep track of the number of ROIs with "bNOT" equal to true
	if (rNOT)
		numberOfNOTs++;

	// Add the ROI to the list
	this->ROIList.push_back(newROI);
}


//----------------------------[ createPolygons ]---------------------------\\

bool vtk2DROIFiberFilter::createPolygons(ROISettings * newROI)
{
	// Get the poly data of the newly added ROI
	vtkPolyData * newPD = (*newROI).ROI;

	// Get the number of polygons (number of lines in the input)
	int numberOfPolygons = newPD->GetNumberOfLines();

	// Check if the input contains lines
	if (numberOfPolygons <= 0)
		return false;

	// Get the lines array from the input
	vtkCellArray * newLines  = newPD->GetLines();

	// Check if the lines array exists
	if (!newLines)
		return false;

	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Initialize traversal of the input lines
	newLines->InitTraversal();

	// Loop through all input lines
	for (vtkIdType lineId = 0; lineId < newPD->GetNumberOfLines(); ++lineId)
	{
		// Get the number of point and the list of point IDs for the next ROI
		newLines->GetNextCell(numberOfPoints, pointList);

		// Create a new polygon
		vtkPolygon * newPolygon = vtkPolygon::New();

		// Decrease the number of points by one, since the last point is a 
		// duplicate of the first point.

		numberOfPoints--;

		// We need at least three points
		if (numberOfPoints < 3)
			continue;

		// Set the number of points of the polygon
		newPolygon->GetPointIds()->SetNumberOfIds(numberOfPoints);
		newPolygon->GetPoints()->SetNumberOfPoints(numberOfPoints);

		// Loop through all points in the input line
		for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Set ID of the new point
			newPolygon->GetPointIds()->SetId(ptId, ptId);

			// Get current point coordinates from the input
			double currentPoint[3];
			newPD->GetPoint(pointList[ptId], currentPoint);

			// Copy the point coordinates to the polygon
			newPolygon->GetPoints()->SetPoint(ptId, currentPoint);
		}

		// Add the polygon to the list of polygons for this ROI
		(*newROI).ROIPolygons.push_back(newPolygon);
	}

	// Done!
	return true;
}

void vtk2DROIFiberFilter::Execute()
{
	// Do nothing if no ROIs have been added
	if (this->ROIList.empty())
		return;

	// Get the input
	this->input = this->GetInput();

	// Check if the input exists
	if (!(this->input))
		return;

	// Get the lines array of the input
	vtkCellArray * inputLines = input->GetLines();

	// Check if the lines exist
	if (!inputLines)
		return;

	// Get the output of the filter
	this->output = this->GetOutput();

	// Check if the output exists
	if (!(this->output))
		return;

	// Create a new point array for the output
	this->outputPoints = vtkPoints::New();
	this->outputPoints->SetDataType(this->input->GetPoints()->GetDataType());
	this->output->SetPoints(this->outputPoints);
	this->outputPoints->Delete();

	// Create a new cell array for the output
	this->outputLines = vtkCellArray::New();
	this->output->SetLines(this->outputLines);
	this->outputLines->Delete();

	// Get the cell data arrays of the in- and output, and prepare them for copying.
	this->inCD  = this->input->GetCellData();
	this->outCD = this->output->GetCellData();
	this->inCD->CopyAllOn();
	this->outCD->CopyAllocate(this->inCD);

	// Get the point data arrays of the in- and output, and prepare them for copying.
	this->inPD  = this->input->GetPointData();
	this->outPD = this->output->GetPointData();
	this->inPD->CopyAllOn();
	this->outPD->CopyAllocate(this->inPD);

	// We cannot cut off the fibers at the first and last ROI encountered if the 
	// number of ROIs for which "bNOT" is false is less than two.

	if ((this->ROIList.size() - this->numberOfNOTs) < 2)
		this->CutFibersAtROI = false;

	// Create one boolean for each ROI
	bool * fiberCrossesROI = new bool[this->ROIList.size()];

	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Initialize traversal of the input fibers
	inputLines->InitTraversal();

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < input->GetNumberOfLines(); ++lineId)
	{
		// Get the number of point and the list of point IDs for the next ROI
		inputLines->GetNextCell(numberOfPoints, pointList);

		// We need at least two fiber points
		if (numberOfPoints < 2)
			continue;

		double p1[3];	// First fiber point
		double p2[3];	// Second fiber point
		double iX[3];	// Point of intersection with the ROI

		// Reset the point IDs of the first and last ROI intersection
		this->firstID = -1;
		this->lastID  = -1;

		// Iterator for the list of ROIs
		std::list<ROISettings>::iterator ROIIter;

		// Index for the "fiberCrossesROI" array
		int ROIID;

		// If "bNOT" is false, "fiberCrossesROI[ROIID]" is initialized to
		// false, and will be set to true when an intersection with that ROI
		// is encountered. If "bNOT" is true, "fiberCrossesROI[ROIID]" is set
		// to true, and will remain true, since we stop as soon as an inter-
		// section with a "NOT" ROI is encountered.

		for (ROIIter = this->ROIList.begin(), ROIID = 0; ROIIter != this->ROIList.end(); ++ROIIter, ++ROIID)
		{
			fiberCrossesROI[ROIID] = (*ROIIter).bNOT;
		}

		// At the start, the fiber is niether valid nor invalid
		bool fiberIsValid   = false;
		bool fiberIsInvalid = false;

		// In the special case that we only have "NOT" ROIs, all fibers start out
		// as valid, and are invalidated only when crosses a ROI.
		if (this->ROIList.size() == this->numberOfNOTs)
		{
			fiberIsValid = true;
		}

		// Get the first fiber point
		input->GetPoint(pointList[0], p1);

		// Loop through all remaining fiber points
		for (vtkIdType ptId = 1; ptId < numberOfPoints; ++ptId)
		{
			// Get the current fiber point
			input->GetPoint(pointList[ptId], p2);
		
			// Loop through all ROIs of the filter
			for (ROIIter = this->ROIList.begin(), ROIID = 0; ROIIter != this->ROIList.end(); ++ROIIter, ++ROIID)
			{
				// Check if the ROI contains polygons
				if ((*ROIIter).ROIPolygons.empty())
					continue;

				// Create an iterator for the list of polygons
				std::list<vtkPolygon *>::iterator PolygonIter;

				// Loop through all polygons associated with the current ROI. By default, 
				// each ROI contains only one polygon. If a ROI contains more than one 
				// polygon, the fiber only has to pass through one of these polygons.

				for (	PolygonIter = (*ROIIter).ROIPolygons.begin(); 
						PolygonIter != (*ROIIter).ROIPolygons.end(); 
						++PolygonIter)
				{
					// Get the current polygon
					this->currentROI = (*PolygonIter);

					// Check if the current line segment crosses the polygon. Also, we
					// only enter this if-statement if 1) we're dealing with a non-"NOT"
					// ROI that has not yet been crossed, or 2) we're dealing with a "NOT"
					// ROI. This essentially ignores repeated crossings of non-"NOT" ROIs,
					// which improves behaviour when "CutFibersAtROI" is true.

					if (this->lineIntersectsROI(p1, p2, iX) && 
							((fiberCrossesROI[ROIID] == false && (*ROIIter).bNOT == false)	|| 
							 (fiberCrossesROI[ROIID] == true  && (*ROIIter).bNOT == true)	))
					{
						// If the fiber intersected a "NOT" ROI, the fiber is invalid,
						// and we can immediately break from the for-loop.

						if ((*ROIIter).bNOT)
						{
							fiberIsInvalid = true;
							break;
						}

						// Set the corresponding boolean to true
						fiberCrossesROI[ROIID] = true;

						// If the first point of intersection has not yet been set,
						// store the information of the current point of intersection
						// in the "firstID" and "firstX" variables.

						if (this->firstID == -1)
						{
							this->firstID = ptId;
							this->firstX[0] = iX[0];
							this->firstX[1] = iX[1];
							this->firstX[2] = iX[2];
						}

						// Otherwise, store them in "lastID" and "lastX". These values will
						// be overwritten when another ROI is intersected, making sure that
						// the output fibers will run between the first and last ROI
						// encountered along the path of the fibers.

						else
						{
							this->lastID = ptId;
							this->lastX[0] = iX[0];
							this->lastX[1] = iX[1];
							this->lastX[2] = iX[2];
						}

						// Check if all booleans in "fiberCrossesROI" are true
						fiberIsValid = true;

						for (int i = 0; i < (int) this->ROIList.size(); ++i)
						{
							fiberIsValid &= fiberCrossesROI[i];
						}

						// Different polygons inside one ROI poly data object are OR'd, so we 
						// can break the "PolygonIter" for-loop as soon as we find one intersection.

						break;

					} // if [Line Intersection]

				} // for [Polygons]

				// If the fiber is invalid (because we intersected a "NOT" ROI), we immediately
				// break from the for-loop. We do the same when it is deemed valid and there are
				// no "NOT" ROIs (when there are "NOT" ROIs, we always need to fully check each fiber).
				
				if ((fiberIsValid && this->numberOfNOTs == 0) || fiberIsInvalid)
					break;

			} // for [ROIs]

			if ((fiberIsValid && this->numberOfNOTs == 0) || fiberIsInvalid)
				break;

			// Update "p1"

			p1[0] = p2[0];
			p1[1] = p2[1];
			p1[2] = p2[2];

		} // for [Fiber Points]

		// If the fiber is valid (it has intersected all non-"NOT" ROIs), and it
		// is not invalid (it has not intersected a "NOT" ROI), we add it to the output.

		if (fiberIsValid && !fiberIsInvalid)
		{
			this->writeFiberToOutput(numberOfPoints, pointList, lineId);
		}

	} // for [Fibers]

	// Remove the boolean array
	delete[] fiberCrossesROI;
}


//--------------------------[ lineIntersectsROI ]--------------------------\\

bool vtk2DROIFiberFilter::lineIntersectsROI(double p1[], double p2[], double * iX)
{
	// "T", "pCoords" and "subId" are needed to call "IntersectWithLine", 
	// but their values are not actually used.

	double T;
	double pCoords[3];
	int subId;

	// Point of intersection with the ROI
	double intersectX[3];

	// Check if the line segment between "p1" and "p2" intersects the ROI
	bool result = this->currentROI->IntersectWithLine(p1, p2, 0.0, T, intersectX, pCoords, subId);

	// Store the point of intersection if necessary
	if (this->CutFibersAtROI && iX && result)
	{
		iX[0] = intersectX[0];
		iX[1] = intersectX[1];
		iX[2] = intersectX[2];
	}

	return result;
}


//-------------------------[ :writeFiberToOutput ]-------------------------\\

void vtk2DROIFiberFilter::writeFiberToOutput(vtkIdType numberOfPoints, vtkIdType * pointList, vtkIdType lineID)
{
	// Current fiber point coordinates
	double X[3];

	// Create a list of the output point IDs
	vtkIdList * outList = vtkIdList::New();

	vtkIdType newPointID;	// ID of newly added point
	vtkIdType newLineID;	// ID of newly added fiber

	// Start and end of the fiber; by default, we use "0" and "numberOfPoints",
	// respectively, but when "CutFibersAtROI" is true, and the corresponding
	// first and last ID have been set, we use those IDs instead.

	vtkIdType maxPointId = (this->CutFibersAtROI && this->firstID != -1) ? (this->lastID)  : (numberOfPoints);
	vtkIdType minPointId = (this->CutFibersAtROI && this->lastID  != -1) ? (this->firstID) : (0);

	// If the fiber starts at "firstX", we first add this point to the output
	if (this->CutFibersAtROI && this->firstID != -1)
	{
		// Copy point of intersection to "X"
		X[0] = this->firstX[0];
		X[1] = this->firstX[1];
		X[2] = this->firstX[2];

		// Add the point of intersection to the output
		newPointID = this->outputPoints->InsertNextPoint(X);

		// Insert the new point ID into the list
		outList->InsertNextId(newPointID);

		// Copy the point data of the point "firstID" (i.e., the first fiber point
		// after the point of intersection) to the output point data of the
		// point of intersection.

		this->outPD->CopyData(this->inPD, pointList[this->firstID], newPointID);
	}

	// Loop through the fiber points
	for (vtkIdType ptId = minPointId; ptId < maxPointId; ptId++)
	{
		// Get the point coordinates from the input
		this->input->GetPoint(pointList[ptId], X);

		// Store the point in the output
		newPointID = this->outputPoints->InsertNextPoint(X);

		// Add the new point ID to the list
		outList->InsertNextId(newPointID);

		// Copy the point data from input to output
		this->outPD->CopyData(this->inPD, pointList[ptId], newPointID);
	}

	// If the fiber ends at "lastX", we now add this point to the output
	if (this->CutFibersAtROI && this->lastID != -1)
	{
		// Copy point of intersection to "X"
		X[0] = this->lastX[0];
		X[1] = this->lastX[1];
		X[2] = this->lastX[2];

		// Add the point of intersection to the output
		newPointID = this->outputPoints->InsertNextPoint(X);

		// Insert the new point ID into the list
		outList->InsertNextId(newPointID);

		// Copy the point data of the point "lastID - 1" (i.e., the last fiber point
		// before the point of intersection) to the output point data of the
		// point of intersection.

		this->outPD->CopyData(this->inPD, pointList[this->lastID - 1], newPointID);
	}

	// Create a new lines using the list of point IDs
	newLineID = this->outputLines->InsertNextCell(outList);

	// Copy the cell data of the input fiber to the output fiber
	this->outCD->CopyData(this->inCD, lineID, newLineID);

	// Delete the list of point IDs
	outList->Delete();
}


} // namespace bmia