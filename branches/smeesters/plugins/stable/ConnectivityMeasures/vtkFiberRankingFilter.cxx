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
 * vtkFiberRankingFilter.cxx
 *
 * 2011-05-13	Evert van Aart
 * - First version.
 *
 * 2011-08-22	Evert van Aart
 * - Fixed a bug caused by incorrect iteration through the fiber map.
 * - Added a progress bar.
 *
 */


/** Includes */

#include "vtkFiberRankingFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkFiberRankingFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkFiberRankingFilter::vtkFiberRankingFilter()
{
	// Set default options
	this->measure			= ConnectivityMeasuresPlugin::RM_FiberEnd;
	this->outputMethod		= ConnectivityMeasuresPlugin::RO_BestPercentage;
	this->numberOfFibers	= 1;
	this->percentage		= 10;
	this->useSingleValue	= true;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkFiberRankingFilter::~vtkFiberRankingFilter()
{

}


//-------------------------------[ Execute ]-------------------------------\\

void vtkFiberRankingFilter::Execute()
{
	// Get the input
	vtkPolyData * input = this->GetInput();

	if(!input)
	{
		vtkErrorMacro(<< "Input has not been set.");
		return;
	}

	// Check if the input contains point data
	vtkPointData * inputPD = input->GetPointData();

	if (!inputPD)
	{
		vtkErrorMacro(<< "Input does not have point data.");
		return;
	}

	// Check if the input contains scalars
	vtkDataArray * inputScalars = inputPD->GetScalars();

	if (!inputScalars)
	{
		vtkErrorMacro(<< "Input does not have a scalar array.");
		return;
	}

	// Get the points of the input
	vtkPoints * inputPoints = input->GetPoints();

	if (!inputPoints)
	{
		vtkErrorMacro(<< "Input does not have points.");
		return;
	}

	// Get the lines array of the input
	vtkCellArray * inputLines = input->GetLines();

	if (!inputLines)
	{
		vtkErrorMacro(<< "Input does not have lines.");
		return;
	}

	// Get the output
	vtkPolyData * output = this->GetOutput();

	if (!output)
	{
		vtkErrorMacro(<< "Output has not been set.");
		return;
	}

	// Check if the output contains point data
	vtkPointData * outputPD = output->GetPointData();

	if (!outputPD)
	{
		vtkErrorMacro(<< "Output does not have point data.");
		return;
	}

	// Create a scalar array for the output scalar values
	vtkDataArray * outputScalars = vtkDataArray::CreateDataArray(inputScalars->GetDataType());
	outputScalars->SetNumberOfComponents(1);
	outputPD->SetScalars(outputScalars);
	outputScalars->Delete();

	// Create a point set for the output
	vtkPoints * outputPoints = vtkPoints::New();
	output->SetPoints(outputPoints);
	outputPoints->Delete();

	// Create a line array for the output
	vtkCellArray * outputLines = vtkCellArray::New();
	output->SetLines(outputLines);
	outputLines->Delete();

	// Map used to store fiber indices (value) and their connectivity measure values (key)
	QMap<double, vtkIdType> fiberMap;

	// Initialize traversal of the input fibers
	inputLines->InitTraversal();

	// Number of points in the current fiber, and a list of its point IDs
	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < inputLines->GetNumberOfCells(); ++lineId)
	{
		// Get the data of the current fiber
		inputLines->GetNextCell(numberOfPoints, pointList);

		double currentMeasure = 0.0;

		// Use the CM value of the last point
		if (this->measure == ConnectivityMeasuresPlugin::RM_FiberEnd)
		{
			vtkIdType lastPointId = pointList[numberOfPoints - 1];
			currentMeasure = inputScalars->GetTuple1(lastPointId);
		}
		// Compute the average CM value
		else if (this->measure == ConnectivityMeasuresPlugin::RM_Average)
		{
			for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
			{
				currentMeasure += inputScalars->GetTuple1(pointList[pointId]) / (double) numberOfPoints;
			}
		}

		// Add the fiber ID and the CM value to the map
		fiberMap.insert(currentMeasure, lineId);
	}

	int numberOfOutputFibers;

	// Use the fixed number of fibers...
	if (this->outputMethod == ConnectivityMeasuresPlugin::RO_BestNumber)
	{
		numberOfOutputFibers = this->numberOfFibers;
	}

	// ...or compute it as a percentage of the input fibers...
	else if (this->outputMethod == ConnectivityMeasuresPlugin::RO_BestPercentage)
	{
		numberOfOutputFibers = (int) (((double) this->percentage / 100.0) * (double) inputLines->GetNumberOfCells());
	}

	// ...or simply use all input fibers.
	else if (this->outputMethod == ConnectivityMeasuresPlugin::RO_AllFibers)
	{
		numberOfOutputFibers = inputLines->GetNumberOfCells();
	}

	// Make sure there's at least one output fiber
	if (numberOfOutputFibers <= 0)
		numberOfOutputFibers = 1;

	// We cannot have more output fibers than input fibers
	if (numberOfOutputFibers > inputLines->GetNumberOfCells())
		numberOfOutputFibers = inputLines->GetNumberOfCells();

	int rFiberIndex = 0;
	QMap<double, vtkIdType>::iterator rIter = fiberMap.end();

	// Setup the progress bar
	int progressStep = numberOfOutputFibers / 25;
	progressStep += (progressStep == 0) ? 1 : 0;
	this->SetProgressText("Ranking fibers based on Connectivity Measure values...");
	this->UpdateProgress(0.0);

	// Loop through the last "numberOfOutputFibers" elements of the map. Since the
	// map stores its entries in ascending order for the keys (which are the CM
	// values in our case), the last items represent the strongest fibers (highest CM).

	while (rIter != fiberMap.begin())
	{
		// Decrement the iterator
		rIter--;

		// Update the progress bar
		if ((rFiberIndex % progressStep) == 0)
		{
			this->UpdateProgress((double) rFiberIndex / (double) numberOfOutputFibers);
		}

		// Get the ID of the current fiber
		vtkIdType currentFiberId = rIter.value();

		// Get the cell representing the fiber, and the number of points in this fiber
		vtkCell * currentCell = input->GetCell(currentFiberId);
		int numberOfFiberPoints = currentCell->GetNumberOfPoints();

		// Create an ID list for the output fiber
		vtkIdList * newFiberList = vtkIdList::New();

		// Current point coordinates
		double p[3];

		// Current scalar value (CM value)
		double scalar;

		// Loop through all points in the fiber
		for (int pointId = 0; pointId < numberOfFiberPoints; ++pointId)
		{
			// Get the point ID of the current fiber point
			vtkIdType currentPointId = currentCell->GetPointId(pointId);

			// Copy the point coordinates to the output
			inputPoints->GetPoint(currentPointId, p);
			vtkIdType newPointId = outputPoints->InsertNextPoint(p);
			newFiberList->InsertNextId(newPointId);

			if (this->useSingleValue)
			{
				// Use the key of the fiber (ranking measure) as the scalar value...
				scalar = rIter.key();
			}
			else
			{
				// ...or use the input scalar value
				scalar = inputScalars->GetTuple1(currentPointId);
			}

			// Copy the scalar value to the output
			outputScalars->InsertNextTuple1(scalar);
		}

		// Add the new fiber to the output
		outputLines->InsertNextCell(newFiberList);
		newFiberList->Delete();

		// Break if we've reached the desired number of fibers
		if (++rFiberIndex == numberOfOutputFibers)
			break;

	} // for [Every output fiber]

	// Finalize the progress bar
	this->UpdateProgress(1.0);
}


} // namespace bmia
