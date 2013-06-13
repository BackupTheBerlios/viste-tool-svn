
/** Includes */

#include "vtkFiberSelectionFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkFiberSelectionFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkFiberSelectionFilter::vtkFiberSelectionFilter()
{
	// Set default options

}


//------------------------------[ Destructor ]-----------------------------\\

vtkFiberSelectionFilter::~vtkFiberSelectionFilter()
{

}


//-------------------------------[ Execute ]-------------------------------\\

void vtkFiberSelectionFilter::Execute()
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
	outputScalars->SetName(inputScalars->GetName());
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

	// Number of points in the current fiber, and a list of its point IDs
	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < inputLines->GetNumberOfCells(); ++lineId)
	{
        // Get the data of the current fiber
        vtkCell * currentCell = input->GetCell(lineId);
        int numberOfFiberPoints = currentCell->GetNumberOfPoints();

        // Get cell containing fiber
		inputLines->GetNextCell(numberOfPoints, pointList);

		// Evaluate if the fiber should be included in the output fibers
		bool excludeFiber = this->EvaluateFiber(currentCell,inputScalars);
		if(!excludeFiber)
            continue;

		// Create an ID list for the output fiber
		vtkIdList * newFiberList = vtkIdList::New();

		// Current scalar value
		double scalar;

		// Current point coordinates
		double p[3];

        // Loop through all points in the fiber
		for (int pointId = 0; pointId < numberOfFiberPoints; ++pointId)
		{
            // Get the point ID of the current fiber point
			vtkIdType currentPointId = currentCell->GetPointId(pointId);

			// Copy the point coordinates to the output
			inputPoints->GetPoint(currentPointId, p);
			vtkIdType newPointId = outputPoints->InsertNextPoint(p);
			newFiberList->InsertNextId(newPointId);

            // Get the scalar value
			scalar = inputScalars->GetTuple1(currentPointId);

			// Copy the scalar value to the output
			outputScalars->InsertNextTuple1(scalar);
		}

		// Add the new fiber to the output
		outputLines->InsertNextCell(newFiberList);
	}
}

bool vtkFiberSelectionFilter::EvaluateFiber(vtkCell* cell, vtkDataArray* inputScalars)
{
    int numberOfFiberPoints = cell->GetNumberOfPoints();

    // critera
    bool excludeFiber = false;
    double averageScore = 0.0;

    // Loop through all points in the fiber
    for (int pointId = 0; pointId < numberOfFiberPoints; ++pointId)
    {
        // Get the point ID of the current fiber point
        vtkIdType currentPointId = cell->GetPointId(pointId);

        // Get the scalar value
        double scalar = inputScalars->GetTuple1(currentPointId);

        // Average value of fiber
        averageScore += scalar;
    }

    // finish critera
    averageScore /= numberOfFiberPoints;

    // evaluate average score.
    // must be within selected range
    if(averageScore < averageScoreRange[0] || averageScore > averageScoreRange[1])
        excludeFiber = true;

    return excludeFiber;
}

} // namespace bmia
