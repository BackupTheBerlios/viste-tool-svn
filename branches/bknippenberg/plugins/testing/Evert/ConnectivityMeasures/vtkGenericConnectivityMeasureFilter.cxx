/*
 * vtkGenericConnectivityMeasureFilter.cxx
 *
 * 2011-05-12	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "vtkGenericConnectivityMeasureFilter.h"


namespace bmia
{


vtkStandardNewMacro(vtkGenericConnectivityMeasureFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkGenericConnectivityMeasureFilter::vtkGenericConnectivityMeasureFilter()
{
	// Initialize pointers to NULL
	this->auxImage		= NULL;
	this->auxScalars	= NULL;
	this->outScalars	= NULL;

	// By default, we do not need an auxiliary image, but this should be
	// set by all subclasses of this class.

	this->auxImageIsRequired = false;

	// Set default parameters
	this->normalize			= true;
	this->currentPoint[0]	= 0.0;
	this->currentPoint[1]	= 0.0;
	this->currentPoint[2]	= 0.0;
	this->previousPoint[0]	= 0.0;
	this->previousPoint[1]	= 0.0;
	this->previousPoint[2]	= 0.0;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkGenericConnectivityMeasureFilter::~vtkGenericConnectivityMeasureFilter()
{

}


//-------------------------------[ Execute ]-------------------------------\\

void vtkGenericConnectivityMeasureFilter::Execute()
{
	// If an auxiliary image is required, check if it has been set
	if (this->auxImageIsRequired && this->auxImage == NULL)
	{
		vtkErrorMacro(<< "Input image has not been set, but is not optional.");
		return;
	}

	// If so, initialize this image
	if (this->auxImageIsRequired && this->auxImage)
	{
		if (!(this->initAuxImage()))
		{
			return;
		}
	}

	// Get the input fibers
	vtkPolyData * input = this->GetInput();

	if (!input)
	{
		vtkErrorMacro(<< "Input fibers have not been set.");
		return;
	}

	// Get the input points
	vtkPoints * inputPoints = input->GetPoints();

	if (!inputPoints)
	{
		vtkErrorMacro(<< "Input fibers do not contain points.");
		return;
	}

	// Get the output fibers
	vtkPolyData * output = this->GetOutput();

	if (!output)
	{
		vtkErrorMacro(<< "Output fibers have not been set.");
		return;
	}

	// Get the output point data
	vtkPointData * outputPD = output->GetPointData();

	if (!outputPD)
	{
		vtkErrorMacro(<< "Output does not contain point data.");
		return;
	}

	// Copy the entire input the output
	output->DeepCopy(input);

	// Create a new output scalar array
	this->outScalars = vtkDoubleArray::New();
	this->outScalars->SetNumberOfComponents(1);
	this->outScalars->SetNumberOfTuples(inputPoints->GetNumberOfPoints());

	// Clear the array to 0.0.
	for (vtkIdType ptId = 0; ptId < this->outScalars->GetNumberOfTuples(); ++ptId)
	{
		this->outScalars->SetTuple1(ptId, 0.0);
	}

	// Add the scalar array to the output
	outputPD->SetScalars(this->outScalars);
	this->outScalars->Delete();

	// Get the fiber array
	vtkCellArray * fibers = output->GetLines();
	vtkIdType numberOfFibers = fibers->GetNumberOfCells();

	// Initialize traversal of the fibers
	fibers->InitTraversal();

	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Loop through all fibers
	for (vtkIdType fiberId = 0; fiberId < numberOfFibers; ++fiberId)
	{
		// Get the number of points and the list of point IDs of the current fiber
		fibers->GetNextCell(numberOfPoints, pointList);

		// Do nothing if the fiber is empty
		if (numberOfPoints == 0)
			continue;

		// Loop through all points of the current fiber
		for (vtkIdType pointId = 0; pointId < numberOfPoints; ++pointId)
		{
			// Compute the scalar measure for the current point
			this->updateConnectivityMeasure(pointList[pointId], pointId);
		}
	}

	// If we should normalize the measure value, do so now
	if (this->normalize)
	{
		double range[2];

		this->outScalars->GetRange(range);

		if (range[1] > range[0])
		{
			for (vtkIdType ptId = 0; ptId < this->outScalars->GetNumberOfTuples(); ++ptId)
			{
				// Get the current value
				double scalarValue = this->outScalars->GetTuple1(ptId);

				// Normalize the value to the range 0-1
				scalarValue = (scalarValue - range[0]) / (range[1] - range[0]);

				// Put the value back in the array
				this->outScalars->SetTuple1(ptId, scalarValue);
			}
		}
	}
}


//----------------------[ updateConnectivityMeasure ]----------------------\\

void vtkGenericConnectivityMeasureFilter::updateConnectivityMeasure(vtkIdType fiberPointId, int pointNo)
{
	// Empty, implemented in subclasses
}


//-----------------------------[ initAuxImage ]----------------------------\\

bool vtkGenericConnectivityMeasureFilter::initAuxImage()
{
	// Check if we've got point data
	if (!(this->auxImage->GetPointData()))
	{
		vtkErrorMacro(<< "Input image does not contain point data.");
		return false;
	}

	// Get the default scalars array of the point data
	this->auxScalars = this->auxImage->GetPointData()->GetScalars();

	if (!(this->auxScalars))
	{
		vtkErrorMacro(<< "Input image does not contain a scalar array.");
		return false;
	}

	return true;
}


} // namespace bmia
