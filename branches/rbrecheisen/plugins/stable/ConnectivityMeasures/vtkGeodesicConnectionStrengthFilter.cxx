/*
 * vtkGeodesicConnectionStrengthFilter.cxx
 *
 * 2011-05-12	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "vtkGeodesicConnectionStrengthFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkGeodesicConnectionStrengthFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkGeodesicConnectionStrengthFilter::vtkGeodesicConnectionStrengthFilter()
{
	// Set pointers to NULL
	this->auxImage		= NULL;
	this->auxScalars	= NULL;
	this->outScalars	= NULL;
	this->currentCell	= NULL;

	// For this Connectivity Measure, we need an image (DTI data)
	this->auxImageIsRequired = true;

	// Set default parameter values
	this->normalize			= true;
	this->currentPoint[0]	= 0.0;
	this->currentPoint[1]	= 0.0;
	this->currentPoint[2]	= 0.0;
	this->previousPoint[0]	= 0.0;
	this->previousPoint[1]	= 0.0;
	this->previousPoint[2]	= 0.0;
	this->den				= 0.0;
	this->nom				= 0.0;
	this->firstPointId		= 0;
	this->currentCellId		= -1;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkGeodesicConnectionStrengthFilter::~vtkGeodesicConnectionStrengthFilter()
{

}


//----------------------------[ initAuxImage ]-----------------------------\\

bool vtkGeodesicConnectionStrengthFilter::initAuxImage()
{
	// Make sure that the image has point data
	if (!(this->auxImage->GetPointData()))
	{
		vtkErrorMacro(<< "Input image does not contain point data.");
		return false;
	}

	// Try to get the "Tensors" array
	this->auxScalars = this->auxImage->GetPointData()->GetArray("Tensors");

	if (!(this->auxScalars))
	{
		vtkErrorMacro(<< "Input image does not contain a tensor array.");
		return false;
	}

	// The tensor array should have six components
	if (this->auxScalars->GetNumberOfComponents() != 6)
	{
		vtkErrorMacro(<< "Wrong number of components for input image scalars.");
		return false;
	}

	// Valid DTI image
	return true;	
}


//----------------------[ updateConnectivityMeasure ]----------------------\\

void vtkGeodesicConnectionStrengthFilter::updateConnectivityMeasure(vtkIdType fiberPointId, int pointNo)
{
	// Update the previous point
	this->previousPoint[0] = this->currentPoint[0];
	this->previousPoint[1] = this->currentPoint[1];
	this->previousPoint[2] = this->currentPoint[2];

	// Get the coordinates of the current point
	this->GetOutput()->GetPoint(fiberPointId, this->currentPoint);

	// If this is the first point in the fiber...
	if (pointNo == 0)
	{
		// ...reset the parameters...
		this->den = 0.0;
		this->nom = 0.0;
		this->currentCellId = -1;
		this->currentCell = NULL;

		// ...set the previous point to the current point...
		this->previousPoint[0] = this->currentPoint[0];
		this->previousPoint[1] = this->currentPoint[1];
		this->previousPoint[2] = this->currentPoint[2];

		// ...and store the ID of the first point
		this->firstPointId = fiberPointId;

		return;
	}

	// Compute the current step
	double step[3] = {	this->currentPoint[0] - this->previousPoint[0], 
						this->currentPoint[1] - this->previousPoint[1], 
						this->currentPoint[2] - this->previousPoint[2]	};

	// Add the step length to the nominator
	this->nom += vtkMath::Norm(step);

	// Tensor and inverse of tensor
	double  tensor[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double iTensor[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

	// Cell sub-ID, not used
	int subId;

	// Parametric coordinates, not used
	double pCoords[3];

	// Weights used for interpolation
	double weights[8];

	// Find the cell containing the current fiber point, starting from the current cell
	vtkIdType newCellId = this->auxImage->FindCell(this->currentPoint, this->currentCell, this->currentCellId, 
														0.00001, subId, pCoords, weights);

	// Return if the point is not within any cell
	if (newCellId < 0)
		return;

	// If the current cell has changed, store the new cell ID and pointer
	if (newCellId != this->currentCellId)
	{
		this->currentCellId = newCellId;
		this->currentCell = this->auxImage->GetCell(this->currentCellId);
	}

	// Interpolate the DTI tensor at the current position
	for (int i = 0; i < 8; ++i)
	{
		double tempTensor[6];
		this->auxScalars->GetTuple(this->currentCell->GetPointId(i), tempTensor);

		tensor[0] += weights[i] * tempTensor[0];
		tensor[1] += weights[i] * tempTensor[1];
		tensor[2] += weights[i] * tempTensor[2];
		tensor[3] += weights[i] * tempTensor[3];
		tensor[4] += weights[i] * tempTensor[4];
		tensor[5] += weights[i] * tempTensor[5];
	}

	// Inverse the interpolated tensor
	if (!(vtkTensorMath::Inverse(tensor, iTensor)))
		return;

	// Increment the denominator
	this->den += sqrt(	(step[0] * iTensor[0] + step[1] * iTensor[1] + step[2] * iTensor[2]) * step[0] + 
						(step[0] * iTensor[1] + step[1] * iTensor[3] + step[2] * iTensor[4]) * step[1] + 
						(step[0] * iTensor[2] + step[1] * iTensor[4] + step[2] * iTensor[5]) * step[2]	);

	// Store the scalar value
	this->outScalars->SetTuple1(fiberPointId, (this->den == 0) ? 0.0 : (this->nom / this->den));

	// If this was the second point, copy the value to the first point as well,
	// since it is not possible to compute a unique value for the first point.

	if (pointNo == 1)
	{
		this->outScalars->SetTuple1(this->firstPointId, (this->den == 0) ? 0.0 : (this->nom / this->den));
	}
}


} // namespace bmia
