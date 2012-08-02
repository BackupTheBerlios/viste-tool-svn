/*
 * streamlineTracker.cxx
 *
 * 2010-09-17	Evert van Aart
 * - First version. 
 *
 * 2010-10-05	Evert van Aart
 * - Replaced "std::list" by "std::vector", which increases performance.
 * - Additional performance optimizations.
 *
 * 2011-03-14	Evert van Aart
 * - Fixed a bug in which it was not always detected when a fiber moved to a new
 *   voxel. Because of this, the fiber tracking process kept using the data of the
 *   old cell, resulting in fibers that kept going in areas of low anisotropy.
 *
 * 2011-03-16	Evert van Aart
 * - Fixed a bug that could cause crashes if a fiber left the volume. 
 *
 */


/** Includes */

#include "streamlineTracker.h"


namespace bmia {



//-----------------------------[ Constructor ]-----------------------------\\

streamlineTracker::streamlineTracker()
{
	// Set pointers to NULL
	this->dtiImageData		= NULL;
	this->aiImageData		= NULL;
	this->dtiTensors		= NULL;
	this->aiScalars			= NULL;
	this->parentFilter		= NULL;
	this->cellDTITensors	= NULL;
	this->cellAIScalars		= NULL;

	// Set parameters to default values
	this->stepSize			= 0.1;
	this->tolerance			= 1.0;

	// Setup the pointers of the temporary tensor
	this->tempTensor[0] = tempTensor0;
	this->tempTensor[1] = tempTensor1;
	this->tempTensor[2] = tempTensor2;

	// Setup the eigenvector matrix
	this->eigenVectors[0] = eigenVectors0;
	this->eigenVectors[1] = eigenVectors1;
	this->eigenVectors[2] = eigenVectors2;
}


//-----------------------------[ Destructor ]------------------------------\\

streamlineTracker::~streamlineTracker()
{
	// Set pointers to NULL
	this->dtiImageData		= NULL;
	this->aiImageData		= NULL;
	this->dtiTensors		= NULL;
	this->aiScalars			= NULL;
	this->parentFilter		= NULL;

	// Delete the cell arrays
	this->cellDTITensors->Delete();
	this->cellAIScalars->Delete();

	// Set the array pointers to NULL
	this->cellDTITensors = NULL;
	this->cellAIScalars  = NULL;
}


//--------------------------[ initializeTracker ]--------------------------\\

void streamlineTracker::initializeTracker(	vtkImageData * rDTIImageData, 
											vtkImageData *  rAIImageData, 
											vtkDataArray * rDTITensors, 
											vtkDataArray *  rAIScalars, 
											vtkFiberTrackingFilter * rParentFilter, 
											double rStepSize,
											double rTolerance							)
{
	// Store input values
	this->dtiImageData	= rDTIImageData;
	this->aiImageData	=  rAIImageData;
	this->dtiTensors	= rDTITensors;
	this->aiScalars		=  rAIScalars;	
	this->parentFilter	= rParentFilter;
	this->stepSize		= rStepSize;
	this->tolerance		= rTolerance;

	// Create the cell arrays
	this->cellDTITensors = vtkDataArray::CreateDataArray(this->dtiTensors->GetDataType());
	this->cellAIScalars  = vtkDataArray::CreateDataArray(this->aiScalars->GetDataType());

	// Set number of components and tuples of the cell arrays
	this->cellDTITensors->SetNumberOfComponents(6);
	this->cellDTITensors->SetNumberOfTuples(8);

	this->cellAIScalars->SetNumberOfComponents(this->aiScalars->GetNumberOfComponents());
	this->cellAIScalars->SetNumberOfTuples(8);
}


//----------------------------[ calculateFiber ]---------------------------\\

void streamlineTracker::calculateFiber(int direction, std::vector<streamlinePoint> * pointList)
{
	vtkCell *	currentCell			= NULL;						// Cell of current point
	vtkIdType	currentCellId		= 0;						// Id of current cell
	double		closestPoint[3]		= {0.0, 0.0, 0.0};			// Used in "EvaluatePosition"
	double		pointDistance		= 0.0;						// Used in "EvaluatePosition"
	double		stepDistance		= 0.0;						// Length of current step
	int			subId				= 0;						// Used in "FindCell"
	double		pCoords[3]			= {0.0, 0.0, 0.0};			// Used in "FindCell"
	double		testDot				= 1.0;						// Dot product for current step
	bool		firstStep			= true;						// True during first integration step

	// Interpolation weights
	double *	weights = new double[8];

	// Initialize interpolation weights
	for (int i = 0; i < 8; ++i)
	{
		weights[i] = 0.0;
	}

	// Check if there's a point in the point list
	if (!pointList->empty())
	{
		// Get the first point, and clear the list
		currentPoint = pointList->front();
		pointList->clear();

		// Find the cell containing the seed point
		currentCellId = this->dtiImageData->FindCell(currentPoint.X, NULL, 0, this->tolerance, subId, pCoords, weights);
		currentCell = this->dtiImageData->GetCell(currentCellId);

		// Set the actual step size, depending on the voxel size
		this->step = direction * this->stepSize * sqrt((double) currentCell->GetLength2());

		// Load the tensors and AI values of the cell into the "cellDTITensors" and
		// "cellAIScalars" arrays, respectively
		this->dtiTensors->GetTuples(currentCell->PointIds, this->cellDTITensors);
		this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );

		// Interpolate the DTI tensors at the seed point position
		this->interpolateTensor(currentTensor, weights, this->cellDTITensors);

		// Compute eigenvector from interpolated tensor
		this->getEigenvectors(currentTensor, &currentPoint);

		// Set the total distance to zero
		currentPoint.D = 0.0;

		// Interpolate the AI scalar at the seed point position
		this->interpolateScalar(&(currentPoint.AI), weights);
	
		// Re-add the seed point (which now contains eigenvectors and AI)
		pointList->push_back(currentPoint);

		// Set the previous point equal to the current point
		prevPoint = currentPoint;

		// Initialize the previous segment to zero
		this->prevSegment[0] = 0.0;
		this->prevSegment[1] = 0.0;
		this->prevSegment[2] = 0.0;

		// Loop until a stopping condition is met
		while (1) 
		{
			// Compute the next point of the fiber using a second-order RK-step.
			if (!this->solveIntegrationStep(currentCell, currentCellId, weights))
				break;

			// Check if we've moved to a new cell
			vtkIdType newCellId = this->dtiImageData->FindCell(nextPoint.X, currentCell, currentCellId, 
															this->tolerance, subId, pCoords, weights);

			// If we're in a new cell, and we're still inside the volume...
			if (newCellId >= 0 && newCellId != currentCellId)
			{
				// ...store the ID of the new cell...
				currentCellId = newCellId;

				// ...set the new cell pointer...
				currentCell = this->dtiImageData->GetCell(currentCellId);

				// ...and fill the cell arrays with the data of the new cell
				this->dtiTensors->GetTuples(currentCell->PointIds, this->cellDTITensors);
				this->aiScalars->GetTuples( currentCell->PointIds, this->cellAIScalars );
			}
			// If we've left the volume, break here
			else if (newCellId == -1)
			{
				break;
			}

			// Compute interpolated tensor at new position
			this->interpolateTensor(currentTensor, weights, this->cellDTITensors);

			// Compute eigenvectors from interpolated tensor
			this->getEigenvectors(currentTensor, &(this->nextPoint));

			// Align the main eigenvector of the new point with the previous segment (not with
			// the MEV of the previous point, since the fact that we're using RK2 means that 
			// the actual step direction does not equal the MEV).

			if (vtkMath::Dot(this->newSegment, this->nextPoint.V0) < 0.0)
			{
				nextPoint.V0[0] *= -1.0;
				nextPoint.V0[1] *= -1.0;
				nextPoint.V0[2] *= -1.0;
			}

			// Interpolate the AI value at the current position
			if (currentCellId >= 0)
			{
				this->interpolateScalar(&(nextPoint.AI), weights);
			}

			// If this wasn't the first step, compute the dot product between the 
			// last two fiber segments.
	
			if (!firstStep)
			{
				// Compute the dot product
				testDot = vtkMath::Dot(this->prevSegment, this->newSegment);
			}
			// If this was the first step, set the default value
			else
			{
				testDot = 1.0;
			}

			// Update the total fiber length
			stepDistance = sqrt((double) vtkMath::Distance2BetweenPoints(currentPoint.X, nextPoint.X));
			this->nextPoint.D = this->currentPoint.D + stepDistance;

			// Call "continueTracking" function of parent filter to determine if
			// one of the stopping criteria has been met.

			if (!(this->parentFilter->continueTracking(&(this->nextPoint), testDot, currentCellId)))
			{
				// If so, stop tracking.
				break;
			}

			// Add the new point to the point list
			pointList->push_back(this->nextPoint);

			// If necessary, increase size of the point list
			if (pointList->size() == pointList->capacity())
			{
				pointList->reserve(pointList->size() + 1000);
			}

			// Update the current and previous points
			this->prevPoint = this->currentPoint;
			this->currentPoint = this->nextPoint;

			// Update the previous line segment
			this->prevSegment[0] = this->newSegment[0];
			this->prevSegment[1] = this->newSegment[1];
			this->prevSegment[2] = this->newSegment[2];

			// This is no longer the first step
			firstStep = false;
		}
	}

	delete [] weights;
}


//-------------------------[ solveIntegrationStep ]------------------------\\

bool streamlineTracker::solveIntegrationStep(vtkCell * currentCell, vtkIdType currentCellId, double * weights)
{
	streamlinePoint intermediatePoint;		// Intermediate fiber point, used in RK2 solver
	vtkIdType		intermediateCellId;		// Id of the cell containing the intermediate position
	int				subId;					// Used in "FindCell"
	double			pCoords[3];				// Used in "FindCell"

	// Compute intermediate position using an Euler step
	intermediatePoint.X[0] = currentPoint.X[0] + this->step * currentPoint.V0[0];
	intermediatePoint.X[1] = currentPoint.X[1] + this->step * currentPoint.V0[1];
	intermediatePoint.X[2] = currentPoint.X[2] + this->step * currentPoint.V0[2];

	// Get the Id of the cell containing the intermediate position
	intermediateCellId = dtiImageData->FindCell(intermediatePoint.X, currentCell, currentCellId, 
													this->tolerance, subId, pCoords, weights);

	if (intermediateCellId == -1)
		return false;

	// If the intermediate position is in the same cell as the current position...
	if (intermediateCellId == currentCellId)
	{
		// ...compute the interpolated tensor using the existing cell data array...
		this->interpolateTensor(currentTensor, weights, this->cellDTITensors);
	}
	// ...if not, create a new data array for the new cell
	else
	{
		// Create a new array with the same type and number of components as the DTI image
		vtkDataArray * intermediateDTITensors = vtkDataArray::CreateDataArray(dtiTensors->GetDataType());
		intermediateDTITensors->SetNumberOfComponents(6);
		intermediateDTITensors->SetNumberOfTuples(8);

		// Get pointer to the new cell
		vtkCell * intermediateCell = dtiImageData->GetCell(intermediateCellId);

		// Load cell data into the new array
		dtiTensors->GetTuples(intermediateCell->PointIds, intermediateDTITensors);

		// Interpolate tensor using the new cell data array
		this->interpolateTensor(currentTensor, weights, intermediateDTITensors);

		// Delete the intermediate array
		intermediateDTITensors->Delete();
	}

	// Get eigenvectors of the intermediate point
	this->getEigenvectors(currentTensor, &intermediatePoint);

	// Flip eigenvectors if needed
	this->fixVectors(&currentPoint, &intermediatePoint);

	// Compute the new segment of the fiber
	this->newSegment[0] = (currentPoint.V0[0] + intermediatePoint.V0[0]) * (this->step / 2.0);
	this->newSegment[1] = (currentPoint.V0[1] + intermediatePoint.V0[1]) * (this->step / 2.0);
	this->newSegment[2] = (currentPoint.V0[2] + intermediatePoint.V0[2]) * (this->step / 2.0);

	// Align the new segment with the previous segment
	if (vtkMath::Dot(this->prevSegment, this->newSegment) < 0.0)
	{
		this->newSegment[0] *= -1.0;
		this->newSegment[1] *= -1.0;
		this->newSegment[2] *= -1.0;
	}

	// Compute the next point

	this->nextPoint.X[0] = this->currentPoint.X[0] + this->newSegment[0];
	this->nextPoint.X[1] = this->currentPoint.X[1] + this->newSegment[1];
	this->nextPoint.X[2] = this->currentPoint.X[2] + this->newSegment[2];

	// Normalize the new line segment
	vtkMath::Normalize(this->newSegment);

	return true;
}


//-------------------------[ getEigenvectors ]--------------------------\\

void streamlineTracker::getEigenvectors(double * iTensor, streamlinePoint * point)
{
	// Setup the copy of the input tensor
	this->tempTensor[0][0] = iTensor[0];
	this->tempTensor[0][1] = iTensor[1];
	this->tempTensor[0][2] = iTensor[2];
	this->tempTensor[1][0] = iTensor[1];
	this->tempTensor[1][1] = iTensor[3];
	this->tempTensor[1][2] = iTensor[4];
	this->tempTensor[2][0] = iTensor[2];
	this->tempTensor[2][1] = iTensor[4];
	this->tempTensor[2][2] = iTensor[5];

	// Use the Jacobi function to get the eigenvectors
	vtkMath::Jacobi(this->tempTensor, this->eigenValues, this->eigenVectors);

	// Copy eigenvectors to output
	point->V0[0] = this->eigenVectors[0][0];
	point->V0[1] = this->eigenVectors[1][0];
	point->V0[2] = this->eigenVectors[2][0];

	point->V1[0] = this->eigenVectors[0][1];
	point->V1[1] = this->eigenVectors[1][1];
	point->V1[2] = this->eigenVectors[2][1];

	point->V2[0] = this->eigenVectors[0][2];
	point->V2[1] = this->eigenVectors[1][2];
	point->V2[2] = this->eigenVectors[2][2];
}


//------------------------------[ fixVectors ]-----------------------------\\

void streamlineTracker::fixVectors(streamlinePoint * oldPoint, streamlinePoint * newPoint)
{
	// Main eigenvector
	if (vtkMath::Dot(oldPoint->V0, newPoint->V0) < 0.0)
	{
		newPoint->V0[0] *= -1.0;
		newPoint->V0[1] *= -1.0;
		newPoint->V0[2] *= -1.0;
	}

	// Second eigenvector
	if (vtkMath::Dot(oldPoint->V1, newPoint->V1) < 0.0)
	{
		newPoint->V1[0] *= -1.0;
		newPoint->V1[1] *= -1.0;
		newPoint->V1[2] *= -1.0;
	}

	// Third eigenvector
	if (vtkMath::Dot(oldPoint->V2, newPoint->V2) < 0.0)
	{
		newPoint->V2[0] *= -1.0;
		newPoint->V2[1] *= -1.0;
		newPoint->V2[2] *= -1.0;
	}
}

//--------------------------[ interpolateTensor ]--------------------------\\

void streamlineTracker::interpolateTensor(double * interpolatedTensor, double * weights, vtkDataArray * currentCellDTITensors)
{
	// Set the output tensor to zero
	for (int j = 0; j < 6; ++j)
	{
		interpolatedTensor[j] = 0.0;
	}

	// For all eight surrounding voxels...
	for (int i = 0; i < 8; ++i)
	{
		// ...get the corresponding tensor...
		currentCellDTITensors->GetTuple(i, this->auxTensor);

		// ...and add it to the interpolated tensor
		for (int j = 0; j < 6; ++j)
		{
			interpolatedTensor[j] += weights[i] * this->auxTensor[j];
		}
	}
}


//--------------------------[ interpolateScalar ]--------------------------\\

void streamlineTracker::interpolateScalar(double * interpolatedScalar, double * weights)
{
	// Set the output to zero
	(*interpolatedScalar) = 0.0;

	// For all eight surrounding voxels...
	for (int i = 0; i < 8; ++i)
	{
		// ...get the corresponding scalar...
		double tempScalar = this->cellAIScalars->GetTuple1(i);;

		// ...and add it to the interpolated scalar
		(*interpolatedScalar) += weights[i] * tempScalar;
	}
}

} // namespace bmia