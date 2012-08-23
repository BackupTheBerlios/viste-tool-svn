/*
 * geodesicFiberTracker.cxx
 *
 * 2011-06-01	Evert van Aart
 * - First Version.
 *
 */


/** Includes */

#include "geodesicFiberTracker.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

geodesicFiberTracker::geodesicFiberTracker()
{
	// Initialize pointers to NULL
	this->pp	= NULL;
	this->gfnh	= NULL;

	// Set default tracking parameters
	this->cellDiagonal	= 1.0;
	this->mySolver		= vtkFiberTrackingGeodesicFilter::OS_RK2_Heun;
}


//------------------------------[ Destructor ]-----------------------------\\

geodesicFiberTracker::~geodesicFiberTracker()
{
	// Delete the fiber neighborhood
	if (this->gfnh)
		delete this->gfnh;
}


//----------------------------[ calculateFiber ]---------------------------\\

void geodesicFiberTracker::calculateFiber(std::vector<streamlinePoint> * pointList)
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

	// Compute the diagonal of a cell using the image's spacing
	double spacing[3];
	this->dtiImageData->GetSpacing(spacing);
	this->cellDiagonal = sqrt(spacing[0] * spacing[0] + spacing[1] * spacing[1] + spacing[2] * spacing[2]);

	// Index of the current step
	int stepID = 0;

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
		this->step = this->stepSize * sqrt((double) currentCell->GetLength2());

		// Set the total distance to zero
		currentPoint.D = 0.0;

		// Create and setup the fiber neighborhood
		this->gfnh = new geodesicFiberNeighborhood;
		this->gfnh->setDTIImage(this->dtiImageData);
		this->gfnh->setPreProcessor(this->pp);
		this->gfnh->setScalarArray(this->aiImageData);
		this->gfnh->initializeNeighborhood(currentPoint.X);

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
			// Compute the next point of the fiber using an ODE solver
			if (!this->solveIntegrationStep(currentCell, currentCellId, weights))
				break;

			// Check if we've moved to a new cell
			vtkIdType newCellId = this->dtiImageData->FindCell(nextPoint.X, currentCell, currentCellId, 
																this->tolerance, subId, pCoords, weights);

			// If we're in a new cell, and we're still inside the volume...
			if (newCellId >= 0 && newCellId != currentCellId)
			{
				// Get the bounds of the previous cell
				double oldBounds[6];
				currentCell->GetBounds(oldBounds);

				// Store the new cell ID and get its pointer
				currentCellId = newCellId;
				currentCell = this->dtiImageData->GetCell(currentCellId);

				// Get the bounds of the new cell
				double newBounds[6];
				currentCell->GetBounds(newBounds);

				// Compute how many cells we've moved in each direction
				int dir[3];
				dir[0] = this->round((newBounds[0] - oldBounds[0]) / spacing[0]);
				dir[1] = this->round((newBounds[2] - oldBounds[2]) / spacing[1]);
				dir[2] = this->round((newBounds[4] - oldBounds[4]) / spacing[2]);

				// Try to move the neighborhood according to this direction vector.
				// If this fails - which is usually because we can only move one
				// cell in each direction - completely reset the neighborhood
				// around the coordinates of the next point.

				if (!this->gfnh->move(dir))
					this->gfnh->initializeNeighborhood(this->nextPoint.X);
			}

			// If we've left the volume, break here
			else if (newCellId == -1)
			{
				break;
			}

			// Interpolate the AI value at the current position
			if (currentCellId >= 0)
			{
				nextPoint.AI = this->gfnh->interpolateScalar(weights);
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

			// Update the current and previous points
			this->prevPoint = this->currentPoint;
			this->currentPoint = this->nextPoint;

			// Update the previous line segment
			this->prevSegment[0] = this->newSegment[0];
			this->prevSegment[1] = this->newSegment[1];
			this->prevSegment[2] = this->newSegment[2];

			// This is no longer the first step
			firstStep = false;

			// Increment the step index
			stepID++;

			// Every "MOBILITY_NUMBER_OF_STEPS" steps, we update the mobility of the
			// fiber, and check if it is larger than one. If it is smaller, it means 
			// that the fiber had a net displacement of less than one cell width in
			// the specified amount of cycles, which usually means that it has either
			// come to a full stop ("dX" is zero), or that it's going in circles.

			if ((stepID % geodesicFiberNeighborhood::MOBILITY_NUMBER_OF_STEPS) == 0)
			{
				double mobility = this->gfnh->computeMobility();

				// Break if the mobility is less than one cell width
				if (mobility < 1.0)
					break;
			}
		}
	}

	// Delete the weights
	delete [] weights;

	// Delete the fiber neighborhood
	delete this->gfnh;
	this->gfnh = NULL;
}


//--------------------------------[ round ]--------------------------------\\

int geodesicFiberTracker::round(double x)
{
	// Round to nearest integer
	return int(x > 0.0 ? x + 0.5 : x - 0.5);
}


//-------------------------[ solveIntegrationStep ]------------------------\\

bool geodesicFiberTracker::solveIntegrationStep(vtkCell * currentCell, vtkIdType currentCellId, double * weights)
{
	double CSymbols[18];

	// Compute the 18 unique Christoffel symbols
	this->gfnh->computeChristoffelSymbols(weights, CSymbols);

	// Use the selected ODE solver to compute next fiber position and direction
	switch (this->mySolver)
	{
		case vtkFiberTrackingGeodesicFilter::OS_Euler:
			this->solverEuler(this->currentPoint.dX, this->nextPoint.dX, this->currentPoint.X, this->nextPoint.X, CSymbols);
			break;

		case vtkFiberTrackingGeodesicFilter::OS_RK2_Heun:
			this->solverRK2Heun(this->currentPoint.dX, this->nextPoint.dX, this->currentPoint.X, this->nextPoint.X, CSymbols);
			break;

		case vtkFiberTrackingGeodesicFilter::OS_RK2_MidPoint:
			this->solverRK2Midpoint(this->currentPoint.dX, this->nextPoint.dX, this->currentPoint.X, this->nextPoint.X, CSymbols);
			break;

		case vtkFiberTrackingGeodesicFilter::OS_RK4:
			this->solverRK4(this->currentPoint.dX, this->nextPoint.dX, this->currentPoint.X, this->nextPoint.X, CSymbols);
			break;

		default:
			this->solverEuler(this->currentPoint.dX, this->nextPoint.dX, this->currentPoint.X, this->nextPoint.X, CSymbols);
			break;
	}

	// Store the current fiber segment
	this->newSegment[0] = this->nextPoint.X[0] - this->currentPoint.X[0];
	this->newSegment[1] = this->nextPoint.X[1] - this->currentPoint.X[1];
	this->newSegment[2] = this->nextPoint.X[2] - this->currentPoint.X[2];

	return true;
}


//-----------------------------[ solverEuler ]-----------------------------\\

void geodesicFiberTracker::solverEuler(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols)
{
	// Compute the next position
	nextPosition[0] = currentPosition[0] + this->step * currentDelta[0];
	nextPosition[1] = currentPosition[1] + this->step * currentDelta[1];
	nextPosition[2] = currentPosition[2] + this->step * currentDelta[2];

	// Compute the derivatives in the next point
	this->computeDelta(currentDelta, currentDelta, Csymbols, this->step, nextDelta);
}


//----------------------------[ solverRK2Heun ]----------------------------\\

void geodesicFiberTracker::solverRK2Heun(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols)
{
	double k2[3];	// Direction in next point
	double k3[3];	// Average direction

	// Compute derivatives at the next point (which is reached using a simple Euler step)
	this->computeDelta(currentDelta, currentDelta, Csymbols, this->step, k2);

	// Compute the average of the derivatives in the current point and the next point
	k3[0] = (k2[0] + currentDelta[0])/2;
	k3[1] = (k2[1] + currentDelta[1])/2;
	k3[2] = (k2[2] + currentDelta[2])/2;

	// Use average derivative to compute actual next point
	nextPosition[0] = currentPosition[0] + this->step * k3[0];
	nextPosition[1] = currentPosition[1] + this->step * k3[1];
	nextPosition[2] = currentPosition[2] + this->step * k3[2];

	// Compute new derivatives in next point
	this->computeDelta(currentDelta, k3, Csymbols, this->step, nextDelta);
}


//--------------------------[ solverRK2Midpoint ]--------------------------\\

void geodesicFiberTracker::solverRK2Midpoint(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols)
{
	double k2[3];	// Average direction

	// Compute derivatives at the mid-point between current point and point obtained through Euler step
	this->computeDelta(currentDelta, currentDelta, Csymbols, 0.5 * this->step, k2);

	// Use mid-point derivative to compute next point
	nextPosition[0] = currentPosition[0] + this->step * k2[0];
	nextPosition[1] = currentPosition[1] + this->step * k2[1];
	nextPosition[2] = currentPosition[2] + this->step * k2[2];

	// Compute new derivatives in next point
	this->computeDelta(currentDelta, k2, Csymbols, this->step, nextDelta);
}


//------------------------------[ solverRK4 ]------------------------------\\

void geodesicFiberTracker::solverRK4(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols)
{
	double k1[3];	// Intermediate direction
	double k2[3];	// Intermediate direction
	double k3[3];	// Intermediate direction
	double k4[3];	// Intermediate direction
	double kt[3];	// Average direction

	// Initialize k1
	k1[0] = currentDelta[0];
	k1[1] = currentDelta[1];
	k1[2] = currentDelta[2];

	// Compute k2, k3, and k4 for different values of slope and factor
	this->computeDelta(currentDelta, k1, Csymbols, 0.5 * this->step, k2);
	this->computeDelta(currentDelta, k2, Csymbols, 0.5 * this->step, k3);
	this->computeDelta(currentDelta, k3, Csymbols, this->step, k4);

	// Compute average direction
	kt[0] = (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0;
	kt[1] = (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0;
	kt[2] = (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0;

	// Use average direction to compute next point
	nextPosition[0] = currentPosition[0] + this->step * kt[0];
	nextPosition[1] = currentPosition[1] + this->step * kt[1];
	nextPosition[2] = currentPosition[2] + this->step * kt[2];

	// Compute direction in new point
	this->computeDelta(currentDelta, kt, Csymbols, this->step, nextDelta);
}


//-----------------------------[ computeDelta ]----------------------------\\

void geodesicFiberTracker::computeDelta(double * current, double * slope, double * Csymbols, double factor, double * o)
{
	// Compute new direction based on Neda's formulas
	o[0] = current[0] + factor * (	-(Csymbols[G111] * pow(slope[0], 2))
									-(Csymbols[G112] + Csymbols[G112]) * slope[0] * slope[1] 
									-(Csymbols[G122] * pow(slope[1], 2))
									-(Csymbols[G113] + Csymbols[G113]) * slope[0] * slope[2] 
									-(Csymbols[G123] + Csymbols[G123]) * slope[1] * slope[2] 
									-(Csymbols[G133] * pow(slope[2], 2)));

	o[1] = current[1] + factor * (	-(Csymbols[G211] * pow(slope[0], 2))
									-(Csymbols[G212] + Csymbols[G212]) * slope[0] * slope[1] 
									-(Csymbols[G222] * pow(slope[1], 2)) 
									-(Csymbols[G213] + Csymbols[G213]) * slope[0] * slope[2] 
									-(Csymbols[G223] + Csymbols[G223]) * slope[1] * slope[2] 
									-(Csymbols[G233] * pow(slope[2], 2)));

	o[2] = current[2] + factor * (	-(Csymbols[G311] * pow(slope[0], 2)) 
									-(Csymbols[G312] + Csymbols[G312]) * slope[0] * slope[1] 
									-(Csymbols[G322] * pow(slope[1], 2)) 
									-(Csymbols[G313] + Csymbols[G313]) * slope[0] * slope[2] 
									-(Csymbols[G323] + Csymbols[G323]) * slope[1] * slope[2] 
									-(Csymbols[G333] * pow(slope[2], 2)));

	// Normalize the new direction. This should essentially prevent huge steps 
	// from occurring without affecting the algorithm. If the pre-processing 
	// parameters are well selected, this should not occur in any case. Only 
	// do this if the distance traveled in a single step would be more than 
	// the diagonal of a single cell.

	if (this->step * sqrt(o[0] * o[0] + o[1] * o[1] + o[2] * o[2]) > this->cellDiagonal)
		vtkMath::Normalize(o);

	// In the statistically unlikely case that the newly computed direction is 
	// the exact inverse of the current direction, we may enter an infinite loop 
	// when one of the RK2 solvers is selected, since the average direction will 
	// be zero, and the current fiber position therefore will remain the same. 
	// To remedy this, we halve the new direction, which will at least keep the 
	// fiber moving.

	if (abs(o[0] + current[0]) < 0.001 && abs(o[1] + current[1]) < 0.001 && abs(o[2] + current[2]) < 0.001)
	{
		o[0] *= 0.5;
		o[1] *= 0.5;
		o[2] *= 0.5;
	}
}



} // namespace bmia
