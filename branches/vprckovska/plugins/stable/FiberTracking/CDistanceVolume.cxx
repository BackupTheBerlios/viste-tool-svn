/*
 * CDistanceVolume.cxx
 *
 * 2010-09-21	Evert van Aart
 * - First version. 
 *
 * 2010-10-04	Evert van Aart
 * - Fixed an error that prevented the class from working correctly.
 *
 * 2010-11-10	Evert van Aart
 * - Fixed behaviour near borders. In older versions, a global 1D "index" was
 *   computed from the 3D element coordinates ("dsepCoord"), and we only checked
 *   if "index" was between zero and the total number of indices. If one of the
 *   3D coordinates exceeded the distance volume dimension, the 1D index could 
 *   still be within global range, and the functions would select a distance
 *   element in a completely different part of the volume.
 *
 * 2011-04-21	Evert van Aart
 * - "exactGoodDistance" now immediately returns true if the distance threshold is zero.
 *
 */


/** Includes */

#include "CDistanceVolume.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

CDistanceVolume::CDistanceVolume()
{
	// Set distance volume pointer to NULL
	this->distanceVolume = NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

CDistanceVolume::~CDistanceVolume()
{
	// Delete the distance volume
	if (this->distanceVolume)
	{
		this->clearVolume();
		free(distanceVolume);
	}
}


//-----------------------------[ clearVolume ]-----------------------------\\

void CDistanceVolume::clearVolume()
{
	// Check if the distance volume exists
	if (this->distanceVolume)
	{
		// Loop through all elements in the distance volume
		for (int iClear = 0; iClear < this->dsepSize; iClear++)
		{
			// Pointer to the nest list item
			CDistanceVolume::pointListItem * next;

			// Get first list item of the current distance element
			CDistanceVolume::pointListItem * element = this->distanceVolume[iClear].listHead;

			// While we haven't reached the end of the list...
			while(element)
			{
				// ...get the next element...
				next = element->Next;

				// ...delete the current one...
				delete element;

				// ...and update the "element" pointer
				element = next;
			}

			// Reset the number of items and the first item pointer of the element
			this->distanceVolume[iClear].numberOfItems = 0;
			this->distanceVolume[iClear].listHead      = NULL;

		}
	}
}


//---------------------------[ initializeVolume ]--------------------------\\

void CDistanceVolume::initializeVolume(float rElementDistance, double volumeSize[3])
{
	// Store the element distance
	this->elementDistance = rElementDistance;

	// Compute the number of elements in each direction
	int newDsepDim[3];
	newDsepDim[0] = (int) ceilf(((float) volumeSize[0]) / this->elementDistance);
	newDsepDim[1] = (int) ceilf(((float) volumeSize[1]) / this->elementDistance);
	newDsepDim[2] = (int) ceilf(((float) volumeSize[2]) / this->elementDistance);

	// If a distance volume with the same dimensions as the new one already
	// exists, simply clear the old volume.
	if (	(distanceVolume != NULL) && 
			(newDsepDim[0] == this->dsepDim[0]) &&
			(newDsepDim[1] == this->dsepDim[1]) && 
			(newDsepDim[2] == this->dsepDim[2])		)
	{
		clearVolume();
	}
	else 
	{
		// If a volume (with different dimensions that the new one) exists,
		// we first clear and free it.
		if (this->distanceVolume != NULL)
		{
			clearVolume();
			free(this->distanceVolume);
		}

		// Store volume dimensions
		this->dsepDim[0] = newDsepDim[0];
		this->dsepDim[1] = newDsepDim[1];
		this->dsepDim[2] = newDsepDim[2];

		// Compute the number of elements in the volume
		this->dsepSize = this->dsepDim[0] * this->dsepDim[1] * this->dsepDim[2];

		// Compute the number of elements in one XY slice
		this->XYdsep = this->dsepDim[0] * this->dsepDim[1];

		// Allocate a new distance volume
		this->distanceVolume = (CDistanceVolume::DistanceElement *) 
									malloc(sizeof(CDistanceVolume::DistanceElement) * this->dsepSize);

		// Reset the amount of filled cells
		this->filledCellsCount = 0;

		// For each element in the new volume, set the number of list items
		// to zero, and set the list head pointer to NULL.

		for (int iClear = 0; iClear < this->dsepSize; iClear++)
		{
			this->distanceVolume[iClear].numberOfItems = 0;
			this->distanceVolume[iClear].listHead      = 0;
		}
	}
}


//---------------------------[ calculateMinDist ]--------------------------\\

float CDistanceVolume::calculateMinDist(int index, float * point, float minimumDistance2)
{
	// Smallest distance between the points in the point list of the element 
	// at "index", and the point specified by the "point" coordinates.
	float minDistance = 0.0;

	// Get the first item in the point list of the selected element
	pointListItem * currentElement = distanceVolume[index].listHead;

	// Compute the squared distance between the first list point and the
	// point specified by the "point" coordinates.

	minDistance = 
		(currentElement->p.x - point[0]) * (currentElement->p.x - point[0]) +
		(currentElement->p.y - point[1]) * (currentElement->p.y - point[1]) +
		(currentElement->p.z - point[2]) * (currentElement->p.z - point[2]);

	// Auxilary distance value
	float auxDistance = 0.0;

	// Get the next list item
	currentElement = currentElement->Next;

	// Repeat until we have either reached the end of the list, or we have 
	// encountered a distance smaller than the minimum distance.

	while ((minDistance > minimumDistance2) && (currentElement != NULL))
	{
		// Compute the squared distance between the two points.
		auxDistance =
			(currentElement->p.x - point[0]) * (currentElement->p.x - point[0]) +
			(currentElement->p.y - point[1]) * (currentElement->p.y - point[1]) +
			(currentElement->p.z - point[2]) * (currentElement->p.z - point[2]);

		// If the new distance is smaller than the current minimum distance,
		// update the minimum distance.

		if (auxDistance < minDistance)
		{
			minDistance = auxDistance;
		}

		// Get the next list item
		currentElement = currentElement->Next;
	}

	// Return the minimum distance.
	return minDistance;
}


//---------------------------[ calculateMinDist ]--------------------------\\

float CDistanceVolume::calculateMinDist(int index, float * point, float minimumDistance2, double * closestPoint)
{
	// Smallest distance between the points in the point list of the element 
	// at "index", and the point specified by the "point" coordinates.
	float minDistance = 0.0;

	// Get the first item in the point list of the selected element
	pointListItem * currentElement = distanceVolume[index].listHead;

	// Copy the coordinates to the "closestPoint" array
	closestPoint[0] = (double) currentElement->p.x;
	closestPoint[1] = (double) currentElement->p.y;
	closestPoint[2] = (double) currentElement->p.z;

	// Compute the squared distance between the first list point and the
	// point specified by the "point" coordinates.

	minDistance = 
		(currentElement->p.x - point[0]) * (currentElement->p.x - point[0]) +
		(currentElement->p.y - point[1]) * (currentElement->p.y - point[1]) +
		(currentElement->p.z - point[2]) * (currentElement->p.z - point[2]);

	// Auxilary distance value
	float auxDistance = 0.0;

	// Get the next list item
	currentElement = currentElement->Next;

	// Repeat until we have either reached the end of the list, or we have 
	// encountered a distance smaller than the minimum distance.

	while ((minDistance > minimumDistance2) && (currentElement != NULL))
	{
		// Copy the coordinates to the "closestPoint" array
		closestPoint[0] = (double) currentElement->p.x;
		closestPoint[1] = (double) currentElement->p.y;
		closestPoint[2] = (double) currentElement->p.z;

		// Compute the squared distance between the two points.
		auxDistance =
			(currentElement->p.x - point[0]) * (currentElement->p.x - point[0]) +
			(currentElement->p.y - point[1]) * (currentElement->p.y - point[1]) +
			(currentElement->p.z - point[2]) * (currentElement->p.z - point[2]);

		// If the new distance is smaller than the current minimum distance,
		// update the minimum distance.

		if (auxDistance < minDistance)
		{
			minDistance = auxDistance;
		}

		// Get the next list item
		currentElement = currentElement->Next;
	}

	// Return the minimum distance.
	return minDistance;
}


//-----------------------------[ goodDistance ]----------------------------\\

bool CDistanceVolume::goodDistance(double * point)
{
	// Compute coordinates of the distance element closest to the point
	int dsepCoord[3];
	dsepCoord[0] = (int) floorf((((float) point[0]) / this->elementDistance) + 0.5f);
	dsepCoord[1] = (int) floorf((((float) point[1]) / this->elementDistance) + 0.5f);
	dsepCoord[2] = (int) floorf((((float) point[2]) / this->elementDistance) + 0.5f);

	// Return false if the indices are not within the bounds of the distance volume
	if (dsepCoord[0] < 0 || dsepCoord[0] >= this->dsepDim[0] || 
		dsepCoord[1] < 0 || dsepCoord[1] >= this->dsepDim[1] || 
		dsepCoord[2] < 0 || dsepCoord[2] >= this->dsepDim[2] )
	{
		return false;
	}

	// Compute the index of the distance element
	int index = dsepCoord[0] + dsepCoord[1] * this->dsepDim[0] + dsepCoord[2] * this->XYdsep;

	// Return true if the current element does not yet have any
	// points in its point list, and false otherwise.

	return (this->distanceVolume[index].numberOfItems == 0);
}


//--------------------------[ exactGoodDistance ]--------------------------\\

bool CDistanceVolume::exactGoodDistance(double * point, float minimumDistance2)
{
	if (minimumDistance2 == 0.0)
		return true;

	// Compute coordinates of the distance element closest to the point
	int dsepCoord[3];
	dsepCoord[0] = (int) floorf(((float) point[0]) / this->elementDistance);
	dsepCoord[1] = (int) floorf(((float) point[1]) / this->elementDistance);
	dsepCoord[2] = (int) floorf(((float) point[2]) / this->elementDistance);

	// Return false if the indices are not within the bounds of the distance volume
	if (dsepCoord[0] < 0 || dsepCoord[0] >= this->dsepDim[0] || 
		dsepCoord[1] < 0 || dsepCoord[1] >= this->dsepDim[1] || 
		dsepCoord[2] < 0 || dsepCoord[2] >= this->dsepDim[2] )
	{
		return false;
	}

	// Cast input point to floats
	float fPoint[3];
	fPoint[0] = (float) point[0];
	fPoint[1] = (float) point[1];
	fPoint[2] = (float) point[2];

	// Coordinates of the neighbouring element
	int neighbourCoord[3];

	// Index of neighbouring distance element
	int neighbourIndex;

	// Offset for the index
	int indexOffset[3];

	// Loop through all 27 distance elements around and including the 
	// element specified by "index".

	for (int i = 0; (i < 27); i++)
	{
		// Compute the offset of the index in all three dimensions
		indexOffset[0] =  -1 + (i    % 3);			// X
		indexOffset[1] = (-1 + (i/3) % 3);			// Y
		indexOffset[2] = (-1 + (i/9) % 3);			// Z

		// Compute the coordinates of the neighbouring element
		neighbourCoord[0] = dsepCoord[0] + indexOffset[0];
		neighbourCoord[1] = dsepCoord[1] + indexOffset[1];
		neighbourCoord[2] = dsepCoord[2] + indexOffset[2];

		// Do nothing if the indices are not within the bounds of the distance volume
		if (neighbourCoord[0] < 0 || neighbourCoord[0] >= this->dsepDim[0] || 
			neighbourCoord[1] < 0 || neighbourCoord[1] >= this->dsepDim[1] || 
			neighbourCoord[2] < 0 || neighbourCoord[2] >= this->dsepDim[2] )
		{
			continue;
		}

		// Compute the index of the neighbour
		neighbourIndex = neighbourCoord[0] + neighbourCoord[1] * this->dsepDim[0] + neighbourCoord[2] * this->XYdsep;

		// Minimum distance from "point" to the points of the neighbour
		float minDistance;

		// Check if the corresponding element has a non-empty list of points
		if (this->distanceVolume[neighbourIndex].numberOfItems > 0) 
		{
			// Compute the minimum distance
			minDistance = this->calculateMinDist(neighbourIndex, fPoint, minimumDistance2);

			// If the minimum distance between "point" and the points in "neighbourIndex"
			// is less than the threshold value, we return "false".

			if (minDistance < minimumDistance2)
			{
				return false;
			}
		}
	}

	// If we're here, it means that none of the 27 neighbouring elements contains
	// a point that is too close to the input point, so we return "true".

	return true;
}


//--------------------------[ exactGoodDistance ]--------------------------\\

bool CDistanceVolume::exactGoodDistance(double * point, float minimumDistance2, double * closestPoint)
{
	// Compute coordinates of the distance element closest to the point
	int dsepCoord[3];
	dsepCoord[0] = (int) floorf(((float) point[0]) / this->elementDistance);
	dsepCoord[1] = (int) floorf(((float) point[1]) / this->elementDistance);
	dsepCoord[2] = (int) floorf(((float) point[2]) / this->elementDistance);

	// Return false if the indices are not within the bounds of the distance volume
	if (dsepCoord[0] < 0 || dsepCoord[0] >= this->dsepDim[0] || 
		dsepCoord[1] < 0 || dsepCoord[1] >= this->dsepDim[1] || 
		dsepCoord[2] < 0 || dsepCoord[2] >= this->dsepDim[2] )
	{
		return false;
	}

	// Cast input point to floats
	float fPoint[3];
	fPoint[0] = (float) point[0];
	fPoint[1] = (float) point[1];
	fPoint[2] = (float) point[2];

	// Coordinates of the neighbouring element
	int neighbourCoord[3];

	// Index of neighbouring distance element
	int neighbourIndex;

	// Offset for the index
	int indexOffset[3];

	// Loop through all 27 distance elements around and including the 
	// element specified by "index".

	for (int i = 0; (i < 27); i++)
	{
		// Compute the offset of the index in all three dimensions
		indexOffset[0] =  -1 + (i    % 3);			// X
		indexOffset[1] = (-1 + (i/3) % 3);			// Y
		indexOffset[2] = (-1 + (i/9) % 3);			// Z

		// Compute the coordinates of the neightbouring element
		neighbourCoord[0] = dsepCoord[0] + indexOffset[0];
		neighbourCoord[1] = dsepCoord[1] + indexOffset[1];
		neighbourCoord[2] = dsepCoord[2] + indexOffset[2];

		// Do nothing if the indices are not within the bounds of the distance volume
		if (neighbourCoord[0] < 0 || neighbourCoord[0] >= this->dsepDim[0] || 
			neighbourCoord[1] < 0 || neighbourCoord[1] >= this->dsepDim[1] || 
			neighbourCoord[2] < 0 || neighbourCoord[2] >= this->dsepDim[2] )
		{
			continue;
		}

		// Compute the index of the neighbour
		neighbourIndex = neighbourCoord[0] + neighbourCoord[1] * this->dsepDim[0] + neighbourCoord[2] * this->XYdsep;

		// Minimum distance from "point" to the points of the neighbour
		double minDistance;

		// Check if the neighbour index is within range, and if the
		// corresponding element has a non-empty list of points
		if (this->distanceVolume[neighbourIndex].numberOfItems > 0)
		{
			// Compute the minimum distance, return the closest point
			minDistance = this->calculateMinDist(neighbourIndex, fPoint, minimumDistance2, closestPoint);

			// If the minimum distance between "point" and the points in "neighbourIndex"
			// is less than the threshold value, we return "false".

			if (minDistance < minimumDistance2)
			{
				return false;
			}
		}
	}

	// If we're here, it means that none of the 27 neighbouring elements contains
	// a point that is too close to the input point, so we return "true".

	return true;
}


//--------------------------[ exactGoodDistance ]--------------------------\\

bool CDistanceVolume::exactGoodDistance(double * point, float minimumDistance2, double * closestPoint, double * distance)
{
	// Compute coordinates of the distance element closest to the point
	int dsepCoord[3];
	dsepCoord[0] = (int) floorf(((float) point[0]) / this->elementDistance);
	dsepCoord[1] = (int) floorf(((float) point[1]) / this->elementDistance);
	dsepCoord[2] = (int) floorf(((float) point[2]) / this->elementDistance);

	// Return false if the indices are not within the bounds of the distance volume
	if (dsepCoord[0] < 0 || dsepCoord[0] >= this->dsepDim[0] || 
		dsepCoord[1] < 0 || dsepCoord[1] >= this->dsepDim[1] || 
		dsepCoord[2] < 0 || dsepCoord[2] >= this->dsepDim[2] )
	{
		return false;
	}

	// Cast input point to floats
	float fPoint[3];
	fPoint[0] = (float) point[0];
	fPoint[1] = (float) point[1];
	fPoint[2] = (float) point[2];

	// Coordinates of the neighbouring element
	int neighbourCoord[3];

	// Minimum distance between "point" and points of an element
	float minDistance;

	// Index of neighbouring distance element
	int neighbourIndex;

	// Offset for the index
	int indexOffset[3];

	// Compute initial distance
	(*distance) = (double) (elementDistance * elementDistance);

	// Loop through all 27 distance elements around and including the 
	// element specified by "index".

	for (int i = 0; (i < 27); i++)
	{
		// Compute the offset of the index in all three dimensions
		indexOffset[0] =  -1 + (i    % 3);			// X
		indexOffset[1] = (-1 + (i/3) % 3);			// Y
		indexOffset[2] = (-1 + (i/9) % 3);			// Z

		// Compute the coordinates of the neightbouring element
		neighbourCoord[0] = dsepCoord[0] + indexOffset[0];
		neighbourCoord[1] = dsepCoord[1] + indexOffset[1];
		neighbourCoord[2] = dsepCoord[2] + indexOffset[2];

		// Do nothing if the indices are not within the bounds of the distance volume
		if (neighbourCoord[0] < 0 || neighbourCoord[0] >= this->dsepDim[0] || 
			neighbourCoord[1] < 0 || neighbourCoord[1] >= this->dsepDim[1] || 
			neighbourCoord[2] < 0 || neighbourCoord[2] >= this->dsepDim[2] )
		{
			continue;
		}

		// Compute the index of the neighbour
		neighbourIndex = neighbourCoord[0] + neighbourCoord[1] * this->dsepDim[0] + neighbourCoord[2] * this->XYdsep;
		
		// Check if the neighbour index is within range, and if the
		// corresponding element has a non-empty list of points
		if (this->distanceVolume[neighbourIndex].numberOfItems > 0)
		{
			// Compute the minimum distance, return the closest point
			minDistance = this->calculateMinDist(neighbourIndex, fPoint, minimumDistance2, closestPoint);

			// Update "distance" if needed
			if (minDistance < (*distance))
			{
				(*distance) = (double) minDistance;
			}

			// If the minimum distance between "point" and the points in "neighbourIndex"
			// is less than the threshold value, we return "false". In this version of the
			// function, we first take the root of the (squared) distance variable.

			if (minDistance < minimumDistance2)
			{
				(*distance) = sqrt(*distance);
				return false;
			}
		}
	}

	// Take the root of "distance", since we've been using the squared distance
	(*distance) = sqrt(*distance);

	// If we're here, it means that none of the 27 neighbouring elements contains
	// a point that is too close to the input point, so we return "true".

	return true;
}


//--------------------------[ addPointToDistance ]-------------------------\\

void CDistanceVolume::addPointToDistance(double * point)
{
	// Compute distance element coordinates of the point
	int dsepCoord[3];
	dsepCoord[0] = (int) floorf(((float) point[0]) / this->elementDistance);
	dsepCoord[1] = (int) floorf(((float) point[1]) / this->elementDistance);
	dsepCoord[2] = (int) floorf(((float) point[2]) / this->elementDistance);

	// Return if the indices are not within the bounds of the distance volume
	if (dsepCoord[0] < 0 || dsepCoord[0] >= this->dsepDim[0] || 
		dsepCoord[1] < 0 || dsepCoord[1] >= this->dsepDim[1] || 
		dsepCoord[2] < 0 || dsepCoord[2] >= this->dsepDim[2] )
	{
		return;
	}

	// Copy coordinates to a new point
	Point3f p;
	p.x = (float) point[0];
	p.y = (float) point[1];
	p.z = (float) point[2];

	// Compute index of distance volume element
	int index = dsepCoord[0] + dsepCoord[1] * this->dsepDim[0] + dsepCoord[2] * this->XYdsep;

	// Allocate a new list item
	pointListItem * element = (pointListItem *) malloc(sizeof(pointListItem));

	// Set the coordinates
	element->p = p;

	// Set the "Next" pointer to the "listHead" pointer of the element.
	// This essentially adds the new element to the front of the list.

	element->Next = this->distanceVolume[index].listHead;

	// Set the list head pointer to the new element.
	this->distanceVolume[index].listHead = element;

	// Increment the number of list items
	this->distanceVolume[index].numberOfItems++;

	// If this is the first point of the distance element, update the filled cells count.
	if (this->distanceVolume[index].numberOfItems == 1) 
	{
		filledCellsCount++;
	}
}


//-------------------------[ getPercentageFilled ]-------------------------\\

float CDistanceVolume::getPercentageFilled()
{
	// The percentage (or fraction, since the return value is between 0 and 1)
	// is computed by dividing the number of filled cells (cells containing at
	// least one point) by the total number of cells.

	return ((float) this->filledCellsCount / (float) this->dsepSize);
}

} // namespace bmia
