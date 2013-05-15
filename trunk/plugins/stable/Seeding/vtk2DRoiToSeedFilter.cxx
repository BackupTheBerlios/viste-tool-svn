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
 * vtk2DRoiToSeedFilter.cxx
 *
 * 2010-10-29	Evert van Aart
 * - First Version.
 *
 * 2010-12-15	Evert van Aart
 * - Added support for voxel seeding.
 *
 * 2011-03-16	Evert van Aart
 * - Removed the need to compute the normal for primary planes, making the seeding
 *   more robust for elongated ROIs.
 * - Increased stability for voxel seeding when a ROI is touching the edges of
 *   an image. 
 *
 */

 
/** Includes */

#include "vtk2DRoiToSeedFilter.h"


/** Definitions */

#define ERROR_PRECISION		1.0e-20f	// We consider this to be zero
#define DOT_THRESHOLD		0.95		// Used in "getPlaneNormal"
#define MIN_VECTOR_LENGTH	0.01		// Used in "getPlaneNormal"


namespace bmia {


vtkStandardNewMacro(vtk2DRoiToSeedFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtk2DRoiToSeedFilter::vtk2DRoiToSeedFilter()
{
	// Set pointers to NULL
	this->input			= NULL;
	this->output		= NULL;
	this->inputPolygon	= NULL;

	// Set the seed distance to the default (1.0)
	this->seedDistance	= 1.0;
	this->seedDistance2 = 1.0;
}


//------------------------------[ Destructor ]-----------------------------\\

vtk2DRoiToSeedFilter::~vtk2DRoiToSeedFilter()
{

}


//----------------------------[ getPlaneNormal ]---------------------------\\

bool vtk2DRoiToSeedFilter::getPlaneNormal(double * normal)
{
	// In order to compute the plane equation, we first need a normal on
	// the plane. This is computed using the cross product of two vectors
	// in the plane (AB and AC). Three points are used to construct these
	// vectors. A is always equal to the first line point; for B and C, 
	// we use the following criteria:
	//
	// - AB and AC may not be shorted than some minimum distance.
	// - The dot product between AB and AC should not be larger than some
	//   threshold value, or lower than the negative threshold value.
	//
	// Both criteria aim to ensure a reliable normal: extremelt short vectors
	// tend to induce inaccuracies, and if the angle between the two vectors
	// is near 0 or 180 degrees, the direction of the cross product (normal)
	// is less reliable than when the angle is near 90 degrees.

	double A[3];
	double B[3];
	double C[3];

	// Get the first point, A
	this->inputPolygon->GetPoints()->GetPoint(0, A);

	// ID of the next point in the polygon
	vtkIdType nextId = 1;

	// Repeat until we've found a good candidate for B
	do
	{
		// If we've already reached the end, there are no suitable candidates
		if (nextId == this->inputPolygon->GetPoints()->GetNumberOfPoints())
			return false;

		// Get the next polygon point
		this->inputPolygon->GetPoints()->GetPoint(nextId++, B);

	// Length of AB should exceed a minimum distance
	} while (vtkMath::Distance2BetweenPoints(A, B) < MIN_VECTOR_LENGTH);

	double AB[3];
	double AC[3];

	AB[0] = B[0] - A[0];
	AB[1] = B[1] - A[1];
	AB[2] = B[2] - A[2];

	// Normalize the first vector
	vtkMath::Normalize(AB);

	// Dot product between AB and AC
	double dot;

	// Repeat until we've found a good candidate for C
	do
	{
		// If we've already reached the end, there are no suitable candidates
		if (nextId == this->inputPolygon->GetPoints()->GetNumberOfPoints())
			return false;

		// Get the next polygon point
		this->inputPolygon->GetPoints()->GetPoint(nextId++, C);

		// Check if the length of AC exceeds the minimum distance
		if (vtkMath::Distance2BetweenPoints(A, C) < MIN_VECTOR_LENGTH)
			continue;

		AC[0] = C[0] - A[0];
		AC[1] = C[1] - A[1];
		AC[2] = C[2] - A[2];

		// Normalize the second vector
		vtkMath::Normalize(AC);

		// Compute the dot product
		dot = vtkMath::Dot(AB, AC);

	// The angle between AB and AC should not near 0 degrees (dot == 1.0),
	// nor should it be near 180 degress (dot == -1.0).
	} while (dot > DOT_THRESHOLD || dot < -(DOT_THRESHOLD));

	double cross[3];

	// Compute the cross product, and normalize it
	vtkMath::Cross(AB, AC, cross);
	vtkMath::Normalize(cross);

	// Copy the cross product tothe output
	normal[0] = cross[0];
	normal[1] = cross[1];
	normal[2] = cross[2];

	return true;
}


//---------------------------[ computeIncrement ]--------------------------\\

void vtk2DRoiToSeedFilter::computeIncrement(double V, double W, double * incA, double * incB)
{
	// Computing the correct increments for each direction allows us to place a 
	// regular grid of seed points with square cells onto a (region in an) oblique 
	// plane. Let {x0, y0, z0} be a point on the plane, with the plane equation:
	// 
	//		a * x0 + b * y0 + c * z0 + d = 0
	//
	// When moving from one column to the next in our seed point grid, we increment 
	// the primary dimension with an offset "incA". At this point, the coordinate of the tertiary
	// dimension should increase with a value "incB", such that the plane equation still holds 
	// for the new point. Let X by the primary dimension, and let Z be the tertiary dimension
	// (default situation for planes with a, b, and c all non-zero. We have:
	// 
	//		a * (x0 + incA) + b * y0 + c * (z0 + incB) + d = 0
	// 
	// From this, we get "incB = -incA * (a/c)". Using the knowledge that the length of
	// the step (i.e., "sqrt(incA^2 + incB^2)") should be equal to the set seed distance,
	// we can create the following output functions (SD = Seed Distance):
	//
	//		incA^2 = (SD^2) / (1 + (a^2 / c^2))
	//		incB^2 = (SD^2 - incA^2)
	//
	// So, if we move "incA" along the primary dimension and "incB" along the tertiary 
	// dimension, we wind up in a point on the plane, "SD" away from the previous point.
	// This works the same way for the secondary and tertiary dimension. Input variables
	// V and W are equal to "a", "b" or "c", depending on the configuration of the dimen-
	// sions. For example, if the ordering of dimensions is Z > X > Y, and we want to find 
	// the increments for the secondary and tertiary dimension (i.e., if we increase X by 
	// incA, we should increase Y by incB), V will be "a" and W will be "b".

	// Squared values for "V" and "W"
	double V2 = V * V;
	double W2 = W * W;

	// Check if the squared values are none-zero. The opposite should never happen.
	if (fabs(V2) < ERROR_PRECISION || fabs(W2) < ERROR_PRECISION)
	{
		(*incA) = this->seedDistance;
		(*incB) = 0.0;
		return;
	}

	// Compute the output functions as described above
	(*incA) = (this->seedDistance2) / (1 + (V2 / W2));
	(*incB) = (this->seedDistance2 - (*incA));

	// Compute the root to obtain the actual increments
	(*incA) = sqrtf(*incA);
	(*incB) = sqrtf(*incB);
}


void vtk2DRoiToSeedFilter::getLineBounds(vtkIdType numberOfPoints, vtkIdType * pointList)
{	
	// Initialize the values for the bounds
	this->bounds[0] =  1000000.0;
	this->bounds[1] = -1000000.0;
	this->bounds[2] =  1000000.0;
	this->bounds[3] = -1000000.0;
	this->bounds[4] =  1000000.0;
	this->bounds[5] = -1000000.0;

	// Loop through all points in the line
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		// Current line point
		double currentPoint[3];

		// Get the current line point
		this->input->GetPoint(pointList[ptId], currentPoint);

		// Update maximum/minimum if necessary
		if (currentPoint[0] < this->bounds[0])	this->bounds[0] = currentPoint[0];	// X_min
		if (currentPoint[0] > this->bounds[1])	this->bounds[1] = currentPoint[0];	// X_max
		if (currentPoint[1] < this->bounds[2])	this->bounds[2] = currentPoint[1];	// Y_min
		if (currentPoint[1] > this->bounds[3])	this->bounds[3] = currentPoint[1];	// Y_max
		if (currentPoint[2] < this->bounds[4])	this->bounds[4] = currentPoint[2];	// Z_min
		if (currentPoint[2] > this->bounds[5])	this->bounds[5] = currentPoint[2];	// Z_max
	}
}


//-------------------------------[ Execute ]-------------------------------\\

void vtk2DRoiToSeedFilter::Execute()
{
	// Get the input (as poly data)
	this->input = vtkPolyData::SafeDownCast(this->GetInput());

	// Check if the input has been set
	if (!(this->input))
		return;

	// Get the output of the filter
	this->output = this->GetOutput();

	// Check if the output exists
	if (!(this->output))
		return;

	// Create a point array for the output
	vtkPoints * outputPoints = vtkPoints::New(VTK_DOUBLE);
	this->output->SetPoints(outputPoints);

	// Get the number of ROIs in the input, which is equal to the 
	// number of lines in the polydata

	int numberOfROIs = this->input->GetNumberOfLines();
	
	// Get the lines array from the input
	vtkCellArray * inputLines = this->input->GetLines();

	// Check if the lines exist
	if (!inputLines)
		return;

	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Initialize traversal of the ROIs
	inputLines->InitTraversal();

	// Loop through all ROIs
	for (vtkIdType lineId = 0; lineId < this->input->GetNumberOfLines(); ++lineId)
	{
		// Get the number of point and the list of point IDs for the next ROI
		inputLines->GetNextCell(numberOfPoints, pointList);

		// Get the bounds (minima and maxima for each direction)
		this->getLineBounds(numberOfPoints, pointList);

		// Compute the differences in the bounds
		double dx = this->bounds[1] - this->bounds[0];
		double dy = this->bounds[3] - this->bounds[2];
		double dz = this->bounds[5] - this->bounds[4];

		// If the difference is zero in more than one direction, we cannot use
		// this ROI, since it is one-dimensional.
		if ( (dx < ERROR_PRECISION && dy < ERROR_PRECISION) ||
			 (dx < ERROR_PRECISION && dz < ERROR_PRECISION) ||
			 (dy < ERROR_PRECISION && dz < ERROR_PRECISION) )
		{
			continue;
		}

		// Create a new polygon
		this->inputPolygon = vtkPolygon::New();

		// Decrease the number of points by one, since the last point is a 
		// duplicate of the first point.

		numberOfPoints--;

		// We need at least three points
		if (numberOfPoints < 3)
			continue;

		// Set the number of points of the polygon
		this->inputPolygon->GetPointIds()->SetNumberOfIds(numberOfPoints);
		this->inputPolygon->GetPoints()->SetNumberOfPoints(numberOfPoints);

		// Loop through all points in the input line
		for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Set ID of the new point
			this->inputPolygon->GetPointIds()->SetId(ptId, ptId);

			// Get current point coordinates from the input
			double currentPoint[3];
			this->input->GetPoint(pointList[ptId], currentPoint);

			// Copy the point coordinates to the polygon
			this->inputPolygon->GetPoints()->SetPoint(ptId, currentPoint);
		}

		// If the difference between maximum and minimum in one direction is
		// zero, it means that the plane the ROI is in is parallel to one of the 
		// three primary planes, so we can use a simple approach to selecting 
		// the seed point locations. If not, we're dealing with an oblique plane,
		// and the seed point generation becomes slightly more complex

		if (dx < ERROR_PRECISION || dy < ERROR_PRECISION || dz < ERROR_PRECISION)
		{
			this->generateSeedsOnPrimaryPlane();
		}
		else
		{
			this->generateSeedsOnObliquePlane();
		}

		// Delete the current polygon
		this->inputPolygon->Delete();
	}
}


//---------------------[ generateSeedsOnPrimaryPlane ]---------------------\\

void vtk2DRoiToSeedFilter::generateSeedsOnPrimaryPlane()
{
	// Compute the normal of the plane
	double n[3];
	
	if ((this->bounds[1] - this->bounds[0]) < ERROR_PRECISION)
	{
		n[0] = 1.0;
		n[1] = 0.0;
		n[2] = 0.0;
	}
	else if ((this->bounds[3] - this->bounds[2]) < ERROR_PRECISION)
	{
		n[0] = 0.0;
		n[1] = 1.0;
		n[2] = 0.0;
	}
	else if ((this->bounds[5] - this->bounds[4]) < ERROR_PRECISION)
	{
		n[0] = 0.0;
		n[1] = 0.0;
		n[2] = 1.0;
	}

	// Using seed distance
	if (this->seedMethod == SeedingPlugin::RST_Distance)
	{
		// Loop through all three dimensions, incrementing with "seedDistance" in each step.
		for (double X = this->bounds[0]; X <= this->bounds[1]; X += this->seedDistance)
		{
			for (double Y = this->bounds[2]; Y <= this->bounds[3]; Y += this->seedDistance)
			{
				for (double Z = this->bounds[4]; Z <= this->bounds[5]; Z += this->seedDistance)
				{
					// Current seed point
					double seedPoint[3] = {X, Y, Z};

					// We use the "pointInPolygon" function to determine whether or not
					// the current seed point is located within the ROI. Since this function
					// is hardly even documented in the VTK documentation, I've included a
					// description of its arguments here.
					//
					// - x[]		Coordinates of the seed point
					// - numPts		Number of points in the polygon
					// - pts		The point data of the polygon, cast to a "double *"
					// - bounds		Bounds of the polygon
					// - n			Normal of the polygon

					int isInside = this->inputPolygon->PointInPolygon(seedPoint, this->inputPolygon->GetNumberOfPoints(),
									static_cast<double *>(this->inputPolygon->GetPoints()->GetData()->GetVoidPointer(0)),
									this->bounds, n);

					// If the point is inside the ROI, save it to the output
					if (isInside == 1)
					{
						this->output->GetPoints()->InsertNextPoint(X, Y, Z);
					}
				}
			}
		}
	}
	// Using voxel seeding
	else
	{
		// Check if the voxel image has been set
		if (!this->voxels)
		{
			vtkErrorMacro(<< "Voxel seeding image not set!");
			return;
		}

		// Get the spacing of the image
		double spacing[3];
		this->voxels->GetSpacing(spacing);

		// Limit the bounds of the ROI to those of the image, otherwise the ID
		// of the last and/or first point may be "-1" (i.e., point not found)
		// for ROIs touching the edges of the image

		double voxelBounds[6];
		this->voxels->GetBounds(voxelBounds);
		if (this->bounds[0] < voxelBounds[0])	this->bounds[0] = voxelBounds[0];
		if (this->bounds[1] > voxelBounds[1])	this->bounds[1] = voxelBounds[1];
		if (this->bounds[2] < voxelBounds[2])	this->bounds[2] = voxelBounds[2];
		if (this->bounds[3] > voxelBounds[3])	this->bounds[3] = voxelBounds[3];
		if (this->bounds[4] < voxelBounds[4])	this->bounds[4] = voxelBounds[4];
		if (this->bounds[5] > voxelBounds[5])	this->bounds[5] = voxelBounds[5];
	
		// Get the voxel closest to one corner of the rectangle containing the ROI (using minima)
		double firstPoint[3];
		vtkIdType firstPointId = this->voxels->FindPoint(bounds[0], bounds[2], bounds[4]);
		this->voxels->GetPoint(firstPointId, firstPoint);

		// Get the voxel closest to the opposite corner of the rectangle (using maxima)
		double lastPoint[3];
		vtkIdType lastPointId = this->voxels->FindPoint(bounds[1], bounds[3], bounds[5]);
		this->voxels->GetPoint(lastPointId, lastPoint);

		// Some variables that are needed to call "IntersectWithLine", but aren't used otherwise
		double	t;
		double	x[3];
		double	pcoords[3];
		int		subid;

		// Loop through all three dimensions, incrementing with the spacing in each step.
		for (double X = firstPoint[0]; X <= lastPoint[0]; X += spacing[0])
		{
			for (double Y = firstPoint[1]; Y <= lastPoint[1]; Y += spacing[1])
			{
				for (double Z = firstPoint[2]; Z <= lastPoint[2]; Z += spacing[2])
				{
					// Current seed point
					double seedPoint[3] = {X, Y, Z};

					// Create a line through the ROI, intersecting it at {X, Y, Z}. We use
					// the normal of the plane containing the ROI to do so.

					double p1[3] = {X - n[0], Y - n[1], Z - n[2]};
					double p2[3] = {X + n[0], Y + n[1], Z + n[2]};

					// Check if the created line intersects the ROI. Due to rounding errors and
					// limited precision, the voxel coordinates are not necessarily equal to the
					// coordinates of the ROI point. For example, if all points in a ROI have an
					// X-coordinate of 100.0, the voxel coordinates in that plane may be reported
					// as having an X-coordinate of 99.9999 or 100.00001 or something similar. This 
					// would cause the "PointInPolygon" function to fail. The "IntersectWithLine"
					// function, however, allows us to ignore this inaccuracy.

					int isInside = this->inputPolygon->IntersectWithLine(p1, p2, 0.1, t, x, pcoords, subid);

					// If the point is inside the ROI, save it to the output
					if (isInside == 1)
					{
						this->output->GetPoints()->InsertNextPoint(X, Y, Z);
					}
				}
			}
		}
	}
}


//---------------------[ generateSeedsOnObliquePlane ]---------------------\\

void vtk2DRoiToSeedFilter::generateSeedsOnObliquePlane()
{
	// Voxel seeding is impossible if the plane is not one of the primary planes
	if (this->seedMethod == SeedingPlugin::RST_Voxel)
	{
		vtkErrorMacro(<< "Cannot apply voxel seeding on an oblique plane!");
		return;
	}

	// Plane equation coordinates. Plane equations are of the form
	//    "c[0] * x + c[1] * y + c[2] * z + d = 0"

	double c[3];
	double d;

	// Compute the normal of the polygon
	double n[3];

	if (!(this->getPlaneNormal(n)))
		return;

	// Set the plane equation coefficients to the normal components
	c[0] = n[0];
	c[1] = n[1];
	c[2] = n[2];

	// Get the first point of the polygon, and use it compute "d"
	double firstPoint[3];
	this->inputPolygon->GetPoints()->GetPoint(0, firstPoint);
	d = -c[0] * firstPoint[0] - c[1] * firstPoint[1] - c[2] * firstPoint[2];

	// We define a primary, secondary and tertiary dimension. The dimension
	// with the smallest (absolute) corresponding coefficient in the plane 
	// equation is defined as the primary dimension; the second smallest 
	// coefficient is the secondary dimension, and the largest is the 
	// tertiary dimension. For example, if c[0] = 0.0, c[1] = 7.0 and 
	// c[2] = -3.0, X is the primary dimension, and Y and Z are the secondary
	// and tertiary dimensions, respectively.

	int dim1;	// Primary
	int dim2;	// Secondary
	int dim3;	// Tertiary

	// We create a regular grid on the ROI as follows: Whenever we move from
	// one column to the next, we increment the primary dimension by "incDim1", 
	// and the tertiary by "incDim1to3". These two increments have been computed 
	// by "computeIncrement" to ensure that the distance between the current
	// point and the next one is equal to the seed distance. Moving from one row
	// to the next increments the secondary dimension by "incDim2" and the tertiary
	// by "incDim2to3". Sorting the dimensions by coefficient size ensures correct
	// behavior when one of the coefficients is zero.

	double incDim1 = 0.0;
	double incDim2 = 0.0;
	double incDim1to3 = 0.0;
	double incDim2to3 = 0.0;

	// If "c[0]" is the smallest coefficient, the order is either "X > Y > Z" or "X > Z > Y"
	if (fabs(c[DIM_X]) <= fabs(c[DIM_Y]) && fabs(c[DIM_X]) <= fabs(c[DIM_Z]))
	{
		dim1 = DIM_X;

		if (fabs(c[DIM_Y]) <= fabs(c[DIM_Z]))
		{
			dim2 = DIM_Y;
			dim3 = DIM_Z;
		}
		else
		{
			dim2 = DIM_Z;
			dim3 = DIM_Y;
		}
	}
	// If "c[1]" is the smallest coefficient, the order is either "Y > X > Z" or "Y > Z > X"
	else if (fabs(c[DIM_Y]) <= fabs(c[DIM_X]) && fabs(c[DIM_Y]) <= fabs(c[DIM_Z]))
	{
		dim1 = DIM_Y;

		if (fabs(c[DIM_X]) <= fabs(c[DIM_Z]))
		{
			dim2 = DIM_X;
			dim3 = DIM_Z;
		}
		else
		{
			dim2 = DIM_Z;
			dim3 = DIM_X;
		}
	}
	// If "c[2]" is the smallest coefficient, the order is either "Z > X > Y" or "Z > Y > X"
	else 
	{
		dim1 = DIM_Z;

		if (fabs(c[DIM_X]) < fabs(c[DIM_Y]))
		{
			dim2 = DIM_X;
			dim3 = DIM_Y;
		}
		else
		{
			dim2 = DIM_Y;
			dim3 = DIM_X;
		}
	}

	// If the smallest coefficient is zero, we can move on the plane in the corresponding 
	// dimension without changing the other two coordinates.
	if (fabs(c[dim1]) < ERROR_PRECISION)
		incDim1 = this->seedDistance;
	// Otherwise, moving along "dim1" requires an increment/decrement along "dim3"
	else
		this->computeIncrement(c[dim1], c[dim3], &incDim1, &incDim1to3);

	// Moving along "dim2" always requires a change in "dim3".
	this->computeIncrement(c[dim2], c[dim3], &incDim2, &incDim2to3);

	// We create candidate seed points in a rectangle covering the entire ROI, after which
	// we check whether or not a candidate is inside the ROI. We start in one corner of the
	// rectangle, the origin. We use the minima of the primary and secondary dimensions
	// as the origin's coordinates in those dimensions, and we then compute the origin's
	// coordinate in the tertiary dimension using the plane equation.

	double origin[3];
	origin[dim1] = this->bounds[dim1*2];
	origin[dim2] = this->bounds[dim2*2];
	origin[dim3] = (c[dim1] * origin[dim1] + c[dim2] * origin[dim2] + d) / (-c[dim3]);

	// Point coordinates of the point at the start of a row
	double rowStart[3] = {origin[0], origin[1], origin[2]};

	// Current seed point coordinates
	double seedPoint[3] = {origin[0], origin[1], origin[2]};

	// Loop through all rows (secondary dimension)
	while (seedPoint[dim2] <= this->bounds[2*dim2 + 1])
	{
		// Initialize the seed point to the start of the current row
		seedPoint[0] = rowStart[0];
		seedPoint[1] = rowStart[1];
		seedPoint[2] = rowStart[2];

		// Loop through all columns (primary dimension)
		while (seedPoint[dim1] <= this->bounds[2*dim1 + 1])
		{
			// Check whether the current seed point is inside the ROI. See  
			// "generateSeedsOnPrimaryPlane" for function documentation.

			int isInside = this->inputPolygon->PointInPolygon(seedPoint, this->inputPolygon->GetNumberOfPoints(),
								static_cast<double *>(this->inputPolygon->GetPoints()->GetData()->GetVoidPointer(0)),
								this->bounds, n);

			// If the seed point is within the ROI, add it to the output
			if (isInside == 1)
			{
				this->output->GetPoints()->InsertNextPoint(seedPoint[0], seedPoint[1], seedPoint[2]);
			}

			// Increment the primary dimension by "incDim1", and increment the tertiary
			// dimension by "incDim1to3" to ensure that we remain on the same plane.
			seedPoint[dim1] += incDim1;
			seedPoint[dim3] += incDim1to3;
		}

		// Increment the secondary dimension by "incDim2", and increment the tertiary
		// dimension by "incDim2to3" to ensure that we remain on the same plane.
		rowStart[dim2] += incDim2;
		rowStart[dim3] += incDim2to3;
	}
}


} // namespace bmia


/** Undefine Temporary Definitions */

#undef ERROR_PRECISION
#undef DOT_THRESHOLD
#undef MIN_VECTOR_LENGTH
