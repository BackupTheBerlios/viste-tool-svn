/*
 * SphereTriangulator.cxx
 *
 * 2011-04-11	Evert van Aart
 * - First version. Currently uses the "vtkDelaunay3D" filter, which seems to
 *   produce a good triangulation. If a more sophisticated algorithm is needed
 *   further down the road, it can be added to this class. 
 * 
 */
 

/** Includes - VTK */

#include "SphereTriangulator.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

SphereTriangulator::SphereTriangulator()
{

}


//------------------------------[ Destructor ]-----------------------------\\

SphereTriangulator::~SphereTriangulator()
{

}


//----------------------[ triangulateFromAnglesArray ]---------------------\\

bool SphereTriangulator::triangulateFromAnglesArray(vtkDoubleArray * anglesArray, vtkIntArray * triangles)
{
	// Create a point set for the unit vectors
	vtkPoints * unitVectors = vtkPoints::New();

	double * angles = NULL;
	double p[3];

	// For each set of angles (azimuth and zenith), compute the unit vector (i.e.,
	// 3D coordinates of the point on the unit sphere).

	for (int angleId = 0; angleId < anglesArray->GetNumberOfTuples(); ++angleId)
	{
		angles = anglesArray->GetTuple2(angleId);
		p[0] = cosf(angles[0]) * cosf(vtkMath::Pi() / 2.0f - angles[1]);
		p[1] = sinf(angles[0]) * cosf(vtkMath::Pi() / 2.0f - angles[1]);
		p[2] = sinf(vtkMath::Pi() / 2.0f - angles[1]);
		unitVectors->InsertNextPoint(p[0], p[1], p[2]);
	}

	// Compute the triangulation
	return this->triangulateFromUnitVectors(unitVectors, triangles);
}


//----------------------[ triangulateFromUnitVectors ]---------------------\\

bool SphereTriangulator::triangulateFromUnitVectors(vtkPoints * unitVectors, vtkIntArray * triangles)
{
	// Reset the triangles array, and make sure it has three components
	triangles->Reset();
	triangles->SetNumberOfComponents(3);

	// Create a polydata object using the point set of unit vectors
	vtkPolyData * unitPD = vtkPolyData::New();
	unitPD->SetPoints(unitVectors);

	// Create a Delaunay triangulation filter, and use the new polydata as its input
	vtkDelaunay3D * delaunay = vtkDelaunay3D::New();
	delaunay->SetInput(unitPD);

	// Make sure that no points are merged, and that all triangles are included in
	// the output, regardless of their size.

	delaunay->SetTolerance(0.0);
	delaunay->SetAlpha(0.0);
	delaunay->BoundingTriangulationOff();

	// Run the triangulation filter
	delaunay->Update();

	// Create a surface filter and run it. This will convert the unstructured grid
	// created by the Delaunay filter, which has four points per cell, to a polydata
	// object with three points per cell (i.e., triangles).

	vtkDataSetSurfaceFilter * surfaceFilter = vtkDataSetSurfaceFilter::New();
	surfaceFilter->SetInput(delaunay->GetOutput());
	surfaceFilter->Update();
	vtkPolyData * glyphPD = surfaceFilter->GetOutput();

	// Check if the surface filter succeeded
	if (!glyphPD)
	{
		unitPD->Delete();
		delaunay->Delete();
		surfaceFilter->Delete();
		return false;
	}

	// Get the polygons of the polydata
	vtkCellArray * glyphCells = glyphPD->GetPolys();

	// Check if the polydata contains polygons
	if (!glyphCells)
	{
		unitPD->Delete();
		delaunay->Delete();
		surfaceFilter->Delete();
		return false;
	}

	vtkIdType numberOfPoints;
	vtkIdType * pointList;

	// Loop through all polygons
	for (vtkIdType cellId = 0; cellId < glyphCells->GetNumberOfCells(); ++cellId)
	{
		// Get the number of points of the current polygon, and a list of the indices
		glyphCells->GetNextCell(numberOfPoints, pointList);

		// We should always have three points per polygon
		if (numberOfPoints != 3)
		{
			unitPD->Delete();
			delaunay->Delete();
			surfaceFilter->Delete();
			return false;
		}

		// The problem with the "vtkDataSetSurfaceFilter" is that it re-arranges
		// the points; it does not change the 3D coordinates, but it does change
		// their order. Since the order of the indices matters (it should match
		// the order of the spherical directions in the discrete sphere function
		// volume), we must restore the order of the points. To do so, we get the
		// 3D coordinates of a point in the output polydata set, and use those
		// coordinates to find the index of that point in the original set of
		// unit vectors. The three resulting indices are then inserted into the
		// triangles array.

		double * p;
		p = glyphPD->GetPoints()->GetPoint(pointList[0]);
		vtkIdType outId0 = unitPD->FindPoint(p[0], p[1], p[2]);
		p = glyphPD->GetPoints()->GetPoint(pointList[1]);
		vtkIdType outId1 = unitPD->FindPoint(p[0], p[1], p[2]);
		p = glyphPD->GetPoints()->GetPoint(pointList[2]);
		vtkIdType outId2 = unitPD->FindPoint(p[0], p[1], p[2]);

		triangles->InsertNextTuple3(outId0, outId1, outId2);
	}

	// Done, delete temporary objects
	unitPD->Delete();
	delaunay->Delete();
	surfaceFilter->Delete();

	return true;
}


} // namespace bmia
