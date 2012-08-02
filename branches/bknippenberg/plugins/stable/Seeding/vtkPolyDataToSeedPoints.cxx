/*
 * vtkPolyDataToSeedPoints.cxx
 *
 * 2011-04-18	Evert van Aart
 * - First Version.
 *
 */


/** Includes */

#include "vtkPolyDataToSeedPoints.h"


namespace bmia {


vtkStandardNewMacro(vtkPolyDataToSeedPoints);


//-------------------------------[ Execute ]-------------------------------\\

void vtkPolyDataToSeedPoints::Execute()
{
	// Get the input polydata.
	vtkPolyData * inPD = vtkPolyData::SafeDownCast(this->GetInput());

	if (!inPD)
		return;

	// Get the output
	vtkUnstructuredGrid * outGrid = this->GetOutput();

	if (!outGrid)
		return;

	// Reset the output
	outGrid->Reset();

	// Add a point array to the output
	vtkPoints * outPoints = vtkPoints::New();
	outGrid->SetPoints(outPoints);
	outPoints->Delete();

	int numberOfPoints = inPD->GetNumberOfPoints();

	double p[3];

	// Copy all input points to the output
	for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
	{
		inPD->GetPoint(ptId, p);
		outPoints->InsertNextPoint(p);
	}
}


} // namespace bmia
