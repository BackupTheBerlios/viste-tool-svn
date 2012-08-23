/*
 * vtkScalarVolumeToSeedPoints.cxx
 *
 * 2011-05-10	Evert van Aart
 * - First version
 *
 */


/** Includes */

#include "vtkScalarVolumeToSeedPoints.h"


namespace bmia {


vtkStandardNewMacro(vtkScalarVolumeToSeedPoints);


//-----------------------------[ Constructor ]-----------------------------\\

vtkScalarVolumeToSeedPoints::vtkScalarVolumeToSeedPoints()
{
	// Initialize variables
	this->minThreshold = 0.0;
	this->maxThreshold = 1.0;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkScalarVolumeToSeedPoints::~vtkScalarVolumeToSeedPoints()
{

}


//-------------------------------[ Execute ]-------------------------------\\

void vtkScalarVolumeToSeedPoints::Execute()
{
	// Get the input data set and cast it to an image
	vtkImageData * image = vtkImageData::SafeDownCast(this->GetInput());

	// If the input has not been set, OR if it's not an image, we throw an error
	if (!image)
	{
		vtkErrorMacro(<< "Input image has not been set!");
		return;
	}

	// Get the point data of the input
	vtkPointData * imagePD = image->GetPointData();

	// Check if the input has been set
	if (!imagePD)
	{
		vtkErrorMacro(<< "Input image does not contain point data!");
		return;
	}

	// Get the scalar array
	vtkDataArray * imageScalars = imagePD->GetScalars();

	// Check if the input has been set
	if (!imageScalars)
	{
		vtkErrorMacro(<< "Input image does not contain a scalar array!");
		return;
	}

	// Get the number of seed points
	vtkIdType numberOfPoints = image->GetNumberOfPoints();

	// Get the output seed points
	vtkUnstructuredGrid * output = this->GetOutput();

	// Create a new point set
	vtkPoints * newPoints = vtkPoints::New();
	newPoints->SetDataTypeToDouble();

	// Add the points to the output data set
	output->SetPoints(newPoints);
	newPoints->Delete();

	// Point coordinates
	double p[3];

	// Current scalar value
	double scalar;

	// Loop through all voxels
	for (vtkIdType i = 0; i < numberOfPoints; ++i)
	{
		// Get the voxel scalar value
		scalar = imageScalars->GetTuple1(i);

		// If the scalar value lies between the two thresholds...
		if (scalar >= this->minThreshold && scalar <= this->maxThreshold)
		{
			// ...get the coordinates of its voxel...
			image->GetPoint(i, p);

			// ...and add those coordinates to the output.
			newPoints->InsertNextPoint(p);
		}

	} // for [every voxel]
}


} // namespace bmia
