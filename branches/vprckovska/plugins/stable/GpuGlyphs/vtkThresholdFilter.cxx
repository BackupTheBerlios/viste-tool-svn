/*
 * vtkThresholdFilter.cxx
 *
 * 2011-01-10	Evert van Aart
 * - First version
 *
 */


/** Includes */

#include "vtkThresholdFilter.h"


namespace bmia {


vtkStandardNewMacro(vtkThresholdFilter);


vtkThresholdFilter::vtkThresholdFilter()
{
	// Initialize variables
	this->measure	= HARDIMeasures::GA;
	this->threshold = 0.5;
	this->invert	= false;
	this->shData	= NULL;

	// Initialize the outputs
	this->SetNumberOfOutputs(2);
	this->SetNthOutput(0, NULL);
	this->SetNthOutput(1, NULL);
}


vtkThresholdFilter::~vtkThresholdFilter()
{

}


void vtkThresholdFilter::setOutputs(vtkPointSet * dtiSeeds, vtkPointSet * hardiSeeds)
{
	// Store the output point sets
	this->SetNthOutput(0, dtiSeeds);
	this->SetNthOutput(1, hardiSeeds);
}


void vtkThresholdFilter::forceExecute()
{
	// Force the filter to execute
	this->Execute();
}


void vtkThresholdFilter::Execute()
{
	// Check if the SH image has been set
	if (!(this->shData))
	{
		vtkErrorMacro(<< "No input SH image!");
		return;
	}

	// Get the point data of the input
	vtkPointData * inputPD = shData->GetPointData();

	// Check if the input has been set
	if (!inputPD)
	{
		vtkErrorMacro(<< "Input SH image does not contain point data!");
		return;
	}

	// Get the spherical harmonics coefficients
	vtkDataArray * SHCoefficientsArray = inputPD->GetScalars();

	// Check if the input has been set
	if (!SHCoefficientsArray)
	{
		vtkErrorMacro(<< "Input SH image does not contain an array with SH coefficients!");
		return;
	}

	// Get the seed points
	vtkPointSet * inSeeds = this->GetInput();

	// Check if the seed points have been set
	if (!inSeeds)
	{
		vtkErrorMacro(<<"No input seed points!");
		return;
	}

	// Force the seed point set to update
	inSeeds->Update();

	// Get the number of seed points
	vtkIdType numberOfPoints = inSeeds->GetNumberOfPoints();

	// Get the output point sets
	vtkPointSet * outSeedsDTI   = this->GetOutput(0);
	vtkPointSet * outSeedsHARDI = this->GetOutput(1);

	// Check if the outputs have been set
	if (!outSeedsDTI || !outSeedsHARDI)
	{
		vtkErrorMacro(<<"Output data set pointer not set!");
		return;
	}

	// Create a point array for the output
	vtkPoints * outSeedsDTIPoints = vtkPoints::New(VTK_DOUBLE);
	outSeedsDTI->SetPoints(outSeedsDTIPoints);
	outSeedsDTIPoints->Delete();

	// Create a point array for the output
	vtkPoints * outSeedsHARDIPoints = vtkPoints::New(VTK_DOUBLE);
	outSeedsHARDI->SetPoints(outSeedsHARDIPoints);
	outSeedsHARDIPoints->Delete();

	// Point coordinates
	double p[3];

	// Spherical Harmonics coefficients
	double * coefficients;

	// Point index
	vtkIdType pointId;

	// Create an object for computing HARDI measures
	HARDIMeasures * HMeasures = new HARDIMeasures;

	// Loop through all seed points
	for (vtkIdType i = 0; i < numberOfPoints; ++i)
	{
		// Get the current seed point
		inSeeds->GetPoint(i, p);

		// Find the point ID of the seed point in the input image
		pointId = shData->FindPoint(p);

		// Do nothing if the point was not found
		if (pointId == -1) 
		{
			continue;
		}

		// Get the SH coefficient of the point
		coefficients  = SHCoefficientsArray->GetTuple(pointId);    

		// Check if the first coefficient is positive
		if (coefficients[0] <= 0.0)
		{
			continue;
		} 

		// Get the selected HARDI measure for the current voxel
		double m = HMeasures->HARDIMeasure((int) this->measure, coefficients, 4);

		// Decide whether this seed point should be a DTI glyphs or not
		bool toDTI = (this->invert && m > this->threshold) || (!this->invert && m < this->threshold);

		// Add the point to the DTI or HARDI data set
		if (toDTI)
		{
			outSeedsDTI->GetPoints()->InsertNextPoint(p);
		}
		else
		{
			outSeedsHARDI->GetPoints()->InsertNextPoint(p);
		}

	} // for [every point]

	// Delete the measures object
	delete HMeasures;

}

}