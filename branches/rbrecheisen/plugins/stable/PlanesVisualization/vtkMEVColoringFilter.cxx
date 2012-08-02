/*
 * vtkMEVColoringFilter.cxx
 *
 * 2006-04-13	Tim Peeters
 * - First version
 *
 * 2006-08-03	Tim Peeters
 * - Fixed bug. It now finally works with
 *     vtkPointData * weightingPD = this->WeightingVolume->GetPointData();
 *   instead of
 *     vtkPointData * weightingPD = input->GetPointData();
 * - Call "this->WeightingVolume->Update()" before it is used.
 *
 *  2007-10-17	Tim Peeters
 *  - Added "ShiftValues".
 *
 *  2007-10-19	Tim Peeters
 *  - Call "SetNumberOfScalarComponents(3)" on output. Some filters check
 *    for this value for the input, and it was not set correctly.
 *
 * 2011-01-14	Evert van Aart
 * - Fixed a huge memory leak. The 3-component vector ("in_vector") was allocated using "new"
 *   for every single point, but never deleted. Replaced it with a static array, because there
 *   was no reason to have "in_vector" be dynamically allocated.
 *
 * 2011-04-14	Evert van Aart
 * - Weight values are now automatically normalized to the range 0-1.
 * - Cleaned up code, added comments. 
 *
 * 2011-07-12	Evert van Aart
 * - Removed the warning about "SetScalarType". I'm still not exactly
 *   sure why it cause problems, but we really do not need to call it, so I simply
 *   removed it altogether.
 *
 */


/** Includes */

#include "vtkMEVColoringFilter.h"


namespace bmia {


int vtkMEVColoringFilter::xIndexRGB = 0;
int vtkMEVColoringFilter::yIndexRGB = 1;
int vtkMEVColoringFilter::zIndexRGB = 2;


vtkStandardNewMacro(vtkMEVColoringFilter);


//-----------------------------[ Constructor ]-----------------------------\\

vtkMEVColoringFilter::vtkMEVColoringFilter()
{
	this->WeightingVolume = NULL;
	this->ShiftValues = false;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkMEVColoringFilter::~vtkMEVColoringFilter()
{
	this->WeightingVolume = NULL;
}


vtkCxxSetObjectMacro(vtkMEVColoringFilter, WeightingVolume, vtkImageData);


//----------------------------[ SimpleExecute ]----------------------------\\

void vtkMEVColoringFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
{
	// Check the input and output
	if (!input)
	{
		vtkErrorMacro(<<"No input!");
		return;
	}

	if (!output)
	{
		vtkErrorMacro(<<"No output!");
		return;
	}

	// Check if the in- and output contain point data
	vtkPointData * inPD = input->GetPointData();

	if (!inPD)
	{
		vtkErrorMacro(<<"No input point data!");
		return;
	}

	vtkPointData * outPD = output->GetPointData();

	if (!outPD)
	{
		vtkErrorMacro(<<"No output point data!");
		return;
	}

	// Get the number of points in the input image
	int numberOfPoints = input->GetNumberOfPoints();

	if (numberOfPoints < 1)
	{
		vtkWarningMacro(<<"No data to extract!");
		return;
	}

	// Get the array containing the main eigenvectors
	vtkDataArray * EigenvectorArray;
	EigenvectorArray = inPD->GetScalars("Eigenvector 1");

	if (!EigenvectorArray)
	{
		vtkErrorMacro(<<"Input point data does not have 'Eigenvector 1' array!");
		return;
	}

	// Check if the number of points matches the number of eigenvectors
	if (numberOfPoints != EigenvectorArray->GetNumberOfTuples())
	{
		vtkErrorMacro(<<"Number of tuples for Eigenvector 1 array does not match number of points!");
		return;
	}

	// Check if the eigenvectors have three components
	if (EigenvectorArray->GetNumberOfComponents() != 3)
	{
		vtkErrorMacro(<<"Number of components in Eigenvector 1 array is not 3!");
		return;
	}

	vtkDataArray * weightingArray = NULL;
	double weightRange[2];

	// Check if the weighting volume has been set
	if (this->WeightingVolume)
	{
		// Update the weighting volume, just to be sure
		this->WeightingVolume->Update();

		// Check if the number of points is correct
		if (numberOfPoints != this->WeightingVolume->GetNumberOfPoints())
		{
			vtkErrorMacro(	<< "Number of points " << this->WeightingVolume->GetNumberOfPoints()
							<< " in weighting volume differs from number "
							<< "of points in input volume " << numberOfPoints << "!");
			return;
		}

		// Get the point data of the weighting volume
		vtkPointData * weightingPD = this->WeightingVolume->GetPointData();

		if (!weightingPD)
		{
			vtkErrorMacro(<<"Input weighting dataset has no point data!");
			return;
		}

		// Get the scalars of the weighting value
		weightingArray = weightingPD->GetScalars();

		if (!weightingArray)
		{
			vtkErrorMacro(<< "Input weighting dataset does not have a scalar array!");
			return;
		}

		// Check if we've got scalar data for every voxel
		if (numberOfPoints != weightingArray->GetNumberOfTuples())
		{
			vtkErrorMacro(	<< "Number of tuples in weighting array does not match the "
							<< "number of tuples in input dataset!");
			return;
		}

		// Check if we've got one component per voxel
		if (weightingArray->GetNumberOfComponents() != 1)
		{
			vtkErrorMacro(<<"Number of components in weighting array must be one!");
			return;
		}

		// Get the range
		weightingArray->GetRange(weightRange);

		if((weightRange[1] - weightRange[0]) <= 0.0)
		{
			vtkErrorMacro(<<"Invalid scalar range for the weight volume!");
			return;
		}

	} // if [weighting volume]

	// Create the color array
	vtkUnsignedCharArray * colorArray = vtkUnsignedCharArray::New();
	colorArray->SetNumberOfComponents(3);
	colorArray->SetNumberOfTuples(numberOfPoints);

	// Setup the output
	output->CopyStructure(input);

	// Index of the current point
	vtkIdType ptId;

	// Output RGB color
	unsigned char outputColor[3];

	// Current weight
	double weight = 1.0;

	// Temporary color value
	double tempColor;

	// Eigenvector read from the input volume
	double * inputEigenvector;

	// Same eigenvector, but with order correction
	double fixedEigenvector[3];

	// Loop through all points in the image
	for (ptId = 0; ptId < numberOfPoints; ptId++)
	{
		// Get the current eigenvector
		inputEigenvector = EigenvectorArray->GetTuple3(ptId);

		// Swap order of the eigenvector elements if necessary
		fixedEigenvector[0] = inputEigenvector[vtkMEVColoringFilter::xIndexRGB];
		fixedEigenvector[1] = inputEigenvector[vtkMEVColoringFilter::yIndexRGB];
		fixedEigenvector[2] = inputEigenvector[vtkMEVColoringFilter::zIndexRGB];

		// Compute the weight
		if (weightingArray)
		{
			// Get the current weight
			weight = weightingArray->GetTuple1(ptId);

			// Normalize weight to the range 0-1 using the scalar range
			weight = (weight - weightRange[0]) / (weightRange[1] - weightRange[0]);
		}

		// Loop through the three color components
		for (int i = 0; i < 3; i++)
		{
			// Shift values if required
			if (this->ShiftValues)
			{
				tempColor = (fixedEigenvector[i] / 2.0) + 0.5;
			}
			// Otherwise, just take the absolute value
			else
			{
				tempColor = fabs(fixedEigenvector[i]);
			}

			// Multiply the color component by the weight
			tempColor *= weight;

			// Convert the color component to an unsigned character
			tempColor *= 255.0;
			outputColor[i] = (unsigned char) tempColor;

		} // for [three color components]

		// Store the color in the output array
		colorArray->SetTupleValue(ptId, outputColor);

	} // for [all voxels]

	// Store the color array
	outPD->SetScalars(colorArray);
	colorArray->Delete();

}


} // namespace bmia
