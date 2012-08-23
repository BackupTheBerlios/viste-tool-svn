/**
 * vtkTensorToScalarFilter.cxx
 *
 * 2012-03-16	Ralph Brecheisen
 * - First version.
 *
 */

// Includes

#include "vtkTensorToScalarFilter.h"

// Includes tensor math

#include "TensorMath/vtkTensorMath.h"

// Includes VTK

#include <vtkPointData.h>
#include <vtkDoubleArray.h>

namespace bmia {

//-----------------------[ SimpleExecute ]-------------------------\\

void vtkTensorToScalarFilter::SimpleExecute(vtkImageData * input, vtkImageData * output)
{
    // Initialize progress updates
	this->UpdateProgress(0.0);

	if (!input)
	{
		vtkErrorMacro(<<"No input has been set!");
		return;
	}

	if (!output)
	{
		vtkErrorMacro(<<"No output has been set!");
		return;
	}

    // Get point data of the input tensors
	vtkPointData * inPD = input->GetPointData();

	if (!inPD)
	{
		vtkErrorMacro(<<"Input does not contain point data!");
		return;
	}

    // Get input tensor array
	vtkDataArray * inTensors = inPD->GetArray("Tensors");

	if (!inTensors)
	{
		vtkWarningMacro(<<"Input data has no tensors!");
		return;
	}

    // Get point data of the output scalars
	vtkPointData * outPD = output->GetPointData();

	if (!outPD)
	{
		vtkErrorMacro(<<"Output does not contain point data!");
		return;
	}

	int numberOfPoints = input->GetNumberOfPoints();

	if (numberOfPoints != inTensors->GetNumberOfTuples())
	{
		vtkErrorMacro(<<"Size mismatch between point data and tensor array!");
		return;
	}

	if (numberOfPoints < 1)
	{
		vtkWarningMacro(<<"Number of points in the input is not positive!");
		return;
	}

    // Define output scalar array
    vtkDoubleArray * outArray = vtkDoubleArray::New();
    outArray->SetNumberOfComponents(1);
    outArray->SetNumberOfTuples(numberOfPoints);

    // ID of the current point
	vtkIdType ptId;

    // Loop through all points of the image
	for(ptId = 0; ptId < numberOfPoints; ++ptId)
	{
        // Current tensor value
        double tensor[6];

        // Get tensor value at current point
		inTensors->GetTuple(ptId, tensor);

        // Check if tensor is NULL. This check is not necessary but can
        // save time on sparse datasets
		if (vtkTensorMath::IsNullTensor(tensor))
		{
            // Set output value to zero
            outArray->SetTuple1(ptId, 0.0);
		}

        // Compute the output scalar value
		else
		{
			double outScalar = this->ComputeScalar(tensor);

            // Add scalar value to output array
            outArray->SetTuple1(ptId, outScalar);
		}

        // Update progress value
        if(ptId % 50000 == 0)
        {
            this->UpdateProgress(((float) ptId) / ((float) numberOfPoints));
        }
	}

    // Add scalars to the output
    outPD->SetScalars(outArray);

    outArray->Delete();
    outArray = NULL;

    // We're done!
    this->UpdateProgress(1.0);
}

} // namespace bmia
