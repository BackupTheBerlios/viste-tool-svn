/*
 * geodesicPreProcessor.cxx
 *
 * 2011-05-25	Evert van Aart
 * - First version. 
 *
 */


/** Includes */

#include "geodesicPreProcessor.h"


namespace bmia {


double geodesicPreProcessor::PP_CLOSE_TO_ZERO = 0.0001;


//-----------------------------[ Constructor ]-----------------------------\\

geodesicPreProcessor::geodesicPreProcessor()
{
	// Set pointers to NULL
	this->inTensorImage			= NULL;
	this->inTensorArray			= NULL;
	this->fullPPTensorArray		= NULL;
	this->fullInvTensorArray	= NULL;
	this->scalarArray			= NULL;

	// Set default parameters
	this->enablePP				= false;
	this->gain					= 2000;
	this->sharpenMethod			= SM_None;
	this->sharpenThreshold		= 0.1;
	this->exponent				= 2;
}


//------------------------------[ Destructor ]-----------------------------\\

geodesicPreProcessor::~geodesicPreProcessor()
{
	// Delete the intermediate arrays
	if (this->fullPPTensorArray)
		this->fullPPTensorArray->Delete();

	if (this->fullInvTensorArray)
		this->fullInvTensorArray->Delete();
}


//----------------------------[ setInputImage ]----------------------------\\

bool geodesicPreProcessor::setInputImage(vtkImageData * rImage)
{
	// Set pointers to NULL
	this->inTensorImage = NULL;
	this->inTensorArray = NULL;

	// Check if the image is suitable
	if (!rImage)
		return false;

	if (!(rImage->GetPointData()))
		return false;

	if (!(rImage->GetPointData()->GetArray("Tensors")))
		return false;

	if (rImage->GetPointData()->GetArray("Tensors")->GetNumberOfComponents() != 6)
		return false;

	// Success, store the pointers
	this->inTensorImage = rImage;
	this->inTensorArray = rImage->GetPointData()->GetArray("Tensors");

	return true;
}


//----------------------------[ setScalarImage ]---------------------------\\

bool geodesicPreProcessor::setScalarImage(vtkImageData * rImage)
{
	// Set the pointer to NULL
	this->scalarArray = NULL;

	// Check if this image is suitable
	if (!rImage)
		return false;

	if (!(rImage->GetPointData()))
		return false;

	if (!(rImage->GetPointData()->GetScalars()))
		return false;

	if (rImage->GetPointData()->GetScalars()->GetNumberOfComponents() != 1)
		return false;

	// Success, store the pointer
	this->scalarArray = rImage->GetPointData()->GetScalars();

	return true;
}


//-------------------------[ preProcessFullImage ]-------------------------\\

void geodesicPreProcessor::preProcessFullImage()
{
	// If pre-processing is disabled...
	if (this->enablePP == false)
	{
		// ...use the input tensor array as the output
		this->fullPPTensorArray = this->inTensorArray;
		return;
	}

	// Get the number of tensors
	int numberOfTensors = this->inTensorArray->GetNumberOfTuples();

	// Input and output tensors
	double inTensor[6];
	double outTensor[6];

	// Tensor determinant
	double det;

	// Create a new array for the pre-processed tensors
	this->scalarArray = vtkDataArray::CreateDataArray(VTK_DOUBLE);
	this->scalarArray->SetNumberOfComponents(6);
	this->scalarArray->SetNumberOfTuples(numberOfTensors);

	// Loop through all DTI tensors
	for (int i = 0; i <numberOfTensors; ++i)
	{
		// Get the input tensor
		this->inTensorArray->GetTuple(i, inTensor);

		// Apply the gain to the input tensor
		outTensor[0] = inTensor[0] * this->gain;
		outTensor[1] = inTensor[1] * this->gain;
		outTensor[2] = inTensor[2] * this->gain;
		outTensor[3] = inTensor[3] * this->gain;
		outTensor[4] = inTensor[4] * this->gain;
		outTensor[5] = inTensor[5] * this->gain;

		// Compute the determinant of the tensor
		det = outTensor[0] * outTensor[3] * outTensor[5] + outTensor[1] * outTensor[4] * outTensor[2] + outTensor[1] * outTensor[4] * outTensor[2] - 
			  outTensor[2] * outTensor[3] * outTensor[2] - outTensor[1] * outTensor[1] * outTensor[5] - outTensor[0] * outTensor[4] * outTensor[4];

		// Use the identity tensor if the determinant is zero
		if (det < PP_CLOSE_TO_ZERO && det > -PP_CLOSE_TO_ZERO)
		{
			outTensor[0] = 1.0;
			outTensor[1] = 0.0;
			outTensor[2] = 0.0;
			outTensor[3] = 1.0;
			outTensor[4] = 0.0;
			outTensor[5] = 1.0;
			det = 1;			
		}

		// Check if the anisotropy index of the point is lower than the threshold
		if (this->scalarArray && this->sharpenMethod != SM_None)
		{
			if (this->scalarArray->GetTuple1(i) < this->sharpenThreshold)
			{
				// Apply sharpening by ways of tensor exponentiation
				if (this->sharpenMethod == SM_Exponent || this->sharpenMethod == SM_TraceDivAndExp)
				{
					this->powerInputTensor(outTensor);
				}

				// Apply sharpening by dividing the elements by the tensor's trace
				if (this->sharpenMethod == SM_TraceDivision || this->sharpenMethod == SM_TraceDivAndExp)
				{
					// Compute the trace
					double trace = (outTensor[0] + outTensor[3] + outTensor[5]);

					if (trace == 0.0)
						trace = 1.0;

					// Divide all elements by trace
					outTensor[0] /= trace;
					outTensor[1] /= trace;
					outTensor[2] /= trace;
					outTensor[3] /= trace;
					outTensor[4] /= trace;
					outTensor[5] /= trace;
				}
			}
		}

		// Add the pre-processed tensor to the array
		this->scalarArray->InsertTuple(i, outTensor); 
	}
}


//------------------------[ preProcessSingleTensor ]-----------------------\\

void geodesicPreProcessor::preProcessSingleTensor(vtkIdType pointId, double * outTensor)
{
	// If we've already pre-processed all tensors, use the stored tensor
	if (this->fullPPTensorArray)
	{
		this->fullPPTensorArray->GetTuple(pointId, outTensor);
		return;
	}

	// If pre-processing is disabled, return the input tensor
	if (this->enablePP == false)
	{
		this->inTensorArray->GetTuple(pointId, outTensor);
		return;
	}

	double inTensor[6];
	double det;

	// Get the input tensor
	this->inTensorArray->GetTuple(pointId, inTensor);

	// Apply the gain to the input tensor
	outTensor[0] = inTensor[0] * this->gain;
	outTensor[1] = inTensor[1] * this->gain;
	outTensor[2] = inTensor[2] * this->gain;
	outTensor[3] = inTensor[3] * this->gain;
	outTensor[4] = inTensor[4] * this->gain;
	outTensor[5] = inTensor[5] * this->gain;

	// Compute the determinant of the tensor
	det = outTensor[0] * outTensor[3] * outTensor[5] + outTensor[1] * outTensor[4] * outTensor[2] + outTensor[1] * outTensor[4] * outTensor[2] - 
		  outTensor[2] * outTensor[3] * outTensor[2] - outTensor[1] * outTensor[1] * outTensor[5] - outTensor[0] * outTensor[4] * outTensor[4];

	// Use the identity tensor if the determinant is zero
	if (det < PP_CLOSE_TO_ZERO && det > -PP_CLOSE_TO_ZERO)
	{
		outTensor[0] = 1.0;
		outTensor[1] = 0.0;
		outTensor[2] = 0.0;
		outTensor[3] = 1.0;
		outTensor[4] = 0.0;
		outTensor[5] = 1.0;
		det = 1;			
	}

	// Check if the anisotropy index of the point is lower than the threshold
	if (this->scalarArray && this->sharpenMethod != SM_None && pointId < this->scalarArray->GetNumberOfTuples())
	{
		if (this->scalarArray->GetTuple1(pointId) < this->sharpenThreshold)
		{
			// Apply sharpening by ways of tensor exponentiation
			if (this->sharpenMethod == SM_Exponent || this->sharpenMethod == SM_TraceDivAndExp)
			{
				this->powerInputTensor(outTensor);
			}

			// Apply sharpening by dividing the elements by the tensor's trace
			if (this->sharpenMethod == SM_TraceDivision || this->sharpenMethod == SM_TraceDivAndExp)
			{
				// Compute the trace
				double trace = (outTensor[0] + outTensor[3] + outTensor[5]);

				if (trace == 0.0)
					trace = 1.0;

				// Divide all elements by trace
				outTensor[0] /= trace;
				outTensor[1] /= trace;
				outTensor[2] /= trace;
				outTensor[3] /= trace;
				outTensor[4] /= trace;
				outTensor[5] /= trace;
			}
		}
	}
}


//---------------------------[ powerInputTensor ]--------------------------\\

void geodesicPreProcessor::powerInputTensor(double * T)
{
	// Auxiliary tensors
	double tempTA[6];
	double tempTB[6];

	// Copy input tensor to tempTA
	tempTA[0] = T[0];
	tempTA[1] = T[1];
	tempTA[2] = T[2];
	tempTA[3] = T[3];
	tempTA[4] = T[4];
	tempTA[5] = T[5];

	// Repeat (N - 1) times, where O = T^N
	for (int i = 0; i < (this->exponent - 1); i++)
	{
		// Matrix multiplication: tempTB = tempTA * T
		tempTB[0] = tempTA[0] * T[0] + tempTA[1] * T[1] + tempTA[2] * T[2];
		tempTB[1] = tempTA[0] * T[1] + tempTA[1] * T[3] + tempTA[2] * T[4];
		tempTB[2] = tempTA[0] * T[2] + tempTA[1] * T[4] + tempTA[2] * T[5];
		tempTB[3] = tempTA[1] * T[1] + tempTA[3] * T[3] + tempTA[4] * T[4];
		tempTB[4] = tempTA[1] * T[2] + tempTA[3] * T[4] + tempTA[4] * T[5];
		tempTB[5] = tempTA[2] * T[2] + tempTA[4] * T[4] + tempTA[5] * T[5];

		// Copy tempTB back to tempTA
		tempTA[0] = tempTB[0];
		tempTA[1] = tempTB[1];
		tempTA[2] = tempTB[2];
		tempTA[3] = tempTB[3];
		tempTA[4] = tempTB[4];
		tempTA[5] = tempTB[5];
	}

	// Copy the final result back to T
	T[0] = tempTA[0];
	T[1] = tempTA[1];
	T[2] = tempTA[2];
	T[3] = tempTA[3];
	T[4] = tempTA[4];
	T[5] = tempTA[5];
}


//---------------------------[ invertFullImage ]---------------------------\\

void geodesicPreProcessor::invertFullImage()
{
	this->fullInvTensorArray = NULL;

	// We require the full pre-processed tensor image
	if (!(this->fullPPTensorArray))
		return;

	// Get the number of tensors
	int numberOfTensors = this->fullPPTensorArray->GetNumberOfTuples();

	// Create an output array for the inverse tensors
	this->fullInvTensorArray = vtkDataArray::CreateDataArray(VTK_DOUBLE);
	this->fullInvTensorArray->SetNumberOfComponents(6);
	this->fullInvTensorArray->SetNumberOfTuples(numberOfTensors);

	double inTensor[6];
	double outTensor[6];

	// Loop through all tensors
	for (int i = 0; i < numberOfTensors; ++i)
	{
		// Get the pre-processed tensor
		this->fullPPTensorArray->GetTuple(i, inTensor);

		// Invert the tensor, and add it to the output
		this->invertTensor(inTensor, outTensor);
		this->fullInvTensorArray->InsertTuple(i, outTensor);
	}
}


//--------------------------[ invertSingleTensor ]-------------------------\\

void geodesicPreProcessor::invertSingleTensor(vtkIdType pointId, double * ppTensor, double * outTensor)
{
	// If we've already inverted all tensors, use the stored tensor
	if (this->fullInvTensorArray)
	{
		this->fullInvTensorArray->GetTuple(pointId, outTensor);
		return;
	}

	// Otherwise, invert it now
	this->invertTensor(ppTensor, outTensor);
}


//-----------------------------[ invertTensor ]----------------------------\\

void geodesicPreProcessor::invertTensor(double * iT, double * oT)
{
	// Compute the determinant of the tensor
	double det = iT[0] * iT[3] * iT[5] + iT[1] * iT[4] * iT[2] + iT[1] * iT[4] * iT[2] - 
				 iT[2] * iT[3] * iT[2] - iT[1] * iT[1] * iT[5] - iT[0] * iT[4] * iT[4];

	// If the determinant is close to zero, reset the tensor to the identity matrix
	if (det < PP_CLOSE_TO_ZERO && det > -PP_CLOSE_TO_ZERO)
	{
		oT[0] = 1.0;
		oT[1] = 0.0;
		oT[2] = 0.0;
		oT[3] = 1.0;
		oT[4] = 0.0;
		oT[5] = 1.0;
		det = 1;
	}
	else
	{
		// Compute the inverse of the tensor. We use the analytic solution, in which we divide the
		// transposed matrix of cofactors by the determinant of the 3 x 3 tensor. 
		oT[0] =  (iT[3] * iT[5] - iT[4] * iT[4]) / det;
		oT[1] = -(iT[1] * iT[5] - iT[4] * iT[2]) / det;
		oT[2] =  (iT[1] * iT[4] - iT[3] * iT[2]) / det;
		oT[3] =  (iT[0] * iT[5] - iT[2] * iT[2]) / det;
		oT[4] = -(iT[0] * iT[4] - iT[1] * iT[2]) / det;
		oT[5] =  (iT[0] * iT[3] - iT[1] * iT[1]) / det;

	}
}


} // namespace bmia
