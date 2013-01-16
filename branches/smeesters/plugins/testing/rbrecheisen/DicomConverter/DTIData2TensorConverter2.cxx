#include "DTIData2TensorConverter2.h"
#include <sstream>

///////////////////////////////////////////////
DTIData2TensorConverter2::DTIData2TensorConverter2()
{
	this->Input             = 0;
	this->GradientsExtended = 0;
	this->Gradients         = 0;
	this->NumberOfGradients = 0;
	this->NumberOfB0Slices  = 0;
	this->TensorSlices      = 0;
	this->MaskEnabled       = true;
	this->MaskValue         = 0.0;
	this->BValue            = 0.0;
	this->Rows              = 0;
	this->Columns           = 0;
	this->B0SliceFirstIndex = -1;
}

///////////////////////////////////////////////
DTIData2TensorConverter2::~DTIData2TensorConverter2()
{
	if(this->Input != 0)
	{
		this->Input->clear();
		delete this->Input;
	}

	if(this->Gradients != 0)
		gsl_matrix_free(this->Gradients);

	if(this->GradientsExtended != 0)
		gsl_matrix_free(this->GradientsExtended);
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetInput(vector<DTISliceGroup *> *input)
{
	if(this->Input != 0)
	{
		this->Input->clear();
		delete this->Input;
	}

	this->Input = 0;
	this->Input = input;

	// Retrieve rows and columns from first slice.
	DTISlice *slice = this->Input->at(0)->GetSliceAt(0);
	this->Rows      = slice->GetRows();
	this->Columns   = slice->GetColumns();
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetNumberOfB0Slices(int numberslices)
{
	this->NumberOfB0Slices = numberslices;
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetB0SliceFirstIndex(int index)
{
	this->B0SliceFirstIndex = index;
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetGradients(gsl_matrix * gradients)
{
	if(this->Gradients != 0)
		gsl_matrix_free(this->Gradients);
	this->Gradients = gradients;

	// Get number of gradients in matrix. This is the total number of gradients
	// loaded from file (including gradient 0,0,0).
	this->NumberOfGradients = (int) this->Gradients->size1;

	if(this->GradientsExtended != 0)
		gsl_matrix_free(this->GradientsExtended);
	this->GradientsExtended = gsl_matrix_calloc(this->NumberOfGradients - this->NumberOfB0Slices, 6);

	int index = 0;
	for(int i = 0; i < this->NumberOfGradients; i++)
	{
		double grad0 = gsl_matrix_get(this->Gradients, i, 0);
		double grad1 = gsl_matrix_get(this->Gradients, i, 1);
		double grad2 = gsl_matrix_get(this->Gradients, i, 2);

		// Skip if encounter (0,0,0) gradient. This corresponds to B0 slice.
		if(grad0 == 0.0 && grad1 == 0.0 && grad2 == 0.0)
			continue;

		double len = grad0*grad0 + grad1*grad1 + grad2*grad2;

		gsl_matrix_set(this->GradientsExtended, index, 0, grad0 * grad0 / len);
		gsl_matrix_set(this->GradientsExtended, index, 1, grad1 * grad1 / len);
		gsl_matrix_set(this->GradientsExtended, index, 2, grad2 * grad2 / len);
		gsl_matrix_set(this->GradientsExtended, index, 3, 2 * grad0 * grad1 / len);
		gsl_matrix_set(this->GradientsExtended, index, 4, 2 * grad1 * grad2 / len);
		gsl_matrix_set(this->GradientsExtended, index, 5, 2 * grad0 * grad2 / len);
		index++;
	}
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetMaskEnabled(bool enabled)
{
	this->MaskEnabled = enabled;
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetMaskValue(double maskvalue)
{
	this->MaskValue = maskvalue;
}

///////////////////////////////////////////////
void DTIData2TensorConverter2::SetBValue(double bvalue)
{
	this->BValue = bvalue;
}

///////////////////////////////////////////////
bool DTIData2TensorConverter2::Execute()
{
	const char *func = "DTIData2TensorConverter::Execute";

	// Check if the required parameters have been set.
	if(this->Input == 0)
	{
		__DTIMACRO_LOG(func << ": No input data to convert" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	if(this->Gradients == 0)
	{
		__DTIMACRO_LOG(func << ": No gradients specified" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	if(this->NumberOfB0Slices == 0)
	{
		__DTIMACRO_LOG(func << ": Number of B0 slices not set" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	if(this->B0SliceFirstIndex == -1)
	{
		__DTIMACRO_LOG(func << ": First B0 slice index not set. Assuming index 0 (zero)" << endl, ERROR, DTIUtils::LogLevel);
		this->B0SliceFirstIndex = 0;
	}

	if(this->MaskEnabled && this->MaskValue == 0.0)
	{
		__DTIMACRO_LOG(func << ": Masking enabled with ZERO mask value" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	// Apply masking if enabled. Use only B0 slices to calculate the mask
	// setting. Calculating the mask setting requires knowing the Rows and
	// Columns of each slice. So, we retrieve them here also.
	double masksetting = 0.0;

	if(this->MaskEnabled)
		masksetting = this->CalculateMaskSetting();
	
	__DTIMACRO_LOG(func << ": Mask setting = " << masksetting << endl, ALWAYS, DTIUtils::LogLevel);

	// Compute inverse of extended gradient matrix. Because this is a non-square matrix we
	// need to use the pseudo-inverse.
	gsl_matrix *gradientsinverse = this->CalculatePseudoInverse(this->GradientsExtended);

	// For each slice group...
	vector<DTISliceGroup *>::iterator iter;
	int tensorcount = 0;

	for(iter = this->Input->begin(); iter != this->Input->end(); iter++)
	{
		// Get current slice group and its size.
		DTISliceGroup *slicegroup = (*iter);
		int numberofslices        = slicegroup->GetSize();

		// Get first B0 slice in this slice group. 
		DTISlice *b0slice = slicegroup->GetSliceAt(this->B0SliceFirstIndex);

		// Get slice's pixel data.
		unsigned short *b0pixels = b0slice->GetData();

		if(this->MaskEnabled)
		{
			for(int k = 0; k < numberofslices; k++)
			{
				double grad0 = gsl_matrix_get(this->Gradients, k, 0);
				double grad1 = gsl_matrix_get(this->Gradients, k, 1);
				double grad2 = gsl_matrix_get(this->Gradients, k, 2);

				// Skip slice if it's a B0 slice (gradient 0,0,0).
				if(grad0 == 0.0 && grad1 == 0.0 && grad2 == 0.0)
					continue;

				// Get pixel data for current slice.
				unsigned short *gradientpixels = slicegroup->GetSliceAt(k)->GetData();

				// Apply the mask to the pixel data. This changes the masked gradient pixels
				// in the k-th slice of the slice group.
				this->ApplyMask(gradientpixels, b0pixels, masksetting);
			}
		}

		// Create tensor slice object to store the calcualted tensor values.
		DTITensorSlice *tensorslice = new DTITensorSlice();
		tensorslice->SetRows(this->Rows);
		tensorslice->SetColumns(this->Columns);

		for(int i = 0; i < this->Rows; i++)
		{
			for(int j = 0; j < this->Columns; j++)
			{
				// Compute tensor
				gsl_matrix *tensor = this->ComputeTensor(i, j, slicegroup, b0pixels, gradientsinverse);
				tensorslice->SetTensorAt(tensor, i, j);

				//std::cout 
				//	<< gsl_matrix_get( tensor, 0, 0 ) << " " 
				//	<< gsl_matrix_get( tensor, 1, 0 ) << " " 
				//	<< gsl_matrix_get( tensor, 2, 0 ) << " "
				//	<< gsl_matrix_get( tensor, 3, 0 ) << " "
				//	<< gsl_matrix_get( tensor, 4, 0 ) << " "
				//	<< gsl_matrix_get( tensor, 5, 0 ) << std::endl;
			}
		}

		// Now that all pixels of the tensor slice have a tensor value we can store the
		// tensor slice in the list.
		if(this->TensorSlices == 0)
		{
			this->TensorSlices = new vector<DTITensorSlice *>;
		}

		this->TensorSlices->push_back(tensorslice);

		__DTIMACRO_LOG(func << ": Created tensor slice " << tensorcount << endl, ALWAYS, DTIUtils::LogLevel);
		tensorcount++;
	}

	// Clean up inverse gradient matrix.
	gsl_matrix_free(gradientsinverse);
	return true;
}

///////////////////////////////////////////////
vector<DTITensorSlice *> *DTIData2TensorConverter2::GetOutput()
{
	const char *func = "DTIData2TensorConverter::GetOutput";

	// Check if we can return something at all.
	if(this->TensorSlices == 0)
	{
		__DTIMACRO_LOG(func << ": Tensors have not been calculated yet" << endl, ERROR, DTIUtils::LogLevel);
		return 0;
	}

	return this->TensorSlices;
}

///////////////////////////////////////////////
double DTIData2TensorConverter2::CalculateMaskSetting()
{
	const char *func = "DTIData2TensorConverter::CalculateMaskSetting";

	// Check if user specified mask value.
	if(this->MaskValue == 0.0)
	{
		__DTIMACRO_LOG(func << ": No mask value set" << endl, ERROR, DTIUtils::LogLevel);
		return 0.0;
	}

	unsigned short minvolume = (unsigned short) 99999999;
	unsigned short maxvolume = (unsigned short) 0;

	// Find the B0 slice in each slice group and retrieve its minimum and maximum
	// value. Then, of all B0 slices take the global minimum and maximum values to
	// calculate the mask setting.
	vector<DTISliceGroup *>::iterator iter;

	for(iter = this->Input->begin(); iter != this->Input->end(); iter++)
	{
		DTISliceGroup *slicegroup = (*iter);
		unsigned short *b0pixels = slicegroup->GetSliceAt(this->B0SliceFirstIndex)->GetData();

		// Find minimum and maximum values in this B0 slice data.
		unsigned short minslice = DTIUtils::GetMinValue(b0pixels, this->Rows * this->Columns);
		unsigned short maxslice = DTIUtils::GetMaxValue(b0pixels, this->Rows * this->Columns);

		// Compare values with the absolute (volume) minimum and maximum
		if(minslice < minvolume)
			minvolume = minslice;
		if(maxslice > maxvolume)
			maxvolume = maxslice;
	}

	// Calculate mask setting to use based on minimum and maximum values found
	// in all B0 slices.
	double masksetting = minvolume + this->MaskValue * (maxvolume - minvolume);
	return masksetting;
}

///////////////////////////////////////////////
gsl_matrix *DTIData2TensorConverter2::CalculatePseudoInverse(gsl_matrix *matrix)
{
	int nrgrads = this->NumberOfGradients - this->NumberOfB0Slices;

	// Jacobi SVD
	gsl_matrix *U = gsl_matrix_calloc(nrgrads, 6);
	gsl_matrix *V = gsl_matrix_calloc(6, 6);
	gsl_vector *s = gsl_vector_calloc(6);
	gsl_matrix_memcpy(U, matrix);
	gsl_linalg_SV_decomp_jacobi(U, V, s);

	gsl_matrix *S = gsl_matrix_calloc(6, 6);
	for(int i = 0; i < 6; i++)
		gsl_matrix_set(S, i, i, gsl_vector_get(s, i));

	// transpose S, U and V
	gsl_matrix *ST = gsl_matrix_calloc(6, 6);
	gsl_matrix *UT = gsl_matrix_calloc(6, nrgrads);
	gsl_matrix *VT = gsl_matrix_calloc(6, 6);
	gsl_matrix_transpose_memcpy(ST, S);
	gsl_matrix_transpose_memcpy(UT, U);
	gsl_matrix_transpose_memcpy(VT, V);

	// multiply S and ST
	gsl_matrix *STS = gsl_matrix_calloc(6, 6);
	gsl_linalg_matmult(ST, S, STS);

	// invert by LU decomposition
	int signum;
	gsl_matrix *LU = gsl_matrix_calloc(6, 6);
	gsl_matrix *STSI = gsl_matrix_calloc(6, 6);
	gsl_permutation *p = gsl_permutation_calloc(6);
	gsl_matrix_memcpy(LU, STS);
	gsl_linalg_LU_decomp(LU, p, &signum);
	gsl_linalg_LU_invert(LU, p, STSI);

	// calculate pseudo inverse
	gsl_matrix *STUT = gsl_matrix_calloc(6, nrgrads);
	gsl_matrix *STSISTUT = gsl_matrix_calloc(6, nrgrads);
	gsl_matrix *VSTSISTUT = gsl_matrix_calloc(6, nrgrads);
	gsl_linalg_matmult(ST, UT, STUT);
	gsl_linalg_matmult(STSI, STUT, STSISTUT);
	gsl_linalg_matmult(V, STSISTUT, VSTSISTUT);

	// cleanup
	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(S);
	gsl_matrix_free(UT);
	gsl_matrix_free(VT);
	gsl_matrix_free(ST);
	gsl_matrix_free(STS);
	gsl_matrix_free(STSI);
	gsl_matrix_free(LU);
	gsl_matrix_free(STUT);
	gsl_matrix_free(STSISTUT);
	gsl_vector_free(s);
	gsl_permutation_free(p);

	return VSTSISTUT;
}

///////////////////////////////////////////////
unsigned short *DTIData2TensorConverter2::ApplyMask(unsigned short *gradientpixels, unsigned short *b0pixels, double masksetting)
{
	// For each pixel in the gradient pixels perform a thresholding with the
	// mask setting.
	for(int i = 0; i < this->Rows * this->Columns; i++)
	{
		if(((double) b0pixels[i]) < masksetting)
		{
			gradientpixels[i] = 0;
		}
	}

	return gradientpixels;
}

///////////////////////////////////////////////
gsl_matrix *DTIData2TensorConverter2::ComputeTensor(int i, int j, DTISliceGroup *slicegroup, unsigned short *b0pixels, gsl_matrix *gradinv)
{
	int nrslices = slicegroup->GetSize();

	int position = i * this->Columns + j;
	gsl_matrix *ADC = gsl_matrix_calloc(this->NumberOfGradients - this->NumberOfB0Slices, 1);

	int index = 0;
	for(int k = 0; k < nrslices; k++)
	{
		double grad0 = gsl_matrix_get(this->Gradients, k, 0);
		double grad1 = gsl_matrix_get(this->Gradients, k, 1);
		double grad2 = gsl_matrix_get(this->Gradients, k, 2);

		// Skip slice if it's a B0 slice (gradient 0,0,0).
		if(grad0 == 0.0 && grad1 == 0.0 && grad2 == 0.0)
			continue;

		// Get pixel data for current slice. If masking was enabled, these pixels
		// may be masked.
		unsigned short *gradientpixels = slicegroup->GetSliceAt(k)->GetData();

		// Calculate ADC value.
		double adcvalue = 0.0;
		if(gradientpixels[position] != 0 && b0pixels[position] != 0)
		{
			double s  = (double) gradientpixels[position];
			double s0 = (double) b0pixels[position];
			adcvalue = -(1.0 / this->BValue) * log( s / s0 );

//			if( s / s0 <= 1.0 )
//			{
//				std::cout << "(" << i << "," << j << ") Illegal s/s0 ratio!!!!" <<
//						"s = " << s << ", s0 = " << s0 << std::endl;
//			}
		}

		// Store ADC value in ADC vector
		gsl_matrix_set(ADC, index, 0, adcvalue);
		index++;
	}

	// Calculate inner product of inverse gradient matrix and the ADC map
	// to obtain a 6-valued tensor.
	gsl_matrix *tensor = gsl_matrix_calloc(6, 1);
	gsl_linalg_matmult(gradinv, ADC, tensor);

	return tensor;
}
