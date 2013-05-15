#ifndef __DTIData2TensorConverter_h
#define __DTIData2TensorConverter_h

#include "DTIUtils.h"
#include "DTITensorSlice.h"
#include "DTISlice.h"
#include "DTISliceGroup.h"
#include "gsl/gsl_linalg.h"
#include <vector>

using namespace std;

//---------------------------------------------------------------------------
//! \file   DTIData2TensorConverter.h
//! \class  DTIData2TensorConverter
//! \author Ralph Brecheisen
//! \brief  Converts DTI slice data to tensor data.
//---------------------------------------------------------------------------
class DTIData2TensorConverter
{
public:

	//-------------------------------------------------------------------
	//! Constructor.
	//-------------------------------------------------------------------
	DTIData2TensorConverter()
	{
		this->Input                        = 0;
		this->Gradients                    = 0;
		this->NumberOfGradients            = 0;
		this->NumberOfB0Slices             = 0;
		this->TensorSlices                 = 0;
		this->B0SliceIndexArray            = 0;
		this->GradientSliceFirstIndexArray = 0;
		this->GradientSliceLastIndexArray  = 0;
		this->MaskEnabled                  = true;
		this->MaskValue                    = 0.0;
		this->BValue                       = 0.0;
		this->Rows                         = 0;
		this->Columns                      = 0;
	}

	//-------------------------------------------------------------------
	//! Destructor.
	//-------------------------------------------------------------------
	~DTIData2TensorConverter()
	{
		if(this->Input != 0)
		{
			this->Input->clear();
			delete this->Input;
		}

		if(this->Gradients != 0)
			gsl_matrix_free(this->Gradients);
	}

	//-------------------------------------------------------------------
	//! Sets the input data.
	//-------------------------------------------------------------------
	virtual void SetInput(vector<DTISliceGroup *> *input)
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

	//-------------------------------------------------------------------
	//! Sets the gradient matrix to use for the tensor calculation. Each
	//! gradient vector is converted to 6-components.
	//-------------------------------------------------------------------
	virtual void SetGradients(gsl_matrix * gradients)
	{
		if(this->Gradients != 0)
			gsl_matrix_free(this->Gradients);

		// Get number of gradients in matrix.
		this->NumberOfGradients = (int) gradients->size1;

		this->Gradients = 0;
		this->Gradients = gsl_matrix_calloc(this->NumberOfGradients, 6);

		// Construct 6-component gradient matrix from the 3-component
		// matrix. Don't ask why...
		for(int i = 0; i < this->NumberOfGradients; i++)
		{
			double grad0 = gsl_matrix_get(gradients, i, 0);
			double grad1 = gsl_matrix_get(gradients, i, 1);
			double grad2 = gsl_matrix_get(gradients, i, 2);

			double len = grad0*grad0 + grad1*grad1 + grad2*grad2;

			gsl_matrix_set(this->Gradients, i, 0, grad0 * grad0 / len);
			gsl_matrix_set(this->Gradients, i, 1, grad1 * grad1 / len);
			gsl_matrix_set(this->Gradients, i, 2, grad2 * grad2 / len);
			gsl_matrix_set(this->Gradients, i, 3, 2 * grad0 * grad1 / len);
			gsl_matrix_set(this->Gradients, i, 4, 2 * grad1 * grad2 / len);
			gsl_matrix_set(this->Gradients, i, 5, 2 * grad0 * grad2 / len);
		}
	}

	//-------------------------------------------------------------------
	//! Sets number of B0 slices in single slice group of dataset.
	//-------------------------------------------------------------------
	virtual void SetNumberOfB0Slices(int numberslices)
	{
		this->NumberOfB0Slices = numberslices;
	}

	//-------------------------------------------------------------------
	//! Sets position index of B0 in single slice group. First position
	//! index in slice group is always 0.
	//-------------------------------------------------------------------
	virtual void SetB0SliceIndex(int index)
	{
		const char *func = "DTIData2TensorConverter::SetB0SliceIndex";

		// Check if number of B0 slices has been set and equals 1.
		if(this->NumberOfB0Slices == 0)
		{
			__DTIMACRO_LOG(func << ": Set number of B0 slices first" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->NumberOfB0Slices > 1)
		{
			__DTIMACRO_LOG(func << ": Number of B0 slices should be 1" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->B0SliceIndexArray == 0)
			this->B0SliceIndexArray  = new int[1];

		this->B0SliceIndexArray[0] = index;
	}

	//-------------------------------------------------------------------
	//! Sets position indices of multiple B0 slices in single slice group.
	//! First position index in slice group is always 0. 
	//-------------------------------------------------------------------
	virtual void SetB0SliceIndexArray(int *indices)
	{
		const char *func = "DTIData2TensorConverter::SetB0SliceIndexArray";

		if(this->NumberOfB0Slices == 0)
		{
			__DTIMACRO_LOG(func << ": Set number of B0 slices first" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->NumberOfB0Slices == 1)
		{
			__DTIMACRO_LOG(func << ": Number of B0 slices should be > 1" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		this->B0SliceIndexArray = indices;
	}

	//-------------------------------------------------------------------
	//! Sets position index of first gradient slice in slice group.
	//-------------------------------------------------------------------
	virtual void SetGradientSliceFirstIndex(int index)
	{
		const char *func = "DTIData2TensorConverter::SetGradientSliceFirstIndex";
		if(this->NumberOfB0Slices == 0)
		{
			__DTIMACRO_LOG(func << ": Set number of B0 slices first" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->NumberOfB0Slices > 1)
		{
			__DTIMACRO_LOG(func << ": Number of B0 slices should be 1" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->GradientSliceFirstIndexArray == 0)
			this->GradientSliceFirstIndexArray = new int[1];

		this->GradientSliceFirstIndexArray[0] = index;
	}

	//-------------------------------------------------------------------
	//! Sets position indices of first gradient slices in single slice 
	//! group in case of multiple B0 slices.
	//-------------------------------------------------------------------
	virtual void SetGradientSliceFirstIndexArray(int *indices)
	{
		const char *func = "DTIData2TensorConverter::SetGradientSliceFirstIndexArray";
		if(this->NumberOfB0Slices == 0)
		{
			__DTIMACRO_LOG(func << ": Set number of B0 slices first" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->NumberOfB0Slices == 1)
		{
			__DTIMACRO_LOG(func << ": Number of B0 slices should be > 1" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		this->GradientSliceFirstIndexArray = indices;
	}

	//-------------------------------------------------------------------
	//! Sets position index of last gradient slice in slice group.
	//-------------------------------------------------------------------
	virtual void SetGradientSliceLastIndex(int index)
	{
		const char *func = "DTIData2TensorConverter::SetGradientSliceLastIndex";
		if(this->NumberOfB0Slices == 0)
		{
			__DTIMACRO_LOG(func << ": Set number of B0 slices first" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->NumberOfB0Slices > 1)
		{
			__DTIMACRO_LOG(func << ": Number of B0 slices should be 1" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->GradientSliceLastIndexArray == 0)
			this->GradientSliceLastIndexArray = new int[1];

		this->GradientSliceLastIndexArray[0] = index;
	}

	//-------------------------------------------------------------------
	//! Sets position indices of last gradient slices in single slice 
	//! group in case of multiple B0 slices.
	//-------------------------------------------------------------------
	virtual void SetGradientSliceLastIndexArray(int *indices)
	{
		const char *func = "DTIData2TensorConverter::SetGradientSliceLastIndexArray";
		if(this->NumberOfB0Slices == 0)
		{
			__DTIMACRO_LOG(func << ": Set number of B0 slices first" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		if(this->NumberOfB0Slices == 1)
		{
			__DTIMACRO_LOG(func << ": Number of B0 slices should be > 1" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		this->GradientSliceLastIndexArray = indices;
	}

	//-------------------------------------------------------------------
	//! Sets masking state.
	//-------------------------------------------------------------------
	virtual void SetMaskEnabled(bool enabled)
	{
		this->MaskEnabled = enabled;
	}

	//-------------------------------------------------------------------
	//! Sets mask value.
	//-------------------------------------------------------------------
	virtual void SetMaskValue(double maskvalue)
	{
		this->MaskValue = maskvalue;
	}

	//-------------------------------------------------------------------
	//! Sets B-value.
	//-------------------------------------------------------------------
	virtual void SetBValue(double bvalue)
	{
		this->BValue = bvalue;
	}

	//-------------------------------------------------------------------
	//! Executes converter by computing a tensor volume from the input
	//! data and associated gradient information.
	//-------------------------------------------------------------------
	virtual bool Execute()
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

		if(this->B0SliceIndexArray == 0)
		{
			__DTIMACRO_LOG(func << ": No B0 slice index(es) specified" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		if(this->GradientSliceFirstIndexArray == 0)
		{
			__DTIMACRO_LOG(func << ": No gradient slice first index(es) specified" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		if(this->GradientSliceLastIndexArray == 0)
		{
			__DTIMACRO_LOG(func << ": No gradient slice last index(es) specified" << endl, ERROR, DTIUtils::LogLevel);
			return false;
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

		// Compute inverse of gradient matrix. Because this is a non-square matrix we
		// need to use the pseudo-inverse.
		gsl_matrix *gradientsinverse = this->CalculatePseudoInverse(this->Gradients);

		// For each slice group...
		vector<DTISliceGroup *>::iterator iter;
		int tensorcount = 0;

		for(iter = this->Input->begin(); iter != this->Input->end(); iter++)
		{
			// Get current slice group and its size.
			DTISliceGroup *slicegroup = (*iter);
			int numberofslices        = slicegroup->GetSize();

			// Get first B0 slice in this slice group. 
			DTISlice *b0slice = slicegroup->GetSliceAt(this->B0SliceIndexArray[0]);

			// Get slice's pixel data.
			unsigned short *b0pixels = b0slice->GetData();

			// Apply mask to each gradient slice in the slice group. Skip all B0
			// slices (if there are more than one).
			if(this->MaskEnabled)
			{
				for(int k = 0; k < numberofslices; k++)
				{
					// Skip by default. If k is inside any of the gradient slice ranges do
					// not skip but continue with applying the mask.
					bool skip = true;

					for(int i = 0; i < this->NumberOfB0Slices; i++)
					{
						if(	k >= this->GradientSliceFirstIndexArray[i] &&
							k <= this->GradientSliceLastIndexArray[i])
						{
							skip = false;
						}
					}

					// Value k was outside all of the ranges so it must have been
					// referring to a B0 slice. Skip this iteration therefore.
					if(skip)
						continue;

					// Get this slice's pixel data.
					unsigned short *gradientpixels = slicegroup->GetSliceAt(k)->GetData();

					// Apply the mask to the pixel data.
					gradientpixels = this->ApplyMask(gradientpixels, b0pixels, masksetting);
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
					int position = i * this->Columns + j;	

					// Create ADC map that contains elements for each gradient direction.
					// We use a GSL matrix for easy linear algebra calculations.
					gsl_matrix *ADC = gsl_matrix_calloc(this->NumberOfGradients, 1);

					// Get the (i,j)-th pixel of each slice in the slice group. We can skip
					// the B0 slice(s).
					int index = 0;

					for(int k = 0; k < numberofslices; k++)
					{
						// Skip by default. If k is inside any of the gradient slice ranges do
						// not skip but continue with applying the mask.
						bool skip = true;

						for(int i = 0; i < this->NumberOfB0Slices; i++)
						{
							if(	k >= this->GradientSliceFirstIndexArray[i] &&
								k <= this->GradientSliceLastIndexArray[i])
							{
								skip = false;
							}
						}

						// Value k was outside all of the ranges so it must have been
						// referring to a B0 slice. Skip this iteration therefore.
						if(skip)
							continue;

						// Get pixel data for current slice.
						unsigned short *gradientpixels = slicegroup->GetSliceAt(k)->GetData();

						// Calculate ADC value.
						double adcvalue = 0.0;

						if(gradientpixels[position] != 0 && b0pixels[position] != 0)
							adcvalue = -(1.0 / this->BValue) * log(gradientpixels[position] / ((double) b0pixels[position]));

						gsl_matrix_set(ADC, index, 0, adcvalue);
						index++;
					}

					// Calculate inner product of inverse gradient matrix and the ADC map
					// to obtain a 6-valued tensor.
					gsl_matrix *tensor = gsl_matrix_calloc(6, 1);

					gsl_linalg_matmult(gradientsinverse, ADC, tensor);

					// Store the tensor in the tensor slice.
					tensorslice->SetTensorAt(tensor, i, j);

					// Cleanup ADC map.
					gsl_matrix_free(ADC);
				}
			}

			// Now that all pixels of the tensor slice have a tensor value we can store the
			// tensor slice in the list.
			if(this->TensorSlices == 0)
				this->TensorSlices = new vector<DTITensorSlice *>;

			this->TensorSlices->push_back(tensorslice);

			__DTIMACRO_LOG(func << ": Created tensor slice " << tensorcount << endl, ALWAYS, DTIUtils::LogLevel);
			tensorcount++;
		}

		// Clean up inverse gradient matrix.
		gsl_matrix_free(gradientsinverse);

		return true;
	}

	//-------------------------------------------------------------------
	//! Returns the tensor output.
	//-------------------------------------------------------------------
	vector<DTITensorSlice *> *GetOutput()
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

protected:

	//-------------------------------------------------------------------
	//! Calculates mask setting based on all B0 slices in the volume. If
	//! multiple B0 slices are in a single slice group only the first B0
	//! will be used.
	//-------------------------------------------------------------------
	double CalculateMaskSetting()
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
		int count = 0;

		for(iter = this->Input->begin(); iter != this->Input->end(); iter++)
		{
			DTISliceGroup *slicegroup = (*iter);
			unsigned short *b0pixels = slicegroup->GetSliceAt(this->B0SliceIndexArray[0])->GetData();

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

	//-------------------------------------------------------------------
	//! Calculates pseudo inverse of non-square matrix.
	//-------------------------------------------------------------------
	gsl_matrix *CalculatePseudoInverse(gsl_matrix *matrix)
	{
		// Jacobi SVD
		gsl_matrix *U = gsl_matrix_calloc(this->NumberOfGradients, 6);
		gsl_matrix *V = gsl_matrix_calloc(6, 6);
		gsl_vector *s = gsl_vector_calloc(6);
		gsl_matrix_memcpy(U, matrix);
		gsl_linalg_SV_decomp_jacobi(U, V, s);

		gsl_matrix *S = gsl_matrix_calloc(6, 6);
		for(int i = 0; i < 6; i++)
			gsl_matrix_set(S, i, i, gsl_vector_get(s, i));

		// transpose S, U and V
		gsl_matrix *ST = gsl_matrix_calloc(6, 6);
		gsl_matrix *UT = gsl_matrix_calloc(6, this->NumberOfGradients);
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
		gsl_matrix *STUT = gsl_matrix_calloc(6, this->NumberOfGradients);
		gsl_matrix *STSISTUT = gsl_matrix_calloc(6, this->NumberOfGradients);
		gsl_matrix *VSTSISTUT = gsl_matrix_calloc(6, this->NumberOfGradients);
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

	//-------------------------------------------------------------------
	//! Applies mask to gradient pixels using the B0 pixels and the given
	//! mask setting.
	//-------------------------------------------------------------------
	unsigned short *ApplyMask(unsigned short *gradientpixels, unsigned short *b0pixels, double masksetting)
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

	vector<DTISliceGroup *>  *Input;
	vector<DTITensorSlice *> *TensorSlices;
	gsl_matrix *Gradients;

	int    NumberOfB0Slices;
	int    NumberOfGradients;
	int   *B0SliceIndexArray;
	int   *GradientSliceFirstIndexArray;
	int   *GradientSliceLastIndexArray;
	int    Rows;
	int    Columns;
	bool   MaskEnabled;
	double MaskValue;
	double BValue;
};

#endif