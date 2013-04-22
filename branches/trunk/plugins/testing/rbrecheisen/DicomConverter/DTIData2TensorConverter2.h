#ifndef __DTIData2TensorConverter2_h
#define __DTIData2TensorConverter2_h

#include "DTIUtils.h"
#include "DTITensorSlice.h"
#include "DTISlice.h"
#include "DTISliceGroup.h"
#include "DTIData2TensorConverter.h"
#include "gsl/gsl_linalg.h"
#include <vector>

using namespace std;

//---------------------------------------------------------------------------
//! \file   DTIData2TensorConverter2.h
//! \class  DTIData2TensorConverter2
//! \author Ralph Brecheisen
//! \brief  Converts DTI slice data to tensor data.
//---------------------------------------------------------------------------
class DTIData2TensorConverter2 : public DTIData2TensorConverter
{
public:

	//-------------------------------------------------------------------
	//! Constructor.
	//-------------------------------------------------------------------
	DTIData2TensorConverter2();

	//-------------------------------------------------------------------
	//! Destructor.
	//-------------------------------------------------------------------
	~DTIData2TensorConverter2();

	//-------------------------------------------------------------------
	//! Sets the input data.
	//-------------------------------------------------------------------
	virtual void SetInput(vector<DTISliceGroup *> *input);

	//-------------------------------------------------------------------
	//! Sets the gradient matrix to use for the tensor calculation. Each
	//! gradient vector is converted to 6-components.
	//-------------------------------------------------------------------
	virtual void SetGradients(gsl_matrix * gradients);

	//-------------------------------------------------------------------
	//! Sets number of B0 slices in single slice group of dataset.
	//-------------------------------------------------------------------
	virtual void SetNumberOfB0Slices(int numberslices);

	//-------------------------------------------------------------------
	//! Sets first B0 slice index.
	//-------------------------------------------------------------------
	virtual void SetB0SliceFirstIndex(int index);

	//-------------------------------------------------------------------
	//! Sets masking state.
	//-------------------------------------------------------------------
	virtual void SetMaskEnabled(bool enabled);

	//-------------------------------------------------------------------
	//! Sets mask value.
	//-------------------------------------------------------------------
	virtual void SetMaskValue(double maskvalue);

	//-------------------------------------------------------------------
	//! Sets B-value.
	//-------------------------------------------------------------------
	virtual void SetBValue(double bvalue);

	//-------------------------------------------------------------------
	//! Executes converter by computing a tensor volume from the input
	//! data and associated gradient information.
	//-------------------------------------------------------------------
	virtual bool Execute();

	//-------------------------------------------------------------------
	//! Returns the tensor output.
	//-------------------------------------------------------------------
	vector<DTITensorSlice *> *GetOutput();

protected:

	//-------------------------------------------------------------------
	//! Calculates mask setting based on all B0 slices in the volume. If
	//! multiple B0 slices are in a single slice group only the first B0
	//! will be used.
	//-------------------------------------------------------------------
	double CalculateMaskSetting();

	//-------------------------------------------------------------------
	//! Calculates pseudo inverse of non-square matrix.
	//-------------------------------------------------------------------
	gsl_matrix *CalculatePseudoInverse(gsl_matrix *matrix);

	//-------------------------------------------------------------------
	//! Applies mask to gradient pixels using the B0 pixels and the given
	//! mask setting.
	//-------------------------------------------------------------------
	unsigned short *ApplyMask(unsigned short *gradientpixels, unsigned short *b0pixels, double masksetting);

	//-------------------------------------------------------------------
	//! Computes 6-valued tensor at position (row,col). Tensor is stored 
        //! in (1 x 6) matrix. Following parameters are used:
        //! i: Image row
        //! j: Image column
	//! slicegroup: Current slice group
	//! b0pixels: S0 signal pixels
	//! gradinv: Inverse of extended gradient matrix (nrgrads x 6)
	//-------------------------------------------------------------------
	gsl_matrix *ComputeTensor(
		int row, 
		int col, 
		DTISliceGroup *slicegroup, 
		unsigned short *b0pixels, gsl_matrix *gradinv);

	vector<DTISliceGroup *>  *Input;
	vector<DTITensorSlice *> *TensorSlices;
	gsl_matrix *Gradients;
	gsl_matrix *GradientsExtended;

	int    NumberOfB0Slices;
	int    NumberOfGradients;
	int    Rows;
	int    Columns;
	int    B0SliceFirstIndex;
	bool   MaskEnabled;
	double MaskValue;
	double BValue;
};

#endif