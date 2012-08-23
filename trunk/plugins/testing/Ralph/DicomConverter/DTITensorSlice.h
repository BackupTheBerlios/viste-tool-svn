#ifndef __DTITensorSlice_h
#define __DTITensorSlice_h

#include "DTIUtils.h"
#include "DTISlice.h"
#include "gsl/gsl_linalg.h"

//---------------------------------------------------------------------------
//! \file   DTITensorSlice.h
//! \class  DTITensorSlice
//! \author Ralph Brecheisen
//! \brief  Stores tensors in 2D array.
//---------------------------------------------------------------------------
class DTITensorSlice : public DTISlice
{
public:

	//-------------------------------------------------------------------
	//! Constructor.
	//-------------------------------------------------------------------
	DTITensorSlice()
	{
		this->Rows    = 0;
		this->Columns = 0;
		this->Tensors = 0;
	}

	//-------------------------------------------------------------------
	//! Destructor.
	//-------------------------------------------------------------------
	~DTITensorSlice()
	{
		if(this->Tensors != 0)
			for(int i = 0; i < (this->Rows * this->Columns); i++)
				gsl_matrix_free(this->Tensors[i]);
	}

	//-------------------------------------------------------------------
	//! Set/get methods.
	//-------------------------------------------------------------------
	__DTIMACRO_SETGET(Rows,    int);
	__DTIMACRO_SETGET(Columns, int);

	//-------------------------------------------------------------------
	//! Sets tensor at given row/column position.
	//-------------------------------------------------------------------
	virtual void SetTensorAt(gsl_matrix *tensor, int row, int column)
	{
		const char *func = "DTITensorSlice::SetTensorAt";
		if(this->Rows == 0 || this->Columns == 0)
		{
			__DTIMACRO_LOG(func << ": No rows or columns specified" << endl, ERROR, DTIUtils::LogLevel);
			return;
		}

		// If not already done so, allocate memory space for rows x columns 
		// tensors.
		if(this->Tensors == 0)
			this->Tensors = new gsl_matrix*[this->Rows * this->Columns];

		this->Tensors[row * this->Columns + column] = tensor;
	}

	//-------------------------------------------------------------------
	//! Returns tensor at given row/column position.
	//-------------------------------------------------------------------
	virtual gsl_matrix *GetTensorAt(int row, int column)
	{
		const char *func = "DTITensorSlice::GetTensorAt";
		if(this->Tensors == 0)
		{
			__DTIMACRO_LOG(func << ": Tensor slice still empty" << endl, ERROR, DTIUtils::LogLevel);
			return 0;
		}

		return this->Tensors[row * this->Columns + column];
	}

	//-------------------------------------------------------------------
	//! Not relevant for tensor slices.
	//-------------------------------------------------------------------
	virtual double *GetPixelSpacing()
	{
		return 0;
	}

	//-------------------------------------------------------------------
	//! Not relevant for tensor slices.
	//-------------------------------------------------------------------
	virtual double GetSliceThickness()
	{
		return 0.0;
	}

protected:

	int Rows;
	int Columns;
	gsl_matrix **Tensors;
};

#endif