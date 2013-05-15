#ifndef __DTISlice_h
#define __DTISlice_h

#include "DTIUtils.h"

//---------------------------------------------------------------------------
//! \file   DTISlice.h
//! \class  DTISlice
//! \author Ralph Brecheisen
//! \brief  Represents single DTI slice.
//---------------------------------------------------------------------------
class DTISlice
{
public:

	//-------------------------------------------------------------------
	//! Constructor.
	//-------------------------------------------------------------------
	DTISlice()
	{
		this->Data = NULL;
		this->SliceLocation = 0.0;
		this->SliceThickness = 0.0;
		this->Rows = 0;
		this->Columns = 0;
		this->PixelSpacing = new double[2];
		this->PixelSpacing[0] = 0.0;
		this->PixelSpacing[1] = 0.0;
	}

	//-------------------------------------------------------------------
	//! Destructor.
	//-------------------------------------------------------------------
	~DTISlice()
	{
	}

	//-------------------------------------------------------------------
	//! Set/get methods.
	//-------------------------------------------------------------------
	virtual void SetData(unsigned short *data)
	{
		this->Data = data;
	}

	virtual void SetSliceLocation(double location)
	{
		this->SliceLocation = location;
	}

	virtual void SetSliceThickness(double thickness)
	{
		this->SliceThickness = thickness;
	}

	virtual void SetPixelSpacing(double * spacing)
	{
		this->PixelSpacing[0] = spacing[0];
		this->PixelSpacing[1] = spacing[1];
	}

	virtual void SetRows(int rows)
	{
		this->Rows = rows;
	}

	virtual void SetColumns(int columns)
	{
		this->Columns = columns;
	}

	virtual unsigned short *GetData() { return this->Data; }
	virtual double GetSliceLocation() { return this->SliceLocation; }
	virtual double GetSliceThickness() { return this->SliceThickness; }
	virtual double *GetPixelSpacing() { return this->PixelSpacing; }
	virtual int GetRows() { return this->Rows; }
	virtual int GetColumns() { return this->Columns; }

protected:

	unsigned short *Data;
	double SliceLocation;
	double SliceThickness;
	double *PixelSpacing;
	int Rows;
	int Columns;
};

#endif
