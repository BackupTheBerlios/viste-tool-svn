#ifndef __DTISliceGroup_h
#define __DTISliceGroup_h

#include "DTISlice.h"
#include "DTIUtils.h"

#include <vector>

using namespace std;

//---------------------------------------------------------------------------
//! \file   DTISliceGroup.h
//! \class  DTISliceGroup
//! \author Ralph Brecheisen
//! \brief  Stores DTI slices that share common slice location.
//---------------------------------------------------------------------------
class DTISliceGroup
{
public:

	//---------------------------------------------------------------------------
	//! Constructor.
	//---------------------------------------------------------------------------
	DTISliceGroup()
	{
		this->SliceLocation = 0.0;
		this->Slices = new vector<DTISlice *>;
	}

	//---------------------------------------------------------------------------
	//! Destructor.
	//---------------------------------------------------------------------------
	~DTISliceGroup()
	{
		if(this->Slices)
		{
			this->Slices->clear();
			delete this->Slices;
		}
	}

	//---------------------------------------------------------------------------
	//! Set/get methods.
	//---------------------------------------------------------------------------
	__DTIMACRO_SETGET(SliceLocation, double);

	//---------------------------------------------------------------------------
	//! Adds pixel data to slice group. If required, the slices are ordered by
	//! instance number.
	//---------------------------------------------------------------------------
	virtual bool AddSlice(DTISlice *slice)
	{
		const char *func = "DTISliceGroup::AddSlice";

		// Just append the slice data to the end of the list. This will cause the files
		// to be ordered according to filename.
		this->Slices->push_back(slice);
		return true;
	}

	//---------------------------------------------------------------------------
	//! Returns slice at given position.
	//---------------------------------------------------------------------------
	DTISlice *GetSliceAt(int position)
	{
		const char *func = "DTISliceGroup::GetSliceAt";

		// Check if position not out of bounds.
		if(position < 0 || position >= this->GetSize())
		{
			__DTIMACRO_LOG(func << ": Position out of range" << endl, ERROR, DTIUtils::LogLevel);
			return 0;
		}

		return this->Slices->at(position);
	}

	//---------------------------------------------------------------------------
	//! Returns size of slice group.
	//---------------------------------------------------------------------------
	int GetSize()
	{
		return this->Slices->size();
	}

protected:

	double SliceLocation;
	vector<DTISlice *> *Slices;
};

#endif
