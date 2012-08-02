#ifndef __DTIDICOMSliceGroup_h
#define __DTIDICOMSliceGroup_h

#include "DTISliceGroup.h"
#include "DTIUtils.h"
#include "DTIDICOMSlice.h"

//---------------------------------------------------------------------------
//! \file   DTIDICOMSliceGroup.h
//! \class  DTIDICOMSliceGroup
//! \author Ralph Brecheisen
//! \brief  Stores DTI DICOM slices that share common slice location.
//---------------------------------------------------------------------------
class DTIDICOMSliceGroup : public DTISliceGroup
{
public:

	//---------------------------------------------------------------------------
	//! Constructor.
	//---------------------------------------------------------------------------
	DTIDICOMSliceGroup() : DTISliceGroup()
	{
		this->OrderedByInstanceNumber = false;
	}

	//---------------------------------------------------------------------------
	//! Destructor.
	//---------------------------------------------------------------------------
	~DTIDICOMSliceGroup()
	{
	}

	//---------------------------------------------------------------------------
	//! Set/get methods.
	//---------------------------------------------------------------------------
	__DTIMACRO_SETIS(OrderedByInstanceNumber, bool);

	//---------------------------------------------------------------------------
	//! Adds DTI DICOM slice to list.
	//---------------------------------------------------------------------------
	virtual bool AddSlice(DTIDICOMSlice *slice)
	{
		const char *func = "DTIDICOMSliceGroup::AddSlice";

		if(this->IsOrderedByInstanceNumber())
		{
			// Find first instance number greater than this one. If found
			// we insert it just before.
			long instancenumber = slice->GetInstanceNumber();

			vector<DTISlice *>::iterator iter;
			bool found = false;

			for(iter = this->Slices->begin(); iter != this->Slices->end(); iter++)
			{
				DTIDICOMSlice *tmp = (DTIDICOMSlice *) (*iter);

				if(tmp->GetInstanceNumber() > instancenumber)
				{
					this->Slices->insert(iter, slice);
					found = true;
					break;
				}
			}

			// If slice with larger instance number can't be found, this one
			// must be greatest so append it to the end of the list.
			if(!found)
				this->Slices->push_back(slice);
			return true;
		}

		// Without instance number ordering we just add the slice
		// to the back of the list. This causes the slices to be
		// ordering by filename.
		this->Slices->push_back(slice);
		return true;
	}

protected:

	bool OrderedByInstanceNumber;
};

#endif
