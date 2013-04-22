#ifndef __DTIDICOMSlice_h
#define __DTIDICOMSlice_h

#include "DTISlice.h"

//-------------------------------------------------------------------------
//! \file   DTIDICOMSlice.h
//! \class  DTIDICOMSlice
//! \author Ralph Brecheisen
//! \brief  Stores pixel data of DTI DICOM slice together with and instance
//!         number.
//-------------------------------------------------------------------------
class DTIDICOMSlice : public DTISlice
{
public:

	//---------------------------------------------------------------------
	//! Constructor.
	//---------------------------------------------------------------------
	DTIDICOMSlice();

	//---------------------------------------------------------------------
	//! Destructor.
	//---------------------------------------------------------------------
	virtual ~DTIDICOMSlice();

	//---------------------------------------------------------------------
	//! Set and get methods.
	//---------------------------------------------------------------------
	__DTIMACRO_SETGET( InstanceNumber, int );
	__DTIMACRO_SETGET( ImageAngulationAP, double );
	__DTIMACRO_SETGET( ImageAngulationFH, double );
	__DTIMACRO_SETGET( ImageAngulationRL, double );

	const std::string GetImagePlaneOrientation() const
	{
		return this->ImagePlaneOrientation;
	};

	void SetImageOrientationPatient( double * orientation )
	{
		for( int i = 0; i < 6; ++i )
			this->ImageOrientationPatient[i] = orientation[i];
	};

	double * GetImageOrientationPatient()
	{
		return this->ImageOrientationPatient;
	};

private:

	double * ImageOrientationPatient;

	// Philips-specific private tags
	double ImageAngulationAP;
	double ImageAngulationFH;
	double ImageAngulationRL;
	std::string ImagePlaneOrientation;

	int InstanceNumber;
};

#endif
