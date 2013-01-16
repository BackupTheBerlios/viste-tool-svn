#include "DTIDICOMSlice.h"

//---------------------------------------------------------------------
DTIDICOMSlice::DTIDICOMSlice() : DTISlice()
{
	this->InstanceNumber = 0;
	this->ImageAngulationAP = 0.0;
	this->ImageAngulationFH = 0.0;
	this->ImageAngulationRL = 0.0;
	this->ImagePlaneOrientation = "";

	this->ImageOrientationPatient = new double[6];
	for( int i = 0; i < 6; ++i )
		this->ImageOrientationPatient[i] = 0.0;
}

//---------------------------------------------------------------------
DTIDICOMSlice::~DTIDICOMSlice()
{
}
