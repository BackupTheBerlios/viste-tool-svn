#include "DTIDICOMPhilipsReader.h"

#include <gdcmPrivateTag.h>

//---------------------------------------------------------------------
DTIDICOMPhilipsReader::DTIDICOMPhilipsReader()
{
	this->ImageAngulationAPGroupID       = 8197;
	this->ImageAngulationAPElementID     = 4096;	// Hexadecimal: (2005,1000)
	this->ImageAngulationFHGroupID       = 8197;
	this->ImageAngulationFHElementID     = 4097;	// Hexadecimal: (2005,1001)
	this->ImageAngulationRLGroupID       = 8197;
	this->ImageAngulationRLElementID     = 4098;	// Hexadecimal: (2005,1002)
	this->ImagePlaneOrientationGroupID   = 8193;
	this->ImagePlaneOrientationElementID = 4107;	// Hexadecimal: (2001,100B)
}

//---------------------------------------------------------------------
DTIDICOMPhilipsReader::~DTIDICOMPhilipsReader()
{
}

//---------------------------------------------------------------------
gsl_matrix * DTIDICOMPhilipsReader::GetGradientTransform()
{
	const char * func = "DTIDICOMPhilipsReader::GetGradientTransform() ";

	gdcm::PrivateTag tag[4];
	tag[0] = gdcm::PrivateTag( 0x2005, 0x00, "Philips MR Imaging DD 005" ); // ImageAngulationAP
	tag[1] = gdcm::PrivateTag( 0x2005, 0x01, "Philips MR Imaging DD 005" ); // ImageAngulationFH
	tag[2] = gdcm::PrivateTag( 0x2005, 0x02, "Philips MR Imaging DD 005" ); // ImageAngulationRL
	tag[3] = gdcm::PrivateTag( 0x2005, 0x0B, "Philips MR Imaging DD 005" ); // ImagePlaneOrientation

	DTISliceGroup * group = this->SliceGroups->at( 0 );
	DTISlice * slice = group->GetSliceAt( 0 );
	DTIDICOMSlice * dicomSlice = static_cast<DTIDICOMSlice *>( slice );

	return NULL;
}
