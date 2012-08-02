#ifndef __DTIDICOMPhilipsReader_h
#define __DTIDICOMPhilipsReader_h

#include "DTIDICOMReader.h"
#include "DTIUtils.h"

// Some flipping matrices for different patient positions.
#define __SAGGITAL    {{0,  0, -1}, { 0, -1, 0}, {1, 0, 0}}
#define __CORONAL     {{0, -1,  0}, { 0,  0, 1}, {1, 0, 0}}
#define __TRANSVERSAL {{0, -1,  0}, {-1,  0, 0}, {0, 0, 1}}

// PI
#define PI 3.1415926535f

//-------------------------------------------------------------------------
//! \file   DTIDICOMPhilipsReader.h
//! \class  DTIDICOMPhilipsReader
//! \author Ralph Brecheisen
//! \brief  Reads Philips-specific DICOM Diffusion Tensor Imaging files.
//-------------------------------------------------------------------------
class DTIDICOMPhilipsReader : public DTIDICOMReader
{
public:

	//---------------------------------------------------------------------
	//! Constructor.
	//---------------------------------------------------------------------
	DTIDICOMPhilipsReader();

	//---------------------------------------------------------------------
	//! Destructor.
	//---------------------------------------------------------------------
	virtual ~DTIDICOMPhilipsReader();

	//---------------------------------------------------------------------
	//! Set and get methods.
	//---------------------------------------------------------------------
	__DTIMACRO_SETGET( ImageAngulationAPGroupID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationAPElementID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationFHGroupID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationFHElementID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationRLGroupID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationRLElementID, unsigned short);
	__DTIMACRO_SETGET( ImagePlaneOrientationGroupID, unsigned short);
	__DTIMACRO_SETGET( ImagePlaneOrientationElementID, unsigned short);

	//---------------------------------------------------------------------
	//! Returns gradient transformation matrix based on Philips-specific
	//! DICOM attributes.
	//---------------------------------------------------------------------
	virtual gsl_matrix * GetGradientTransform();

private:

	unsigned short ImageAngulationAPGroupID;
	unsigned short ImageAngulationAPElementID;
	unsigned short ImageAngulationFHGroupID;
	unsigned short ImageAngulationFHElementID;
	unsigned short ImageAngulationRLGroupID;
	unsigned short ImageAngulationRLElementID;
	unsigned short ImagePlaneOrientationGroupID;
	unsigned short ImagePlaneOrientationElementID;
};

#endif
