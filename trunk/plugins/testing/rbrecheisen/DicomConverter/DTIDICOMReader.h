#ifndef __DTIDICOMReader_h
#define __DTIDICOMReader_h

#include "DTIUtils.h"
#include "DTIDICOMSliceGroup.h"

#include "gsl/gsl_linalg.h"

#include <iostream>
#include <vector>

#include "gdcmFile.h"
#include "gdcmImage.h"

#include <QString>

//-------------------------------------------------------------------------
//! \file   DTIDICOMReader.h
//! \class  DTIDICOMReader
//! \author Ralph Brecheisen
//! \brief  Reads DICOM Diffusion Tensor Imaging files.
//-------------------------------------------------------------------------
class DTIDICOMReader
{
public:

	//---------------------------------------------------------------------
	//! Constructor.
	//---------------------------------------------------------------------
	DTIDICOMReader();

	//---------------------------------------------------------------------
	//! Destructor.
	//---------------------------------------------------------------------
	virtual ~DTIDICOMReader();

	//---------------------------------------------------------------------
	//! Set and get methods.
	//---------------------------------------------------------------------
	void SetFilePath( const QString path )
	{
		this->FilePath = path;
	};

	void SetFilePrefix( const QString prefix )
	{
		this->FilePrefix = prefix;
	};

	void SetFileExtension( const QString extension )
	{
		this->FileExtension = extension;
	};

	__DTIMACRO_SETGET( FileFirstIndex, int );
	__DTIMACRO_SETGET( FileLastIndex, int );
	__DTIMACRO_SETGET( FileNumberOfDigits, int );
	__DTIMACRO_SETGET( GradientXGroupID, unsigned short );
	__DTIMACRO_SETGET( GradientYGroupID, unsigned short );
	__DTIMACRO_SETGET( GradientZGroupID, unsigned short );
	__DTIMACRO_SETGET( GradientXElementID, unsigned short );
	__DTIMACRO_SETGET( GradientYElementID, unsigned short );
	__DTIMACRO_SETGET( GradientZElementID, unsigned short );
	__DTIMACRO_SETGET( PatientPositionGroupID, unsigned short );
	__DTIMACRO_SETGET( PatientPositionElementID, unsigned short );
	__DTIMACRO_SETGET( ImageOrientationPatientGroupID, unsigned short );
	__DTIMACRO_SETGET( ImageOrientationPatientElementID, unsigned short );
	__DTIMACRO_SETIS( OrderedByInstanceNumber, bool );
	__DTIMACRO_SETIS( Philips, bool );

	// Philips-specific

	__DTIMACRO_SETGET( ImageAngulationAPGroupID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationAPElementID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationFHGroupID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationFHElementID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationRLGroupID, unsigned short);
	__DTIMACRO_SETGET( ImageAngulationRLElementID, unsigned short);
	__DTIMACRO_SETGET( ImagePlaneOrientationGroupID, unsigned short);
	__DTIMACRO_SETGET( ImagePlaneOrientationElementID, unsigned short);

	//---------------------------------------------------------------------
	//! Loads DICOM data from file and stores all slices that share a
	//! common slice location in a single slice group.
	//---------------------------------------------------------------------
	virtual bool LoadData();

	//---------------------------------------------------------------------
	//! Loads gradients from DICOM header.
	//---------------------------------------------------------------------
	virtual bool LoadGradients( int nrGrads );

	//---------------------------------------------------------------------
	//! Returns Patient-To-Image transformation matrix based on the
	//! ImageOrientationPatient attribute in the DICOM header.
	//---------------------------------------------------------------------
	virtual gsl_matrix * GetGradientTransform();

	//---------------------------------------------------------------------
	//! Returns pixel spacing.
	//---------------------------------------------------------------------
	virtual double * GetPixelSpacing();

	//---------------------------------------------------------------------
	//! Returns slice thickness.
	//---------------------------------------------------------------------
	virtual double GetSliceThickness();

	//---------------------------------------------------------------------
	//! Returns the output of this reader. The output is structured as a
	//! vector of DTISliceGroup objects.
	//---------------------------------------------------------------------
	virtual std::vector< DTIDICOMSliceGroup * > * GetOutput();

	//---------------------------------------------------------------------
	//! Returns the gradients extracted from the DICOM header. If
	//! LoadGradients() was never called, this is reported and NULL is
	//! returned.
	//---------------------------------------------------------------------
	virtual gsl_matrix * GetGradients();

	//---------------------------------------------------------------------
	//! Prints contents of slice groups to output stream. It also checks
	//! whether each slice group has same size.
	//---------------------------------------------------------------------
	virtual void PrintInfo( std::ostream & ostr = std::cout );

protected:

	//---------------------------------------------------------------------
	//! Adds pixel data of given DICOM file to slice group with same slice
	//! location. If the slice group cannot be found, it will be created
	//! and added to the list. Depending on the configuration, adding a
	//! slice group is determined by instance numbering or not.
	//---------------------------------------------------------------------
	virtual bool AddSlice( gdcm::File & file, gdcm::Image & image, const char * fileName );

	//---------------------------------------------------------------------
	//! Returns Patient-To-Image transformation matrix based on Philips-
	//! specific attributes.
	//---------------------------------------------------------------------
	virtual gsl_matrix * GetGradientTransformPhilips();

protected:

	std::vector< DTIDICOMSliceGroup * > * SliceGroups;

private:

//	char * FilePath;
//	char * FilePrefix;
//	char * FileExtension;

	QString FilePath;
	QString FilePrefix;
	QString FileExtension;

	int FileFirstIndex;
	int FileLastIndex;
	int FileNumberOfDigits;

	unsigned short GradientXGroupID;
	unsigned short GradientYGroupID;
	unsigned short GradientZGroupID;
	unsigned short GradientXElementID;
	unsigned short GradientYElementID;
	unsigned short GradientZElementID;
	unsigned short PatientPositionGroupID;
	unsigned short PatientPositionElementID;
	unsigned short ImageOrientationPatientGroupID;
	unsigned short ImageOrientationPatientElementID;

	// Philips-specific:

	unsigned short ImageAngulationAPGroupID;
	unsigned short ImageAngulationAPElementID;
	unsigned short ImageAngulationFHGroupID;
	unsigned short ImageAngulationFHElementID;
	unsigned short ImageAngulationRLGroupID;
	unsigned short ImageAngulationRLElementID;
	unsigned short ImagePlaneOrientationGroupID;
	unsigned short ImagePlaneOrientationElementID;

	bool OrderedByInstanceNumber;
	bool Philips;

	double * PixelSpacing;
	double SliceThickness;

	gsl_matrix * GradientMatrix;
};

#endif
