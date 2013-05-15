#include "DTIDICOMReader.h"
#include "DTIDICOMSlice.h"
#include "DTIDICOMSliceGroup.h"
#include "DTIUtils.h"

#include <gdcmImageReader.h>
#include <gdcmAttribute.h>
#include <gdcmDicts.h>
#include <gdcmDict.h>
#include <gdcmGlobal.h>
#include <gdcmPrivateTag.h>

#include <QString>

#include <stdlib.h>

//---------------------------------------------------------------------------
DTIDICOMReader::DTIDICOMReader()
{
	this->SliceGroups = new std::vector< DTIDICOMSliceGroup * >;

//	this->FilePath                         = NULL;
//	this->FilePrefix                       = NULL;
//	this->FileExtension                    = NULL;

	this->FileFirstIndex                   = -1;
	this->FileLastIndex                    = -1;
	this->FileNumberOfDigits               = -1;

	this->OrderedByInstanceNumber          = false;
	this->Philips                          = false;

	this->GradientXGroupID                 = 25;
	this->GradientXElementID               = 4283;		// Hexadecimal: (0019,10BB)
	this->GradientYGroupID                 = 25;
	this->GradientYElementID               = 4284;		// Hexadecimal: (0019,10BC)
	this->GradientZGroupID                 = 25;
	this->GradientZElementID               = 4285;		// Hexadecimal: (0019,10BD)
	this->PatientPositionGroupID           = 24;
	this->PatientPositionElementID         = 20736;		// Hexadecimal: (0018,5100)
	this->ImageOrientationPatientGroupID   = 32;
	this->ImageOrientationPatientElementID = 55;		// Hexadecimal: (0020,0037)

	this->ImageAngulationAPGroupID         = 8197;
	this->ImageAngulationAPElementID       = 4096;	// Hexadecimal: (2005,1000)
	this->ImageAngulationFHGroupID         = 8197;
	this->ImageAngulationFHElementID       = 4097;	// Hexadecimal: (2005,1001)
	this->ImageAngulationRLGroupID         = 8197;
	this->ImageAngulationRLElementID       = 4098;	// Hexadecimal: (2005,1002)
	this->ImagePlaneOrientationGroupID     = 8193;
	this->ImagePlaneOrientationElementID   = 4107;	// Hexadecimal: (2001,100B)

	this->GradientMatrix                   = NULL;

	this->PixelSpacing = new double[2];
	this->PixelSpacing[0] = 0.0;
	this->PixelSpacing[1] = 0.0;

	this->SliceThickness  = 0.0;
}

//---------------------------------------------------------------------------
DTIDICOMReader::~DTIDICOMReader()
{
	this->SliceGroups->clear();
	delete this->SliceGroups;
}

//---------------------------------------------------------------------------
bool DTIDICOMReader::LoadData()
{
	const char * func = "DTIDICOMReader::LoadData";

	// Append forward or backslash to filepath if necessary.
	if( this->FilePath.contains( '/' ) )
	{
		if( this->FilePath.endsWith( '/' ) == false )
			this->FilePath.append( '/' );
	}
	else if( this->FilePath.contains( '\\' ) )
	{
		if( this->FilePath.endsWith( '\\' ) == false )
			this->FilePath.append( '\\' );
	}
	else
	{
		std::cout << func << " filepath has no slashes, weird....";
		std::cout << "getting out of here!" << std::endl;
		return false;
	}

	// Filename first and last indices are mandatory.
	if(this->FileFirstIndex == -1)
	{
		std::cout << func << ": No first index specified" << std::endl;
		return false;
	}

	if(this->FileLastIndex == -1)
	{
		std::cout << func << ": No last index specified" << std::endl;
		return false;
	}

	// If no number of digits was specified, we assume no zero-padding.
	if(this->FileNumberOfDigits == -1)
		this->FileNumberOfDigits = 0;

	// Calculate the estimated number of files. This is an estimate because file
	// indices may be missing. These files will then be skipped.
	//int estimatednumberoffiles = this->FileLastIndex - this->FileFirstIndex + 2;
	int estimatednumberoffiles = this->FileLastIndex + 1;

	for(int i = this->FileFirstIndex; i < estimatednumberoffiles; i++)
	{
		// Build an indexed filename based on the given parameters (prefix, index,
		// number of digits and file extension)
		char *filename = DTIUtils::BuildIndexedFileName(
			(char *) this->FilePath.toStdString().c_str(),
			(char *) this->FilePrefix.toStdString().c_str(),
			i,
			this->FileNumberOfDigits,
			(char *) this->FileExtension.toStdString().c_str() );

		// If the file does not exist, it can probably be skipped. Sometimes file
		// naming causes some indices to be missing.
		if(!DTIUtils::FileExists(filename))
		{
			std::cout << func << ": Skipping non-existing file " << filename << std::endl;
			continue;
		}

		// Load DICOM file with the given name. If an error occurs do not just skip it
		// but quit the function. If a file exists but cannot be loaded something serious
		// is going on.
		gdcm::ImageReader reader;
		reader.SetFileName( filename );

		if( reader.Read() == false )
		{
			std::cout << func << ": Could not open DICOM file for reading " << filename << std::endl;
			return false;
		}

		if( this->AddSlice( reader.GetFile(), reader.GetImage(), filename ) ==  false )
		{
			std::cout << func << ": Could not add DICOM file " << filename << std::endl;
			return false;
		}

		// Everything went ok with adding the DICOM file so report it
		std::cout << func << ": Added slice " << filename << std::endl;
	}

	return true;
}

//---------------------------------------------------------------------------
bool DTIDICOMReader::LoadGradients( int nrGrads )
{
	return true;
}

//---------------------------------------------------------------------------
gsl_matrix * DTIDICOMReader::GetGradientTransform()
{
	const char *func = "DTIDICOMReader::GetGradientTransform";

	// If the manufacturer is Philips, we return the gradient transform
	// based on the Philips private tags. These tags have already been
	// extracted in the AddSlice() function.

	//if( this->IsPhilips() )
	//	return this->GetGradientTransformPhilips();

	// Get ImageOrientationPatient attribute from the DICOM file. Retrieve DICOM
	// file of the first slice in the first slice group. We simply assume here that
	// the attribute exists in the file (should be standard DICOM).
	DTIDICOMSlice *slice = (DTIDICOMSlice *) ((DTISliceGroup *) this->SliceGroups->at(0))->GetSliceAt(0);

	double *iop = slice->GetImageOrientationPatient();

	// Compute cross-product of both vectors.
	double *u = iop;
	double *v = iop + 3;
	double *w = new double[3];
	w[0] =  u[1] * v[2] - v[1] * u[2];
	w[1] = -u[0] * v[2] + v[0] * u[2];
	w[2] =  u[0] * v[1] - v[0] * u[1];

	// Put vectors column-wise in matrix.
	gsl_matrix *A = gsl_matrix_calloc(3, 3);
	gsl_matrix_set(A, 0, 0, u[0]);
	gsl_matrix_set(A, 0, 1, v[0]);
	gsl_matrix_set(A, 0, 2, w[0]);
	gsl_matrix_set(A, 1, 0, u[1]);
	gsl_matrix_set(A, 1, 1, v[1]);
	gsl_matrix_set(A, 1, 2, w[1]);
	gsl_matrix_set(A, 2, 0, u[2]);
	gsl_matrix_set(A, 2, 1, v[2]);
	gsl_matrix_set(A, 2, 2, w[2]);

	// Compute matrix inverse by LU decomposition to obtain our transformation.
	int signum;
	gsl_permutation *p = gsl_permutation_calloc(3);
	gsl_matrix *Ainv   = gsl_matrix_calloc(3, 3);

	gsl_linalg_LU_decomp(A, p, &signum);
	gsl_linalg_LU_invert(A, p, Ainv);

	std::cout << func << ": Transformation matrix:" << std::endl;
	std::cout << func << ": (" << gsl_matrix_get(Ainv, 0, 0) << "," <<gsl_matrix_get(Ainv, 0, 1) << "," << gsl_matrix_get(Ainv, 0, 2) << ")" << std::endl;
	std::cout << func << ": (" << gsl_matrix_get(Ainv, 1, 0) << "," <<gsl_matrix_get(Ainv, 1, 1) << "," << gsl_matrix_get(Ainv, 1, 2) << ")" << std::endl;
	std::cout << func << ": (" << gsl_matrix_get(Ainv, 2, 0) << "," <<gsl_matrix_get(Ainv, 2, 1) << "," << gsl_matrix_get(Ainv, 2, 2) << ")" << std::endl;

	// Clear temporary data structures.
	gsl_matrix_free(A);
	gsl_permutation_free(p);

	return Ainv;
}

//---------------------------------------------------------------------------
gsl_matrix * DTIDICOMReader::GetGradientTransformPhilips()
{
	return NULL;
}

//---------------------------------------------------------------------------
double * DTIDICOMReader::GetPixelSpacing()
{
	return this->PixelSpacing;
}

//---------------------------------------------------------------------------
double DTIDICOMReader::GetSliceThickness()
{
	return this->SliceThickness;
}

//---------------------------------------------------------------------------
std::vector< DTIDICOMSliceGroup *> * DTIDICOMReader::GetOutput()
{
	const char * func = "DTIDICOMReader::GetOutput()";
	if( this->SliceGroups == NULL )
		std::cout << func << ": data not loaded yet" << std::endl;
	return this->SliceGroups;
}

//---------------------------------------------------------------------------
gsl_matrix * DTIDICOMReader::GetGradients()
{
	return NULL;
}

//---------------------------------------------------------------------------
void DTIDICOMReader::PrintInfo( std::ostream & ostr )
{
	// Print out the number of slice groups found in total. This should match the
	// number of slice locations in the scan.
	ostr << "DTIDICOMReader::PrintInfo" << endl;
	ostr << "    Number of slicegroups: " << this->SliceGroups->size() << endl;

	vector<DTIDICOMSliceGroup *>::iterator iter;
	int index = 0;
	int numberofslices = 0;

	for(iter = this->SliceGroups->begin(); iter != this->SliceGroups->end(); iter++)
	{
		DTIDICOMSliceGroup *slicegroup = (DTIDICOMSliceGroup *) (*iter);
		cout << "    Size slicegroup " << index << ": " << slicegroup->GetSize() << endl;
		index++;
	}

	ostr << endl;
}

//---------------------------------------------------------------------------
bool DTIDICOMReader::AddSlice( gdcm::File & file, gdcm::Image & image, const char * filename )
{
	const char * func = "DTIDICOMReader::AddSlice";

	gdcm::Attribute< 0x0020, 0x1041 > attribSliceLocation;
	gdcm::Attribute< 0x0018, 0x0050 > attribSliceThickness;
	gdcm::Attribute< 0x0020, 0x0013 > attribInstanceNumber;
	gdcm::Attribute< 0x0028, 0x0030 > attribPixelSpacing;
	gdcm::Attribute< 0x0028, 0x0010 > attribRows;
	gdcm::Attribute< 0x0028, 0x0011 > attribColumns;
	gdcm::Attribute< 0x0020, 0x0037 > attribImageOrientationPatient;

	gdcm::DataSet & dataset = file.GetDataSet();

	std::stringstream str;
	if( dataset.FindDataElement( attribSliceLocation.GetTag() ) == false )
		str << "SliceLocation ";
	if( dataset.FindDataElement( attribSliceThickness.GetTag() ) == false )
		str << "SliceThickness ";
	if( dataset.FindDataElement( attribInstanceNumber.GetTag() ) == false )
		str << "InstanceNumber ";
	if( dataset.FindDataElement( attribPixelSpacing.GetTag() ) == false )
		str << "PixelSpacing ";
	if( dataset.FindDataElement( attribRows.GetTag() ) == false )
		str << "Rows ";
	if( dataset.FindDataElement( attribColumns.GetTag() ) == false )
		str << "Columns";
	if( dataset.FindDataElement( attribImageOrientationPatient.GetTag() ) == false )
		str << "ImageOrientationPatient";

	if( str.str().empty() == false )
	{
		std::cout << func << ": Could not find required DICOM tags: " << std::endl;
		std::cout << str.str() << std::endl;
		return false;
	}

	attribSliceLocation.Set( dataset );
	attribSliceThickness.Set( dataset );
	attribInstanceNumber.Set( dataset );
	attribPixelSpacing.Set( dataset );
	attribRows.Set( dataset );
	attribColumns.Set( dataset );
	attribImageOrientationPatient.Set( dataset );

	// Get slice location
	double slicelocation = attribSliceLocation.GetValue();

	// Get slice location
	double slicethickness = attribSliceThickness.GetValue();

	// Get instance number
	long instancenumber = 0;
	if( this->IsOrderedByInstanceNumber() )
		instancenumber = attribInstanceNumber.GetValue();

	// Get pixel data
	int bufferLength = image.GetBufferLength();
	char * buffer = new char[bufferLength];
	image.GetBuffer( buffer );
	unsigned short * pixels = (unsigned short *) buffer;

	// Get rows and columns
	int rows = attribRows.GetValue();
	int columns = attribColumns.GetValue();

	// Get pixel spacing
	double * spacing = (double *) attribPixelSpacing.GetValues();

	// These parameters are needed when writing the .DTI header later on
	if( this->PixelSpacing[0] == 0.0 && this->PixelSpacing[1] == 0.0 )
	{
		this->PixelSpacing[0] = spacing[0];
		this->PixelSpacing[1] = spacing[1];
	}

	if( this->SliceThickness == 0.0 )
	{
		this->SliceThickness = slicethickness;
	}

	// Get image orientation patient
	double * orientation = (double *) attribImageOrientationPatient.GetValues();

	// Get Philips-specific private tags, if required
	if( this->IsPhilips() )
	{
		gdcm::Tag tag[4];
		tag[0] = gdcm::Tag( 0x2005, 0x1000 ); // ImageAngulationAP
		tag[1] = gdcm::Tag( 0x2005, 0x1001 ); // ImageAngulationFH
		tag[2] = gdcm::Tag( 0x2005, 0x1002 ); // ImageAngulationRL
		tag[3] = gdcm::Tag( 0x2001, 0x100B ); // ImagePlaneOrientation

		if(	dataset.FindDataElement( tag[0] ) == false ||
			dataset.FindDataElement( tag[1] ) == false ||
			dataset.FindDataElement( tag[2] ) == false ||
			dataset.FindDataElement( tag[3] ) == false )
		{
			std::cout << func << " could not find Philips private tags" << std::endl;
			return false;
		}

		std::cout	<< dataset.GetDataElement( tag[0] ).GetByteValue() << " "
					<< dataset.GetDataElement( tag[1] ).GetByteValue() << " "
					<< dataset.GetDataElement( tag[2] ).GetByteValue() << " "
					<< dataset.GetDataElement( tag[3] ).GetByteValue() << std::endl;

		std::cout	<< dataset.GetDataElement( tag[0] ).GetValue() << " "
					<< dataset.GetDataElement( tag[1] ).GetValue() << " "
					<< dataset.GetDataElement( tag[2] ).GetValue() << " "
					<< dataset.GetDataElement( tag[3] ).GetValue() << std::endl;

//		const gdcm::ByteValue * value[4];
//		value[0] = dataset.GetDataElement( tag[0] ).GetByteValue();
//		value[1] = dataset.GetDataElement( tag[1] ).GetByteValue();
//		value[2] = dataset.GetDataElement( tag[2] ).GetByteValue();
//		value[3] = dataset.GetDataElement( tag[3] ).GetByteValue();
//
//		std::string str[4];
//		str[0] = std::string( value[0]->GetPointer(), value[0]->GetLength() );
//		str[1] = std::string( value[1]->GetPointer(), value[1]->GetLength() );
//		str[2] = std::string( value[2]->GetPointer(), value[2]->GetLength() );
//		str[3] = std::string( value[3]->GetPointer(), value[3]->GetLength() );
//
//		float f[3];
//		f[0] = atof( value[0]->GetPointer() );
//		f[1] = atof( value[1]->GetPointer() );
//		f[2] = atof( value[2]->GetPointer() );
//
//		std::cout << f[0] << " " << f[1] << " " << f[2] << " " << str[3] << std::endl;

		// THIS IS STRANGE. I CAN READ OUT THE (2001,100b) ATTRIBUTE BUT
		// NOT THE 2005 ATTRIBUTES. DON'T KNOW WHY....
	}

	// Create new DICOM slice
	DTIDICOMSlice *slice = new DTIDICOMSlice();
	slice->SetData(pixels);
	slice->SetRows(rows);
	slice->SetColumns(columns);
	slice->SetPixelSpacing(spacing);
	slice->SetSliceLocation(slicelocation);
	slice->SetSliceThickness(slicethickness);
	slice->SetInstanceNumber(instancenumber);
	slice->SetImageOrientationPatient(orientation);

	// Add the DICOM file to the list of slice groups. If this list is still empty
	// we can just create a new slice group and add the DICOM file to it. We use the
	// file index for internal ordering.
	if(this->SliceGroups->size() == 0)
	{
		DTIDICOMSliceGroup *slicegroup = new DTIDICOMSliceGroup();
		slicegroup->SetSliceLocation(slicelocation);
		slicegroup->SetOrderedByInstanceNumber(this->IsOrderedByInstanceNumber());

		if(!slicegroup->AddSlice(slice))
		{
			std::cout << func << ": Could not add DTI DICOM slice " << filename << " to slice group" << std::endl;
			return false;
		}

		this->SliceGroups->push_back(slicegroup);
		return true;
	}

	// If list is not empty search for correct location by comparing slice locations.
	vector<DTIDICOMSliceGroup *>::iterator iter;
	for(iter = this->SliceGroups->begin(); iter != this->SliceGroups->end(); iter++)
	{
		DTIDICOMSliceGroup *slicegroup = (DTIDICOMSliceGroup *) (*iter);

		// If the slice locations match we have found the slice group to which the
		// DICOM file belongs!
		if(slicegroup->GetSliceLocation() == slicelocation)
		{
			if(!slicegroup->AddSlice(slice))
			{
				std::cout << func << ": Could not add DTI DICOM slice " << filename << " to list" << std::endl;
				return false;
			}

			return true;
		}

		// If the slice group's slice location is greater than that of our DICOM
		// file this means that no slice group exists yet for this DICOM file. We
		// should create it and insert just before this slice group.
		if(slicegroup->GetSliceLocation() > slicelocation)
		{
			DTIDICOMSliceGroup *slicegroup = new DTIDICOMSliceGroup();
			slicegroup->SetSliceLocation(slicelocation);
			slicegroup->SetOrderedByInstanceNumber(this->IsOrderedByInstanceNumber());

			if(!slicegroup->AddSlice(slice))
			{
				std::cout << func << ": Could not add DTI DICOM slice " << filename << " to list" << std::endl;
				return false;
			}

			// Insert the new slice group just before the current one.
			this->SliceGroups->insert(iter, slicegroup);

			return true;
		}

	}

	// If no slice group was found with a slice location greater than this DICOM
	// file we can create a new slice group and add it to the end of the list
	DTIDICOMSliceGroup *slicegroup = new DTIDICOMSliceGroup();
	slicegroup->SetSliceLocation(slicelocation);
	slicegroup->SetOrderedByInstanceNumber(this->IsOrderedByInstanceNumber());

	if(!slicegroup->AddSlice(slice))
	{
		std::cout << func << ": Could not add DTI DICOM slice " << filename << " to list" << std::endl;
		return false;
	}

	this->SliceGroups->push_back(slicegroup);
	return true;
}
