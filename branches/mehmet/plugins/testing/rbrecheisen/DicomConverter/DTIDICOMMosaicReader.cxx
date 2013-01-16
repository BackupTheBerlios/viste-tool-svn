#include "DTIDICOMMosaicReader.h"

///////////////////////////////////////////////
DTIDICOMMosaicReader::DTIDICOMMosaicReader() : DTIDICOMReader()
{
	this->NumberOfImagesPerFile = 0;
}

///////////////////////////////////////////////
DTIDICOMMosaicReader::~DTIDICOMMosaicReader()
{
}

///////////////////////////////////////////////
bool DTIDICOMMosaicReader::LoadData()
{
	const char *func = "DTIDICOMMosaicReader::LoadData";

	if(this->FilePath == 0)
	{
		__DTIMACRO_LOG(func << ": No file path specified. Assuming current directory." << endl, DEBUG, DTIUtils::LogLevel);
		this->FilePath = "./";
	}
	
	this->FilePath = DTIUtils::AppendSlashToPath(this->FilePath);
	
	if(this->FilePrefix == 0)
	{
		__DTIMACRO_LOG(func << ": No file extension specified. Assuming none." << endl, DEBUG, DTIUtils::LogLevel);
		this->FilePrefix = "";
	}
	
	if(strcmp(this->FileExtension, ".") == 0)
		this->FileExtension = 0;
	
	if(this->FileFirstIndex == -1)
	{
		__DTIMACRO_LOG(func << ": No first index specified" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}
	
	if(this->FileLastIndex == -1)
	{
		__DTIMACRO_LOG(func << ": No last index specified" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	if(this->FileNumberOfDigits == -1)
		this->FileNumberOfDigits = 0;

	if(this->NumberOfImagesPerFile == 0)
	{
		__DTIMACRO_LOG(func << ": Number of images per file not specified" << endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	for(int i = this->FileFirstIndex; i < (this->FileLastIndex + 1); i++)
	{
		char *filename = DTIUtils::BuildIndexedFileName(
			this->FilePath,
			this->FilePrefix, 
			i, 
			this->FileNumberOfDigits, 
			this->FileExtension);

		if(!DTIUtils::FileExists(filename))
		{
			__DTIMACRO_LOG(func << ": File " << filename << " does not exist" << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		DcmFileFormat *dicomfile = new DcmFileFormat();
		OFCondition status = dicomfile->loadFile(filename);

		if(status.bad())
		{
			__DTIMACRO_LOG(func << ": Could not load DICOM file " << filename << endl, ERROR, DTIUtils::LogLevel);
			__DTIMACRO_LOG(func << ": " << dicomfile->error().text() << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}

		if(!this->AddFile(dicomfile, filename))
		{
			__DTIMACRO_LOG(func << ": Could not add dicom file " << filename << endl, ERROR, DTIUtils::LogLevel);
			return false;
		}
		__DTIMACRO_LOG(func << ": Added file " << filename << endl, ALWAYS, DTIUtils::LogLevel);
	}

	return true;
}

///////////////////////////////////////////////
bool DTIDICOMMosaicReader::AddFile(DcmFileFormat *dicomfile, const char *filename)
{
	const char *func = "DTIDICOMMosaicReader::AddFile";
	
	unsigned long numberofpixels;
	unsigned short *pixels = 0;
	unsigned short columns;
	unsigned short rows;
	double slicethickness = 0;

	if(this->LayoutColumns == 0 || this->LayoutRows == 0)
	{
		__DTIMACRO_LOG(func << ": No image layout specified"<< endl, ERROR, DTIUtils::LogLevel);
		return false;
	}

	dicomfile->loadAllDataIntoMemory();
	dicomfile->getDataset()->findAndGetUint16Array(DCM_PixelData, (const Uint16 *&) pixels, &numberofpixels);
	dicomfile->getDataset()->findAndGetFloat64(DCM_SliceThickness, slicethickness);
	dicomfile->getDataset()->findAndGetUint16(DCM_Rows, rows);
	dicomfile->getDataset()->findAndGetUint16(DCM_Columns, columns);

	rows /= this->GetLayoutRows();
	columns /= this->GetLayoutColumns();

	// If list of slice groups still empty, add slice group for each image
	// in the DICOM file.
	if(this->SliceGroups->size() == 0)
	{
		for(int i = 0; i < this->NumberOfImagesPerFile; i++)
		{
			DTIDICOMSliceGroup *slicegroup = new DTIDICOMSliceGroup();
			slicegroup->SetSliceLocation(i * slicethickness);
			slicegroup->SetOrderedByInstanceNumber(false);

			this->SliceGroups->push_back(slicegroup);
		}
	}

	// Extract pixel data from DICOM file
	for(int i = 0; i < this->NumberOfImagesPerFile; i++)
	{
		unsigned short *imagepixels = this->ExtractImagePixels(pixels, i, rows, columns);

		DTIDICOMSlice *slice = new DTIDICOMSlice();
		slice->SetData(imagepixels);
		slice->SetFile(dicomfile);
		slice->SetRows(rows);
		slice->SetColumns(columns);
		slice->SetSliceLocation(i * slicethickness);

		DTIDICOMSliceGroup *slicegroup = this->SliceGroups->at(i);
		slicegroup->AddSlice(slice);
		//__DTIMACRO_LOG(func << ": Added slice " << i << " from file "<< filename << endl, ALWAYS, DTIUtils::LogLevel);
	}

	return true;
}

///////////////////////////////////////////////
unsigned short *DTIDICOMMosaicReader::ExtractImagePixels(unsigned short *pixels, int i, int rows, int columns)
{
	unsigned short *image = new unsigned short[rows*columns];

	int lrow  = (int) floor(((double) i) / this->LayoutColumns);
	int lcol  = i - lrow*this->LayoutColumns;
	int start = lrow*rows*columns*this->LayoutColumns + lcol*columns;

	for(int k = 0; k < rows; k++)
	{
		memcpy(image+(k*columns), pixels+start, columns*sizeof(unsigned short));
		start += (columns*this->LayoutColumns);
	}

	return image;
}