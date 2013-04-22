#ifndef __DTIDICOMMosaicReader_h
#define __DTIDICOMMosaicReader_h

#include "DTIUtils.h"
#include "DTIDICOMReader.h"

//---------------------------------------------------------------------------
//! \file   DTIDICOMMosaicReader.h
//! \class  DTIDICOMMosaicReader
//! \author Ralph Brecheisen
//! \brief  Reads DICOM Mosaic with multiple images packed in a file.
//---------------------------------------------------------------------------
class DTIDICOMMosaicReader : public DTIDICOMReader
{
public:
	DTIDICOMMosaicReader();
	~DTIDICOMMosaicReader();

	virtual bool LoadData();
	
	__DTIMACRO_SETGET(NumberOfImagesPerFile, int);
	__DTIMACRO_SETGET(LayoutRows, int);
	__DTIMACRO_SETGET(LayoutColumns, int);

protected:
	virtual bool AddFile(DcmFileFormat *dicomfile, const char *filename);
	virtual unsigned short *ExtractImagePixels(unsigned short *pixels, int i, int rows, int columns);
	
	int NumberOfImagesPerFile;
	int LayoutRows;
	int LayoutColumns;
};

#endif