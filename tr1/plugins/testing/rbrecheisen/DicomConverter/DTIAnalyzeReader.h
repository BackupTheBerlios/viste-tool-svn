#ifndef __DTIAnalyzeReader_h
#define __DTIAnalyzeReader_h

#include <vector>
#include "gsl/gsl_linalg.h"

class DTISliceGroup;
class DTISlice;

typedef struct
{
	float Real;
	float Imaginary;
} ComplexNr;

class DTIAnalyzeReader
{
	static const int None = 0;
	static const int Unknown = 0;
	static const int Binary = 1;
	static const int UnsignedChar = 2;
	static const int SignedShort = 4;
	static const int SignedInt = 8;
	static const int Float = 16;
	static const int Complex = 32;
	static const int Double = 64;
	static const int RGB = 128;
	static const int All = 255;

	struct AnalyzeHeader
	{
		struct HeaderKey
		{
			int HeaderSize;
			char DataType[10];
			char DatabaseName[18];
			int Extents;
			short int SessionError;
			char Regular;
			char HeaderKeyUnused;
		};

		struct ImageInfo
		{
			short int Dimensions[8];
			unsigned char VoxelUnits[4];
			unsigned char CalibrationUnits[8];
			short int Unused1;
			short int DataType;
			short int BitsPerPixel;
			short int DimensionsUnused;
			float PixelDimensions[8];
			float VoxelOffset;
			float FUnused1;
			float FUnused2;
			float FUnused3;
			float CalibrationMax;
			float CalibrationMin;
			float Compressed;
			float Verified;
			int GlobalMax;
			int GlobalMin;
		};

		struct DataHistory
		{
			char Description[80];
			char AuxiliaryFile[24];
			char Orientation;
			char Originator[10];
			char Generated[10];
			char ScanNumber[10];
			char PatientId[10];
			char ExpirationDate[10];
			char ExpirationTime[10];
			char HistoryUnused[3];
			int Views;
			int VolumesAdded;
			int StartField;
			int FieldSkip;
			int OMax;
			int OMin;
			int SMax;
			int SMin;
		};

		struct HeaderKey Key;
		struct ImageInfo Info;
		struct DataHistory History;
	};

public:

	DTIAnalyzeReader();
	virtual ~DTIAnalyzeReader();

	void SetNumberOfVolumes( int iNr )
	{
		m_iNrVolumes = iNr;
	};

	void SetFirstIndex( int iIdx )
	{
		m_iFirstIdx = (iIdx < 0 ) ? 0 : iIdx;
	};

	void SetFilePath( char * pPath )
	{
		m_pPath = pPath;
	};

	void SetFilePrefix( char * pPref )
	{
		m_pPrefix = pPref;
	};

	void SetGradientFilePath( char * pPath )
	{
		m_pGradientPath = pPath;
	};

	void SetGradientFileName( char * pName )
	{
		m_pGradientName = pName;
	};

	void SetNumberOfGradients( int iNr )
	{
		m_iNrGradients = iNr;
	};

	bool LoadData();
	bool LoadGradients();

	double * GetPixelSpacing()
	{
		return m_pPixelSpacing;
	};

	double GetSliceThickness()
	{
		return m_dSliceThickness;
	};

	std::vector<DTISliceGroup *> * GetOutput()
	{
		return m_pSliceGroups;
	};

	gsl_matrix * GetGradients()
	{
		return m_pGradientMatrix;
	};

	gsl_matrix * GetGradientTransform();

	void PrintHeader( struct AnalyzeHeader * structHeader );

private:

	struct AnalyzeHeader * LoadHeader( char * pFileName );
	void LoadImageData( char * pFileName, struct AnalyzeHeader * structHeader );
	void Clear();

	void SwapHeader( struct AnalyzeHeader * structHeader );
	void SwapLong( unsigned char * );
	void SwapShort( unsigned char * );

	void AddSlice( unsigned short * pPixels, int iSliceNr, struct AnalyzeHeader * structHeader );
	DTISliceGroup * FindSliceGroup( int iSliceNr );

	std::vector<DTISliceGroup *> * m_pSliceGroups;
	char * m_pPath, * m_pPrefix, * m_pGradientPath, * m_pGradientName;
	int m_iNrVolumes;
	int m_iFirstIdx;
	int m_iNrGradients;
	bool m_bHeaderSwapped;
	double * m_pPixelSpacing, m_dSliceThickness;
	gsl_matrix * m_pGradientMatrix;
};

#endif
