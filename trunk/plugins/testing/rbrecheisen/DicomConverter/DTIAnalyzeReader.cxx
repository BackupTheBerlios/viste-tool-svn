#include "DTIAnalyzeReader.h"
#include "DTISliceGroup.h"
#include "DTISlice.h"
#include <assert.h>

////////////////////////////////////////////////////////////////////////////////////
DTIAnalyzeReader::DTIAnalyzeReader() : 
	m_pSliceGroups( 0 ), 
	m_iNrVolumes( 0 ),
	m_iFirstIdx( 0 ),
	m_iNrGradients( 0 ),
	m_pPath( 0 ),
	m_pPrefix( 0 ),
	m_pGradientPath( 0 ),
	m_pGradientName( 0 ),
	m_pGradientMatrix( 0 ),
	m_bHeaderSwapped( false ),
	m_pPixelSpacing( 0 ),
	m_dSliceThickness( 0 )
{
}

////////////////////////////////////////////////////////////////////////////////////
DTIAnalyzeReader::~DTIAnalyzeReader()
{
	Clear();
}

////////////////////////////////////////////////////////////////////////////////////
bool DTIAnalyzeReader::LoadData()
{
	assert( m_pPath );
	assert( m_pPrefix );
	assert( m_iNrVolumes > 0 );

	if( ! m_pSliceGroups )
		m_pSliceGroups = new std::vector<DTISliceGroup *>;
	if( ! m_pSliceGroups->empty() )
		Clear();

	char * fileBase = DTIUtils::AppendSlashToPath( m_pPath );
	fileBase = 
		DTIUtils::Concatenate( fileBase, m_pPrefix );

	for( int i = m_iFirstIdx; i < (m_iFirstIdx + m_iNrVolumes); i++ )
	{
		int nrChars = strlen( fileBase ) + 8;
		char	* fileHeader = new char[nrChars],
				* fileImageData = new char[nrChars];
		sprintf( fileHeader, "%s%d.hdr", fileBase, i );
		sprintf( fileImageData, "%s%d.img", fileBase, i );

		struct AnalyzeHeader * header = LoadHeader( fileHeader );
		LoadImageData( fileImageData, header );
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////////
bool DTIAnalyzeReader::LoadGradients()
{
	assert( m_iNrGradients > 0 );

	if( m_pGradientMatrix != NULL )
		gsl_matrix_free( m_pGradientMatrix );
	m_pGradientMatrix = gsl_matrix_calloc( m_iNrGradients, 3 );

	char * fileName = DTIUtils::AppendSlashToPath( m_pGradientPath );
	fileName = 
		DTIUtils::Concatenate( fileName, m_pGradientName );

	FILE *f = fopen( fileName, "rt" );
	if( ! f )
	{
		std::cout << "DTIAnalyzeReader::LoadGradients() could not open gradients file " 
			<< fileName << std::endl;
		return false;
	}

	int n = 0;
	int nrline = 0;

	while( ! feof( f ) )
	{
		char *line = (char *) calloc( 128, sizeof(char) );
		fgets( line, 128, f );
		if( ferror( f ) )
		{
			std::cout << "DTIAnalyzeReader::LoadGradients() error reading line "
				<< line << std::endl;
			break;
		}

		nrline++;
		if( line[0] == '#' || line[0] == '\n' || line == "" || line[0] == '\0' || line[0] == '\t' )
			continue;

		double x = DTIUtils::StringToDouble( DTIUtils::Trim( strtok( line, "," ) ) );
		double y = DTIUtils::StringToDouble( DTIUtils::Trim( strtok( 0, "," ) ) );
		double z = DTIUtils::StringToDouble( DTIUtils::Trim( strtok( 0, "," ) ) );
		std::cout << "DTIAnalyzeReader::LoadGradients() " << x << ", " << y << ", " << z << std::endl;

		double len = sqrt( x*x + y*y + z*z );
		if( len > 0.0 )
		{
			x = x / len;
			y = y / len;
			z = z / len;
		}

		gsl_matrix_set( m_pGradientMatrix, n, 0, x );
		gsl_matrix_set( m_pGradientMatrix, n, 1, y );
		gsl_matrix_set( m_pGradientMatrix, n, 2, z );
		n++;
	}

	fclose( f );
	return true;
}

////////////////////////////////////////////////////////////////////////////////////
struct DTIAnalyzeReader::AnalyzeHeader * DTIAnalyzeReader::LoadHeader( char * pFileName )
{
	FILE * fp = fopen( pFileName, "rb" );
	if( ! fp )
	{
		std::cout << "DTIAnalyzeReader::LoadHeader() cannot open file " 
			<< pFileName << std::endl;
		return 0;
	}

	struct AnalyzeHeader * header = new struct AnalyzeHeader;
	fread( header, 1, sizeof(struct AnalyzeHeader), fp );
	if( header->Info.Dimensions[0] < 0 || header->Info.Dimensions[0] > 15 )
		SwapHeader( header );

	if( ! m_pPixelSpacing )
		m_pPixelSpacing = new double[2];
	m_pPixelSpacing[0] = header->Info.PixelDimensions[1];
	m_pPixelSpacing[1] = header->Info.PixelDimensions[2];
	m_dSliceThickness  = header->Info.PixelDimensions[3];

	return header;
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::LoadImageData( char * pFileName, struct AnalyzeHeader * structHeader )
{
	short int rows = 0, columns = 0, slices = 0;
	rows    = structHeader->Info.Dimensions[2];
	columns = structHeader->Info.Dimensions[1];
	slices  = structHeader->Info.Dimensions[3];
	short int dataType = structHeader->Info.DataType;

	FILE * f = fopen( pFileName, "rb" );
	if( ! f )
	{
		std::cout << "DTDIAnalyzeReader::LoadImageData() could not open file "
			<< pFileName << std::endl;
		return;
	}

	switch( dataType )
	{
	case Binary:
		break;
	case UnsignedChar:
		break;
	case SignedShort:
	{
		std::cout << "DTDIAnalyzeReader::LoadImageData() Added volume " << pFileName << std::endl;
		unsigned short * voxels = new unsigned short[2 * rows * columns * slices];

		int nr = fread( voxels, sizeof(unsigned short), rows * columns * slices, f );
		if( nr != rows * columns * slices )
			std::cout << "DTDIAnalyzeReader::LoadImageData() could not read all voxels" << std::endl;

		int k = 0;
		for( int i = 0; i < rows * columns * slices; i += (rows * columns) )
		{
			unsigned short * pixels = 
				new unsigned short[rows * columns];
			memcpy( pixels, voxels + i, rows * columns * sizeof(unsigned short) );
			AddSlice( pixels, k, structHeader );
			k++;
		}

		delete [] voxels;
		break;
	}
	case SignedInt:
		break;
	case Float:
		break;
	case Double:
		break;
	}

	fclose( f );
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::AddSlice( unsigned short * pPixels, int iSliceNr, struct AnalyzeHeader * structHeader )
{
	short int rows = 0, columns = 0;
	rows    = structHeader->Info.Dimensions[2];
	columns = structHeader->Info.Dimensions[1];

	double spacing[3];
	spacing[0] = static_cast<double>(structHeader->Info.PixelDimensions[1]);
	spacing[1] = static_cast<double>(structHeader->Info.PixelDimensions[2]);
	spacing[2] = static_cast<double>(structHeader->Info.PixelDimensions[3]);

	DTISlice * slice = new DTISlice;
	slice->SetRows( rows );
	slice->SetColumns( columns );
	slice->SetPixelSpacing( spacing[0], spacing[1] );
	slice->SetData( pPixels );
	slice->SetSliceLocation( (double) iSliceNr );
	slice->SetSliceThickness( spacing[2] );

	DTISliceGroup * sliceGroup = FindSliceGroup( iSliceNr );
	if( ! sliceGroup )
	{
		sliceGroup = new DTISliceGroup;
		sliceGroup->SetSliceLocation( (double) iSliceNr );
		m_pSliceGroups->push_back( sliceGroup );
	}

	sliceGroup->AddSlice( slice );
}

////////////////////////////////////////////////////
DTISliceGroup * DTIAnalyzeReader::FindSliceGroup( int iSliceNr )
{
	DTISliceGroup * sliceGroup = 0;

	std::vector<DTISliceGroup *>::iterator iter = m_pSliceGroups->begin();
	for( ; iter != m_pSliceGroups->end(); iter++ )
	{
		if( (*iter)->GetSliceLocation() == ((double) iSliceNr) )
		{
			sliceGroup = (*iter);
			break;
		}
	}

	return sliceGroup;
}

////////////////////////////////////////////////////
gsl_matrix * DTIAnalyzeReader::GetGradientTransform()
{
	gsl_matrix *m = gsl_matrix_calloc( 3, 3 );
	
	gsl_matrix_set(m, 0, 0, 1.0);
	gsl_matrix_set(m, 0, 1, 0.0);
	gsl_matrix_set(m, 0, 2, 0.0);
	gsl_matrix_set(m, 1, 0, 0.0);
	gsl_matrix_set(m, 1, 1, 1.0);
	gsl_matrix_set(m, 1, 1, 0.0);
	gsl_matrix_set(m, 2, 0, 0.0);
	gsl_matrix_set(m, 2, 1, 0.0);
	gsl_matrix_set(m, 2, 2, 1.0);

	return m;
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::Clear()
{
	if( ! m_pSliceGroups )
		return;
	if( ! m_pSliceGroups->empty() )
	{
		std::vector<DTISliceGroup *>::iterator iter = m_pSliceGroups->begin();
		for( ; iter != m_pSliceGroups->end(); iter++ )
			delete (*iter);
		m_pSliceGroups->clear();
	}
	delete m_pSliceGroups;
	m_pSliceGroups = 0;
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::PrintHeader( struct AnalyzeHeader * structHeader )
{
	// structHeader key

	printf ( "HeaderSize %d\n", structHeader->Key.HeaderSize );
	printf ( "DataType %s\n", structHeader->Key.DataType );
	printf ( "DatabaseName %s\n", structHeader->Key.DatabaseName );
	printf ( "Extents %d\n", structHeader->Key.Extents );
	printf ( "SessionError %d\n", structHeader->Key.SessionError );
	printf ( "Regular %c\n", structHeader->Key.Regular );
	printf ( "HeaderKeyUnused %c\n", structHeader->Key.HeaderKeyUnused );

	// image dimensions

	for ( int i = 0; i < 8; i++ )
	{
		printf ( "Dimensions[%d] %d\n", i, structHeader->Info.Dimensions[i] );
	}

	char str[128];

	strncpy ( str, (const char *) structHeader->Info.VoxelUnits, 4 );
	printf ( "VoxelUnits %s\n", str );
	strncpy ( str, (const char *) structHeader->Info.CalibrationUnits, 8 );
	printf ( "CalibrationUnits %s\n", str );
	printf ( "Unused1 %d\n", structHeader->Info.Unused1 );
	printf ( "DataType %d\n", structHeader->Info.DataType );
	printf ( "BitsPerPixel %d\n", structHeader->Info.BitsPerPixel );

	for ( int i = 0; i < 8; i++ )
	{
		printf ( "PixelDimensions[%d] %d\n", i, structHeader->Info.PixelDimensions[i] );
	}

	printf ( "VoxelOffset %6.4\n", structHeader->Info.VoxelOffset );
	printf ( "FUnused1 %6.4f\n", structHeader->Info.FUnused1 );
	printf ( "FUnused2 %6.4f\n", structHeader->Info.FUnused2 );
	printf ( "FUnused3 %6.4f\n", structHeader->Info.FUnused3 );
	printf ( "CalibrationMax %6.4f\n", structHeader->Info.CalibrationMax );
	printf ( "CalibrationMin %6.4f\n", structHeader->Info.CalibrationMin );
	printf ( "Compressed %d\n", structHeader->Info.Compressed );
	printf ( "Verified %d\n", structHeader->Info.Verified );
	printf ( "GlobalMax %d\n", structHeader->Info.GlobalMax );
	printf ( "GlobalMin %d\n", structHeader->Info.GlobalMin );

	// data history

	strncpy ( str, (const char *) structHeader->History.Description, 80 );
	printf ( "Description %s\n", str );
	strncpy ( str, (const char *) structHeader->History.AuxiliaryFile, 24 );
	printf ( "AuxiliaryFile %s\n", str );
	printf ( "Orientation %d\n", structHeader->History.Orientation );
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::SwapHeader( struct AnalyzeHeader * structHeader )
{
	SwapLong ( (unsigned char *) & structHeader->Key.HeaderSize );
	SwapLong ( (unsigned char *) & structHeader->Key.Extents );
	SwapShort( (unsigned char *) & structHeader->Key.SessionError );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[0] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[1] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[2] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[3] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[4] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[5] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[6] );
	SwapShort( (unsigned char *) & structHeader->Info.Dimensions[7] );
	SwapShort( (unsigned char *) & structHeader->Info.Unused1 );
	SwapShort( (unsigned char *) & structHeader->Info.DataType );
	SwapShort( (unsigned char *) & structHeader->Info.BitsPerPixel );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[0] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[1] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[2] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[3] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[4] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[5] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[6] );
	SwapLong ( (unsigned char *) & structHeader->Info.PixelDimensions[7] );
	SwapLong ( (unsigned char *) & structHeader->Info.VoxelOffset );
	SwapLong ( (unsigned char *) & structHeader->Info.FUnused1 );
	SwapLong ( (unsigned char *) & structHeader->Info.FUnused2 );
	SwapLong ( (unsigned char *) & structHeader->Info.FUnused3 );
	SwapLong ( (unsigned char *) & structHeader->Info.CalibrationMax );
	SwapLong ( (unsigned char *) & structHeader->Info.CalibrationMin );
	SwapLong ( (unsigned char *) & structHeader->Info.Compressed );
	SwapLong ( (unsigned char *) & structHeader->Info.Verified );
	SwapShort( (unsigned char *) & structHeader->Info.DimensionsUnused );
	SwapLong ( (unsigned char *) & structHeader->Info.GlobalMax );
	SwapLong ( (unsigned char *) & structHeader->Info.GlobalMin );

	m_bHeaderSwapped = true;
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::SwapLong( unsigned char * pBytes )
{
	unsigned char b0, b1, b2, b3;

	b0 = * pBytes;
	b1 = * (pBytes + 1);
	b2 = * (pBytes + 2);
	b3 = * (pBytes + 3);

	* pBytes = b3;
	* (pBytes + 1) = b2;
	* (pBytes + 2) = b1;
	* (pBytes + 3) = b0;
}

////////////////////////////////////////////////////////////////////////////////////
void DTIAnalyzeReader::SwapShort( unsigned char * pBytes )
{
	unsigned char b0, b1;

	b0 = * pBytes;
	b1 = * (pBytes + 1);

	* pBytes = b1;
	* (pBytes + 1) = b0;
}
