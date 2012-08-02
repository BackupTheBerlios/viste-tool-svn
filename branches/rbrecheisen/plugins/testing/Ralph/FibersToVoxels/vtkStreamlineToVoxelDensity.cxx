#include "vtkStreamlineToVoxelDensity.h"
#include "vtkMathExtensions.h"

#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkUnsignedShortArray.h"

#include <algorithm>
#include <cstdio>

///////////////////////////////////////////////////////////////////////
vtkStreamlineToVoxelDensity::vtkStreamlineToVoxelDensity() : m_Streamlines(NULL),
	m_MeanVoxelDensity(0), m_StandardDeviationVoxelDensity(0)
{
	m_Dimensions[0] = 0;
	m_Dimensions[1] = 0;
	m_Dimensions[2] = 0;

	m_Binary = 0;
	
	m_Spacing[0] = 0.0;
	m_Spacing[1] = 0.0;
	m_Spacing[2] = 0.0;

    m_Scores = NULL;
}

///////////////////////////////////////////////////////////////////////
vtkStreamlineToVoxelDensity::~vtkStreamlineToVoxelDensity()
{
	if ( m_Streamlines )
		m_Streamlines->Delete();
}

///////////////////////////////////////////////////////////////////////
void vtkStreamlineToVoxelDensity::SetInput(vtkPolyData *data)
{
	m_Streamlines = data;
}

///////////////////////////////////////////////////////////////////////
void vtkStreamlineToVoxelDensity::SetDimensions(int dimX, int dimY, int dimZ)
{
	m_Dimensions[0] = dimX;
	m_Dimensions[1] = dimY;
	m_Dimensions[2] = dimZ;
}

///////////////////////////////////////////////////////////////////////
void vtkStreamlineToVoxelDensity::SetSpacing(double dX, double dY, double dZ)
{
	m_Spacing[0] = dX;
	m_Spacing[1] = dY;
	m_Spacing[2] = dZ;
}

///////////////////////////////////////////////////////////////////////
void vtkStreamlineToVoxelDensity::SetScores( double * scores, int count )
{
    if( m_Scores != NULL )
        delete [] m_Scores;
    m_Scores = new double[count];
    for( int i = 0; i < count; ++i )
        m_Scores[i] = scores[i];
}

///////////////////////////////////////////////////////////////////////
vtkImageData *vtkStreamlineToVoxelDensity::GetOutput()
{
    if( m_Dimensions[0] <= 0.0 || m_Dimensions[1] <= 0.0 || m_Dimensions[2] <= 0.0 )
        return NULL;
    if( m_Spacing[0] <= 0.0 || m_Spacing[1] <= 0.0 || m_Spacing[2] <= 0.0 )
        return NULL;
    if( ! m_Streamlines )
        return NULL;

    if(m_Scores)
        std::cout << "vtkStreamlineToVoxelDensity::GetOutput() using scores" << std::endl;

    int dims[3], size;
    dims[0] = m_Dimensions[0];
    dims[1] = m_Dimensions[1];
    dims[2] = m_Dimensions[2];
    size = dims[0] * dims[1] * dims[2];

    int dimsPad[3], sizePad;
    dimsPad[0] = this->NextPowerOfTwo( dims[0] );
    dimsPad[1] = this->NextPowerOfTwo( dims[1] );
    dimsPad[2] = dims[2];
    sizePad = dimsPad[0] * dimsPad[1] * dimsPad[2];

    unsigned short * voxelsPad = new unsigned short[sizePad];
    for( int i = 0; i < sizePad; ++i )
        voxelsPad[i] = 0;

    vtkIdType nrPtIds, * ptIds = NULL;
    vtkPoints * points = m_Streamlines->GetPoints();
    vtkCellArray * lines  = m_Streamlines->GetLines();

    int cellIdx = 0;

    lines->InitTraversal();
    while ( lines->GetNextCell( nrPtIds, ptIds ) )
    {
        int prevIdx = -1;

        for( int i = 0; i < nrPtIds; ++i )
        {
            double pt[3];
            points->GetPoint( ptIds[i], pt );

            int x = (int) floor( pt[0] / m_Spacing[0] );
            int y = (int) floor( pt[1] / m_Spacing[1] );
            int z = (int) floor( pt[2] / m_Spacing[2] );

            int index = z * dimsPad[0] * dimsPad[1] + y * dimsPad[0] + x;
            if( index == prevIdx )
                continue;

            if( m_Binary )
            {
                voxelsPad[index] = 1;
            }
            else if( m_Scores )
            {
                double score = m_Scores[cellIdx];
                unsigned short value = (unsigned short) floor(score * 255.0);
                if( value > voxelsPad[index] )
                    voxelsPad[index] = value;
            }
            else
            {
                voxelsPad[index]++;
            }

            prevIdx = index;
        }

        cellIdx++;
    }

    vtkImageData * data = vtkImageData::New();
    data->SetOrigin( 0, 0, 0 );
    data->SetScalarTypeToUnsignedShort();
    data->SetNumberOfScalarComponents( 1 );
    data->SetSpacing( m_Spacing[0], m_Spacing[1], m_Spacing[2] );
    data->SetExtent( 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1 );
    data->AllocateScalars();

    unsigned short * voxels = (unsigned short *) data->GetScalarPointer();
    for( int z = 0; z < dims[2]; ++z )
    {
        for( int y = 0; y < dims[1]; ++y )
        {
            for( int x = 0; x < dims[0]; ++x )
            {
                int index = z * dims[0] * dims[1] + y * dims[0] + x;
                int indexPad = z * dimsPad[0] * dimsPad[1] + y * dimsPad[0] + x;
                voxels[index] = voxelsPad[indexPad];
            }
        }
    }

    return data;
}

//////////////////////////////////////////////////////////////////////
int vtkStreamlineToVoxelDensity::NextPowerOfTwo( int N )
{
    int k = 2;
    if( N == 0 ) return 1;
    while( k < N )
        k *= 2;
    return k;
}
