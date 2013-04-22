// Includes DTI tool
#include <vtkImageDataToDistanceTransform2.h>

// Includes VTK
#include <vtkObjectFactory.h>

// Includes PBA
#include <PBA3D/dtengine.h>
#include <PBA3D/gpudefs.h>
#include <PBA3D/utils.h>
#include <PBA3D/pba/pba3D.h>

vtkCxxRevisionMacro( vtkImageDataToDistanceTransform2, "$Revision: 1.0 $" );
vtkStandardNewMacro( vtkImageDataToDistanceTransform2 );

//////////////////////////////////////////////////////////////////////
vtkImageDataToDistanceTransform2::vtkImageDataToDistanceTransform2()
{
	this->DistanceTransform = 0;
	this->Voronoi = 0;
	this->Threshold = 0.0f;
	this->Data = 0;
	this->InputVoronoi = 0;
}

//////////////////////////////////////////////////////////////////////
vtkImageDataToDistanceTransform2::~vtkImageDataToDistanceTransform2()
{
	// Unregister input data
	if( this->Data )
		this->Data->UnRegister( this );
	this->Data = 0;

	// Delete distance transform
	if( this->DistanceTransform )
		this->DistanceTransform->Delete();
	this->DistanceTransform = 0;

	// Delete voronoi
	if( this->Voronoi )
		this->Voronoi->Delete();
	this->Voronoi = 0;
}

//////////////////////////////////////////////////////////////////////
void vtkImageDataToDistanceTransform2::Upload()
{
	// Check that we have a valid image
	if( ! this->Data )
	{
		std::cout << "vtkImageDataToDistanceTransform2::Execute() ";
		std::cout << "input volume not set!" << std::endl;
		return;
	}

	// Get image dimensions. If these are not a power of two,
	// update them such that there are
	int dims[3];
	this->Data->GetDimensions( dims );
	int size = dims[0] * dims[1] * dims[2];

	// Get image voxel spacing
	double spacing[3];
	this->Data->GetSpacing( spacing );

	// Check that we have scalar data
	if( ! this->Data->GetScalarPointer() )
	{
		std::cout << "vtkImageDataToDistanceTransform2::Execute() ";
		std::cout << "input data has no voxels!" << std::endl;
		return;
	}

	// Check that we have data of right type
	std::vector< float * > * vertices = 0;
	if( this->Data->GetScalarType() == VTK_UNSIGNED_CHAR )
	{
		unsigned char * voxels = (unsigned char *) this->Data->GetScalarPointer();
		vertices = this->UnsignedCharVoxelsToVertices( voxels, dims[0], dims[1], dims[2] );
	}
	else if( this->Data->GetScalarType() == VTK_UNSIGNED_SHORT )
	{
		unsigned short * voxels = (unsigned short *) this->Data->GetScalarPointer();
		vertices = this->UnsignedShortVoxelsToVertices( voxels, dims[0], dims[1], dims[2] );
	}
	else if( this->Data->GetScalarType() == VTK_FLOAT )
	{
		float * voxels = (float *) this->Data->GetScalarPointer();
		vertices = this->FloatVoxelsToVertices( voxels, dims[0], dims[1], dims[2] );
	}
	else
	{
		std::cout << "vtkImageDataToDistanceTransform2::Execute() ";
		std::cout << "unsupported data type!" << std::endl;
		return;
	}

	// Compute distance transform dimensions
	int newDims[3];
	newDims[0] = this->NextPowerOfTwo( dims[0] );
	newDims[1] = this->NextPowerOfTwo( dims[1] );
	newDims[2] = this->NextPowerOfTwo( dims[2] );

	// Initialize CUDA data
	this->InputVoronoi = 0;
	int maxVertices = newDims[0] * newDims[1] * newDims[2];
	//dtcuda_initialization( maxVertices, newDims[0], newDims[1], newDims[2] );
	dtcuda_initialization2( maxVertices, newDims[0], newDims[1], newDims[2], spacing[0], spacing[1], spacing[2] );
	dtcuda_bindBuffer( this->InputVoronoi );

	// Initialize input Voronoi with markers
	for( int i = 0; i < maxVertices; ++i )
		this->InputVoronoi[i] = -1;

	// Encode vertex positions into Voronoi diagram
	int pid = 0;
	for( int i = 0; i < vertices->size(); ++i )
	{
		float * vertex = vertices->at( i );

		int x = vertex[0];
		int y = vertex[1];
		int z = vertex[2];

		int id  = z * newDims[0] * newDims[1] + y * newDims[0] + x;
		if( this->InputVoronoi[id] != -1 )
			continue;
		this->InputVoronoi[id] = ENCODE( x, y, z );
		inputPoints[pid] = this->InputVoronoi[id];
		pid++;
	}

	// Delete vertices
	for( int i = 0; i < vertices->size(); ++i )
	{
		float * vertex = vertices->at( i );
		delete [] vertex;
		vertex = 0;
	}

	vertices->clear();
	delete vertices;
}

//////////////////////////////////////////////////////////////////////
void vtkImageDataToDistanceTransform2::Execute()
{
	this->Upload();

	// Compute Voronoi diagram
	pba3DVoronoiDiagram( this->InputVoronoi, outputVoronoi, phase1Band, phase2Band, phase3Band );

	// Compute distance transform
	dtcuda_bindBuffer( outputDT );
	pba3DDT( outputVoronoi, outputDT );

	this->Download();
}

//////////////////////////////////////////////////////////////////////
void vtkImageDataToDistanceTransform2::Download()
{
	if( outputDT == 0 )
	{
		std::cout << "vtkImageDataToDistanceTransform2::Download() ";
		std::cout << "No output DT!" << std::endl;
		return;
	}

	if( outputVoronoi == 0 )
	{
		std::cout << "vtkImageDataToDistanceTransform2::Download() ";
		std::cout << "No output voronoi!" << std::endl;
		return;
	}

	// Get data spacing from original input data
	double spacing[3];
	this->Data->GetSpacing( spacing );

	// Get data dimensions
	int dims[3];
	this->Data->GetDimensions( dims );

	// Compute power-of-two dimensions
	int newDims[3];
	newDims[0] = this->NextPowerOfTwo( dims[0] );
	newDims[1] = this->NextPowerOfTwo( dims[1] );
	newDims[2] = this->NextPowerOfTwo( dims[2] );

	// Create new VTK image data
	if( this->DistanceTransform )
		this->DistanceTransform->Delete();
	this->DistanceTransform = vtkImageData::New();
	this->DistanceTransform->SetOrigin( 0, 0, 0 );
	this->DistanceTransform->SetScalarTypeToFloat();
	this->DistanceTransform->SetNumberOfScalarComponents( 1 );
	this->DistanceTransform->SetSpacing( spacing[0], spacing[1], spacing[2] );
	this->DistanceTransform->SetExtent( 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1 );
	this->DistanceTransform->AllocateScalars();

	// Copy distance transform values to new VTK image
	float * pointer = (float *) this->DistanceTransform->GetScalarPointer();
	for( int k = 0; k < dims[2]; ++k )
	{
		for( int i = 0; i < dims[1]; ++i )
		{
			for( int j = 0; j < dims[0]; ++j )
			{
				int idx0 = k * newDims[0] * newDims[1] + i * newDims[0] + j;
				int idx1 = k * dims[0] * dims[1] + i * dims[0] + j;
				pointer[idx1] = outputDT[idx0] < 0.0f ? 0.0 : outputDT[idx0];
			}
		}
	}

	// Create new VTK image data for voronoi
	if( this->Voronoi )
		this->Voronoi->Delete();
	this->Voronoi = vtkImageData::New();
	this->Voronoi->SetOrigin( 0, 0, 0 );
	this->Voronoi->SetScalarTypeToInt();
	this->Voronoi->SetNumberOfScalarComponents( 1 );
	this->Voronoi->SetSpacing( spacing[0], spacing[1], spacing[2] );
	this->Voronoi->SetExtent( 0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1 );
	this->Voronoi->AllocateScalars();

	// Copy voronoi information into VTK image
	int * pointer2 = (int *) this->Voronoi->GetScalarPointer();
	for( int k = 0; k < dims[2]; ++k )
	{
		for( int i = 0; i < dims[1]; ++i )
		{
			for( int j = 0; j < dims[0]; ++j )
			{
				int idx0 = k * newDims[0] * newDims[1] + i * newDims[0] + j;
				int idx1 = k * dims[0] * dims[1] + i * dims[0] + j;
				pointer2[idx1] = outputVoronoi[idx0];
			}
		}
	}

	// Clean up CUDA stuff
	dtcuda_deinitialization();
}

//////////////////////////////////////////////////////////////////////
std::vector< float * > * vtkImageDataToDistanceTransform2::UnsignedCharVoxelsToVertices( unsigned char * voxels, int dimX, int dimY, int dimZ )
{
	std::vector< float * > * vertices = new std::vector< float * >;
	for( int k = 0; k < dimZ; ++k )
	{
		for( int i = 0; i < dimY; ++i )
		{
			for( int j = 0; j < dimX; ++j )
			{
				// Get voxel value. Only non-zero values and values
				// above the given threshold are mapped to a vertex
				int idx = k * dimX * dimY + i * dimX + j;
				unsigned char & value = voxels[idx];
				if( value && value > this->GetThreshold() )
				{
					float * vertex = new float[3];
					vertex[0] = j;
					vertex[1] = i;
					vertex[2] = k;
					vertices->push_back( vertex );
				}
			}
		}
	}

	return vertices;
}

//////////////////////////////////////////////////////////////////////
std::vector< float * > * vtkImageDataToDistanceTransform2::UnsignedShortVoxelsToVertices( unsigned short * voxels, int dimX, int dimY, int dimZ )
{
	std::vector< float * > * vertices = new std::vector< float * >;
	for( int k = 0; k < dimZ; ++k )
	{
		for( int i = 0; i < dimY; ++i )
		{
			for( int j = 0; j < dimX; ++j )
			{
				// Get voxel value. Only non-zero values and values
				// above the given threshold are mapped to a vertex
				int idx = k * dimX * dimY + i * dimX + j;
				unsigned short & value = voxels[idx];
				if( value && value > this->GetThreshold() )
				{
					float * vertex = new float[3];
					vertex[0] = j;
					vertex[1] = i;
					vertex[2] = k;
					vertices->push_back( vertex );
				}
			}
		}
	}

	return vertices;
}

//////////////////////////////////////////////////////////////////////
std::vector< float * > * vtkImageDataToDistanceTransform2::FloatVoxelsToVertices( float * voxels, int dimX, int dimY, int dimZ )
{
	std::vector< float * > * vertices = new std::vector< float * >;
	for( int k = 0; k < dimZ; ++k )
	{
		for( int i = 0; i < dimY; ++i )
		{
			for( int j = 0; j < dimX; ++j )
			{
				// Get voxel value. Only non-zero values and values
				// above the given threshold are mapped to a vertex
				int idx = k * dimX * dimY + i * dimX + j;
				float & value = voxels[idx];
				if( value && value > this->GetThreshold() )
				{
					float * vertex = new float[3];
					vertex[0] = j;
					vertex[1] = i;
					vertex[2] = k;
					vertices->push_back( vertex );
				}
			}
		}
	}

	return vertices;
}

//////////////////////////////////////////////////////////////////////
void vtkImageDataToDistanceTransform2::SetInput( vtkImageData * data )
{
	if( this->Data )
		this->Data->UnRegister( this );
	this->Data = data;
	if( this->Data )
		this->Data->Register( this );
}

//////////////////////////////////////////////////////////////////////
vtkImageData * vtkImageDataToDistanceTransform2::GetInput()
{
	return this->Data;
}

//////////////////////////////////////////////////////////////////////
void vtkImageDataToDistanceTransform2::SetThreshold( float threshold )
{
	this->Threshold = threshold;
}

//////////////////////////////////////////////////////////////////////
float vtkImageDataToDistanceTransform2::GetThreshold()
{
	return this->Threshold;
}

//////////////////////////////////////////////////////////////////////
vtkImageData * vtkImageDataToDistanceTransform2::GetOutput()
{
	return this->DistanceTransform;
}

//////////////////////////////////////////////////////////////////////
vtkImageData * vtkImageDataToDistanceTransform2::GetOutputVoronoi()
{
	return this->Voronoi;
}

//////////////////////////////////////////////////////////////////////
int vtkImageDataToDistanceTransform2::NextPowerOfTwo( int number )
{
	int k = 2;
	if( number == 0 ) return 1;
	while( k < number )
		k *= 2;
	return k;
}
