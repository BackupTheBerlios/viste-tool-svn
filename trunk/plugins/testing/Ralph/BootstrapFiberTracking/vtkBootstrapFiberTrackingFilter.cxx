// Includes DTITool
#include <vtkBootstrapFiberTrackingFilter.h>
#include <vtkFiberTrackingFilter.h>
#include <vtkDTIReader2.h>

// Includes VTK
#include <vtkObjectFactory.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataReader.h>
#include <vtkAppendPolyData.h>
#include <vtkIntArray.h>

// Includes C++
#include <sstream>

// Definition of cache directory. This directory is used to
// store intermediate computation results
#define CACHE_DIRECTORY "/Users/Ralph/Temp/Cache/"

vtkCxxRevisionMacro(vtkBootstrapFiberTrackingFilter, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkBootstrapFiberTrackingFilter );

//////////////////////////////////////////////////////////////////////
vtkBootstrapFiberTrackingFilter::vtkBootstrapFiberTrackingFilter()
{
	this->MaximumPropagationDistance = 0.0f;
	this->IntegrationStepLength = 0.0f;
	this->SimplificationStepLength = 0.0f;
	this->MinimumFiberSize = 0.0;
	this->StopAIValue = 0.0;
	this->StopDegrees = 0.0;
	this->SeedPoints = 0;
	this->Anisotropy = 0;
	this->NumberOfBootstrapIterations = 0;
	this->FiberIds = 0;
	this->Output = 0;
}

//////////////////////////////////////////////////////////////////////
vtkBootstrapFiberTrackingFilter::~vtkBootstrapFiberTrackingFilter()
{
	// Delete seed points
	if( this->SeedPoints )
		this->SeedPoints->UnRegister( this );
	this->SeedPoints = 0;

	// Delete anistropy volume
	if( this->Anisotropy )
		this->Anisotropy->UnRegister( this );
	this->Anisotropy = 0;

	// Delete output
	if( this->Output )
		this->Output->UnRegister( this );
	this->Output = 0;

	// Clear list of fiber ID's
	if( this->FiberIds )
		this->FiberIds->UnRegister( this );
	this->FiberIds = 0;

	// Clear list of filenames
	this->FileNames.clear();
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetMaximumPropagationDistance( float distance )
{
	this->MaximumPropagationDistance = distance;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetIntegrationStepLength( float stepLength )
{
	this->IntegrationStepLength = stepLength;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetSimplificationStepLength( float stepLength )
{
	this->SimplificationStepLength = stepLength;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetMinimumFiberSize( float fiberSize )
{
	this->MinimumFiberSize = fiberSize;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetStopAIValue( float aiValue )
{
	this->StopAIValue = aiValue;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetStopDegrees( float degrees )
{
	this->StopDegrees = degrees;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetSeedPoints( vtkDataSet * seedPoints )
{
	if( this->SeedPoints )
		this->SeedPoints->UnRegister( this );
	this->SeedPoints = seedPoints;
	if( this->SeedPoints )
	{
		this->SeedPoints->Register( this );
		this->SeedPoints->Update();
	}
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetAnisotropyIndexImage( vtkImageData * anisotropy )
{
	if( this->Anisotropy )
		this->Anisotropy->UnRegister( this );
	this->Anisotropy = anisotropy;
	if( this->Anisotropy )
	{
		this->Anisotropy->Register( this );
		this->Anisotropy->Update();
	}
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetNumberOfBootstrapIterations( int numberOfIterations )
{
	this->NumberOfBootstrapIterations = numberOfIterations;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::SetFileNames( std::vector< std::string > & fileNames )
{
	std::vector< std::string >::iterator iter = fileNames.begin();
	for( ; iter != fileNames.end(); ++iter )
	{
		this->FileNames.push_back( (*iter) );
	}
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::Update()
{
	// Index used for naming written polydata files
	int index = 0;

	// Process each bootstrap tensor volume
	std::vector< std::string >::iterator iter = this->FileNames.begin();
	for( ; iter != this->FileNames.end(); ++iter )
	{
		std::cout << "vtkBootstrapFiberTrackingFilter::Update() ";
		std::cout << "processing " << (*iter) << std::endl;

		// Create DTI reader to read the tensor data
		bmia::vtkDTIReader2 * reader = bmia::vtkDTIReader2::New();
		reader->SetFileDimensionality( 3 );
		reader->SetFileName( (*iter).c_str() );
		reader->Update();

		// Pass the fiber tracking parameters to the internal
		// fiber tracking algorithm
		bmia::vtkFiberTrackingFilter * filter = bmia::vtkFiberTrackingFilter::New();
		filter->SetMaximumPropagationDistance( this->MaximumPropagationDistance );
		filter->SetIntegrationStepLength( this->IntegrationStepLength );
		filter->SetMinimumFiberSize( this->MinimumFiberSize );
		filter->SetStopAIValue( this->StopAIValue );
		filter->SetStopDegrees( this->StopDegrees );
		filter->SetSeedPoints( this->SeedPoints );
		filter->SetAnisotropyIndexImage( this->Anisotropy );
		filter->SetInput( reader->GetOutput() );
		filter->Update();

		// Get streamlines from the filter
		vtkPolyData * fibers = filter->GetOutput();

		// Update the list of fiber ID's. For each streamline an index
		// is appended to this list. For each fiber set the index starts
		// at zero (see header file for more explanation)
		this->UpdateFiberIds( fibers );

		// Construct filename for temporarily writing the polydata
		std::stringstream stream;
		stream << CACHE_DIRECTORY << index << ".vtk";
		std::string fileName = stream.str();
		index++;

		// Write fibers to disk
		vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
		writer->SetFileName( fileName.c_str() );
		writer->SetFileTypeToBinary();
		writer->SetInput( fibers );
		writer->Write();
		writer->Delete();

		// Delete stuff
		filter->Delete();
	}

	// Build new polydata set
	this->BuildOutput();
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::UpdateFiberIds( vtkPolyData * fibers )
{
	if( ! this->FiberIds )
	{
		this->FiberIds = vtkIntArray::New();
		this->FiberIds->Register( this );
	}

	// Add index to list of fiber ID's, starting at zero (see
	// header file for more information)
	for( int i = 0; i < fibers->GetNumberOfCells(); ++i )
	{
		this->FiberIds->InsertNextValue( i );
	}
}

//////////////////////////////////////////////////////////////////////
vtkIntArray * vtkBootstrapFiberTrackingFilter::GetFiberIds()
{
	return this->FiberIds;
}

//////////////////////////////////////////////////////////////////////
void vtkBootstrapFiberTrackingFilter::BuildOutput()
{
	// Define index for creating the filenames
	int index = 0;

	// Create polydata appender
	vtkAppendPolyData * appender = vtkAppendPolyData::New();

	// For each file, load its corresponding polydata object
	std::vector< std::string >::iterator iter = this->FileNames.begin();
	for( ; iter != this->FileNames.end(); ++iter )
	{
		// Build filename
		std::stringstream stream;
		stream << CACHE_DIRECTORY << index << ".vtk";
		std::string fileName = stream.str();
		index++;

		// Create polydata reader
		vtkPolyDataReader * reader = vtkPolyDataReader::New();
		reader->SetFileName( fileName.c_str() );
		reader->Update();

		// Append polydata to appender
		appender->AddInput( reader->GetOutput() );
		reader->Delete();

		// Delete polydata file from cache directory. If this fails,
		// something else is going on, so quit the loop
		if( remove( fileName.c_str() ) != 0 )
		{
			std::cout << "vtkBootstrapFiberTrackingFilter::BuildOutput() ";
			std::cout << "could not remove polydata" << std::endl;
			break;
		}
	}

	// Update appender just in case...
	appender->Update();

	// Set output polydata and delete the appender (hope it
	// does not crash)
	this->Output = appender->GetOutput();
	this->Output->Register( this );
	appender->Delete();
}

//////////////////////////////////////////////////////////////////////
vtkPolyData * vtkBootstrapFiberTrackingFilter::GetOutput()
{
	if( this->Output == 0 )
		this->Update();

	if( this->Output )
		this->Output->Update();

	return this->Output;
}

#undef CACHE_DIRECTORY
