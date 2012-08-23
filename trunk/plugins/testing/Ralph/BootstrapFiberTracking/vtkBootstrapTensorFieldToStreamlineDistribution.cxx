/**
 * vtkBootstrapTensorFieldToStreamlineDistribution.cxx
 * by Ralph Brecheisen
 *
 * 2009-12-07	Ralph Brecheisen
 * - First version
 */
#include "vtkBootstrapTensorFieldToStreamlineDistribution.h"

#include "vtkObjectFactory.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyData.h"
#include "vtkExecutive.h"
#include "vtkAppendPolyData.h"
#include "vtkUnsignedShortArray.h"
#include "vtkIntArray.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

#include "vtkDTIReader2.h"
#include "vtkTensorToEigensystemFilter.h"
#include "vtkEigenvaluesToAnisotropyFilter.h"
#include "vtkStreamlineToSimplifiedStreamline.h"
#include "AnisotropyMeasures.h"

#include <cassert>
#include <cstdio>

#define FILENAME_LENGTH 128

#ifdef _WIN32
#define CACHE "M:/datasets/bootstrapping/cache/temporary"
#else
//#define CACHE "/home/rbrecheis/datasets/bootstrapping/cache/temporary"
#define CACHE "/Users/Ralph/Downloads/Cache"
#endif

namespace bmia {

	vtkStandardNewMacro( vtkBootstrapTensorFieldToStreamlineDistribution );

	///////////////////////////////////////////////////////////////////////////////////////
	vtkBootstrapTensorFieldToStreamlineDistribution::vtkBootstrapTensorFieldToStreamlineDistribution()
	{
		this->StreamlineFilter = NULL;
		this->SeedPoints = NULL;
		this->FileNames = NULL;
		this->SimplificationStepLength = 1.0f;
		this->StepLength = 0.1f;
		this->AnisotropyThreshold = 0.2f;
		this->AngularThreshold = 90.0f;
		this->MinimumFiberLength = 0.0f;
		this->MaximumFiberLength = 400.0f;
		this->AnisotropyMeasure = AnisotropyMeasures::Cl;
		this->SeedPointIds = NULL;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	vtkBootstrapTensorFieldToStreamlineDistribution::~vtkBootstrapTensorFieldToStreamlineDistribution()
	{
		if( this->StreamlineFilter )
		{
			this->StreamlineFilter->SetSource( 0 );
			this->StreamlineFilter->UnRegister( this );
		}

		if( this->FileNames )
			delete this->FileNames;
		if( this->SeedPoints )
			this->SeedPoints->UnRegister( this );

		this->StreamlineFilter = NULL;
		this->SeedPoints = NULL;
		this->FileNames = NULL;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::Execute()
	{
		// Check input parameters
		assert( this->GetTracker() );
		assert( this->GetSource() );
		assert( this->GetFileNames() );
		assert( this->GetFileNames()->size() > 0 );
		assert( this->GetSimplificationStepLength() > 0.0f );
		assert( this->GetStepLength() > 0.0f );
		assert( this->GetAnisotropyThreshold() > 0.0f );
		assert( this->GetAngularThreshold() > 0.0f );
		assert( this->GetMinimumFiberLength() >= 0.0f );
		assert( this->GetMaximumFiberLength() > 0.0f );
		assert( this->GetAnisotropyMeasure() >= 0 );
		assert( this->GetAnisotropyMeasure() < AnisotropyMeasures::numberOfMeasures );

		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::Execute()" << std::endl;

		int index = 0;
		double progress = 0.0;
		double increment = 0.80 / this->GetFileNames()->size();

		int nrSeedPoints = this->GetSource()->GetNumberOfPoints();

		// Initialize progress
		this->SetProgressText( "Running bootstrap fiber tracking" );
		this->UpdateProgress( progress );

		// Assign ID's to each seed point by inserting a scalar array into
		// the seed points data
		this->InsertSeedPointIds();

		// Loop through the list of file names, load each corresponding tensor dataset
		// and perform fiber tracking on it. Write the intermediate streamlines to file
		// and then collect them together in a single poly data object
		std::vector<std::string>::iterator iter = this->GetFileNames()->begin();

        int bla = 0;

		for( ; iter != this->GetFileNames()->end(); iter++ )
		{
			const char * fileName = iter->c_str();
			std::cout << "Reading filename " << fileName << std::endl;

			// Read tensor dataset
			vtkDTIReader2 * reader = vtkDTIReader2::New();
			reader->SetFileDimensionality( 3 );
			reader->SetFileName( fileName );
			reader->Update();

			// Check for tensor data
			vtkImageData * bla = reader->GetOutput();
			vtkPointData * pd  = bla->GetPointData();
			vtkDataArray * ts  = pd->GetTensors();

			if( ts )
			{
				std::cout << "Tensors ok" << std::endl;
			}
			else
			{
				std::cout << "Tensor not ok" << std::endl;
				return;
			}

			// Set anistropy index image for the streamline filter
			vtkTensorToEigensystemFilter * eigenFilter = vtkTensorToEigensystemFilter::New();
			eigenFilter->SetInput( reader->GetOutput() );
			eigenFilter->Update();

			vtkEigenvaluesToAnisotropyFilter * anisotropyFilter = vtkEigenvaluesToAnisotropyFilter::New();
			anisotropyFilter->SetMeasure( this->GetAnisotropyMeasure() );
			anisotropyFilter->SetInput( eigenFilter->GetOutput() );
			anisotropyFilter->Update();

			// Set properties of tracker. Make sure to enable copying of seed point scalars
			// because these contain the seedpoint ID's to which each streamline belongs
			this->GetTracker()->SetCopySeedPointScalars( true );
			this->GetTracker()->SetInput( reader->GetOutput() );
			this->GetTracker()->SetStopAnisotropyIndexImage( anisotropyFilter->GetOutput() );
			this->GetTracker()->SetSource( this->GetSource() );
			this->GetTracker()->SetStopAIValue( this->GetAnisotropyThreshold() );
			this->GetTracker()->SetStopDegrees( this->GetAngularThreshold() );
			this->GetTracker()->SetMinimumFiberSize( this->GetMinimumFiberLength() );
			this->GetTracker()->SetMaximumPropagationDistance( this->GetMaximumFiberLength() );
			this->GetTracker()->SetIntegrationStepLength( this->GetStepLength() );
			this->GetTracker()->Update();

			// Simplify the streamline by removing points that practically
			// lie on a straight line
			vtkStreamlineToSimplifiedStreamline * simplifier = vtkStreamlineToSimplifiedStreamline::New();
			simplifier->SetStepLength( this->GetSimplificationStepLength() );
			simplifier->SetInput( this->GetTracker()->GetOutput() );
			simplifier->Update();

			vtkPolyData * lines = simplifier->GetOutput();

			// Update list of seedpoint ID's and then remove the seedpoint scalars from the
			// streamlines before writing them to file.
            this->UpdateSeedPointIds( lines );
			this->RemoveSeedPointIds( lines );

            bla++;

			// Write streamlines to disk. If we keep everything in memory we will run out of 
			// memory and the application will crash
			char tmp[FILENAME_LENGTH];
			sprintf( tmp, "%s/%d.vtk", CACHE, index );

			vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
			writer->SetFileName( tmp );
			writer->SetInput( lines );
			writer->Write();

			// Clean up
			reader->Delete();
			eigenFilter->Delete();
			anisotropyFilter->Delete();
			writer->Delete();
			simplifier->Delete();

			std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::Execute() finished tracing " << fileName << std::endl;

			index++;

			progress += increment;
			this->UpdateProgress( progress );
		}

		// Build output by loading the cached streamlines and collecting them
		// into a single poly dataset
		this->BuildOutput();

		// Finalize progress
		this->SetProgressText( "Finished" );
		this->UpdateProgress( 1.0 );

		// Clear cache directory. We should really check whether everything has gone
		// correctly otherwise it might be wiser to keep the cache?
		this->ClearCacheDirectory();

		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::Execute() finished all" << std::endl;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::BuildOutput()
	{
		// Check preconditions
		assert( this->GetFileNames() );
		assert( this->GetFileNames()->size() > 0 );

		// Update progress
		double progress = 0.8;
		double increment = 0.15 / this->GetFileNames()->size();
		this->UpdateProgress( progress );

		// Load streamline datasets from file and append them together into a
		// single poly dataset
		vtkAppendPolyData * appender = vtkAppendPolyData::New();

		for( int i = 0; i < static_cast<int>(this->GetFileNames()->size()); i++ )
		{
			char fileName[FILENAME_LENGTH];
			sprintf( fileName, "%s/%d.vtk", CACHE, i );

			vtkPolyDataReader * reader = vtkPolyDataReader::New();
			reader->SetFileName( fileName );
			reader->Update();

			appender->AddInput( reader->GetOutput() );
			reader->Delete();

			progress += increment;
			this->UpdateProgress( progress );
		}

		appender->Update();

		// Copy appended polydata to algorithm output and then clean up 
		// the appender
		vtkPolyData * data = appender->GetOutput();
		vtkPolyData * output = this->GetOutput();
		output->DeepCopy( data );

		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::BuildOutput() total number of lines: " <<
			output->GetNumberOfLines() << std::endl;

		appender->Delete();
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetTracker( vtkTensorFieldToStreamline * _filter )
	{
		assert( _filter );
		if( this->StreamlineFilter == _filter )
			return;
		if( this->StreamlineFilter )
			this->StreamlineFilter->UnRegister( this );
		this->StreamlineFilter = _filter;
		this->StreamlineFilter->Register( this );
	}

	///////////////////////////////////////////////////////////////////////////////////////
	vtkTensorFieldToStreamline * vtkBootstrapTensorFieldToStreamlineDistribution::GetTracker()
	{
		return this->StreamlineFilter;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetStepLength( float _stepLen )
	{
		assert( _stepLen > 0.0f );
		this->StepLength = _stepLen;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	float vtkBootstrapTensorFieldToStreamlineDistribution::GetStepLength()
	{
		return this->StepLength;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetAnisotropyThreshold( float _threshold )
	{
		assert( _threshold > 0.0f );
		assert( _threshold <= 1.0f );
		this->AnisotropyThreshold = _threshold;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	float vtkBootstrapTensorFieldToStreamlineDistribution::GetAnisotropyThreshold()
	{
		return this->AnisotropyThreshold;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetAngularThreshold( float _degrees )
	{
		assert( _degrees > 0.0f );
		assert( _degrees < 360.0f );
		this->AngularThreshold = _degrees;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	float vtkBootstrapTensorFieldToStreamlineDistribution::GetAngularThreshold()
	{
		return this->AngularThreshold;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetMinimumFiberLength( float _length )
	{
		assert( _length >= 0.0f );
		this->MinimumFiberLength = _length;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	float vtkBootstrapTensorFieldToStreamlineDistribution::GetMinimumFiberLength()
	{
		return this->MinimumFiberLength;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetMaximumFiberLength( float _length )
	{
		assert( _length > 0.0f );
		this->MaximumFiberLength = _length;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	float vtkBootstrapTensorFieldToStreamlineDistribution::GetMaximumFiberLength()
	{
		return this->MaximumFiberLength;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetAnisotropyMeasure( int _measure )
	{
		assert( _measure >= 0 );
		assert( _measure < AnisotropyMeasures::numberOfMeasures );
		this->AnisotropyMeasure = _measure;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	int vtkBootstrapTensorFieldToStreamlineDistribution::GetAnisotropyMeasure()
	{
		return this->AnisotropyMeasure;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetSimplificationStepLength( float _stepLen )
	{
		assert( _stepLen > 0.0f );
		assert( _stepLen > this->GetStepLength() );
		this->SimplificationStepLength = _stepLen;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	float vtkBootstrapTensorFieldToStreamlineDistribution::GetSimplificationStepLength()
	{
		return this->SimplificationStepLength;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetFileNames( std::vector<std::string> * _fileNames )
	{
		assert( _fileNames );
		assert( _fileNames->size() > 0 );

		if( this->FileNames )
			delete this->FileNames;
		this->FileNames = new std::vector<std::string>;

		std::vector<std::string>::iterator iter = _fileNames->begin();
		for( ; iter != _fileNames->end(); iter++ )
		{
			std::string str( (*iter) );
			this->FileNames->push_back( str );
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////
	std::vector<std::string> * vtkBootstrapTensorFieldToStreamlineDistribution::GetFileNames()
	{
		return this->FileNames;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::SetSource( vtkDataSet * _seedPoints )
	{
		this->vtkProcessObject::SetNthInput( 1, _seedPoints );
	}

	///////////////////////////////////////////////////////////////////////////////////////
	vtkDataSet * vtkBootstrapTensorFieldToStreamlineDistribution::GetSource()
	{
		return (vtkDataSet *) (this->Inputs[1]);
	}

	///////////////////////////////////////////////////////////////////////////////////////
	std::vector<int> * vtkBootstrapTensorFieldToStreamlineDistribution::GetSeedPointIds() const
	{
		return this->SeedPointIds;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::InsertSeedPointIds()
	{
		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::InsertSeedPointIds()" << std::endl;

		assert( this->GetSource() );

		// If we already have scalar data inside our seed points, this will be
		// problematic later on, so raise an error
		vtkDataArray * scalars = this->GetSource()->GetPointData()->GetScalars();
		if( scalars != NULL )
		{
			vtkErrorMacro( << "Seed points already have scalar data" );
			return;
		}

		// Create array of unsigned shorts
		int nrSeeds = this->GetSource()->GetNumberOfPoints();
		vtkIntArray * ids = vtkIntArray::New();
		ids->Allocate( nrSeeds );

		// Assign ID's starting at zero and running until (nrSeeds - 1)
		int seedIdx = 0;
		for( int i = 0; i < nrSeeds; i++ )
		{
			ids->InsertNextTupleValue( & seedIdx );
			seedIdx++;
		}

		this->GetSource()->GetPointData()->SetScalars( ids );

		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::InsertSeedPointIds() inserted "
			<< nrSeeds << " seed points" << std::endl;

		ids->Delete();
	}

	///////////////////////////////////////////////////////////////////////////////////////
    void vtkBootstrapTensorFieldToStreamlineDistribution::UpdateSeedPointIds( vtkPolyData * _lines )
	{
		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::UpdateSeedPointIds()" << std::endl;

		assert( _lines != NULL );
		
		if( _lines->GetNumberOfCells() == 0 )
			return;
			
		assert( _lines->GetPointData()->GetScalars() != NULL );

		// Get scalar array containing seedpoint ID's
		vtkIntArray * scalars = 
			vtkIntArray::SafeDownCast( _lines->GetPointData()->GetScalars() );
		assert( scalars != NULL );

		// Run through the cell array and for each streamline (defined by list of
		// point ID's) we store the corresponding scalar in our list of seed ID's
		vtkCellArray * cells = _lines->GetLines();
		int nrCells = _lines->GetNumberOfCells();
		vtkIdType cellIdx = 0;

		for( int i = 0; i < nrCells; i++ )
		{
			vtkIdType nrPtIds = 0;
			vtkIdType * ptIds = NULL;
			cells->GetCell( cellIdx, nrPtIds, ptIds );
			cellIdx += (nrPtIds + 1);

			// Get seed ID from first streamline point
            int seedIdx;
			scalars->GetTupleValue( ptIds[0], & seedIdx );

			if( this->SeedPointIds == NULL )
				this->SeedPointIds = new std::vector<int>;
            this->SeedPointIds->push_back( seedIdx );
        }
	}

	///////////////////////////////////////////////////////////////////////////////////////
	vtkPolyData * vtkBootstrapTensorFieldToStreamlineDistribution::RemoveSeedPointIds( vtkPolyData * _lines )
	{
		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::RemoveSeedPointIds()" << std::endl;

		assert( _lines != NULL );
		
		if( _lines->GetNumberOfCells() == 0 )
			return _lines;
		
		assert( _lines->GetPointData()->GetScalars() != NULL );

		// Is this enough?
		_lines->GetPointData()->SetScalars( NULL );
		_lines->Update();

		if( _lines->GetPointData()->GetScalars() )
			std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::RemoveSeedPointIds() number of scalars: " <<
				_lines->GetPointData()->GetScalars()->GetNumberOfTuples() << std::endl;
		else
			std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::RemoveSeedPointIds() no scalars" << std::endl;

		return _lines;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	void vtkBootstrapTensorFieldToStreamlineDistribution::ClearCacheDirectory()
	{
		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::ClearCacheDirectory()" << std::endl;

		int nr = static_cast<int>(this->GetFileNames()->size());
		for( int i = 0; i < nr; i++ )
		{
			char tmp[FILENAME_LENGTH];
			sprintf( tmp, "%s/%d.vtk", CACHE, i );

			if( remove( tmp ) != 0 )
			{
				std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::ClearCacheDirectory() "
					<< "could not delete file " << tmp << std::endl;
				return;
			}
		}

		std::cout << "vtkBootstrapTensorFieldToStreamlineDistribution::ClearCacheDirectory() "
			<< "finished clearing cache directory" << std::endl;
	}

} // namespace bmia

#undef CACHE
