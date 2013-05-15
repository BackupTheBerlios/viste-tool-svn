#define _CRT_SECURE_NO_WARNINGS

/**
 * vtkBootstrapStreamlineToDistanceTableFilter.cxx
 * by Ralph Brecheisen
 *
 * 2010-01-07	Ralph Brecheisen
 * - First version
 */
#include "vtkBootstrapStreamlineToDistanceTableFilter.h"

#include "vtkObjectFactory.h"
#include "vtkCellArray.h"
#include "vtkPolyDataWriter.h"

#include <cassert>
#include <map>

#ifdef _WIN32
#define CACHE "M:/datasets/bootstrapping/cache/temporary"
#else
#define CACHE "/home/rbrecheis/datasets/bootstrapping/cache/temporary"
#endif

namespace bmia {

	vtkStandardNewMacro( vtkBootstrapStreamlineToDistanceTableFilter );

	////////////////////////////////////////////////////////////////////////
	vtkBootstrapStreamlineToDistanceTableFilter::vtkBootstrapStreamlineToDistanceTableFilter()
	{
		this->Measure = NULL;
		this->Input = NULL;
        this->InputOrg = NULL;
        this->Output = NULL;
        this->OutputOrg = NULL;
		this->OutputBundle = NULL;
		this->NumberOfSeedPoints = 0;
        this->SeedPointIds = NULL;
	}

	////////////////////////////////////////////////////////////////////////
	vtkBootstrapStreamlineToDistanceTableFilter::~vtkBootstrapStreamlineToDistanceTableFilter()
	{
        if( this->Measure )
            this->Measure->UnRegister( this );

		if( this->Input )
			this->Input->UnRegister( this );
        if( this->InputOrg )
            this->InputOrg->UnRegister( this );

		if( this->Output )
			this->Output->Delete();
        if( this->OutputOrg )
            this->OutputOrg->Delete();
		if( this->OutputBundle )
			this->OutputBundle->Delete();
	}

	////////////////////////////////////////////////////////////////////////
	void vtkBootstrapStreamlineToDistanceTableFilter::Execute()
	{
		// Clear output tables

        if( this->Output )
            this->Output->Delete();
        this->Output = vtkDistanceTable::New();

		if( this->OutputOrg )
			this->OutputOrg->Delete();
		this->OutputOrg = vtkDistanceTable::New();

		if( this->OutputBundle )
			this->OutputBundle->Delete();
		this->OutputBundle = vtkDistanceTable::New();

		// Compute distances

		this->GetDistanceMeasure()->SetPoints( this->GetInput()->GetPoints() );

        vtkCellArray * lines = this->GetInput()->GetLines();
        int nrLines = this->GetInput()->GetNumberOfLines();
        int nrSeeds = this->GetNumberOfSeedPoints();

		// Create map linking each seedpoint ID with its list of streamlines
        std::map<int, std::vector<int> > groups;
		vtkIdType offset = 0;

        for( int i = 0; i < nrLines; ++i )
        {
			vtkIdType nrPtIds;
			vtkIdType * ptIds = NULL;
            lines->GetCell( offset, nrPtIds, ptIds );
            int seedId = this->GetSeedPointIds()->at( i );
            groups[seedId].push_back( offset );
            offset += (nrPtIds + 1 );
        }

        // Compute for each seedpoint, the distance of streamlines to the centerline
        // and the original fiber.

        std::map<int, std::vector<int> >::iterator iter = groups.begin();
        for( ; iter != groups.end(); ++iter )
        {
            std::vector<int> & offsets = (*iter).second;
            int seedId = (*iter).first;
            int nrOffsets = static_cast<int>(offsets.size());

			// Set up pair-wise distance table
			double ** distances = new double*[nrOffsets];
			for( int i = 0; i < nrOffsets; ++i )
			{
				distances[i] = new double[nrOffsets];
				for( int j = 0; j < nrOffsets; ++j )
					distances[i][j] = 0.0f;
			}

			for( int i = 0; i < nrOffsets; ++i )
			{
				vtkIdType firstOffset = (*iter).second[i];
				vtkIdType firstNrPtIds;
				vtkIdType * firstPtIds = NULL;
				lines->GetCell( firstOffset, firstNrPtIds, firstPtIds );

				for( int j = 0; j < i; ++j )
				{
					vtkIdType secondOffset = (*iter).second[j];
					vtkIdType secondNrPtIds;
					vtkIdType * secondPtIds = NULL;
					lines->GetCell( secondOffset, secondNrPtIds, secondPtIds );

					double d = this->GetDistanceMeasure()->Compute( firstPtIds, firstNrPtIds, secondPtIds, secondNrPtIds );
					if( d < 0.0f ) d = 0.0f;

					distances[i][j] = d;
					distances[j][i] = d;
				}
			}

			double minimum = VTK_DOUBLE_MAX;
			int centerIdx = -1;

			for( int i = 0; i < nrOffsets; ++i )
			{
				double total = 0.0;
				for( int j = 0; j < nrOffsets; ++j )
					total += distances[i][j];
				if( total < minimum )
				{
					minimum = total;
					centerIdx = i;
				}
			}

			assert( centerIdx != -1 );

			vtkIdType centerOffset = (*iter).second[centerIdx];
			vtkIdType centerNrPtIds;
			vtkIdType * centerPtIds = NULL;
			lines->GetCell( centerOffset, centerNrPtIds, centerPtIds );

			// Store distances to center streamline
			for( int i = 0; i < nrOffsets; ++i )
				this->Output->Add( distances[centerIdx][i], (*iter).second[i] );

			// Store distances to the streamline from original dataset
			for( int i = 0; i < nrOffsets; ++i )
				this->OutputOrg->Add( distances[0][i], (*iter).second[i] );

			for( int i = 0; i < nrOffsets; ++i )
				delete [] distances[i];
			delete [] distances;
		}

		this->Output->Sort();
		this->Output->Normalize();

		this->OutputOrg->Sort();
		this->OutputOrg->Normalize();

		iter = groups.begin();
		for( ; iter != groups.end(); ++iter )
			(*iter).second.clear();
		groups.clear();

		// Set bundle output to be same as original
		this->OutputBundle = this->OutputOrg;

		return;

		// Compute distance matrix for whole set of streamlines

		double ** distances = new double*[nrLines];
		for( int i = 0; i < nrLines; ++i )
			distances[i] = new double[nrLines];

		vtkIdType firstOffset = 0;
        for( int i = 0; i < nrLines; ++i )
        {
			vtkIdType firstNrPtIds;
			vtkIdType * firstPtIds = NULL;
			lines->GetCell( firstOffset, firstNrPtIds, firstPtIds );
			firstOffset += (firstNrPtIds + 1);

			vtkIdType secondOffset = 0;
            for( int j = 0; j < i; ++j )
            {
				vtkIdType secondNrPtIds;
				vtkIdType * secondPtIds = NULL;
                lines->GetCell( secondOffset, secondNrPtIds, secondPtIds );
				secondOffset += (secondNrPtIds + 1);

				double d = this->GetDistanceMeasure()->Compute( firstPtIds, firstNrPtIds, secondPtIds, secondNrPtIds );

				distances[i][j] = d;
				distances[j][i] = d;
            }
        }

		// Compute centerline of whole set of streamlines

		double minimum = VTK_DOUBLE_MAX;

		vtkIdType centerOffset = -1;
		vtkIdType centerIdx = -1;
		offset = 0;

		for( int i = 0; i < nrLines; ++i )
		{
			vtkIdType nrPtIds;
			vtkIdType * ptIds = NULL;
			lines->GetCell( offset, nrPtIds, ptIds );

			double total = 0.0;
			for( int j = 0; j < nrLines; ++j )
				total += distances[i][j];

			if( total < minimum )
			{
				minimum = total;
				centerIdx = i;
				centerOffset = offset;
			}

			offset += (nrPtIds + 1);
		}

		assert( centerIdx != -1 );
		assert( centerOffset != -1 );

		// Get centerline and save it

		vtkIdType centerNrPtIds;
		vtkIdType * centerPtIds = NULL;
		lines->GetCell( centerOffset, centerNrPtIds, centerPtIds );

		offset = 0;
		for( int i = 0; i < nrLines; ++i )
		{
			vtkIdType nrPtIds;
			vtkIdType * ptIds = NULL;
			lines->GetCell( offset, nrPtIds, ptIds );

			this->OutputBundle->Add( distances[centerIdx][i], offset );

			offset += (nrPtIds + 1);
		}

		this->OutputBundle->Sort();
		this->OutputBundle->Normalize();

		for( int i = 0; i < nrLines; ++i )
			delete [] distances[i];
		delete [] distances;
	}

	////////////////////////////////////////////////////////////////////////
	void vtkBootstrapStreamlineToDistanceTableFilter::SetInput( vtkPolyData * _input )
	{
		if( this->Input )
			this->Input->UnRegister( this );
		this->Input = _input;
		if( this->Input )
			this->Input->Register( this );
	}

	////////////////////////////////////////////////////////////////////////
	vtkPolyData * vtkBootstrapStreamlineToDistanceTableFilter::GetInput()
	{
		return this->Input;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkBootstrapStreamlineToDistanceTableFilter::SetInputOrg( vtkPolyData * _input )
	{
        if( this->InputOrg )
            this->InputOrg->UnRegister( this );
        this->InputOrg = _input;
        if( this->InputOrg )
            this->InputOrg->Register( this );
	}

	////////////////////////////////////////////////////////////////////////
    vtkPolyData * vtkBootstrapStreamlineToDistanceTableFilter::GetInputOrg()
	{
        return this->InputOrg;
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkBootstrapStreamlineToDistanceTableFilter::SetDistanceMeasure( vtkDistanceMeasure * _measure )
    {
        if( this->Measure )
            this->Measure->UnRegister( this );
        this->Measure = _measure;
        if( this->Measure )
            this->Measure->Register( this );
    }

    ////////////////////////////////////////////////////////////////////////
    vtkDistanceMeasure * vtkBootstrapStreamlineToDistanceTableFilter::GetDistanceMeasure()
    {
        return this->Measure;
    }

    ////////////////////////////////////////////////////////////////////////
	void vtkBootstrapStreamlineToDistanceTableFilter::SetSeedPointIds( std::vector<int> * _pointIds )
	{
		assert( _pointIds != NULL );
		this->SeedPointIds = _pointIds;
	}

	////////////////////////////////////////////////////////////////////////
	std::vector<int> * vtkBootstrapStreamlineToDistanceTableFilter::GetSeedPointIds() const
	{
		return this->SeedPointIds;
	}

    ////////////////////////////////////////////////////////////////////////
	void vtkBootstrapStreamlineToDistanceTableFilter::SetNumberOfSeedPoints( const int _nrPoints )
	{
		assert( _nrPoints > 0 );
		this->NumberOfSeedPoints = _nrPoints;
	}

	////////////////////////////////////////////////////////////////////////
	int vtkBootstrapStreamlineToDistanceTableFilter::GetNumberOfSeedPoints() const
	{
		return this->NumberOfSeedPoints;
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceTable * vtkBootstrapStreamlineToDistanceTableFilter::GetOutput()
	{
		return this->Output;
	}

	////////////////////////////////////////////////////////////////////////
    vtkDistanceTable * vtkBootstrapStreamlineToDistanceTableFilter::GetOutputOrg()
	{
        return this->OutputOrg;
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceTable * vtkBootstrapStreamlineToDistanceTableFilter::GetOutputBundle()
	{
		return this->OutputBundle;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkBootstrapStreamlineToDistanceTableFilter::SaveCenterLine( vtkPoints * _points, int _nrPtIds, int * _ptIds, const char * _fileName )
    {
        vtkPolyData * fiber = vtkPolyData::New();

        vtkCellArray * cells = vtkCellArray::New();
        cells->Allocate( 1 );

        vtkPoints * points = vtkPoints::New();
        points->Allocate( 2500 );

        vtkIdList * ids = vtkIdList::New();
        ids->Allocate( 2500 );

        for( int i = 0; i < _nrPtIds; i++ )
        {
            double pt[3];
            _points->GetPoint( _ptIds[i], pt );

            vtkIdType id = points->InsertNextPoint( pt[0], pt[1], pt[2] );
            ids->InsertNextId( id );
        }

        cells->InsertNextCell( ids );

        fiber->SetLines( cells );
        fiber->SetPoints( points );

        vtkPolyDataWriter * writer = vtkPolyDataWriter::New();
        writer->SetFileName( _fileName );
        writer->SetInput( fiber );
        writer->Write();

        writer->Delete();
        fiber->Delete();
    }
}

#undef CACHE
