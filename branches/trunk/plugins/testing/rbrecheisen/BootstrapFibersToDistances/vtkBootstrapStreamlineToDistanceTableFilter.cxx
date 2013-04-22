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
#include <vector>
#include <map>

#define CACHE_DIRECTORY "/Users/Ralph/Temp/Cache/"

//namespace bmia {

	vtkStandardNewMacro( vtkBootstrapStreamlineToDistanceTableFilter );

	////////////////////////////////////////////////////////////////////////
	vtkBootstrapStreamlineToDistanceTableFilter::vtkBootstrapStreamlineToDistanceTableFilter()
	{
		this->Measure = NULL;
		this->Input = NULL;
        this->Output = NULL;
		this->FiberIds = NULL;
		this->NumberOfSeedPoints = 0;
	}

	////////////////////////////////////////////////////////////////////////
	vtkBootstrapStreamlineToDistanceTableFilter::~vtkBootstrapStreamlineToDistanceTableFilter()
	{
        if( this->Measure )
            this->Measure->UnRegister( this );

		if( this->Input )
			this->Input->UnRegister( this );

		if( this->Output )
			this->Output->Delete();
	}

	////////////////////////////////////////////////////////////////////////
	void vtkBootstrapStreamlineToDistanceTableFilter::Execute()
	{
		// Clear output tables

        if( this->Output )
            this->Output->Delete();
        this->Output = vtkDistanceTable::New();

		// Compute distances

		this->GetDistanceMeasure()->SetPoints( this->GetInput()->GetPoints() );

        vtkCellArray * lines = this->GetInput()->GetLines();
        int nrLines = this->GetInput()->GetNumberOfLines();
		int nrSeeds = this->GetNumberOfSeedPoints();

		std::cout << "vtkBootstrapStreamlineToDistanceTableFilter::Execute() ";
		std::cout << "creating mapping between ID's and streamlines..." << std::endl;

		// Create map linking each seedpoint ID with its list of streamlines
        std::map<int, std::vector<int> > groups;
		vtkIdType offset = 0;

        for( int i = 0; i < nrLines; ++i )
        {
			vtkIdType nrPtIds;
			vtkIdType * ptIds = NULL;
            lines->GetCell( offset, nrPtIds, ptIds );
			int seedId = this->GetFiberIds()->GetValue( i );
            groups[seedId].push_back( offset );
            offset += (nrPtIds + 1 );
        }

		std::cout << "vtkBootstrapStreamlineToDistanceTableFilter::Execute()";
		std::cout << "computing distances for " << nrSeeds << " seed points..." << std::endl;

		// Compute for each seedpoint, the distance of streamlines to the centerline
        // and the original fiber.
        std::map<int, std::vector<int> >::iterator iter = groups.begin();
        for( ; iter != groups.end(); ++iter )
        {
			std::vector<int> & offsets = (*iter).second;
            int seedId = (*iter).first;
            int nrOffsets = static_cast<int>(offsets.size());

			std::cout << "vtkBootstrapStreamlineToDistanceTableFilter::Execute()";
			std::cout << "computing distances for seed point " << seedId << "..." << std::endl;

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

			for( int i = 0; i < nrOffsets; ++i )
				delete [] distances[i];
			delete [] distances;
		}

		this->Output->Sort();
		this->Output->Normalize();

		iter = groups.begin();
		for( ; iter != groups.end(); ++iter )
			(*iter).second.clear();
		groups.clear();
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
	void vtkBootstrapStreamlineToDistanceTableFilter::SetFiberIds( vtkIntArray * _fiberIds )
	{
		assert( _fiberIds != NULL );
		this->FiberIds = _fiberIds;
	}

	////////////////////////////////////////////////////////////////////////
	vtkIntArray * vtkBootstrapStreamlineToDistanceTableFilter::GetFiberIds()
	{
		return this->FiberIds;
	}

    ////////////////////////////////////////////////////////////////////////
	void vtkBootstrapStreamlineToDistanceTableFilter::SetNumberOfSeedPoints( int _nrPoints )
	{
		assert( _nrPoints > 0 );
		this->NumberOfSeedPoints = _nrPoints;
	}

	////////////////////////////////////////////////////////////////////////
	int vtkBootstrapStreamlineToDistanceTableFilter::GetNumberOfSeedPoints()
	{
		return this->NumberOfSeedPoints;
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceTable * vtkBootstrapStreamlineToDistanceTableFilter::GetOutput()
	{
		return this->Output;
	}
//}

#undef CACHE_DIRECTORY
