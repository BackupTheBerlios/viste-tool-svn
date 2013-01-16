/**
 * vtkBootstrapStreamlineToDistanceTableFilter.h
 * by Ralph Brecheisen
 *
 * 2009-12-07	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkBootstrapStreamlineToDistanceTableFilter_h
#define bmia_vtkBootstrapStreamlineToDistanceTableFilter_h

#include "vtkObject.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"

#include "vtkDistanceMeasure.h"
#include "vtkDistanceTable.h"
#include "vtkDistanceTableCollection.h"

#include <vector>

namespace bmia
{
	/**
	 * This class generates a streamline distance table from a collection of bootstrapped 
	 * streamlines. Each streamline has an associated seedpoint that it originated from.
	 * The output distance table contains, for each streamline, its distance to the center
	 * streamline running from the same seedpoint. So, if we have 1000 streamlines per
	 * seedpoint (because we ran the bootstrapping 1000 times), then one of these streamlines
	 * will be the most central streamline because it has the lowest summed distance to all
	 * the others.
	 */
	class vtkBootstrapStreamlineToDistanceTableFilter : public vtkObject
	{
	public:

		/** Creates new instance of the filter */
		static vtkBootstrapStreamlineToDistanceTableFilter * New();

		/** Sets/gets the input for this filter */
		void SetInput( vtkPolyData * _input );
		vtkPolyData * GetInput();

        /** NOT NEEDED ANYMORE? Sets/gets input of original fibers */
        void SetInputOrg( vtkPolyData * _input );
        vtkPolyData * GetInputOrg();

		/** Sets/gets the list of seed points ID's associated with each streamline */
		void SetSeedPointIds( std::vector<int> * _pointIds );
		std::vector<int> * GetSeedPointIds() const;

		/** Sets/gets the number of seedpoints */
		void SetNumberOfSeedPoints( const int _nrPoints );
		int GetNumberOfSeedPoints() const;

		/** Sets/gets the distance measure to be used for creating the table */
		void SetDistanceMeasure( vtkDistanceMeasure * _measure );
        vtkDistanceMeasure * GetDistanceMeasure();

		/** Executes the filter */
		void Execute();

		/** Returns the output distance table of this filter. These are the
		    distances with respect to the median fiber in each seedpoint */
		vtkDistanceTable * GetOutput();

		/** Returns the output distance table of this filter. These are the
		    distances with respect to the original fiber in each seedpoint */
        vtkDistanceTable * GetOutputOrg();

        /** Returns distances of each streamline with respect to the bundle
            centerline instead of each seedpoint's centerline */
        vtkDistanceTable * GetOutputBundle();

	protected:

		/** Constructor and destructor */
		vtkBootstrapStreamlineToDistanceTableFilter();
		virtual ~vtkBootstrapStreamlineToDistanceTableFilter();

        void SaveCenterLine( vtkPoints * _points, int _nrPtIds, int * _ptIds, const char * _fileName );

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
		vtkBootstrapStreamlineToDistanceTableFilter( const vtkBootstrapStreamlineToDistanceTableFilter & );
		void operator = ( const vtkBootstrapStreamlineToDistanceTableFilter & );

		std::vector<int> * SeedPointIds;

		int NumberOfSeedPoints;

        vtkPolyData * InputOrg; // NOT NEEDED ANYMORE?
		vtkPolyData * Input;
		vtkDistanceTable * Output;
        vtkDistanceTable * OutputOrg;
        vtkDistanceTable * OutputBundle;
		vtkDistanceMeasure * Measure;
	};

} // namespace bmia

#endif // bmia_vtkBootstrapStreamlineToDistanceTableFilter_h
