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
#include "vtkIntArray.h"

#include "vtkDistanceMeasure.h"
#include "vtkDistanceTable.h"

//namespace bmia
//{
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

		/** Sets/gets the list of seed points ID's associated with each streamline */
		void SetFiberIds( vtkIntArray * _fiberIds );
		vtkIntArray * GetFiberIds();

		/** Sets the number of seed points */
		void SetNumberOfSeedPoints( int numberPoints );
		int GetNumberOfSeedPoints();

		/** Sets/gets the distance measure to be used for creating the table */
		void SetDistanceMeasure( vtkDistanceMeasure * _measure );
        vtkDistanceMeasure * GetDistanceMeasure();

		/** Executes the filter */
		void Execute();

		/** Returns the output distance table of this filter. These are the
		    distances with respect to the median fiber in each seedpoint */
		vtkDistanceTable * GetOutput();

	protected:

		/** Constructor */
		vtkBootstrapStreamlineToDistanceTableFilter();

		/** Destructor */
		virtual ~vtkBootstrapStreamlineToDistanceTableFilter();

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
		vtkBootstrapStreamlineToDistanceTableFilter( const vtkBootstrapStreamlineToDistanceTableFilter & );
		void operator = ( const vtkBootstrapStreamlineToDistanceTableFilter & );

		vtkIntArray * FiberIds;

		vtkPolyData * Input;
		vtkDistanceTable * Output;
		vtkDistanceMeasure * Measure;

		int NumberOfSeedPoints;
	};

//} // namespace bmia

#endif // bmia_vtkBootstrapStreamlineToDistanceTableFilter_h
