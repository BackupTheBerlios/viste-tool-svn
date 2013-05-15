/**
 * vtkBootstrapTensorFieldToStreamlineDistribution.h
 * by Ralph Brecheisen
 *
 * 2009-12-07	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkBootstrapTensorFieldToStreamlineDistribution_h
#define bmia_vtkBootstrapTensorFieldToStreamlineDistribution_h

#include "vtkDataSetToPolyDataFilter.h"
#include "vtkAlgorithmOutput.h"
#include "vtkInformation.h"
#include "vtkDataSet.h"

#include "vtkTensorFieldToStreamline.h"

#include <vector>
#include <string>

namespace bmia
{
	/**
	 * Class that loads a set of tensor datasets obtained from a wild bootstrapping
	 * procedure and runs fiber tracking on them. The class does not inherit from
	 * vtkTensorFieldToStreamline because it is not a specialization of it. It simply
	 * repeats fiber tracking on a set of tensor datasets. The most important reason
	 * however, is that we want to be able to use our class with different FT algorithms.
	 * Having it inherit from a specific type of FT algorithm would not make sense in
	 * that case.
	 */
	class vtkBootstrapTensorFieldToStreamlineDistribution : public vtkDataSetToPolyDataFilter
	{
	public:

		/** No comment */
		vtkTypeMacro( vtkBootstrapTensorFieldToStreamlineDistribution, vtkDataSetToPolyDataFilter );

		/** Creates new instance of filter */
		static vtkBootstrapTensorFieldToStreamlineDistribution * New();

		/** Sets/gets streamline filter that traces the stremalines through the dataset */
		void SetTracker( vtkTensorFieldToStreamline * _filter );
		vtkTensorFieldToStreamline * GetTracker();

		/** Sets/gets the integration step length in voxels (default: 0.1) */
		void SetStepLength( float _stepLen );
		float GetStepLength();

		/** Sets/gets the anisotropy threshold between [0,1] */
		void SetAnisotropyThreshold( float _threshold );
		float GetAnisotropyThreshold();

		/** Sets/gets the curve threshold in degrees [0,359] */
		void SetAngularThreshold( float _degrees );
		float GetAngularThreshold();

		/** Sets/gets the minimum fiber length in mm. */
		void SetMinimumFiberLength( float _length );
		float GetMinimumFiberLength();

		/** Sets/gets the maximum fiber length in mm. */
		void SetMaximumFiberLength( float _length );
		float GetMaximumFiberLength();

		/** Sets/gets the anisotropy measure index */
		void SetAnisotropyMeasure( int _measure );
		int GetAnisotropyMeasure();

		/** Sets/gets the simplification step length */
		void SetSimplificationStepLength( float _stepLen );
		float GetSimplificationStepLength();

		/** Sets list of filenames for bootstrap datasets. Internally, a copy is made of
			the list so you can delete the list provided as the function's parameter */
		void SetFileNames( std::vector<std::string> * _fileNames );
		std::vector<std::string> * GetFileNames();

		/** Sets/gets the seed point data source */
		void SetSource( vtkDataSet * _seedPoints );
		vtkDataSet * GetSource();

		/** Returns list of seed-point ID's corresponding to the list of streamlines
		    in the output polydata */
		std::vector<int> * GetSeedPointIds() const;

	protected:

		/** Constructor/destructor */
		vtkBootstrapTensorFieldToStreamlineDistribution();
		virtual ~vtkBootstrapTensorFieldToStreamlineDistribution();

		/** Executes the filter */
		void Execute();

	private:

		/** NOT IMPLEMENTED 
			copy constructor and assignment operator */
		vtkBootstrapTensorFieldToStreamlineDistribution( const vtkBootstrapTensorFieldToStreamlineDistribution & );
		void operator = ( const vtkBootstrapTensorFieldToStreamlineDistribution & );

		/** Builds output by loading cached streamlines from file and collecting
			them into a single poly dataset */
		void BuildOutput();

		/** Assigns an ID to each seed point in the source data. These ID's are later used
		    to identify individual streamlines */
		void InsertSeedPointIds();

		/** Removes redundant seed point ID's from each streamline point, except
		    the first one in the list */
		vtkPolyData * RemoveSeedPointIds( vtkPolyData * _lines );

		/** Updates table of seed-point ID's */
        void UpdateSeedPointIds( vtkPolyData * _lines );

		/** Clears the cache directory from all temporary VTK files */
		void ClearCacheDirectory();

		vtkTensorFieldToStreamline * StreamlineFilter;
		vtkDataSet * SeedPoints;
		std::vector<std::string> * FileNames;

		float SimplificationStepLength;
		float StepLength;
		float AnisotropyThreshold;
		float AngularThreshold;
		float MinimumFiberLength;
		float MaximumFiberLength;

		int AnisotropyMeasure;

		std::vector<int> * SeedPointIds;
	};

} // namespace bmia

#endif // bmia_vtkBootstrapTensorFieldToStreamlineDistribution_h
