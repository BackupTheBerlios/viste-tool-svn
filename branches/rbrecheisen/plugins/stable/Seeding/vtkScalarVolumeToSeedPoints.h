/*
 * vtkScalarVolumeToSeedPoints.h
 *
 * 2011-05-10	Evert van Aart
 * - First version
 *
 */


#ifndef bmia_SeedingPlugin_vtkScalarVolumeToSeedPoints_h
#define bmia_SeedingPlugin_vtkScalarVolumeToSeedPoints_h


/** Includes - VTK */

#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkObjectFactory.h>


namespace bmia {


/** Simple filter that generates a seed point for every voxel with a scalar value
	that lies between a lower and an upper threshold value. Used for volume seeding.
*/

class vtkScalarVolumeToSeedPoints : public vtkDataSetToUnstructuredGridFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkScalarVolumeToSeedPoints, vtkDataSetToUnstructuredGridFilter);

		/** Constructor Call */

		static vtkScalarVolumeToSeedPoints * New();

		/** Set the lower threshold.
			@param rT			Desired lower threshold. */

		void setMinThreshold(double rT)
		{
			minThreshold = rT;
		}

		/** Set the upper threshold.
			@param rT			Desired upper threshold. */

		void setMaxThreshold(double rT)
		{
			maxThreshold = rT;
		}

	protected:

		/** Constructor */

		vtkScalarVolumeToSeedPoints();

		/** Destructor */

		~vtkScalarVolumeToSeedPoints();

		/** Main entry point for the execution of the filter. */

		void Execute();

		/** Lower threshold. */

		double minThreshold;

		/** Upper threshold. */

		double maxThreshold;

}; // class vtkScalarVolumeToSeedPoints


} // namespace bmia


#endif // bmia_SeedingPlugin_vtkScalarVolumeToSeedPoints_h