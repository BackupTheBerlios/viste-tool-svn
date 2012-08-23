/*
 * vtkPolyDataToSeedPoints.h
 *
 * 2011-04-18	Evert van Aart
 * - First Version.
 *
 */


#ifndef bmia_SeedingPlugin_vtkPolyDataToSeedPoints_h
#define bmia_SeedingPlugin_vtkPolyDataToSeedPoints_h


/** Includes - VTK */

#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkObjectFactory.h>


namespace bmia {


/** Very simple class that takes a "vtkPolyData" object as input, and uses it to
	generate a "vtkUnstructuredGrid" object with all points of the polydata. There
	are easier ways to do this, but using this filter has the advantage that the
	pipeline will automatically update if the fibers change. 
*/

class vtkPolyDataToSeedPoints : public vtkDataSetToUnstructuredGridFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkPolyDataToSeedPoints, vtkDataSetToUnstructuredGridFilter);

		/** Constructor Call */

		static vtkPolyDataToSeedPoints * New();

		/** Constructor */

		vtkPolyDataToSeedPoints()
		{
			
		};

		/** Destructor */

		~vtkPolyDataToSeedPoints()
		{
			
		};

	protected:

		/** Main point of entry of the filter. */

		void Execute();

};
		

} // namespace


#endif // bmia_SeedingPlugin_vtkPolyDataToSeedPoints_h
