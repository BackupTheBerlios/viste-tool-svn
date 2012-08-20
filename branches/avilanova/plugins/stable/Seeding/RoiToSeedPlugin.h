/*
 * RoiToSeedPlugin.h
 *
 * 2010-10-29	Evert van Aart
 * - First Version.
 *
 * 2010-12-15	Evert van Aart
 * - Added support for voxel seeding.
 * - Seed distance is now read from data set attributes.
 *
 * 2011-03-16	Evert van Aart
 * - Version 1.0.0.
 * - Removed the need to compute the normal for primary planes, making the seeding
 *   more robust for elongated ROIs.
 * - Increased stability for voxel seeding when a ROI is touching the edges of
 *   an image. 
 *
 */

 
/** ToDo List for "RoiToSeedPlugin"
	Last updated 15-12-2010 by Evert van Aart

    - Add support for 3D ROI and other types/shapes of ROIs when
	  they are implemented.
*/


#ifndef bmia_RoiToSeedPlugin_h
#define bmia_RoiToSeedPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>


namespace bmia {


/** Forward Class Declaration */

class vtk2DRoiToSeedFilter;


/** Plugin that accepts regions of interest (ROIs) as input, and generates
	a set of seed points as output. Seed points are generated on a regular
	1D, 2D or 3D grid (depending on the type of the ROI); the distance between
	the seed points is defined in the attributes of the input data set. 
*/

class RoiToSeedPlugin : 	public plugin::Plugin,
							public data::Consumer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return the current version. */

		QString getPluginVersion()
		{
			return "1.0.0";
		}
	
		/** Seeding method. Either uses seed distance (seed points are planed on a regular
			grid, the spacing is which is controlled by an attribute), or seed voxels 
			(seed points are placed on each voxel in an image within the ROI; the 
			image pointer is passed as an attribute). */

		enum SeedMethod
		{
			SM_Distance = 0,
			SM_Voxels
		};

		/** Constructor */

		RoiToSeedPlugin();

		/** Destructor */

		~RoiToSeedPlugin();

		/** Define behavior when new data set are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */

		void dataSetAdded(data::DataSet * ds);
		void dataSetChanged(data::DataSet * ds);
		void dataSetRemoved(data::DataSet * ds);

private:

		/** Structure containing information about a single ROI. */
	
		struct ROIInfo
		{
			data::DataSet * inputDS;	// Input data set (ROIs)
			vtkPolyData *   inputPD;	// "vtkPolyData" object of input
			data::DataSet * outputDS;	// Output data set (Seed points)

			QString inputName;			// Name of input data set
			QString outputName;			// Name of output data set
		
			// Filter used to generate the seed points
			vtkDataSetToUnstructuredGridFilter * ROIFilter;
		};

		/** List containing information about all ROIs, for future reference. */

		QList<ROIInfo> ROIInfoList;

}; // class RoiToSeedPlugin


} // namespace bmia


#endif // bmia_RoiToSeedPlugin_h