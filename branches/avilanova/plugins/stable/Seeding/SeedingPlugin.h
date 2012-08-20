/*
 * SeedingPlugin.h
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.0.0.
 * - First version. Currently, the GUI controls include support for 2D ROIs, 
 *   scalar volumes, and fibers, but volume seeding has not yet been implemented.
 *
 * 2011-05-10	Evert van Aart
 * - Version 1.1.0.
 * - Added support for volume seeding.
 *
 */


#ifndef bmia_SeedingPlugin_h
#define bmia_SeedingPlugin_h


/** Includes - Main Headers */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_Seeding.h"

/** Includes - Qt */

#include <QDebug>

/** Includes - VTK */

#include <vtkPointSet.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>


/** Forward Declarations */

namespace Ui 
{
	class SeedingForm;
}


namespace bmia {


/** Forward Declarations */

class vtk2DRoiToSeedFilter;
class vtkStreamlineToSimplifiedStreamline;
class vtkPolyDataToSeedPoints;
class vtkScalarVolumeToSeedPoints;


/** This plugin is used to generate seed points. It currently supports these 
	methods for seed point generation:
	- 2D ROIs (using either a fixed distance, or voxel seeding).
	- Fibers (either using the original fiber points, or using simplified fibers
	  in which all line segments have the same length).
    - Scalar volumes (places a seed point at every voxel for which the scalar
	  value lies between an upper and lower threshold value).
    Other methods can be added in the future. The different methods for point
	generation are pretty much independent of each other. The output for all 
	methods is a seed point data set. 
*/

class SeedingPlugin :	public plugin::Plugin,
						public data::Consumer,
						public plugin::GUI
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::GUI)

	public:
	
		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.1.0";
		}

		/** Constructor. */

		SeedingPlugin();

		/** Destructor. */

		~SeedingPlugin();

		/** Initializes the plugin. */

		void init();

		/** Returns the Qt widget that gives the user control.
			This implements the GUI interface. */
    
		QWidget * getGUI();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		void dataSetAdded(data::DataSet * ds);
    
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */

		void dataSetChanged(data::DataSet * ds);

		/** The data manager calls this function whenever an existing
			data set is removed.
			@param ds	Modified data set. */
   
		void dataSetRemoved(data::DataSet * ds);

		/** Different seeding types for 2D ROIs. Distance seeding places all seed
			points on a planar grid with fixed distance in both directions. Voxel
			seeding means that the seed points will be placed on the voxels of 
			a selected image. */

		enum roiSeedingType
		{
			RST_NoSeeding = 0,		/**< No seeding. */
			RST_Distance,			/**< Distance seeding. */
			RST_Voxel				/**< Voxel seeding. */
		};

	protected slots:

		/** Copy the settings of the selected ROI to the GUI. Called when the user
			selects a new ROI, or when the settings of the currently selected ROI
			are changed. Also enables or disables GUI controls based on the settings. */

		void setupGUIForROIs();

		/** Copy the settings of the selected fibers to the GUI. Called when the user
			selects a new fibers, or when the settings of the currently selected fibers
			are changed. Also enables or disables GUI controls based on the settings. */

		void setupGUIForFibers();

		/** Copy the settings of the selected volume to the GUI. Called when the user
			selects a new volume, or when the settings of the currently selected volume
			are changed. Also enables or disables GUI controls based on the settings. */

		void setupGUIForVolumes();

		/** Called when the settings for the currently selected ROI are changed. 
			Updates the corresponding ROI information object, and then update
			the output seed points. */

		void ROISettingsChanged();

		/** Called when the settings for the currently selected fibers are changed. 
			Updates the corresponding fiber information object, and then update
			the output seed points. */

		void FiberSettingsChanged();

		/** Called when the settings for the currently selected volume are changed. 
			Updates the corresponding volume information object, and then update
			the output seed points. */

		void VolumeSettingsChanged();


	private:

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt designer. */

		Ui::SeedingForm * ui;

		/** Information about the seed points for a 2D ROI. */

		struct roiSeedInfo
		{
			roiSeedingType seedType;			/**< Seeding method (see "roiSeedingType"). */
			double distance;					/**< Seed distance. */
			int imageID;						/**< Index of the image used for voxel seeding. */
			data::DataSet * inDS;				/**< Input data set (2D ROI). */
			data::DataSet * outDS;				/**< Output data set (seed points). */
			vtk2DRoiToSeedFilter * filter;		/**< Filter used to generate seed points. */
		};

		/** Information about the seed points for a set of fibers. */

		struct fiberSeedInfo
		{
			bool enable;						/**< Enable or disable fiber seeding. */
			double distance;					/**< fixed seed distance. */
			bool doFixedDistance;				/**< Apply fixed seed distance (simplification). */
			data::DataSet * inDS;				/**< Input data set (2D ROI). */
			data::DataSet * outDS;				/**< Output data set (seed points). */

			/** Simplification filter. When active, all seed points on one fiber
				will be placed a fixed distance from each other. */

			vtkStreamlineToSimplifiedStreamline * simplifyFilter;

			/** Seed filter. Simple filter that generates a seed point set from
				a "vtkPolyData" object. */

			vtkPolyDataToSeedPoints * seedFilter;
		};

		/** Information about the seed points for a scalar volume. */

		struct volumeSeedInfo
		{
			bool enable;						/**< Enable or disable scalar volume seeding. */
			double minValue;					/**< Lower threshold. */
			double maxValue;					/**< Upper threshold. */
			data::DataSet * inDS;				/**< Input data set (Scalar volume). */			
			data::DataSet * outDS;				/**< Output data set (seed points). */
			bool initializedRange;				/**< False until volume is first selected for seeding. */

			/** Seed filter. Generates a seed point at every voxel for which the
				scalar values lies between the two thresholds. */

			vtkScalarVolumeToSeedPoints * seedFilter;
		};

		/** List with seeding information for all 2D ROIs. */

		QList<roiSeedInfo> roiInfoList;

		/** List with seeding information for all fiber sets. */

		QList<fiberSeedInfo> fiberInfoList;

		/** List with seeding information for all scalar volumes. */

		QList<volumeSeedInfo> volumeInfoList;

		/** List of images that can be used to generate voxel seeds. All data sets
			that contain a "vtkImageData" object are added to this list. */

		QList<vtkImageData *> roiVoxelImages;

		/** Connect or disconnect controls for the ROI group.
			@param doConnect	Connect controls if true, disconnect otherwise. */

		void connectControlsForROIs(bool doConnect);

		/** Connect or disconnect controls for the fibers group.
			@param doConnect	Connect controls if true, disconnect otherwise. */

		void connectControlsForFibers(bool doConnect);

		/** Connect or disconnect controls for the volumes group.
			@param doConnect	Connect controls if true, disconnect otherwise. */

		void connectControlsForVolumes(bool doConnect);

		/** Update the seed points for the ROI represented by the input information
			struct. Either updates the output data set, or, if "No Seeding" has been
			selected, removes the existing output data set. 
			@param info			Information about the seed points that need to be updated. */

		void updateROISeeds(roiSeedInfo &info);

		/** Update the seed points for the fiber set represented by the input information
			struct. Either updates the output data set, or, if "No Seeding" has been
			selected, removes the existing output data set. 
			@param info			Information about the seed points that need to be updated. */

		void updateFiberSeeds(fiberSeedInfo &info);

		/** Update the seed points for the scalar volume represented by the input information
			struct. Either updates the output data set, or, if "No Seeding" has been
			selected, removes the existing output data set. 
			@param info			Information about the seed points that need to be updated. */

		void updateVolumeSeeds(volumeSeedInfo &info);

}; // class SeedingPlugin


} // namespace bmia


#endif // bmia_SeedingPlugin_h
