/*
 * HARDIFiberTrackingPlugin.h
 *
 * 2011-10-14	Anna Vilanova
 * - Version 1.0.0.
 * - First version
 *
 * 2013-25-03 Mehmet Yusufoglu, Bart Van Knippenberg
 * - dataSetAdded function also accepts "discrete sphere" as input to the list. 
 *
 */



#ifndef bmia_HARDIFiberTrackingPlugin_h
#define bmia_HARDIFiberTrackingPlugin_h


/** Define the UI class */

namespace Ui 
{
    class HARDIFiberTrackingForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_HARDIFiberTracking.h"

/** Includes - Custom Files */

#include "vtkHARDIFiberTrackingFilter.h"

/** Includes - VTK */

#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkMatrix4x4.h>
#include <vtkCellArray.h>

/** Includes - Qt */

#include <QDebug>
#include <QMessageBox>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QSlider>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSpacerItem>
#include <QGroupBox>
#include <QGridLayout>
#include <QRadioButton>


namespace bmia {



/** Fiber Tracking plugin. This class mainly takes care of managing the
	input data set and the created fiber data sets, in combination with
	the GUI. The actual fiber tracking process is done in the "vtkFiber-
	TrackingFilter" class.
*/

class HARDIFiberTrackingPlugin :public plugin::Plugin,
								public data::Consumer,
								public plugin::GUI
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Get current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.0";
		}

			vtkImageData* getMaximaVolume()
		{
			return  this->maximaImage;
		}

			void setMaximaVolume(vtkImageData* img)
		{
			this->maximaImage = img;
		}


		/** Constructor */

		 HARDIFiberTrackingPlugin();

		 /** Destructor */

		~HARDIFiberTrackingPlugin();

		/*void init()
		{
			this->widget = new QWidget();
			this->ui = new Ui::HARDIFiberTrackingForm();
			this->ui->setupUi(this->widget);

		};*/

		/** Returns a pointer to the GUI object */

		QWidget * getGUI();

		/** Define behavior when new data sets are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */

		void dataSetAdded(data::DataSet * ds);
		void dataSetChanged(data::DataSet * ds);
		void dataSetRemoved(data::DataSet * ds);

	 

	protected slots:

		/** Called when the user presses the "Update" button. */
		void updateFibers(void);

		/** Update the minimum and maximum of the two scalar threshold spin boxes
			to the range of the currently selected scalar volume image. */

		void updateScalarRange();

		/** Change the fiber tracking method. This configures the GUI for the new method
			(e.g., for Whole Volume Seeding, the WVS Options page is added to the toolbox.
			@param index	Index of the new method (within the "TrackingMethod" enum). */

		void changeMethod(int index);

		/** Change the name of the plugin. Used to add "(CUDA)" to the CUDA-version. 
			Implemented in the "_Config_" files. */
			
		void changePluginName();

	private:

		/** The Qt widget returned by "getGUI" */

		QWidget * widget;

		/** The form created by Qt Designer */

		Ui::HARDIFiberTrackingForm * ui;


		
		/** Lists containing pointers to added data set, divided into three
			categories: seed points, HARDI data sets, and Anisotropy Index images. */

		QList<data::DataSet *> seedList;
		QList<data::DataSet *> HARDIDataList;
		QList<data::DataSet *> aiDataList;
		QList<data::DataSet *> maxUnitVecDataList;
		/** List available tracking methods. */

		enum TrackingMethod
		{
			TM_Deterministic = 0,		/**< Deterministic(default). */
		};

		/** For each output data set pointer, we store the pointers of the DTI 
			data set and the ROI data set used to create the output fibers, as
			well as the method used (e.g., ROIs or WVS). We also store the name
			of the data set; if the user manually changes the name of the output
			fibers, this plugin will no longer change its name automatically. */

		struct outputInfo
		{
			data::DataSet * hardi;
			data::DataSet * seed;
			data::DataSet * output;
			TrackingMethod  method;
			QString         oldName;
		};

		/** Main fiber tracking filter for ROI seeding. */

		vtkHARDIFiberTrackingFilter * HARDIFiberTrackingFilter;

		vtkImageData *maximaImage;

		/** List of added output data sets and their related information. */

		QList<outputInfo> outputInfoList;

		/** Returns true if the "outputInfoList" list contains an "outputInfo" item
			whose HARDI data, seed point data, and method all match those of the input
			"newFiberObject". It will also copy the output data set pointer of this
			matching item to "newFiberInfo", and return the index of the matching in
			"outputInfoList". Returning true means that the new output fibers should
			be copied to an existing data set; returning false means that we will create
			a new data instead. 
			@param newFiberInfo		Output information of new fibers. 
			@param outputIndex		List index of matching information object. */

		bool overwriteDataSet(outputInfo * newFiberInfo, int * outputIndex);

		/** Add a new data set to one of the three lists */

		bool addHARDIDataSet(data::DataSet * ds);
		bool addAIDataSet(data::DataSet * ds);
		bool addSeedPoints(data::DataSet * ds);
		bool addMaximaUnitVectorsDataSet(data::DataSet * ds);

		/** Change existing data sets. */

		void changeHARDIDataSet(data::DataSet * ds);
		void changeAIDataSet(data::DataSet * ds);
		void changeMaxUnitVecDataSet(data::DataSet * ds);
		void changeSeedPoints(data::DataSet * ds);

		/** Remove existing data sets. */

		void removeHARDIDataSet(data::DataSet * ds);
		void removeAIDataSet(data::DataSet * ds);
		void removeSeedPoints(data::DataSet * ds);
		void removeMaxUnitVecDataSet(data::DataSet * ds);

		/** Adds output fibers to the data manager. If a set of fibers with the same
			DTI image, seeding region, and tracking method already exists, we overwrite
			the fibers in this existing data set.
			@param fibers			Output fibers.
			@param fiberName		Name of the fiber set. 
			@param method			Tracking method used to generate fibers.
			@param seed				Data set of seeding region. */

		void addFibersToDataManager(vtkPolyData * fibers, QString fiberName, TrackingMethod method, data::DataSet * seed);

		/** Remove all pages except for the first one from the toolbox, and delete
			optional GUI components. The first page is spared, as
			it contains tracking parameters which apply to all methods. */

		void clearToolbox();

		/** Enable all controls that may have been disabled for one of the fiber
			tracking methods. For example, the seed list is disabled for Whole
			Volume Seeding, so when we switch to another method, we need to
			re-enable it. */

		void enableAllControls();

		/** Performs fiber tracking for all Regions of Interest in the "seedList"
			@param HARDIimageData		HARDI data set in discrete sphere form.
			@param aiImageData			AI Scalars */
	    void doDeterministicFiberTracking(vtkImageData * HARDIimageData, vtkImageData * aiImageData, vtkImageData *maxUnitVecData, int i);

		
		/** Setup the GUI for Deterministic. Calls "enableAllControls" and "clearToolbox". */

		void setupGUIForHARDIDeterministic();


}; // class HARDIFiberTrackingPlugin


} // namespace bmia


#endif // bmia_HARDIFiberTrackingPlugin_h
