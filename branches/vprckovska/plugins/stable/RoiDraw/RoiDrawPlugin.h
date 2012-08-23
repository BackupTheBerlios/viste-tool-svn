/*
 * RoiDrawPlugin.h
 *
 * 2011-02-16	Evert van Aart
 * - First version. This is a new approach to the ROI Drawing plugin by Tim 
 *   Peeters. The previous version was plagued by a number of bugs, most of
 *   which were due to the use of a separate drawing window, which is not
 *   fully supported by VTK. This plugin moves drawing the ROIs back to the
 *   main window, which is more stable and more user friendly.
 *
 * 2011-03-09	Evert van Aart
 * - Version 1.0.0.
 * - Automatically set seeding type to "Distance" for loaded ROIs.
 * - Added support for grouping fibers.
 *
 * 2011-03-14	Evert van Aart
 * - Version 1.0.1.
 * - Fixed ROIs occasionnally not showing up in the 2D views.
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.2.
 * - When saving ROIs, plugin now automatically selects the data directory
 *   as defined by the default profile. 
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.1.0.
 * - Outsourced seeding options to the new Seeding plugin.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.1.1.
 * - Improved attribute handling.
 *
 */


#ifndef bmia_RoiDrawPlugin_h
#define bmia_RoiDrawPlugin_h


/** Includes - Main Headers */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkMatrix4x4.h>
#include <vtkActor.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropAssembly.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>
#include <vtkGlyphSource2D.h>
#include <vtkIdList.h>
#include <vtkCellArray.h>
#include <vtkAppendPolyData.h>

/** Includes - GUI */

#include "ui_ROIEdit.h"

/** Includes - Custom Files */

#include "data/DataSet.h"
#include "Helpers/TransformationMatrixIO.h"
#include "vtkImageTracerWidget2.h"
#include "gui/MetaCanvas/vtkMetaCanvasUserEvents.h"
#include "gui/MetaCanvas/vtkMedicalCanvas.h"
#include "gui/MetaCanvas/vtkSubCanvas.h"
#include "Helpers/vtkImageSliceActor.h"
#include "ROIGroupDialog.h"

/** Includes - Qt */

#include <QDebug>
#include <QFileDialog>
#include <QVector>


/** Forward Declarations */

namespace Ui 
{
    class RoiForm;
}


namespace bmia {


/** Forward Class Declarations */

class RoiDrawPluginCallback;


/** This plugin can be used to manipulate (create, rename, draw, delete) Regions 
	of Interest (ROIs), and to specify seeding parameters from the ROIs. It also
	takes care of drawing existing ROIs (either created in this plugin, or loaded
	from a file) on the 3D subcanvas.
*/

class RoiDrawPlugin :	public plugin::AdvancedPlugin,
						public data::Consumer,
						public plugin::Visualization,
						public plugin::GUI
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::plugin::AdvancedPlugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::Visualization)
	Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.1.1";
		}

		/** Constructor */

		RoiDrawPlugin();

		/** Destructor */

		~RoiDrawPlugin();

		/** Initializes the plugin. */

		void init();

		/** Returns the Qt widget that gives the user control.
			This implements the GUI interface. */
    
		QWidget * getGUI();

		/** Return the VTK prop that renders all the geometry.
			This implements the Visualization interface. */
    
		vtkProp * getVtkProp();

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

		/** Update an existing ROI with a new polygon drawn by the user. 
			@param newData		New polydata, drawn by the user.
			@param m			Optional transformation matrix. */

		void updateROI(vtkPolyData * newData, vtkMatrix4x4 * m = NULL);

		/** Returns the 2D subcanvas of the main window with the indicated index.
			Used by the callback class to find the selected subcanvas.
			@param i			Index of the target subcanvas. */

		vtkSubCanvas * getSubcanvas(int i);

		/** Activate the indicated subcanvas, i.e., start drawing on this canvas.
			Switching to a different canvas is only possible when we're actually
			drawing. In practice, this function deletes the existing tracing widget,
			and creates a new one in the target subcanvas. This approach is less than
			elegant, but is more stable than moving an existing widget to a new 
			subcanvas. 
			@param canvasID		Index of the target subcanvas. */

		void activateSubCanvas(int canvasID);

		/** If the user selects a different subcanvas, but we are not actively
			drawing a ROI, we store the ID of the selected subcanvas, and move
			the drawing widget when the user presses the "Start drawing" button. */

		int currentSubCanvas;


	protected slots:

		/** Select the ROI with the specified index. The input parameter
			"index" should be within range of the "roiData" list.
			@param index		Index of target ROI. */

		void selectROI(int index);
    
		/** Allows the user to draw a new ROI. */

		void toggleDrawing();

		/** Create an empty ROI with a default name. */
    
		void createNewROI();

		/** Delete an existing ROI. */

		void deleteRoi();

		/** Write a ROI to an output file. */

		void saveROI();

		/** Rename an existing ROI. */
    
		void rename();

		/** Group together several ROIs. */

		void groupROIs();

		/** Enable or disable controls based on the current GUI settings and
			available data sets. */

		void enableControls();

	private:

		/** Callback class, used to register canvas selection events. */

		RoiDrawPluginCallback * callBack;

		/** The three slices shown in the 2D subcanvasses. */

		vtkImageSliceActor * slices[3];

		/** True if we're currently drawing a ROI; false otherwise. */

		bool isDrawing;

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt designer. */

		Ui::RoiForm * ui;

		/** List of available ROI data sets. */

		QList<data::DataSet *> roiData;

		/** List of available volume data sets. */
    
		QList<data::DataSet *> volumeData;

		/** Structure containing pointers to the tracing widgets for this ROI
			(one for each 2D view), as well as the index of the 2D subcanvas
			currently containing an active tracing widget. */

		struct TracerInformation
		{
			vtkImageTracerWidget2 * widgets[3];
			int currentView;
		};

		/** Vector with tracing widget information for each ROI. */

		QVector<TracerInformation> tracers;

		/** Collection of all the actors that show the ROIs in the 3D view. */
    
		vtkPropAssembly * assembly;

		/** Currently selected ROI. This number corresponds to an index in 
			the "roiData" list, and is -1 if no ROI is selected. */
    
		int selectedRoi;

		/** Return the currently selected ROI data, or NULL if no ROI data is selected. */
    
		data::DataSet * getSelectedRoi();

		/** Helper function to get the VTK actor of a ROI from the assembly. 
			@param index	Index of the desired ROI. */
    
		vtkActor * getROIActor(int index);

}; // class RoiDrawPlugin


/** This is a callback class for the ROI Edit plugin. This plugin needs to know when
	the user selects a new subcanvas, as it will need to move the ROI tracing widget to
	this new subcanvas, which triggers events of type "BMIA_USER_EVENT_SUBCANVAS_SELECTED".
*/

class RoiDrawPluginCallback : public vtkCommand
{
	public:

		/** Constructor Call */

		static RoiDrawPluginCallback * New() { return new RoiDrawPluginCallback; }

		/** Execute the callback. 
			@param caller		Not used.
			@param event		Event ID.
			@param callData		Pointer to the selected subcanvas. */

		void Execute(vtkObject * caller, unsigned long event, void * callData);

		/** Pointer to the ROI drawing plugin. */

		RoiDrawPlugin * plugin;
};


} // namespace bmia


#endif // bmia_RoiDrawPlugin_h
