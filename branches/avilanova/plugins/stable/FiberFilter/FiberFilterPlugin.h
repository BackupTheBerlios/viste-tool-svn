/*
 * FiberFilterPlugin.h
 *
 * 2010-09-29	Tim Peeters
 * - First version
 *
 * 2010-11-09	Evert van Aart
 * - Implemented filtering functionality
 *
 * 2011-01-24	Evert van Aart
 * - Added support for transformation matrices
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.0.
 * - Improved attribute handling.
 *
 */


/** ToDo List for "FiberFilteringPlugin"
	Last updated 09-11-2010 by Evert van Aart

	- Add support for 3D ROIs, once needed.
	- Update the name of the data set when the fibers are updated.
*/


#ifndef bmia_FiberFilterPlugin_h
#define bmia_FiberFilterPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "FiberFilterWidget.h"

/** Includes - Qt */

#include <QDebug>

/** Includes - VTK */

#include <vtkMatrix4x4.h>


namespace bmia {


/** Forward Class Declaration */

class FiberFilterWidget;


/** This plugin allows the user to filter an existing set of fibers - either loaded
	from a file or computed using a fiber tracking technique - through one or more
	Regions of Interest (ROI). This particular class mainly keeps track of the input
	and output data sets. GUI functionality for selecting the ROIs etcetera is handled
	by the "FiberFilterWidget" class; each object of this class represents one tab
	in the GUI of this plugin. Finally, the actual filtering is done in VTK filters
	like "vtk2DROIFiberFilter".
*/

class FiberFilterPlugin :	public plugin::Plugin,
							public plugin::GUI,
							public data::Consumer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::GUI)
    Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return the current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.0";
		}

		/** The number of tabs in the GUI for defining filters. */

		static const int NUM_TABS = 10;

		/** Constructor */

		FiberFilterPlugin();

		/** Destructor */

		~FiberFilterPlugin();
		
		/** Initialize the plugin by creating the filtering tabs. */

		void init();

		/** Return the widget that is shown in the GUI. */

		QWidget * getGUI();

		/** Define behaviour when new data sets are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */

		void dataSetAdded(data::DataSet * ds);
		void dataSetChanged(data::DataSet * ds);
		void dataSetRemoved(data::DataSet * ds);

		/** Returns the ROI at index "i" in the "ROIList". 
			@param i	List index. */

		vtkPolyData * getROI(int i);

		/** Returns the fiber set at index "i" in the "fiberList".
			@param i	List index. */

		vtkPolyData * getFibers(int i);

		/** Returns the transformation matrix (if present) of the input fiber data
			set at position "i" in the list of fibers. */

		vtkMatrix4x4 * getTransformationMatrix(int i);

		/** Attach newly generated fibers to a data set, and add it to the data
			manager. If the filtering widget calling this function has already
			generated data in the past, this data is overwritten; if not, a new
			data set is created. Returns true if data set was succesfully added.
			@param out		Output fibers.
			@param name		New name of the output data set.
			@param filerID	Index of the filtering widget calling this function. 
			@param m		Transformation matrix for output fibers. */

		bool addFibersToDataManager(vtkPolyData * out, QString name, int filterID, vtkMatrix4x4 * m = NULL);

		/** Turn off the visibility of the fiber set used as input for the filter. 
			@param index	Index of the fiber data set in "fiberList". */

		void hideInputFibers(int index);

	private:
    
		/** The Qt widget to return in "getGUI" function. */
    
		QTabWidget * tabWidget;

		/** List of the filtering widgets (tabs). */

		QList<FiberFilterWidget *> filterWidgets;
		
		/** Lists of the input data sets (ROIs and fibers). */
	
		QList<data::DataSet *> ROIList;
		QList<data::DataSet *> FiberList;

		/** List containing pointers to previously generated data sets. Used to overwrite
			data when a filtering widget generates a new set of fibers. */

		data::DataSet ** outputDataSets;
	
}; // class FiberFilterPlugin


} // namespace bmia


#endif // bmia_FiberFilterPlugin_h
