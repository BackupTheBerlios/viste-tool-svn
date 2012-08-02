/*
 * HARDIConverterPlugin.h
 *
 * 2011-08-04	Evert van Aart
 * - Version 1.0.0.
 * - First version. Currently, only SH-to-DSF conversions are supported.
 *
 */


#ifndef bmia_HARDIConverterPlugin_h
#define bmia_HARDIConverterPlugin_h


/** Define the UI class */

namespace Ui 
{
	class HARDIConverterForm;
}

/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_HARDIConverter.h"

/** Includes - Qt */

#include <QInputDialog>
#include <QList>

/** Includes - VTK */

#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkMath.h>

/** Includes - C++ */

#include <vector>

/** Includes - Custom Files */

#include "vtkSH2DSFFilter.h"


namespace bmia {


/** This plugin is used to convert one HARDI data type to another. For example,
	we can convert a volume containing Spherical Harmonics data (SH) to one
	containing Discrete Sphere Functions (DSF).
*/

class HARDIConverterPlugin : 	public plugin::Plugin,
								public data::Consumer,
								public plugin::GUI
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Current Version */

		QString getPluginVersion()
		{
			return "1.0.0";
		}

		/** Constructor */

		HARDIConverterPlugin();

		/** Destructor */

		~HARDIConverterPlugin();

		/** Initialize the plugin. */

		void init();

		/** Returns the Qt widget that gives the user control. This 
			implements the GUI interface. */
    
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

	protected slots:

		/** Enable or disable the GUI controls, based on the current settings. */

		void enableControls();

		/** Rename an existing output data set. */

		void renameOutputDataSet();

		/** Load the data set information of the selected input data set.
			@param index	Index of the input data set. */

		void loadDataInfo(int index);

		/** Apply the HARDI conversion, using one of the available filters. */

		void applyConversion();

	private:

		/** Structure containing information about one input data set. Contains
			the input data set pointer, a list of output data sets, and a pointer
			to the currently selected output data set (which should be in the list).
			If the selected data set pointer is NULL, the output method "New Data
			Set" is used instead. */

		struct dataSetInfo
		{
			data::DataSet * inDS;				/**< Input data set pointer. */
			QList<data::DataSet *> outDSs;		/**< All available output data sets. */
		};

		/** Enumeration for the data type, used to cut down on the number of 
			string comparisons (i.e., "ds->GetKind()"). */

		enum DataType
		{
			DT_DSF = 0,							/**< Discrete Sphere Function. */
			DT_SH,								/**< Spherical Harmonics. */
			DT_Unknown							/**< Should never happen. */
		};

		/** Convert a SH image to a DSF image, using "vtkSH2DSFFilter". */

		void convertSHtoDSF();

		/** Create a new output data set, or overwrite an existing one, based on
			the current settings in the GUI.
			@param outImage		Output image.
			@param name			Target data set name. 
			@param type			Data set type. */

		void createOutput(vtkImageData * outImage, QString name, QString type);

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt Designer. */

		Ui::HARDIConverterForm * ui;

		/** List containing all available input data sets and their outputs. */

		QList<dataSetInfo> dataList;

}; // class HARDIConverterPlugin


} // namespace bmia


#endif // bmia_HARDIConverterPlugin_h
