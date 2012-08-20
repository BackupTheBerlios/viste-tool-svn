/*
 * HARDIConvolutionsPlugin.h
 *
 * 2011-07-22	Evert van Aart
 * - Version 1.0.0.
 * - First version
 *
 * 2011-08-05	Evert van Aart
 * - Version 1.0.1.
 * - Fixed error in the computation of the Duits kernels.
 * - Fixed computation of unit vectors from spherical angles.
 * 
 */


#ifndef bmia_HARDIConvolutionsPlugin_h
#define bmia_HARDIConvolutionsPlugin_h


/** Define the UI class */

namespace Ui 
{
	class HARDIConvolutionsForm;
}

/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "KernelGenerator.h"
#include "vtkHARDIConvolutionFilter.h"

/** Includes - GUI */

#include "ui_HARDIConvolutions.h"

/** Includes - Qt */

#include <QStringList>
#include <QFileDialog>
#include <QInputDialog>
#include <QFile>
#include <QList>

/** Includes - VTK */

#include <vtkDoubleArray.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkMath.h>

/** Includes - C++ */

#include <vector>


namespace bmia {


/** This plugin is used to apply a convolution to HARDI images. The input images are
	all of the type "discrete sphere", meaning that they have an array containing
	all spherical directions (stored as pairs of spherical angles), and one radius
	value for each voxel and direction. Optionally, these data sets can also contain
	a triangle array, which defines the triangulation. Convolution is done using 
	kernels, after which the result (also a "discrete sphere" data set) is added
	to the data manager (i.e., the input is not modified). The kernels can either be 
	loaded from Kernel Image Group (".kig") file, which are also loaded and created
	by this plugin, or they can be computed directly when applying the convolution.
	Based on the Enhancement Kernels by Paulo Rodriguez.
*/

class HARDIConvolutionsPlugin : 	public plugin::Plugin,
									public data::Consumer,
									public plugin::GUI,
									public data::Reader
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::GUI)
	Q_INTERFACES(bmia::data::Reader)

	public:

		/** Current Version */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */

		HARDIConvolutionsPlugin();

		/** Destructor */

		~HARDIConvolutionsPlugin();

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

		/** Returns the list of file extensions supported by this reader plugin. */
    
		QStringList getSupportedFileExtensions();

		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

		/** Load transfer function data from the given file and make it available
			to the data manager.
			@param filename Name if the desired transfer function file. */

		void loadDataFromFile(QString filename);

	protected slots:

		/** Enable or disable the GUI controls, based on the current settings. */

		void enableControls();

		/** Generate the kernels based on the current settings, and write them
			to a set of NIfTI files. First asks the user for a target Kernel
			Image Group (".kig") file. */

		void generateKernels();

		/** Launches a file dialog, and loads the selected ".kig" file. This can
			also be done through the main menu, but this shortcut might be more
			accessible and intuitive for some users. */

		void loadKIGFile();

		/** Load the settings for one input data set to the GUI. Specifically, 
			we create the combo box containing the output data sets for over-
			writing (i.e., the user can select which data set should be over-
			written), and selects the previously selected data set.
			@param index	Index of the input data set. */

		void loadDataInfo(int index);

		/** Rename an existing output data set. */

		void renameOutputDataSet();

		/** Select one of the available output data sets. If the "Overwrite"
			radio button is checked, this data set will be overwritten when
			the convolution is re-applied. This function only updates the
			information of the selected input data set in "dataList".
			@param index	Index of the output data set. */

		void selectOutputDataSet(int index);

		/** Called when the user changes the output method. The output method is
			either "New Data Set" or "Overwrite Existing Data Set". Updates the
			information of the selected input data set in "dataList". */

		void changeOutputMethod();

		/** Apply the convolution. Called when the user clicks the corresponding button. */

		void applyConvolution();

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
			data::DataSet * selectedOutputDS;	/**< Currently selected output data set. */
		};

		/** Decompose a full file name and path into the base name of the file
			(without path and extension), and the absolute path (without file name).
			@param inFN			Input file name and path.
			@param baseName		Output base name.
			@param absolutePath	Output path. */

		void decomposeFileName(QString inFN, QString & baseName, QString & absolutePath);

		/** Setup a newly created generator. Copies options from the GUI (such as the
			kernel type and size), and also adds properties of the currently 
			selected input image (such as its directions and spacing). Called 
			by "generateKernels" (if we want to write the kernels to NIfTI files),
			or by "applyConvolution" (if we want to directly compute the kernels).
			@param generator	Target kernel generator. */

		void setupGenerator(KernelGenerator * generator);

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt Designer. */

		Ui::HARDIConvolutionsForm * ui;

		/** List containing all available input data sets and their outputs. */

		QList<dataSetInfo> dataList;

		/** List containing all available Kernel Image Groups. Kernel Image Groups
			are essentially lists of file names of the NIfTI files containing the
			generated kernel images. */

		QList<QStringList *> kernelImageGroups;

}; // class HARDIConvolutionsPlugin


} // namespace bmia


#endif // bmia_HARDIConvolutionsPlugin_h