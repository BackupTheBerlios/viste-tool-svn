#ifndef bmia_BootstrapFiberTracking_BootstrapFiberTrackingPlugin_h
#define bmia_BootstrapFiberTracking_BootstrapFiberTrackingPlugin_h

// Includes DTITool
#include "DTITool.h"

// Includes plugin UIC
#include "ui_BootstrapFiberTrackingPlugin.h"

// Includes VTK
#include <vtkProp.h>
#include <vtkActor.h>
#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>

// Includes QT
#include <QStringList>
#include <QList>

/** @class BootstrapFiberTrackingPlugin
	@brief Performs fiber tracking on multiple tensor volumes

	This plugin allows fiber tracking on multiple tensor volumes using
	the same seed region. The fibers reconstructed in each volume are
	added to a larger fiber collection. Furthermore, this plugin
	computes distances between fibers and stores these values in a
	table mapping fiber ID to fiber distance. When the @see
	DicomToTensorConverterPlugin has been executed with the bootstrap
	option, only the first tensor volume is added to the DTI tool
	data manager. This volume can be selected in the current plugin
	after which the plugin looks for additional volumes */
namespace bmia
{
	class BootstrapFiberTrackingPlugin :	public plugin::Plugin,
											public plugin::Visualization,
											public plugin::GUI,
											public data::Consumer
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::Visualization )
		Q_INTERFACES( bmia::plugin::GUI )
		Q_INTERFACES( bmia::data::Consumer )

	public:

		/** Constructor */
		BootstrapFiberTrackingPlugin();

		/** Destructor */
		virtual ~BootstrapFiberTrackingPlugin();

		/** Returns plugin's VTK prop
			@return The actor containing the bootstrap fibers */
		vtkProp * getVtkProp();

		/** Returns plugin's QT widget
			@return The widget containing this plugin's UI */
		QWidget * getGUI();

		/** Handles datasets just added to the reposity
			@param dataset The dataset */
		void dataSetAdded( data::DataSet * dataset );

		/** Handles datasets just removed from the reposity
			@param dataset The dataset */
		void dataSetRemoved( data::DataSet * dataset );

		/** Handles datasets just changed in the reposity
			@param dataset The dataset */
		void dataSetChanged( data::DataSet * dataset );

	private:

		/** Sets up signal/slot connections for all GUI components
			of this plugin */
		void setupConnections();

		/** Returns directory path from given filename. If there is no
			directory information in the filename, then an empty string
			is returned
			@param  fileNameAndPath The filename and path
			@return The directory path */
		QString getDirectory( const QString fileNameAndPath );

		/** Returns file name in a given file name and path string
			@param  fileNameAndPath The file name and path
			@return The file name */
		QString getFileName( const QString fileNameAndPath );

		/** Returns a list of bootstrap DTI file names. This assumes a
			number of things: (1) The original file name has either a
			numeric postfix of 0 or 1, or no postfix. (2) No zero-padding
			was used on the bootstrap file names
			@param  directory The directory path to look for file names
			@param  fileName  The base file name
			@return The list of filenames */
		QStringList getBootstrapFileNames( const QString directory, const QString fileName );

	private slots:

		/** Opens directory dialog to select directory where the
			bootstrap files are located */
		void openDir();

		/** Runs the fiber tracking on each bootstrap tensor volume */
		void run();

		/** Opens file dialog for loading a seed region from file */
		void loadROI();

		/** Open file dialog for saving a seed region */
		void saveROI();

	private:

		QStringList						_fileNames;			// The list of bootstrap file names
		QList< vtkImageData * >			_tensorVolumes;		// The list of DTI tensor volumes
		QList< vtkImageData * >			_aiVolumes;			// The list of anisotropy volumes
		QList< vtkUnstructuredGrid * >	_seedRegions;		// The list of seed regions
		QWidget							* _widget;			// The QT widget holding this plugin's GUI
		vtkActor						* _prop;			// The VTK actor representing the fibers
		Ui::BootstrapFiberTrackingForm	* _form;			// UI form describing plugin's GUI
	};
}

#endif
