#ifndef bmia_DicomConverter_DicomConverterPlugin_h
#define bmia_DicomConverter_DicomConverterPlugin_h

// Includes DTITool
#include "DTITool.h"

// Includes plugin UIC
#include "ui_DicomConverterPlugin.h"

// Forward declarations DTIConv2
class DTIConfig;

//--------------------------------------------------------------------
//! @class DicomConverterPlugin
//!
//! @brief Converts DICOM data to DTI tensor data
//!
//! This plugin allows the conversion of DICOM files to DTI
//! tensor data. This functionality was previously available in a
//! separate stand-alone tool called 'dticonv2'. I have now added
//! to the DTItool as a plugin.
//!
//! @todo Add dialog window for managing configuration files
//--------------------------------------------------------------------

namespace bmia
{
	class DicomConverterPlugin :	public plugin::Plugin,
											public plugin::GUI
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::GUI )

	public:

		//------------------------------------------------------------
		//! Constructor
		//------------------------------------------------------------
		DicomConverterPlugin();

		//------------------------------------------------------------
		//! Destructor
		//------------------------------------------------------------
		virtual ~DicomConverterPlugin();

		//------------------------------------------------------------
		//! Returns plugin's QT widget
		//! @return QWidget * The widget containing this plugin's UI
		//------------------------------------------------------------
		QWidget * getGUI();

	private slots:

		//------------------------------------------------------------
		//! Loads the configuration file that describes the DICOM data
		//! which has to be converted to tensor data
		//------------------------------------------------------------
		void loadConfig();

		//------------------------------------------------------------
		//! Runs the actual conversion of the data
		//------------------------------------------------------------
		void convert();

	private:

		//------------------------------------------------------------
		//! Sets up signal/slot connections for all GUI components
		//! of this plugin
		//------------------------------------------------------------
		void setupConnections();

	private:

		Ui::DicomConverterForm * _form;	// UI form describing plugin's GUI
		QWidget							* _widget;	// The QT widget holding this plugin's GUI
		DTIConfig						* _config;	// Configuration file describing DICOM dataset
	};
}

#endif
