#ifndef bmia_DicomReader_DicomReaderPlugin_h
#define bmia_DicomReader_DicomReaderPlugin_h

// Includes DTI tool
#include <DTITool.h>

// Includes QT
#include <QList>
#include <QString>
#include <QStringList>

// Includes GDCM
#include <vtkGDCMImageReader.h>

/** @class DicomReaderPlugin
	@brief This plugin allows reading of DICOM datasets */
namespace bmia
{
	class DicomReaderPlugin :	public plugin::Plugin,
								public plugin::GUI,
								public data::Reader
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::GUI )
		Q_INTERFACES( bmia::data::Reader )

	public:

		/** Constructor and destructor */
		DicomReaderPlugin();
		virtual ~DicomReaderPlugin();

		/** Returns plugin's QT widget */
		QWidget * getGUI();

		/** Returns plugin's supported file extensions */
		QStringList getSupportedFileExtensions();

		/** Returns short description for each supported file type */
		QStringList getSupportedFileDescriptions();

		/** Loads the DICOM data */
		void loadDataFromFile( QString fileName );

	private:

		QWidget * _widget;
		QList< vtkGDCMImageReader * > _readers;
	};
}

#endif
