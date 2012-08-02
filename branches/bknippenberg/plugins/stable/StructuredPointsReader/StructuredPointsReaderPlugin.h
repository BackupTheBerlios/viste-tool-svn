/*
 * StructuredPointsReaderPlugin.h
 *
 * 2010-10-20	Evert van Aart
 * - First version. Added this to allow reading of ".clu" files.
 *
 * 2010-12-10	Evert van Aart
 * - Added automatic shortening of file names.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.0.
 * - Improved progress reporting.
 *
 */


#ifndef bmia_StructuredPointsReaderPlugin_h
#define bmia_StructuredPointsReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsReader.h>


namespace bmia {


/** This class is used to read VTK files containing "STRUCTURED_POINTS" data sets.
	The Clustering Plugin uses files of this type, with extention ".clu", to store
	clustering information. */

class StructuredPointsReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

	public:

		/** Return current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.0";
		}

		/** Constructor */

		StructuredPointsReaderPlugin();

		/** Destructor */

		~StructuredPointsReaderPlugin();

		/** Returns the list of file extensions supported by this reader plugin.
			This function is required by the data::Reader plugin interface. */

		QStringList getSupportedFileExtensions();

		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

		/** Load point data from the given file and make it available to the data manager.
			This function is required by the data::Reader plugin interface. */
	  
		void loadDataFromFile(QString filename);

	protected:

	private:

}; // class StructuredPointsReaderPlugin


} // namespace bmia


#endif // bmia_StructuredPointsReaderPlugin_h
