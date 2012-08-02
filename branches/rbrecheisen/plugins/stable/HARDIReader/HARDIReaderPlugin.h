/*
 * HARDIReaderPlugin.h
 *
 * 2010-11-29	Evert van Aart
 * - First version.
 *
 * 2011-01-24	Evert van Aart
 * - Added support for transformation matrices
 *
 * 2011-04-19	Evert van Aart
 * - Version 1.0.0.
 * - Raw HARDI data is now outputted in the format excepted by the Geometry Glyphs
 *   plugin, so with an array of angles and an array defining the triangles.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


#ifndef bmia_HARDIReaderPlugin_h
#define bmia_HARDIReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkMatrix4x4.h>

/** Includes - Custom Files */

#include "vtkHARDIReader.h"
#include "vtkSHARMReader.h"
#include "Helpers/TransformationMatrixIO.h"

/** Includes - C++ */

#include <string>


namespace bmia {


/** This class is used to read different formats of HARDI data. At the moment, the only
	supported format is the "raw" HARDI data format, which consists of one ".hardi" file 
	and as many ".dat" files as there are gradient directions. Other formats will be added
	in the future. 
*/

class HARDIReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

	public:

		/** Return current version. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */

		HARDIReaderPlugin();

		/** Destructor */

		~HARDIReaderPlugin();

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

}; // class HARDIReaderPlugin


} // namespace bmia


#endif // bmia_HARDIReaderPlugin_h
