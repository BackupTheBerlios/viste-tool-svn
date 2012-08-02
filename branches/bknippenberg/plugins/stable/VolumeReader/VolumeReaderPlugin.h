/*
 * VolumeReaderPlugin.h
 *
 * 2010-02-10	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-01-04	Evert van Aart
 * - Added additional comments, added more descriptive error messages.
 * - Remove "vtkDataHeaderReader" from this plugin, since it was not used.
 *
 * 2011-01-17	Evert van Aart
 * - Fixed reading for files with unsigned characters.
 *
 * - Version 1.0.0.
 * - Enabled reading of decimal spacing values (instead of integers).
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


/** Note from Evert van Aart: 
	In its current implementation, this reader is far from robust, as it
	fails completely when the order of arguments in the ".vol" file does not 
	exactly match the assumed order. The argument names (e.g., "Data.FileName")
	are ignored completely. A better, more robust method would be to first read
	an argument name, check if it is a supported/recognized argument, and then
	parse the argument itself based on its type (e.g., when encountering the 
	"Data.Dimensions" argument, the reader tries to parse three integers). 
	Reimplementation of this reader is not a priority, however, as long as a
	clear formatting guideline for the ".vol" files is provided. 
*/


#ifndef bmia_VolumeReaderPlugin_h
#define bmia_VolumeReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - C++ */

#include <assert.h>

/** Includes - VTK */

#include <vtkImageReader2.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QDebug>


namespace bmia {


/** A plugin for reading Volume (".vol") files. The output is a data set of type 
	"scalar volume", with an optional transformation matrix attached to it
	as an attribute. 
 */


class VolumeReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

	public:
    
		/** Get current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */
    
		VolumeReaderPlugin();

		/** Destructor */
     
		~VolumeReaderPlugin();

		/** Returns the list of file extensions supported by this reader plugin. */
    
		QStringList getSupportedFileExtensions();

		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

		/** Load scalar volume data from the given file and make it available
			to the data manager.
			@param filename		Name of the required file. */
    
		void loadDataFromFile(QString filename);


}; // class VolumeReaderPlugin


} // namespace bmia


#endif // bmia_VolumeReaderPlugin_h
