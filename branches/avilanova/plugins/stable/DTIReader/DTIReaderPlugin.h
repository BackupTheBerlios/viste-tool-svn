/*
 * DTIReaderPlugin.h
 *
 * 2009-11-27	Tim Peeters
 * - First version
 *
 * 2011-01-14	Evert van Aart
 * - Structural information images are now added as separate
 *   scalar volume data sets, which allows them to be visualized
 *   using planes, volume mapping, etceta.
 *
 * 2011-01-24	Evert van Aart
 * - Added support for reading ".tfm" transformation matrix files.
 *
 * 2011-03-31	Evert van Aart
 * - Version 1.0.0.
 * - Allowed the reader to read doubles.
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


#ifndef bmia_DTIReader_DTIReaderPlugin_h
#define bmia_DTIReader_DTIReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Qt */

#include <QDebug>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>

/** Includes - VTK */

#include <vtkExecutive.h>
#include <vtkImageData.h>
#include <vtkMatrix4x4.h>

/** Includes 0 Custom Files */

#include "Helpers/TransformationMatrixIO.h"

namespace bmia {


/** A plugin that reads DTI data from ".dti" and ".dat" files, which use
	our old BMIA DTI file format. The structural data (the "I" file in the DTI
	header) is stored in one data set; the DTI tensors, of which only the six
	unique components are stored (instead of all nine), are stored in another
	data set.
 */

class DTIReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

	public:
		
		/** Return the current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */
    
		DTIReaderPlugin();

		/** Destructor */
    
		~DTIReaderPlugin();

		/** Returns the list of file extensions supported by this reader plugin. */
    
		QStringList getSupportedFileExtensions();

		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

		/** Load DTI data from the given file and make it available to the data manager. */

		void loadDataFromFile(QString filename);
    
}; // class DTIReaderPlugin


} // namespace bmia


#endif // bmia_DTIReader_DTIReaderPlugin_h
