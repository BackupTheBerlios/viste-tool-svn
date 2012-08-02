/*
 * NiftiReaderPlugin.h
 *
 * 2010-08-11	Tim Peeters
 * - First version.
 *
 * 2011-03-21	Evert van Aart
 * - Version 1.0.0.
 * - Turned off debug mode of the reader, reducing its verbosity. 
 * - Reformatted code, added more comments. 
 *
 * 2011-04-04	Evert van Aart
 * - Version 2.0.0.
 * - Completely rebuilt the reader. The new reader is capable of dealing with 
 *   MiND extensions, which are necessary for reading in HARDI data. Furthermore,
 *   the new reader is more easy to extend with support of other data types.
 *
 * 2011-05-10	Evert van Aart
 * - Version 2.1.0.
 * - Added support for spherical harmonics using MiND. 
 *
 * 2011-08-22	Evert van Aart
 * - Version 2.1.1.
 * - Which transformation matrix to use is now determined correctly based on the
 *   "qform_code" and "sform_code" of the NIfTI image.
 *
 */


#ifndef bmia_NiftiReaderPlugin_h
#define bmia_NiftiReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "bmiaNiftiReader.h"

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <QString>
#include <QStringList>
#include <QMessageBox>


namespace bmia {


/** This plugin reads NIfTI (.nii) files, and interprets them. The output is 
	always a data set containing a "vtkImageData" object, but the kind of the data
	set depends on the parameters of the NIfTI file. For example, if the file 
	has six components and its intent code is set to "SYMMATRIX", we interpret
	it as a set of DTI tensors. 
*/

class NiftiReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "2.1.1";
		}

		/** Constructor */
    
		NiftiReaderPlugin();

		/** Destructor */

		~NiftiReaderPlugin();

		/** Returns the list of file extensions supported by this reader plugin. */
    
		QStringList getSupportedFileExtensions();

		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

		/** Read the NIfTI file and make it available to the data manager.
			@param filename		Filename of the NIfTI file. */

		void loadDataFromFile(QString filename);

	private:

		/** Add a data set to the data manager, with optional transformation matrix.
			@param ds			New data set.
			@param m			Optional transformation matrix. */

		void addDataSet(data::DataSet * ds, vtkMatrix4x4 * m);

}; // class NiftiReaderPlugin


} // namespace bmia


#endif // bmia_NiftiReaderPlugin_h
