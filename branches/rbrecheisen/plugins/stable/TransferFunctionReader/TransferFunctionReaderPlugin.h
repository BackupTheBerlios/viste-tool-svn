/*
 * TransferFunctionReaderPlugin.h
 *
 * 2010-03-03	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-01-4	Evert van Aart
 * - Improved error handling, added some comments.
 *
 * 2011-01-17	Evert van Aart
 * - Fixed a bug that prevented transfer functions with a piecewise function
 *   from being loaded correctly.
 * - Transfer functions are now given a shorter name by default.
 *
 */

#ifndef bmia_TransferFunctionReaderPlugin_h
#define bmia_TransferFunctionReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>

/** Includes - Qt */

#include <QFile>
#include <QByteArray>

/** Includes - C++ */

#include <assert.h>


namespace bmia {


/** A plugin for reading Transfer function files. The output consists of a 
	"vtkColorTransferFunction" object stored in a data set of type "transfer function",
	with the minimum and maximum range and an optional "vtkPiecewiseFunction" object
	added as attributes. 
*/


class TransferFunctionReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

	public:

		/** Constructor */
    
		TransferFunctionReaderPlugin();

		/** Destructor */

		~TransferFunctionReaderPlugin();

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

}; // class TransferFunctionReaderPlugin


} // namespace bmia


#endif // bmia_TransferFunctionReaderPlugin_h
