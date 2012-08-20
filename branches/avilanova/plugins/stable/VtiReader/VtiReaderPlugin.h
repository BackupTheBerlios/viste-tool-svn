/*
 * VtiReaderPlugin.h
 *
 * 2010-11-09	Tim Peeters
 * - First version
 *
 * 2011-01-27	Evert van Aart
 * - Added support for loading transformation matrices
 *
 */

#ifndef bmia_VtiReader_VtiReaderPlugin_h
#define bmia_VtiReader_VtiReaderPlugin_h

#include "DTITool.h"

namespace bmia {

/**
 * A plugin for reading VTK imagedata files.
 */
class VtiReaderPlugin : public plugin::Plugin, public data::Reader
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Reader)

public:
    VtiReaderPlugin();
    ~VtiReaderPlugin();

    /**
     * Returns the list of file extensions supported by this reader plugin.
     * This function is required by the data::Reader plugin interface.
     */
    QStringList getSupportedFileExtensions();

	/** Returns a list containing short descriptions of the supported file
		types. The number of descriptions and their order should match those
		of the list returned by "getSupportedFileExtensions". */

	QStringList getSupportedFileDescriptions();

	/**
     * Load geometry data from the given file and make it available
     * to the data manager.
     * This function is required by the data::Reader plugin interface.
     */
    void loadDataFromFile(QString filename);

protected:
private:

}; // class VtiReaderPlugin
} // namespace bmia
#endif // bmia_VtiReader_VtiReaderPlugin_h
