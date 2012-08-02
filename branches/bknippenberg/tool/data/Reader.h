/*
 * Reader.h
 *
 * 2009-10-27	Tim Peeters
 * - First version
 */

#ifndef bmia_data_Reader_h
#define bmia_data_Reader_h

#include <QStringList>

namespace bmia {
namespace data {

/**
 * The interface to be implemented by classes that can
 * read data from files.
 */
class Reader {
public:
    virtual ~Reader() {};

    /**
     * Returns a list of extensions of filenames that are supported
     * by this data reader.
     */
    virtual QStringList getSupportedFileExtensions() = 0;

	/** Returns a list of descriptions for the file type supported
		by this data reader. This is the description that will be used in
		the filter box of the file open dialog (e.g., "Fibers (*.fbs)").
		Must be implemented by all readers. */

	virtual QStringList getSupportedFileDescriptions() = 0;

    /**
     * Load the data from the file with the given filename.
     * Adding data the data manager is the responsibility of the
     * subclass that implements this function.
     */
    virtual void loadDataFromFile(QString filename) = 0;

protected:
private:

}; // class Reader
} // namespace data
} // namespace bmia

#endif // bmia_data_Reader_h
