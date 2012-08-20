/*
 * Reader.h
 *
 * 2009-10-27	Tim Peeters
 * - First version
 */

#ifndef bmia_data_Reader_h
#define bmia_data_Reader_h


/** Includes - Qt */

#include <QStringList>

namespace bmia {
namespace data {

/** Reader plugins can read data files. All reader plugin classes should implement
	the functions of this interface. 
*/

class Reader {

	public:

		/** Destructor */

		virtual ~Reader() {};

		/** Returns a list of extensions of filenames that are supported by this data reader. */
    
		virtual QStringList getSupportedFileExtensions() = 0;

		/** Returns a list of descriptions for the file type supported
			by this data reader. This is the description that will be used in
			the filter box of the file open dialog (e.g., "Fibers (*.fbs)").
			Must be implemented by all readers. */

		virtual QStringList getSupportedFileDescriptions() = 0;

		/** Load the data from the file with the given filename. Adding data 
			the data manager is the responsibility of the subclass that 
			implements this function.
			@param filename	Full file name (with path) of the input file. */
    
		virtual void loadDataFromFile(QString filename) = 0;

	protected:


	private:


}; // class Reader


} // namespace data


} // namespace bmia


#endif // bmia_data_Reader_h
