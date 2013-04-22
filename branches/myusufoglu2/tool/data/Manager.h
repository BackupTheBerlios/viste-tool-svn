/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Manager.h
 *
 * 2009-10-27	Tim Peeters
 * - First version
 *
 * 2009-11-04	Tim Peeters
 * - Add printAllDataSets() function.
 *
 * 2010-10-27	Tim Peeters
 * - Tuple (name, kind) no longer needs to be unique.
 *   Use the DataSet pointers to identify data sets!
 *
 * 2011-01-03	Evert van Aart
 * - Added a function that combines the supported extensions of the readers
 *   with the file descriptions, into a format that can be used as the filter
 *   string of a file dialog.
 *
 */

#ifndef bmia_data_Manager_h
#define bmia_data_Manager_h


/** Includes - Qt */

#include <QStringList>


namespace bmia {


namespace data {


class DataSet;
class Consumer;
class Reader;
class Writer;
 
/** This class iused to manage data sets. Data sets, consumers and readers can 
	be added and removed from a data manager.
*/

class Manager {

	public:
		
		/** Construct a new data manager object that does not have any data 
			sets, consumers or readers. */
    
		Manager();

		/** Destroy the data manager. */
    
		virtual ~Manager();

		/** Add a data set to the manager. The data manager will notify all 
			consumers of the availability of this new data set.
			@param ds		New data set. */

		bool addDataSet(DataSet * ds);

		/** Remove a data set from the manager. The data manager will notify 
			all consumers that the data set has been removed from the data pool.
			@param ds		Removed data set. */
  
		void removeDataSet(DataSet * ds);

		/** Notify the data manager that a data set has changed. The data manager 
			will notify all consumers that the data set has changed.
			@param ds		Modified data set. */

		void dataSetChanged(DataSet * ds);

		/** Returns a list of all data sets with the specified kind. 
			@param kind		Target kind. */
    
		QList<DataSet *> listDataSets(QString kind);

		/** Returns a list that contains all currently loaded data sets. */

		QList<DataSet *> listAllDataSets();

		/** Returns a data set with the given name and kind, or NULL if it was not 
			found. NOTE: The tuple "(name, kind)" does not need to be unique! If 
			there are more than one data sets with the given name and kind, it is 
			not specified which one is returned.
			@param name		Target data set name. 
			@param kind		Target data kind. */

		DataSet * getDataSet(QString name, QString kind);

		/** Prints the name and kind of each data set in the list to cout. To be used for testing. */

		void printAllDataSets();

		/** Adds a data consumer to this data manager. All added consumers will be 
			notified when new data sets are added, removed, or changed.
			@param cons		New consumer. */
		
		void addConsumer(Consumer * cons);

		/** Remove a data consumer from this data manager. The removed consumer 
			will no longer be notified of added, removed, or changed data sets.
			@param cons		Removed consumer. */
    
		void removeConsumer(Consumer * cons);

		/** Add a data reader to this data manager. The supported file extensions 
			of the reader will be added to the supported file extensions of the
			data manager.
			@param reader	New data reader plugin. */
    
		void addReader(Reader * reader);

		/** Remove a data reader from this data manager. The file extensions that 
			were supported by the reader will be removed from the supported file
			extensions of the data manager.
			@param reader	Removed data reader plugin. */

		void removeReader(Reader * reader);

		/** Return a list of all file extensions that are supported by the data 
			manager. The list of returned extensions consists of the concatenation
			of all the lists of supported file extensions of the data writers 
			that were added to this manager. Note that currently we do not check 
			for double extensions, i.e. the same extension supported by different 
			readers. */

			void addWriter(Writer * writer);

		/** Remove a data writer from this data manager. The file extensions that 
			were supported by the reader will be removed from the supported file
			extensions of the data manager.
			@param reader	Removed data reader plugin. */

		void removeWriter(Writer * writer);

		/** Return a list of all file extensions that are supported by the data 
			manager. The list of returned extensions consists of the concatenation
			of all the lists of supported file extensions of the data readers 
			that were added to this manager. Note that currently we do not check 
			for double extensions, i.e. the same extension supported by different 
			readers. */

		QStringList getSupportedFileExtensions();

		/** Returns a list of all supported extensions, complete with file descriptions, 
			i.e. in the format "Fibers (*.fbs)". This list can then be used for the filter 
			list when opening a file dialog. Contains one or more entries per reader. */

		QStringList getSupportedFileExtensionsWithDescriptions();


		/** Return a list of all file extensions that are supported by the data 
			manager. The list of returned extensions consists of the concatenation
			of all the lists of supported file extensions of the data writers 
			that were added to this manager. Note that currently we do not check 
			for double extensions, i.e. the same extension supported by different 
			readers. */

		QStringList getSupportedFileWriteExtensions();

		/** Returns a list of all supported extensions, complete with file descriptions, 
			i.e. in the format "Fibers (*.fbs)". This list can then be used for the filter 
			list when opening a file dialog. Contains one or more entries per reader. */

		QStringList getSupportedFileWriteExtensionsWithDescriptions();


		/** Prints the supported file extensions. */
    
		void printSupportedFileExtensions();

		/** Writes the the data to the specified file. The actual writing of the data 
			will be performed by the data writer that supports the file type.
			@param filename	Target file name. */

		void writeDataToFile(QString filename, DataSet *ds );  

		
		/** Loads the data from the specified file. The actual loading of the data 
			will be performed by the data reader that supports the file type.
			@param filename	Target file name. */

		void loadDataFromFile(QString filename);

	protected:

		QList<DataSet *> dataSets;		/**< All available data sets. */
		QList<Consumer *> consumers;	/**< All active consumer plugins. */
		QList<Reader *> readers;		/**< All active reader plugins. */
		QList<Writer *> writers;		/**< All active reader plugins. */

private:

}; // class Manager


} // namespace data


} // namespace bmia


#endif // bmia_data_Manager_h
