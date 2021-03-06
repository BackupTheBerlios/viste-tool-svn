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
