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

/** 
 * FiberOutputXML.h
 *
 * 2008-01-28	Jasper Levink
 * - First Version.
 *
 * 2010-12-20	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberOutputXML_h
#define bmia_FiberOutputXML_h


/** Includes - Custom Files */

#include "FiberOutput.h"


namespace bmia {


/** Writes the fiber output data to a single ".xml" file. Inherits its main
	functionality - collecting data from fibers and/or ROIs - from its parent class,
	and implements only those functions related to actually writing the output. 
*/

class FiberOutputXML : public FiberOutput {

	public: 
		/** Constructor */
		FiberOutputXML();

		/** Destructor */
		~FiberOutputXML();

	protected:

		/** Run at the start of output, in this case opening the ".xml" file. */
	
		virtual void outputInit();
	
		/** Write the header, in this case as worksheet. */
	
		virtual void outputHeader();
	
		/** Start a new worksheet. 
			@param title	Title of the new worksheet. */

		virtual void outputInitWorksheet(std::string titel = "");

		/** Write row of strings to the output. 
			@param content		Array of output strings.
			@param contentSize	Number of strings. 
			@param styleID		Output style. */
		
		virtual void outputWriteRow(std::string * content = &(std::string)"", int contentSize = 1, int styleID = 0);

		/** Write a row of doubles, with an optional label.
			@param content		Array of output doubles. 
			@param contentSize	Number of doubles.
			@param label		Optional label, printed at the start of the row.
			@param styleID		Output style. */

		virtual void outputWriteRow(double * content, int contentSize = 1, std::string label = "", int styleID = 0); 

		/** Finalize the current worksheet. */
	
		virtual void outputEndWorksheet();
	
		/** Finalize the output. */
	
		virtual void outputEnd();

		/** Excel produces errors if a worksheet name contains more then 31 characters.
			Therefore, we shorten the desired title by taking the rightmost 27 characters
			and adding "..." in front. Nothing happens if the string is shorter than 31.
			@param longName		Original worksheet title. */

		std::string shortenWorksheetName(std::string longName);

		/** Replace reserved characters with underscores. This is used for Excel worksheet
			names, which do not allow certain characters.
			@param roiName	Desired name. */
	
		std::string removeReservedCharacters(std::string roiName);


}; // class FiberOutputXML


} // namespace bmia


#endif // bmia_FiberOutputXML_h