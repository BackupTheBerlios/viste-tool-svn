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