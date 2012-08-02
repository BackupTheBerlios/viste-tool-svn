/** 
 * FiberOutputTXT.h
 *
 * 2008-01-28	Jasper Levink
 * - First Version.
 *
 * 2010-12-20	Evert van Aart
 * - First version for the DTITool3.
 *
 */


#ifndef bmia_FiberOutputTXT_h
#define bmia_FiberOutputTXT_h


/** Includes - Custom Files */

#include "FiberOutput.h"


namespace bmia {


/** Writes the fiber output data to a number of ".txt" files. Inherits its main
	functionality - collecting data from fibers and/or ROIs - from its parent class,
	and implements only those functions related to actually writing the output. 
*/


class FiberOutputTXT : public FiberOutput {

	public: 
	
		/** Constructor */
		FiberOutputTXT();

		/** Destructor */
		~FiberOutputTXT();

	protected:
	
		/** Runs at start of output, in this case determining file prefix. */
	
		virtual void outputInit();

		/** Write the header, in this case as separate file. */
	
		virtual void outputHeader();

		/** Start a new worksheet, in this case as new file. 
			@param titel	Title of the new worksheet. */

		virtual void outputInitWorksheet(std::string titel="");

		/** Write row of std::strings to the output. 
			@param content		Array of output std::strings.
			@param contentSize	Number of std::strings. 
			@param styleID		Not used in this function. */

		virtual void outputWriteRow(std::string * content = &(std::string)"", int contentSize = 1, int styleID = 0);

		/** Write a row of doubles, with an optional label.
			@param content		Array of output doubles. 
			@param contentSize	Number of doubles.
			@param label		Optional label, printed at the start of the row.
			@param styleID		Not used in this function. */

		virtual void outputWriteRow(double * content, int contentSize = 1, std::string label = "", int styleID = 0); 

		/** Finalize the current worksheet. */

		virtual void outputEndWorksheet();

		/** Finalize the output. */
	
		virtual void outputEnd();

	private:
	
		/** Location of the output file. */

		std::string fileLocation;

		/** Prefix of the output file. */

		std::string filePrefix;

		/** Replace reserved characters with underscores. 
			@param roiName	Desired name. */
	
		std::string prepareROINameForFileName(std::string roiName);

}; // class FiberOutputTXT


} // namespace bmia


#endif // bmia_FiberOutputTXT_h