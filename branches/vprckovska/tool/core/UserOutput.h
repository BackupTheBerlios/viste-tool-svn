/*
 * UserOutput.h
 *
 * 2009-11-26	Tim Peeters
 * - First version.
 *
 * 2011-04-26	Evert van Aart
 * - The function "showMessage" now creates a message box.
 * - Improved the way progress bars are handled. Each algorithm gets a separate
 *   progress bar, which will exist until the plugin that created it destroys it.
 *   This way, it's easier to maintain progress bars for algorithms that update
 *   through the VTK pipeline; the algorithm's progress bar will only appear when
 *   the algorithm updates, and will remain hidden the rest of the time.
 * - Tidied up the code.
 *
 */


#ifndef bmia_UserOutput_h
#define bmia_UserOutput_h


/** Includes - Qt */

#include <QString>
#include <QTextStream>
#include <QMessageBox>
#include <QHash>
#include <QtDebug>

/** Includes - VTK */

#include <vtkAlgorithm.h>
#include <vtkCommand.h>

/** Includes - Custom Files */

#include "vtkTextProgressCommand.h"
#include "QVTKProgressCommand.h"


namespace bmia {


/** Class for showing messages and reporting progress to the user. Preferably,
	feedback to the user should be passed through this class (using "core()->out()")
	as much as possible, to keep the program output centralized. 
*/

class UserOutput
{
	public:
    
		/** Constructor. */
    
		UserOutput();

		/** Destructor. */

		~UserOutput();

		/** Show a message to the user. 
			@param msg		The text to show to the user.
			@param title	Optional title for the message dialog
			@param logMsg	If true, message will also be logged. True by default. */
    
		void showMessage(QString msg, QString title = QString(), bool logMsg = true);

		/** Add a message to the log, but don't show it to the user.
			@param msg		The message to add to the log. */
		
		void logMessage(QString msg);

		/** Create a progress bar with optional title and label for a specified
			algorithm. If no title is passed, "DTITool" is used as the window 
			title. If no label is passed, the progress bar uses the progress
			text of the algorithm if available, and the default string ("Progress
			for algorithm [Algorithm Name]") otherwise. Does nothing is the 
			algorithm already has a progress bar. 
			@param algorithm	Algorithm for which a progress bar should be created.
			@param title		Optional window title for the progress bar. 
			@param label		Optional label (progress text). */

		void createProgressBarForAlgorithm(vtkAlgorithm * algorithm, QString title = QString(), QString label = QString());

		/** Delete the progress bar for the input algorithm. Does nothing if the 
			input algorithm doesn't have a progress bar. 
			@param algorithm	Algorithm for which the progress bar should be deleted. */
	
		void deleteProgressBarForAlgorithm(vtkAlgorithm * algorithm);

	protected:

	private:

		/** Text stream used for logging messages. */
    
		QTextStream outStream;

		/** Hash map, used to keep track of the algorithms for which a progress bar
			has been created. The algorithm pointer is used as the key, while the
			progress bar pointer is used as the value. */

		QHash<vtkAlgorithm *, QVTKProgressCommand *> registeredAlgorithms;

}; // class UserOutput


} // namespace bmia


#endif // bmia_UserOutput_h
