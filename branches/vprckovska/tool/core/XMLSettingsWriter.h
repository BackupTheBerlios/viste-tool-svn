/*
 * XMLSettingsWriter.h
 *
 * 2011-03-15	Evert van Aart
 * - First version.
 *
 * 2011-07-18	Evert van Aart
 * - Added support for writing settings (e.g., window size, shortcuts, etc).
 *
 */


#ifndef bmia_XMLSettingsWriter_h
#define bmia_XMLSettingsWriter_h


/** Includes - Qt */

#include <QApplication>
#include <QFile>
#include <QDir>
#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QColor>

/** Includes - Custom Files */

#include "DTIToolProfile.h"
#include "DTIToolSettings.h"


namespace bmia {


/** This class creates a settings XML file ("settings.xml") in the same folder
	as the main application, using the available profiles and general DTITool
	settings. Here, we use simple text streams to generate the output XML 
	data, as they allow for more descriptive formatting than the default 
	XML methods provided by Qt. 
*/

class XMLSettingsWriter
{
	public:

		/** Constructor */

		XMLSettingsWriter();

		/** Destructor */

		~XMLSettingsWriter();

		/** Write the settings of all profiles, as well as the general DTITool
			settings, to the target file. The target file name will be the full 
			path to "settings.xml", located in the same	folder as the main application. 
			@param fileName		Full path to "settings.xml".
			@param profiles		List of all profiles. 
			@param settings		General DTITool settings. */

		QString writeSettings(QString fileName, QList<DTIToolProfile *> profiles, DTIToolSettings * settings);
		

}; // class XMLSettingsWriter


} // namespace bmia


#endif // bmia_XMLSettingsWriter_h