/*
 * XMLSettingsWriter.cxx
 *
 * 2011-03-15	Evert van Aart
 * - First version.
 *
 * 2011-07-18	Evert van Aart
 * - Added support for writing settings (e.g., window size, shortcuts, etc).
 *
 */


/** Includes */

#include "XMLSettingsWriter.h"


using namespace std;


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

XMLSettingsWriter::XMLSettingsWriter()
{

}


//------------------------------[ Destructor ]-----------------------------\\

XMLSettingsWriter::~XMLSettingsWriter()
{

}


//----------------------------[ writeSettings ]----------------------------\\

QString XMLSettingsWriter::writeSettings(QString fileName, QList<DTIToolProfile *> profiles, DTIToolSettings * settings)
{
	// Error string. Remains empty if every goes right.
	QString err = "";

	// Create a file handle with the specified filename
	QFile file(fileName);

	// Try to open the file
	if (!file.open(QIODevice::WriteOnly))
	{
		err = "Could not open file '" + fileName + "'.";
		return err;
	}

	// Create a text stream
	QTextStream out(&file);

	// Output the header and the root tag
	out << "<!DOCTYPE DTIToolSettings>\n";
	out << "<DTIToolSettings>\n";
	out << "\t<profiles>\n";

	// Loop through all profiles
	for (int profileID = 0; profileID < profiles.size(); ++profileID)
	{
		// Get the current profile
		DTIToolProfile * currentProfile = profiles.at(profileID);

		// Output the "profile" tag and its attributes
		out << "\t\t<profile name=\"" << currentProfile->profileName << 
			"\" default=\"" << (currentProfile->isDefault ? "yes" : "no") << "\">\n";

		// Loop through all plugin file names (loaded)
		for (int i = 0; i < currentProfile->pluginLoadList.size(); ++i)
		{
			// Get the filename
			QString currentPluginFileName = currentProfile->pluginLoadList.at(i);

			// Add the "%plugin%" placeholder, if possible
			currentPluginFileName = currentProfile->addPluginPlaceholder(currentPluginFileName);

			// Otherwise, add the "%app%" placeholder, if possible
			currentPluginFileName = currentProfile->addAppPlaceholder(currentPluginFileName);
			
			// Output the "plugin" tag and its attributes
			out << "\t\t\t<plugin file=\"" << currentPluginFileName << "\" load=\"" <<
														"yes\" />\n";
		}

		// Loop through all plugin file names (unloaded)
		for (int i = 0; i < currentProfile->pluginOpenList.size(); ++i)
		{
			QString currentPluginFileName = currentProfile->pluginOpenList.at(i);
			currentPluginFileName = currentProfile->addPluginPlaceholder(currentPluginFileName);
			currentPluginFileName = currentProfile->addAppPlaceholder(currentPluginFileName);
			out << "\t\t\t<plugin file=\"" << currentPluginFileName << "\" load=\"" <<
				"no\" />\n";
		}

		// Loop through all filenames that must be loaded on startup
		for (int i = 0; i < currentProfile->openFileList.size(); ++i)
		{
			QString currentFileName = currentProfile->openFileList.at(i);
			currentFileName = currentProfile->addDataPlaceholder(currentFileName);
			currentFileName = currentProfile->addAppPlaceholder(currentFileName);
			out << "\t\t\t<openfile file=\"" << currentFileName << "\" />\n";
		}

		// Add the plugin directory tag (with "%app%" placeholder, if possible)
		QString pluginDirPH = currentProfile->pluginDir.absolutePath();
		pluginDirPH = currentProfile->addAppPlaceholder(pluginDirPH);
		out << "\t\t\t<folder type=\"plugin\" dir=\"" << pluginDirPH << "\" />\n";

		// Add the data directory tag (with "%app%" placeholder, if possible)
		QString dataDirPH = currentProfile->dataDir.absolutePath();
		dataDirPH = currentProfile->addAppPlaceholder(dataDirPH);
		out << "\t\t\t<folder type=\"data\" dir=\"" << dataDirPH << "\" />\n";

		// Closing tag for this profile
		out << "\t\t</profile>\n";

	} // for [All profiles]

	// Close the profiles tag
	out << "\t</profiles>\n";

	// Open the settings tag
	out << "\t<settings>\n";

	// Write the window size
	out << "\t\t<windowsize w=\"" << settings->windowWidth << "\" h=\"" << settings->windowHeight << 
		"\" maximize=\"" << (settings->maximizeWindow ? "yes" : "no") << "\" />\n";

	// Write the background color
	out << "\t\t<backgroundcolor r=\"" << settings->backgroundColor.red() << "\" g=\"" <<
		settings->backgroundColor.green() << "\" b=\"" << settings->backgroundColor.blue() <<
		"\" gradient=\"" << (settings->gradientBackground ? "yes" : "no") << "\" />\n";

	// Write the opening tag for the GUI shortcuts
	out << "\t\t<guishortcuts>\n";

	// Loop through all gui shortcuts
	for (int i = 0; i < 10; ++i)
	{
		QString pluginName = settings->guiShortcuts[i].plugin;

		// Only write the shortcuts that have been defined
		if (pluginName == "None")
			continue;

		// Translate the position to a string
		DTIToolSettings::GUIPosition pluginPosition = settings->guiShortcuts[i].position;

		QString positionString;

		switch (pluginPosition)
		{
			case DTIToolSettings::GUIP_Top:				positionString = "Top";		break;
			case DTIToolSettings::GUIP_TopExclusive:	positionString = "TopExcl";	break;
			case DTIToolSettings::GUIP_Bottom:			positionString = "Bottom";	break;
			default:									positionString = "Top";		break;
		}

		// Write the shortcut tag to the output
		out << "\t\t\t<guishortcut i=\"" << i << "\" plugin=\"" <<
			pluginName << "\" pos=\"" << positionString << "\" />\n";
	}

	// Write the closing tag for the GUI shortcuts
	out << "\t\t</guishortcuts>\n";

	// Close the settings tag
	out << "\t</settings>\n";

	// Close the root tag
	out << "</DTIToolSettings>\n";

	// flush and close the output
	out.flush();
	file.close();

	// Done!
	return "";
}


} // namespace bmia
