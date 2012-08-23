/*
 * XMLSettingsReader.cxx
 *
 * 2011-03-15	Evert van Aart
 * - First version.
 *
 * 2011-07-18	Evert van Aart
 * - Added support for reading the general DTITool settings.
 * - The introduction of the general settings required a new XML structure, with
 *   "DTIToolSettings" as the root-level tag instead of "profiles". The reader
 *   can still parse the old format, but will immediately convert it to the 
 *   new format.
 *
 */


/** Includes */

#include "XMLSettingsReader.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

XMLSettingsReader::XMLSettingsReader()
{
	// Set pointers to NULL, clear the list
	this->newProfile = NULL;
	this->profiles.clear();
}


//------------------------------[ Destructor ]-----------------------------\\

XMLSettingsReader::~XMLSettingsReader()
{
	// Delete the new profile. If this pointer is non-NULL, it means that the
	// profile was created, but not added to the list due to an error.

	if (this->newProfile)
		delete this->newProfile;

	// Clear the list of profiles. Do not delete them, since the core will still
	// use the profile objects.

	this->profiles.clear();
}


//-----------------------------[ readSettings ]----------------------------\\

QString XMLSettingsReader::readSettings(QString fileName, DTIToolSettings * defaultSettings)
{
	// Error string. Remains empty if every goes right.
	QString err = "";

	// Create a XML document object
	QDomDocument doc("DTIToolSettings");

	// Create a file handle with the specified filename
	QFile file(fileName);

	// Try to open the file
	if (!file.open(QIODevice::ReadOnly))
	{
		err = "Could not open file '" + fileName + "'.";
		return err;
	}

	// Try to parse the XML file
	if (!doc.setContent(&file)) 
	{
		file.close();
		err = "Failed to parse XML file '" + fileName + "'.";
		return err;
	}

	// Close the file
	file.close();

	// The document type (specified by "!DOCTYPE") should be "DTIToolSettings"
	if (doc.doctype().name() != "DTIToolSettings")
	{
		err = "Expected document type 'DTIToolSettings', but found '" + doc.doctype().name() + "'.";
		return err;
	}

	// Get the top-level tag
	QDomElement docElem = doc.documentElement();

	// If the top-level tag is "profiles", this settings file still has the old structure.
	// We first read this old file, and then write it to the new structure.

	if (docElem.tagName() == "profiles")
	{
		// Parse the profiles
		err = this->parseProfiles(docElem);

		if (err.isEmpty() == false)
			return err;

		// Tell the user that we're converting the file to the new format
		QMessageBox::warning(NULL, "XML Settings Reader", "Found 'settings.xml' file with deprecated structure; writing new settings file...");

		XMLSettingsWriter * writer = new XMLSettingsWriter;

		// Write the settings
		err = writer->writeSettings(fileName, this->profiles, defaultSettings);

		if (err.isEmpty())
			return err;
		else
			return "XMLSettingsWriter: " + err;
	}

	// In the new format, the top-level tag is "DTIToolSettings"
	else if (docElem.tagName() != "DTIToolSettings")
	{
		err = "Expect top-level tag 'DTIToolSettings', but found '" + docElem.tagName() + "'.";
		return err;
	}

	// Get the first child node
	QDomNode docNode = docElem.firstChild();

	// Loop through all child tags
	while (docNode.isNull() == false)
	{
		// Convert the node to an element
		QDomElement settingsElement = docNode.toElement();

		// Check if the conversion succeeded
		if (settingsElement.isNull())
		{
			err = "Failed to convert a node within the 'DTIToolSettings' tag to a 'QDomElement' object.";
			return err;
		}

		QString settingsTagName = settingsElement.tagName();

		// Parse the general DTITool settings
		if (settingsTagName == "settings")
		{
			err = this->parseSettings(settingsElement, defaultSettings);

			if (err.isEmpty() == false)
				return err;
		}

		// Parse the profiles
		else if (settingsTagName == "profiles")
		{
			err = this->parseProfiles(settingsElement);
	
			if (err.isEmpty() == false)
				return err;
		}
		else
		{
			err = "Encountered unknown tag type '" + settingsTagName + "'.";
			return err;
		}

		// Get the next node
		docNode = docNode.nextSibling();
	}

	return "";
}


//----------------------------[ parseSettings ]----------------------------\\

QString XMLSettingsReader::parseSettings(QDomElement settingsElement, DTIToolSettings * settings)
{
	// Error string returned by this function
	QString err = "";

	// Get the first child tag of the "profiles" tag
	QDomNode docNode = settingsElement.firstChild();

	// Loop through all profiles
	while (docNode.isNull() == false)
	{
		// Convert the node to an element
		QDomElement currentElement = docNode.toElement();

		// Check if the conversion succeeded
		if (currentElement.isNull())
		{
			err = "Failed to convert a node within the 'settings' tag to a 'QDomElement' object.";
			return err;
		}

		QString currentTagName = currentElement.tagName();

		// Window size
		if (currentTagName == "windowsize")
		{
			// Get the necessary attributes
			QString wString = currentElement.attribute("w", "ERR");
			QString hString = currentElement.attribute("h", "ERR");
			QString maxString = currentElement.attribute("maximize", "ERR");

			// Check if all attributes exist
			if (wString == "ERR" || hString == "ERR" || maxString == "ERR")
			{
				err = "Could not find all required attributes for the window size!";
				return err;
			}

			bool parseOK[3] = {true, true, true};

			// Parse width and height to integers
			int w = wString.toInt(&(parseOK[0]));
			int h = hString.toInt(&(parseOK[1]));

			bool maxWindow = true;

			// Parse the string for maximizing the window
			if (maxString == "yes")			maxWindow = true;
			else if (maxString == "no")		maxWindow = false;
			else							parseOK[2] = false;

			// Check if one of the parsing operations failed
			if (!(parseOK[0] && parseOK[1] && parseOK[2]))
			{
				err = "Failed to parse attributes for the window size!";
				return err;
			}

			// Store the settings
			settings->windowWidth    = w;
			settings->windowHeight   = h;
			settings->maximizeWindow = maxWindow;
		}

		// Background color
		else if (currentTagName == "backgroundcolor")
		{
			QString rString = currentElement.attribute("r", "ERR");
			QString gString = currentElement.attribute("g", "ERR");
			QString bString = currentElement.attribute("b", "ERR");
			QString gradientString = currentElement.attribute("gradient", "ERR");

			if (rString == "ERR" || gString == "ERR" || bString == "ERR" || gradientString == "ERR")
			{
				err = "Could not find all required attributes for the background color!";
				return err;
			}

			bool parseOK[4] = {true, true, true, true};

			// Parse RGB values to integers
			int r = rString.toInt(&(parseOK[0]));
			int g = gString.toInt(&(parseOK[1]));
			int b = bString.toInt(&(parseOK[2]));

			bool gradient = true;

			// Parse the string for applying a background gradient
			if (gradientString == "yes")			gradient = true;
			else if (gradientString == "no")		gradient = false;
			else									parseOK[3] = false;

			// Check if one of the parsing operations failed
			if (!(parseOK[0] && parseOK[1] && parseOK[2] && parseOK[3]))
			{
				err = "Failed to parse attributes for the window size!";
				return err;
			}

			// Store the settings
			settings->backgroundColor = QColor(r, g, b, 255);
			settings->gradientBackground = gradient;
		}

		// GUI Shortcuts
		else if (currentTagName == "guishortcuts")
		{
			err = this->parseGUIShortcuts(currentElement, settings);

			if (err.isEmpty() == false)
				return err;
		}
		else
		{
			err = "Encountered unknown tag type '" + currentTagName + "'.";
			return err;
		}

		// Get the next profile node
		docNode = docNode.nextSibling();
	}

	return "";
}


//--------------------------[ parseGUIShortcuts ]--------------------------\\

QString XMLSettingsReader::parseGUIShortcuts(QDomElement guiElement, DTIToolSettings * settings)
{
	QString err = "";

	QDomNode docNode = guiElement.firstChild();

	// Loop through all nodexs
	while (docNode.isNull() == false)
	{
		QDomElement currentElement = docNode.toElement();

		if (currentElement.isNull())
		{
			err = "Failed to convert a node within the 'guishortcuts' tag to a 'QDomElement' object.";
			return err;
		}

		QString currentTagName = currentElement.tagName();

		// We only expect "guishortcut" tags here
		if (currentTagName != "guishortcut")
		{
			err = "Expected 'guishortcut' tag, found '" + currentTagName + "'.";
			return err;
		}

		// Get the three required attributes
		QString iString      = currentElement.attribute("i", "ERR");
		QString pluginString = currentElement.attribute("plugin", "ERR");
		QString posString    = currentElement.attribute("pos", "ERR");

		// Check if all attributes existed
		if (iString == "ERR" || pluginString == "ERR" || posString == "ERR")
		{
			err = "Could not find all required attributes for the GUI shortcuts!";
			return err;
		}

		bool parseOK;

		// Parse the index to an integer
		int i = iString.toInt(&parseOK);

		// Check if parsing was successful, and if "i" is in the correct range
		if (!parseOK || i < 0 || i > 9)
		{
			err = "Failed to parse shortcut index!";
			return err;
		}

		// Parse the position string
		DTIToolSettings::GUIPosition guiPos;
		if (posString == "Top")				guiPos = DTIToolSettings::GUIP_Top;
		else if (posString == "TopExcl")	guiPos = DTIToolSettings::GUIP_TopExclusive;
		else if (posString == "Bottom")		guiPos = DTIToolSettings::GUIP_Bottom;
		else
		{
			err = "Failed to parse GUI position!";
			return err;
		}

		// Store the settings
		settings->guiShortcuts[i].plugin = pluginString;
		settings->guiShortcuts[i].position = guiPos;

		// Get the next node
		docNode = docNode.nextSibling();
	}

	// Success!
	return "";
}


//----------------------------[ parseProfiles ]----------------------------\\

QString XMLSettingsReader::parseProfiles(QDomElement profilesElement)
{
	// Error string returned by this function
	QString err = "";

	// Get the first child tag of the "profiles" tag
	QDomNode docNode = profilesElement.firstChild();

	// Loop through all profiles
	while (docNode.isNull() == false)
	{
		// Convert the node to an element
		QDomElement profileElement = docNode.toElement();

		// Check if the conversion succeeded
		if (profileElement.isNull())
		{
			err = "Failed to convert a node within the 'profiles' tag to a 'QDomElement' object.";
			return err;
		}

		// At this level, all tags should be of type "profile"
		QString profileTagName = profileElement.tagName();

		if (profileTagName != "profile")
		{
			err = "Expected a 'profile' tag, but found a '" + profileTagName + "' tag.";
			return err;
		}

		// Get the profile name and whether or not it is the default profile
		QString profileName = profileElement.attribute("name", "Unnamed profile");
		QString isDefaultString = profileElement.attribute("default", "no");

		// Create a new profile
		this->newProfile = new DTIToolProfile;
		this->newProfile->profileName = profileName;
		this->newProfile->isDefault = (isDefaultString == "yes");

		// By default, all three directories are the application directory
		QDir appDir = QDir(qApp->applicationDirPath());
		this->newProfile->appDir    = appDir;
		this->newProfile->pluginDir = appDir;
		this->newProfile->dataDir   = appDir;

		// Profile information
		QStringList openFileList;
		QStringList pluginLoadList;
		QStringList pluginOpenList;
		QString     dataFolder		= "";
		QString     pluginFolder	= "";

		// Get the first child of the "profile" tag
		QDomNode profileNode = profileElement.firstChild();

		// Loop through all children tags of the "profile" tag
		while (profileNode.isNull() == false)
		{
			// Convert the current node to an element
			QDomElement profileDataElement = profileNode.toElement();

			// Check if the conversion succeeded
			if (profileDataElement.isNull())
			{
				err = "Failed to convert a node within the '" + profileName + "' tag to a 'QDomElement' object.";
				return err;
			}

			// Get the tag name
			QString dataTag = profileDataElement.tagName();

			// Plugins to be opened and (optionally) loaded on startup
			if (dataTag == "plugin")
			{
				// Get the path to the plugin.
				QString pluginPath = profileDataElement.attribute("file", "");

				if (pluginPath.isEmpty())
				{
					err = "Encountered a 'plugin' tag without a 'file' attribute.";
					return err;
				}

				// Get the "load" attribute, which specifies whether the plugin should be loaded now
				QString loadNowString = profileDataElement.attribute("load", "yes");

				// Add the plugin filename to one of the two strings, depending
				// on the "loadNowString" contents.

				if (loadNowString == "yes")
				{
					pluginLoadList.append(pluginPath);
				}
				else
				{
					pluginOpenList.append(pluginPath);
				}
			}

			// Files to be opened on startup
			else if (dataTag == "openfile")
			{
				// Get the file path from the attributes
				QString filePath = profileDataElement.attribute("file", "");

				if (filePath.isEmpty())
				{
					err = "Encountered a 'openfile' tag without a 'file' attribute.";
					return err;
				}

				// Add the path to the list
				openFileList.append(filePath);
			}

			// Folder definitions
			else if (dataTag == "folder")
			{
				// Get the directory
				QString folderDir = profileDataElement.attribute("dir", "");

				if (folderDir.isEmpty())
				{
					err = "Encountered a 'folder' tag without a 'appDir' attribute.";
					return err;
				}

				// Get the folder type
				QString folderType = profileDataElement.attribute("type", "");

				// Plugin folder, corresponds to the "%plugin%" placeholder
				if (folderType == "plugin")
				{
					pluginFolder = folderDir;
				}
				// Data folder, corresponds to the "%data%" placeholder
				else if (folderType == "data")
				{
					dataFolder = folderDir;
				}
				else
				{
					err = "Encountered unknown folder type '" + folderType + "'.";
					return err;
				}
			}
			else
			{
				err = "Encountered unknown tag type '" + dataTag + "'.";
				return err;
			}

			// Get the next node within the "profile" tag
			profileNode = profileNode.nextSibling();

		} // while [All nodes of one profile]

		// Add the plugin folder definition to the profile. This parses the 
		// "%app%" placeholder, if present, and checks if the directory exists.

		if (pluginFolder.isEmpty() == false)
		{
			if (!(this->newProfile->setPluginDir(pluginFolder)))
			{
				err = "Plugin folder '" + pluginFolder + "' does not exist.";
				return err;
			}
		}

		// Do the same for the data folder definition
		if (dataFolder.isEmpty() == false)
		{
			if (!(this->newProfile->setDataDir(dataFolder)))
			{
				err = "Data folder '" + dataFolder + "' does not exist.";
				return err;
			}
		}

		// Copy, parse and check the plugin list (to be loaded now)
		if (!(this->newProfile->setPluginLoadList(pluginLoadList)))
		{
			err = "Failed to locate one or more plugins for profile '" + profileName + "'.";
			return err;
		}

		// Copy, parse and check the plugin list (to be loaded later)
		if (!(this->newProfile->setPluginOpenList(pluginOpenList)))
		{
			err = "Failed to locate one or more plugins for profile '" + profileName + "'.";
			return err;
		}

		// Copy, parse and check the list of files to be opened on startup
		if (!(this->newProfile->setOpenFileList(openFileList)))
		{
			err = "Failed to locate one or more data files for profile '" + profileName + "'.";
			return err;
		}

		// Add the profile to the list
		this->profiles.append(this->newProfile);
		this->newProfile = NULL;

		// Get the next profile node
		docNode = docNode.nextSibling();

	} // while [All profiles]

	// Success, return an empty error string
	return "";
}


//--------------------------[ deleteAllProfiles ]--------------------------\\

void XMLSettingsReader::deleteAllProfiles()
{
	// Delete the new profile
	if (this->newProfile)
	{
		delete this->newProfile;
		this->newProfile = NULL;
	}

	// Delete all stored profiles
	for (int i = 0; i < this->profiles.size(); ++i)
	{
		DTIToolProfile * currentProfile = this->profiles.at(i);

		delete currentProfile;
	}

	// Clear the list of profiles
	this->profiles.clear();
}


} // namespace bmia
