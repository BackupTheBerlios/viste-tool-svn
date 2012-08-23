/*
 * DTIToolProfile.h
 *
 * 2011-03-18	Evert van Aart
 * - First version
 *
 */


#ifndef bmia_DTIToolProfile_h
#define bmia_DTIToolProfile_h


/** Includes - Qt */

#include <QString>
#include <QStringList>
#include <QFile>
#include <QDir>
#include <QApplication>


namespace bmia {


/** This class represents a so-called "profile", which is a collection of settings
	for the DTITool. These profiles are loaded from the "settings.xml" file which
	is located in the application directory, and they can also be created by the
	user through the Profile Manager. When "settings.xml" contains multiple profiles,
	only the one defined as the default profile is loaded. Profile settings include
	the plugins that should be added and/or loaded on start-up, the data files that
	should be opened, and definitions of the default folders for data and plugins. 
*/

class DTIToolProfile
{
	public:

		/** Constructor */

		DTIToolProfile();

		/** Destructor */

		~DTIToolProfile();

		/** Create a "QDir" directory handle for the plugin directory, based on
			the input string. This string may contain the "%app%" keywords, which
			will be replaced by the application directory path. Returns true if 
			the specified directory exists, false otherwise.
			@param dirString	Path string (absolute or with "%app%" placeholder. */

		bool setPluginDir(QString dirString);

		/** Create a "QDir" directory handle for the data directory, based on
			the input string. This string may contain the "%app%" keywords, which
			will be replaced by the application directory path. Returns true if 
			the specified directory exists, false otherwise.
			@param dirString	Path string (absolute or with "%app%" placeholder. */

		bool setDataDir(QString dirString);

		/** Copies, parses and checks a list of plugin file paths that should be 
			loaded on start-up. Paths may contain placeholders; before adding them
			to the "pluginLoadList" list, this function removes the placeholders,
			thus creating absolute paths. Returns false if it encounters a file
			that does not exist; returns true if all files exist. 
			@param rList		List of file paths. */

		bool setPluginLoadList(QStringList rList);

		/** Copies, parses and checks a list of plugin file paths that should be 
			added (but not loaded) on start-up. Paths may contain placeholders; 
			before adding them to the "pluginOpenList" list, this function 
			removes the placeholders, thus creating absolute paths. Returns 
			false if it encounters a file that does not exist; returns true 
			if all files exist. 
			@param rList		List of file paths. */

		bool setPluginOpenList(QStringList rList);

		/** Copies, parses and checks a list of data file paths that should be 
			opened on start-up. Paths may contain placeholders; before adding them
			to the "openFileList" list, this function removes the placeholders,
			thus creating absolute paths. Returns false if it encounters a file
			that does not exist; returns true if all files exist. 
			@param rList		List of file paths. */

		bool setOpenFileList(QStringList rList);

		
		QString	profileName;			/**< Name of the profile. */
		bool isDefault;					/**< True is this is the default profile. */
		QDir appDir;					/**< Application directory handle. */
		QDir dataDir;					/**< Data directory handle. */
		QDir pluginDir;					/**< Plugin directory handle. */

		/** List of full file paths for plugins that should be loaded on start-up. */

		QStringList pluginLoadList;

		/** List of full file paths for plugins that should be added on start-up. */

		QStringList pluginOpenList;

		/** List of full file paths for data files that should be opened on start-up. */

		QStringList openFileList;

		/** Replace a placeholder by the corresponding directory path, thus turning
			a relative path into an absolute path. If the input string does not 
			contain a placeholder, it will be returned unmodified. 
			@param inString		Input path, absolute or with placeholder. */

		QString removePlaceholders(QString inString);

		/** Checks if the first part of the input path matches the path of the
			application directory, and if so, replace it with the "%app%" 
			placeholder; otherwise, return the unmodified input string. */

		QString addAppPlaceholder(QString inString);

		/** Checks if the first part of the input path matches the path of the
			plugin directory, and if so, replace it with the "%plugin%" 
			placeholder; otherwise, return the unmodified input string. */

		QString addPluginPlaceholder(QString inString);

		/** Checks if the first part of the input path matches the path of the
			data directory, and if so, replace it with the "%data%" 
			placeholder; otherwise, return the unmodified input string. */

		QString addDataPlaceholder(QString inString);

		/** Replace the first part of an absolute file path ("fileName") by a
			target placeholder if it matches the (also absolute) directory "dir".
			Used to create the "%app%", "%plugin%" and "%data%" placeholders. 
			@param fileName		Absolute file path.
			@param dir			Absolute directory path.
			@param placeholder	If "fileName" starts with "dir", replace "dir"
								by this placeholder. */

		QString addPlaceholder(QString fileName, QString dir, QString placeholder);

	private:


		/** On Windows systems, the drive letter (e.g., "C") may sometimes be
			lowercase, and sometimes uppercase, depending on how the path of a file
			or directory was obtained. To prevent these differences from blocking
			the placement of placeholders in the "addPlaceholder" function, this
			function converts the drive letter to uppercase. 
			@param inString		Input path. */

		void fixDriveLetter(QString & inString);

};


}


#endif // bmia_DTIToolProfile_h