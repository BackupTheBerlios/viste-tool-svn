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
 * XMLSettingsReader.h
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


#ifndef bmia_XMLSettingsReader_h
#define bmia_XMLSettingsReader_h


/** Includes - Qt */

#include <QtXml/QDomDocument>
#include <QtXml/QDomElement>
#include <QtXml/QDomNode>
#include <QApplication>
#include <QFile>
#include <QDir>
#include <QString>
#include <QStringList>
#include <QMessageBox>
#include <QColor>

/** Includes - Custom Files */

#include "DTIToolProfile.h"
#include "DTIToolSettings.h"
#include "XMLSettingsWriter.h"


namespace bmia {


/** This class reads one or more profiles from the "settings.xml" file. It is
	called from the core after all the main components of the tool have been
	initialized. The profiles read from "settings.xml" are used by the core to
	load plugins and files. The "settings.xml" file should always be in the 
	same directory as the main application. 
*/


class XMLSettingsReader
{
	public:

		/** Constructor */

		XMLSettingsReader();

		/** Destructor */

		~XMLSettingsReader();

		/** Read the settings XML file. The input file name will be the full path
			to the "settings.xml" file, which should be in the same directory as
			the main application. The profiles read from the file are stored in 
			the "profiles" list. The function returns an error string, which 
			will be empty on success. 
			@param fileName			Full path to "settings.xml". 
			@param defaultSettings	Default DTITool settings. */

		QString readSettings(QString fileName, DTIToolSettings * defaultSettings);

		/** Delete all profiles (if any) read from "settings.xml". Called when
			the reader encounters an error; on success, the profiles are kept
			in memory, so that the core can use them. */

		void deleteAllProfiles();

		/** List of all profiles successfully read from "settings.xml". */

		QList<DTIToolProfile *> profiles;

	private:

		/** Profile currently under construction. If the profile is successfully
			read, this pointer is added to the "profiles" list, and the 
			"newProfile" variable is reset to NULL. */

		DTIToolProfile * newProfile;

		/** Parse the profiles stored within the "profiles" tag. 
			@param profilesElement	XML element representing the "profiles" tag. */

		QString parseProfiles(QDomElement profilesElement);

		/** Parse the general DTITool settings stored within the "settings" tag.
			@param settingsElement	XML element representing the "settings" tag.
			@param settings			Output DTITool settings. */

		QString parseSettings(QDomElement settingsElement, DTIToolSettings * settings);

		/** Parse the GUI shortcuts stored within the "guishortcuts" tag. 
			@param guiElement		XML element representing the "guishortcuts" tag.
			@param settings			Output DTITool settings. */

		QString parseGUIShortcuts(QDomElement guiElement, DTIToolSettings * settings);

}; // class XMLSettingsReader


} // namespace bmia


#endif // bmia_XMLSettingsReader_h
