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