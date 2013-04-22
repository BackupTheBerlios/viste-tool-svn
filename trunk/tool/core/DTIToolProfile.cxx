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
 * DTIToolProfile.cxx
 *
 * 2011-03-18	Evert van Aart
 * - First version
 *
 */


#include "DTIToolProfile.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

DTIToolProfile::DTIToolProfile()
{
	// Default settings for the profile
	this->isDefault   = false;
	this->profileName = "Default";

	// Make all directories equal to the application directory
	this->appDir    = QDir(qApp->applicationDirPath());
	this->pluginDir = this->appDir;
	this->dataDir   = this->appDir;
}


//------------------------------[ Destructor ]-----------------------------\\

DTIToolProfile::~DTIToolProfile()
{
	// Clear the string lists
	this->pluginLoadList.clear();
	this->pluginOpenList.clear();
	this->openFileList.clear();
}


//-----------------------------[ setPluginDir ]----------------------------\\

bool DTIToolProfile::setPluginDir(QString dirString)
{
	// Check if the directory contains the "%app%" placeholder
	if (dirString.left(5) == "%app%")
	{
		// Check if the plugin directory is the same as the application directory
		if (dirString == "%app%")
		{
			this->pluginDir = this->appDir;
		}
		else
		{
			// Replace the "%app%" placeholder with the application directory path
			QString relativePath = dirString.right(dirString.length() - 6);
			QString fullPath = this->appDir.absoluteFilePath(relativePath);
			this->pluginDir = QDir(fullPath);
		}
	}
	else
	{
		// Create a new directory handle from the (absolute) path
		this->pluginDir = QDir(dirString);
	}

	// Check if the directory exists
	return (this->pluginDir.exists());
}


//------------------------------[ setDataDir ]-----------------------------\\

bool DTIToolProfile::setDataDir(QString dirString)
{
	// Similar to "setPluginDir"

	if (dirString.left(5) == "%app%")
	{
		if (dirString == "%app%")
		{
			this->dataDir = this->appDir;
		}
		else
		{
			QString relativePath = dirString.right(dirString.length() - 6);
			QString fullPath = this->appDir.absoluteFilePath(relativePath);
			this->dataDir = QDir(fullPath);
		}
	}
	else
	{
		this->dataDir = QDir(dirString);
	}

	return (this->dataDir.exists());
}


//--------------------------[ setPluginLoadList ]--------------------------\\

bool DTIToolProfile::setPluginLoadList(QStringList rList)
{
	// Loop through all strings in the list
	for (int i = 0; i < rList.size(); ++i)
	{
		QString currentString = rList.at(i);

		// Remove all placeholders from the string (e.g., replace "%plugin%" with
		// the plugin directory path.

		currentString = this->removePlaceholders(currentString);

		QFile tempFile(currentString);

		// Check if the specified file exists
		if (!(tempFile.exists()))
			return false;

		// Add it to the list
		this->pluginLoadList.append(currentString);
	}

	return true;
}


//--------------------------[ setPluginOpenList ]--------------------------\\

bool DTIToolProfile::setPluginOpenList(QStringList rList)
{
	// Similar to "setPluginLoadList"

	for (int i = 0; i < rList.size(); ++i)
	{
		QString currentString = rList.at(i);

		currentString = this->removePlaceholders(currentString);

		QFile tempFile(currentString);

		if (!(tempFile.exists()))
			return false;

		this->pluginOpenList.append(currentString);
	}

	return true;
}


//---------------------------[ setOpenFileList ]---------------------------\\

bool DTIToolProfile::setOpenFileList(QStringList rList)
{
	// Similar to "setPluginLoadList"

	for (int i = 0; i < rList.size(); ++i)
	{
		QString currentString = rList.at(i);

		currentString = this->removePlaceholders(currentString);

		QFile tempFile(currentString);

		if (!(tempFile.exists()))
			return false;

		this->openFileList.append(currentString);
	}

	return true;
}


//-------------------------[ removePlaceholders ]--------------------------\\

QString DTIToolProfile::removePlaceholders(QString inString)
{
	// Replace "%app%" by the application directory path
	if (inString.left(5) == "%app%")
	{
		return this->appDir.absoluteFilePath(inString.right(inString.length() - 6));
	}
	// Replace "%plugin%" by the plugin directory path
	else if (inString.left(8) == "%plugin%")
	{
		return this->pluginDir.absoluteFilePath(inString.right(inString.length() - 9));
	}
	// Replace "%data%" by the data directory path
	else if (inString.left(6) == "%data%")
	{
		return this->dataDir.absoluteFilePath(inString.right(inString.length() - 7));
	}

	// no placeholders found, so return the unmodified input string
	return inString;
}


//--------------------------[ addAppPlaceholder ]--------------------------\\

QString DTIToolProfile::addAppPlaceholder(QString inString)
{
	return this->addPlaceholder(inString, this->appDir.absolutePath(), "%app%");
}


//-------------------------[ addPluginPlaceholder ]------------------------\\

QString DTIToolProfile::addPluginPlaceholder(QString inString)
{
	return this->addPlaceholder(inString, this->pluginDir.absolutePath(), "%plugin%");
}


//--------------------------[ addDataPlaceholder ]-------------------------\\

QString DTIToolProfile::addDataPlaceholder(QString inString)
{
	return this->addPlaceholder(inString, this->dataDir.absolutePath(), "%data%");
}


//----------------------------[ addPlaceholder ]---------------------------\\

QString DTIToolProfile::addPlaceholder(QString fileName, QString dir, QString placeholder)
{
	// Fix the drive letters
	this->fixDriveLetter(fileName);
	this->fixDriveLetter(dir);

	// Also fix the separators
	fileName = QDir::fromNativeSeparators(fileName);
	dir = QDir::fromNativeSeparators(dir);

	// Do nothing if the file name does not start with the target directory
	if (fileName.left(dir.length()) != dir)
		return fileName;

	// If it does, replace the directory with the placeholder
	return (placeholder + fileName.right(fileName.length() - dir.length()));
}


//---------------------------[ fixDriveLetter ]----------------------------\\

void DTIToolProfile::fixDriveLetter(QString & inString)
{
	// String must contain at least three characters
	if (inString.length() > 2)
	{
		// If the second and third characters are ":/" or ":\", and the first
		// character is lowercase, convert the first character (which is a drive
		// letter) to uppercase.

		if (	 inString[1] == QChar(':') && 
				(inString[2] == QChar('/') || inString[2] == QChar('\\')) && 
				inString[0].isLower())
		{
			inString[0] = inString[0].toUpper();
		}
	}
}


} // namespace bmia
