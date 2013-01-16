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
 * Core.cxx
 *
 * 2009-11-05	Tim Peeters
 * - First version
 *
 * 2010-03-17	Tim Peeters
 * - No more singleton
 *
 * 2011-03-18	Evert van Aart
 * - Added support for reading and writing "settings.xml" using XML readers/writers.
 *
 * 2011-04-06	Evert van Aart
 * - Added "getDataDirectory".
 *
 * 2011-04-14	Evert van Aart
 * - Added "disableRendering" and "enableRendering".
 *
 */


/** Includes */

#include "Core.h"
#include "data/Manager.h"
#include "plugin/Manager.h"
#include "gui/MainWindowInterface.h"


namespace bmia {


Core::Core()
{
	qDebug() << "Constructing new Core object (" << this << ")";

	// Set pointers to the other components to NULL. The function that created this
	// core object will (or should) next create a "MainWindow" object; in its 
	// constructor, that object will set the following three pointers.

    this->dataManager   = NULL;
    this->pluginManager = NULL;
    this->userOutput    = NULL;

    this->mainWindow	= NULL;
	this->renderWindow	= NULL;
	this->medicalCanvas	= NULL;
    this->settings      = NULL;

    // No profiles loaded yet
    this->defaultProfile = NULL;

    // Allow rendering by default
	allowRendering = true;
}


Core::~Core()
{
	// Delete the settings
	if (this->settings)
		delete this->settings;

	// Delete all profiles
	for (QList<DTIToolProfile *>::iterator i = this->profiles.begin(); i != this->profiles.end(); ++i)
	{
		if (*i)
			delete (*i);
	}

	// Clear the list of profile
	this->profiles.clear();

	// Clear the following pointers. The main window object will be deleted by the 
	// same function that destroyed this core object, and it in turn will delete
	// the canvas and render window.

	this->mainWindow	= NULL;
	this->renderWindow	= NULL;
	this->medicalCanvas	= NULL;

	// Delete the components that were created by the core
	delete this->pluginManager;
	delete this->dataManager;
	delete this->userOutput;
}


void Core::init()
{
    // Initialize the data manager, plugin manager, and user output
    this->dataManager	= new data::Manager();
    this->pluginManager = new plugin::Manager(this);
    this->userOutput	= new UserOutput();

    // Create default settings
    this->settings = new DTIToolSettings;
}


void Core::applySettings()
{
	// Apply the settings to the main window
	this->mainWindow->applySettings(settings);

	// Get the full file path of the settings file
	QDir appDir = QDir(qApp->applicationDirPath());
	QString settingsFilePath = appDir.absoluteFilePath("settings.xml");

	// Write the modified settings
	XMLSettingsWriter * writer = new XMLSettingsWriter;
	QString err = writer->writeSettings(settingsFilePath, this->profiles, this->settings);

	// Check if anything failed
	if (!(err.isEmpty()))
		QMessageBox::warning(NULL, "DTITool", QString("Writing default 'settings.xml' failed with the following error: \n\n") + err);

	delete writer;
}


void Core::loadSettings()
{
	// Get the application directory, and create a file handle for "settings.xml"
	QDir appDir = QDir(qApp->applicationDirPath());
	QString settingsFilePath = appDir.absoluteFilePath("settings.xml");
	QFile settingsFile(settingsFilePath);

	// If the file does not exist, we create a default one
	if (!settingsFile.exists())
	{
		this->userOutput->showMessage("The file 'settings.xml' was not found, creating default settings file...");

		// Read all available plugin files
		this->pluginManager->readAll(appDir);

		// List of filenames for the plugins
		QStringList pluginFileNameList;

		// Add all available plugins to the list
		for (int i = 0; i < this->pluginManager->getNumberOfPlugins(); ++i)
		{
			pluginFileNameList.append(this->pluginManager->getFileName(i));
		}

		if (pluginFileNameList.isEmpty())
		{
			QMessageBox::warning(NULL, "DTITool", "No plugins found!");
			return;
		}

		// Create a new default profile
		DTIToolProfile * newProfile = new DTIToolProfile;
		newProfile->profileName = "Default";
		newProfile->isDefault = true;

		// Application directory and data directory are the same by default
		newProfile->appDir  = appDir;
		newProfile->dataDir = appDir;

		// Get the directory of the first plugin in the list; this is the plugin directory
		QString firstPluginFileName = pluginFileNameList.at(0);
		QDir firstPluginDir = QDir(firstPluginFileName);
		firstPluginDir.cdUp();
		newProfile->setPluginDir(firstPluginDir.absolutePath());

		// Add all loaded plugins to the profile
		newProfile->setPluginLoadList(pluginFileNameList);

		// Add the default profile to the list
		this->profiles.append(newProfile);

		// Create an XML writer, and write the "settings.xml" file
		XMLSettingsWriter * settingsWriter = new XMLSettingsWriter;
		QString err = settingsWriter->writeSettings(settingsFilePath, this->profiles, this->settings);

		// Check if anything failed
		if (!(err.isEmpty()))
			QMessageBox::warning(NULL, "DTITool", QString("Writing default 'settings.xml' failed with the following error: \n\n") + err);

		delete settingsWriter;
	}
	else
	{
		// Create an XML reader, and read the "settings.xml" file
		XMLSettingsReader * settingsReader = new XMLSettingsReader;
		QString err = settingsReader->readSettings(settingsFilePath, this->settings);

		// Check if something failed
		if (!(err.isEmpty()) || settingsReader->profiles.isEmpty())
		{
			// The reader returned an error
			if (!(err.isEmpty()))
				QMessageBox::warning(NULL, "DTITool", QString("Reading 'settings.xml' failed with the following error: \n\n") + 
											err + QString("\n\nPlease delete 'settings.xml' and try again."));
			// The "settings.xml" file does not contain any profiles
			else
				QMessageBox::warning(NULL, "DTITool", QString("Reading 'settings.xml' failed; No profiles were read.") + 
											QString("\n\nPlease delete 'settings.xml' and try again."));

			// Delete all existing profiles
			settingsReader->deleteAllProfiles();

			// Simply read all available plugins
			this->pluginManager->readAll(appDir);
		}
		else
		{
			// Get the first profile
			this->defaultProfile = settingsReader->profiles.at(0);

			// Loop through all profiles
			for (int i = 0; i < settingsReader->profiles.size(); ++i)
			{
				DTIToolProfile * currentProfile = settingsReader->profiles.at(i);

				// Copy profile pointers to the local list
				this->profiles.append(currentProfile);

				// Set the default profile
				if (currentProfile->isDefault)			
					this->defaultProfile = currentProfile;
			}

			// Make sure "isDeault" is set to true. If no default profile had been
			// set, "isDefault" will now be true for the first profile in the list,
			// and this will be written to the output the next time the user edits
			// the profiles.

			this->defaultProfile->isDefault = true;

			// Loop through all plugins of the profile (to be loaded now)
			for (int i = 0; i < this->defaultProfile->pluginLoadList.size(); ++i)
			{
				// Get the filename
				QString currentPluginFileName = this->defaultProfile->pluginLoadList.at(i);

				// Add and load the plugin
				int newFileID = this->pluginManager->add(currentPluginFileName);

				if (newFileID != -1)
				{
					this->pluginManager->load(newFileID);
				}
			}

			// Loop through all plugins of the profile (to be loaded later)
			for (int i = 0; i < this->defaultProfile->pluginOpenList.size(); ++i)
			{
				// Get the filename
				QString currentPluginFileName = this->defaultProfile->pluginOpenList.at(i);

				// Add the plugin, but do not load it yet
				int newFileID = this->pluginManager->add(currentPluginFileName);
			}

			// Loop through all files that need to be loaded on startup
			for (int i = 0; i < this->defaultProfile->openFileList.size(); ++i)
			{
				// Get the filename
				QString currentOpenFileName = this->defaultProfile->openFileList.at(i);

				// Tell the data manager to load the file
				this->dataManager->loadDataFromFile(currentOpenFileName);
			}
		}

		// Done, delete the reader
		delete settingsReader;
	}

	// Apply the settings now
	this->applySettings();
}


QDir Core::getDataDirectory()
{
	// Either return the data directory of the default profile...
	if (this->defaultProfile)
		return this->defaultProfile->dataDir;

	// ...or use the application directory
	else
		return QDir(qApp->applicationDirPath());
}


data::Manager * Core::data()
{
	return this->dataManager;
}


plugin::Manager * Core::plugin()
{
	return this->pluginManager;
}


UserOutput * Core::out()
{
	return this->userOutput;
}


void Core::setMainWindow(gui::MainWindowInterface * mwi)
{
	this->mainWindow = mwi;
}


void Core::setRenderWindow(vtkRenderWindow * rw)
{
	this->renderWindow = rw;
}


gui::MainWindowInterface * Core::gui()
{
	return this->mainWindow;
}


void Core::render()
{
	// Do nothing if rendering is not allowed
	if (!allowRendering)
		return;

	Q_ASSERT(this->renderWindow);
	this->renderWindow->Render();
}


vtkMedicalCanvas * Core::canvas()
{
	Q_ASSERT(this->medicalCanvas);
	return this->medicalCanvas;
}

void Core::setMedicalCanvas(vtkMedicalCanvas* canvas)
{
	this->medicalCanvas = canvas;
	Q_ASSERT(this->medicalCanvas);
}


} // namespace bmia
