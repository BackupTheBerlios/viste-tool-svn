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
 * Manager.cxx
 *
 * 2009-11-11	Tim Peeters
 * - First version
 *
 * 2010-11-29	Evert van Aart
 * - Added support for loading and writing "profile.ini".
 *
 * 2011-02-03	Evert van Aart
 * - Added support for automatically finding the plugins directory.
 *
 * 2011-10-04   Ralph Brecheisen
 * - Fixed bug in constructor where Q_ASSERT was check member 'core'
 * instead of the parameter 'coreInstance'.
 *
 * 2013-02-07   Mehmet Yusufoglu
 * - Changed the unloadable plugin warning which is called by load(), from a message box warning to terminal log warning.
 */


/** Includes */

#include "plugin/Manager.h"


using namespace std;


namespace bmia {


namespace plugin {


//-----------------------------[ Constructor ]-----------------------------\\

Manager::Manager(Core * coreInstance)
{
    Q_ASSERT(coreInstance);

	// Store the core instance
	this->core = coreInstance;
}


//------------------------------[ Destructor ]-----------------------------\\

Manager::~Manager()
{
	// Unload all plugins
	for (int i = 0; i < this->getNumberOfPlugins(); ++i)
	{
		if (this->isLoaded(i)) 
		{
			this->unload(i);
		}
	}

	// Remove all plugins from the manager
	this->removeAllUnloaded();
}


//---------------------------------[ add ]---------------------------------\\

int Manager::add(QString filename)
{
	// Create a Qt file handle
	QFile file;
	file.setFileName(filename);

	// Check if the input file exists
	if (!file.exists())
	{
		this->core->out()->showMessage("Failed to load plugin because file " + filename + " does not exist!");
		return -1;
	} 

	// Check for duplicate plugins
	if (this->isAdded(filename))
	{
		this->core->out()->showMessage("The plugin you tried to load was already loaded before.");
		return -1;
	}

	// Create a new plugin and add it to the list
	QPluginLoader * loader = new QPluginLoader(filename);
	this->pluginLoaders.append(loader);

	qDebug()<<"Created a plugin loader for file"<<filename;

	// return the index of the newly added plugin loader.
	return (this->pluginLoaders.length()-1);
}


//-------------------------------[ isAdded ]-------------------------------\\

bool Manager::isAdded(QString filename)
{
	// Check for duplicate filenames
	for (int i = 0; i < this->getNumberOfPlugins(); ++i)
	{
		if (filename == this->getFileName(i)) 	
		{
			return true;
		}
	}

	return false;
}


//---------------------------------[ indexOf ]-----------------------------\\

int Manager::indexOf(QString name)
{
	for( int i = 0; i < this->getNumberOfPlugins(); ++i )
	{
		QString pluginName = this->getName( i );
		if( pluginName == name )
			return i;
	}
	return -1;
}


//---------------------------------[ load ]--------------------------------\\

bool Manager::load(int i)
{
	// Check if the index is within range
	Q_ASSERT(i >= 0 && i < this->pluginLoaders.size());

	// Get the loader for the current plugin
	QPluginLoader * loader = this->pluginLoaders.at(i);

	// Load the plugin from file and return an instance
	QObject * loadedQObject = loader->instance();

	// Check if loading was successful 
	if (!loadedQObject)
	{
		this->core->out()->logMessage("Plugin could not be loaded. " + loader->errorString()); 
		return false;
	}
 
	// Cast the output of the loader to a plugin object
	Plugin * loadedPlugin = qobject_cast<Plugin *>(loadedQObject);

	// Check if casting was successful
	if (!loadedPlugin)
	{
		this->core->out()->showMessage("The loaded file " + loader->fileName() + " is not a plugin file. Plugin not loaded.");
		this->unload(i);
		return false;
	}

	// Initialize the plugin
	loadedPlugin->setCoreInstance(this->core);
	this->loadAdvanced(qobject_cast<plugin::AdvancedPlugin*>(loadedPlugin));
	loadedPlugin->init();

	qDebug()<<"Successfully loaded plugin with name"<<loadedPlugin->getName();

	// Handle the different plugin types by casting the plugin and running the appropriate function. 
	// If a plugin is not of a casted type, the cast will return NULL, which causes the function to return.

	this->loadReader(qobject_cast<data::Reader *>(loadedPlugin));
	this->loadConsumer(qobject_cast<data::Consumer *>(loadedPlugin));
	this->loadVisualization(loadedPlugin, qobject_cast<plugin::Visualization *>(loadedPlugin));
	this->loadGUI(loadedPlugin, qobject_cast<plugin::GUI *>(loadedPlugin));

	return true;
}


//--------------------------------[ unload ]-------------------------------\\

void Manager::unload(int i)
{
    qDebug() << "Unloading plugin " << i;
    Q_ASSERT(this->isLoaded(i));

	// Get the plugin pointer
    Plugin * loadedPlugin = this->getPlugin(i);

	// Call the unload function for the different plugin types. If the object cast fails,
	// the pointer will be NULL, and the unloading function will immediately return

    this->unloadReader(qobject_cast<data::Reader *>(loadedPlugin));
    this->unloadConsumer(qobject_cast<data::Consumer *>(loadedPlugin));
    this->unloadVisualization(qobject_cast<plugin::Visualization *>(loadedPlugin));
    this->unloadGUI(loadedPlugin, qobject_cast<plugin::GUI *>(loadedPlugin));
    this->unloadAdvanced(qobject_cast<plugin::AdvancedPlugin *>(loadedPlugin));

	// Unload the plugin
    if (this->pluginLoaders.at(i)->unload()) 
		return;

	// If unloading failed, show an error message
    this->core->out()->showMessage("ERROR: Plugin could not be unloaded!");
}


//------------------------------[ setLoaded ]------------------------------\\

void Manager::setLoaded(int i, bool loaded)
{
	// If true, we need to load the plugin
	if (loaded)
	{
		// Only load the plugin if it is currently unloaded
		if (!(this->isLoaded(i)) )
		{
			this->load(i);
		}
	}

	// If false, we need to unload the plugin
	else
	{
		// Only unload the plugin if it is currently loaded
		if (this->isLoaded(i)) 
		{
			this->unload(i);
		}
	}    
}


//-------------------------------[ isLoaded ]------------------------------\\

bool Manager::isLoaded(int i)
{
	Q_ASSERT(i >= 0 && i < this->pluginLoaders.size());
	return this->pluginLoaders.at(i)->isLoaded();
}


//--------------------------[ getNumberOfPlugins ]-------------------------\\

int Manager::getNumberOfPlugins()
{
	return this->pluginLoaders.size();
}


//--------------------------[ removeAllUnloaded ]--------------------------\\

void Manager::removeAllUnloaded()
{
    QPluginLoader * loader = NULL;

	// Loop through the plugins in reverse order
    for (int i = this->getNumberOfPlugins() - 1; i >=0; --i)
	{
		// Check if the plugin was unloaded
		if (!this->isLoaded(i))
		{
			// If so, delete its loader and remove it from the list
			loader = this->pluginLoaders.at(i);
			delete loader; 
			loader = NULL;
			this->pluginLoaders.removeAt(i);
		}
	}
}


//-----------------------------[ getFileName ]-----------------------------\\

QString Manager::getFileName(int i)
{
	return this->pluginLoaders.at(i)->fileName();
}


//---------------------------[ getShortFileName ]--------------------------\\

QString Manager::getShortFileName(int i)
{
	QString fullName = this->pluginLoaders.at(i)->fileName();

	if (fullName.contains("/"))
	{
		return fullName.right(fullName.length() - fullName.lastIndexOf("/") - 1);
	}
	else if (fullName.contains("\\"))
	{
		return fullName.right(fullName.length() - fullName.lastIndexOf("\\") - 1);
	}

	return fullName;
}


//-------------------------------[ getName ]-------------------------------\\

QString Manager::getName(int i)
{
	return this->getPlugin(i)->getName();
}


//------------------------------[ getPlugin ]------------------------------\\

Plugin * Manager::getPlugin(int i)
{
	Q_ASSERT(this->isLoaded(i));

	// Get the instance from the loader, and cast it to a plugin pointer
	QObject * instance = this->pluginLoaders.at(i)->instance();
	Q_ASSERT(instance);
	Plugin * plugin = qobject_cast<Plugin *>(instance);
	Q_ASSERT(plugin);
	return plugin;
}


//-----------------------------[ getFeatures ]-----------------------------\\

QStringList Manager::getFeatures(int i)
{
	// Get the plugin
	Plugin * plugin = this->getPlugin(i);

	// List of features
	QStringList list;

	// First add the version
	QString pluginVersion = "Version: ";
	pluginVersion += plugin->getPluginVersion();
	list.append(pluginVersion);

	// Add the plugin type(s)
	if (qobject_cast<plugin::AdvancedPlugin *>(plugin)) 
		list.append("Advanced");

	if (qobject_cast<data::Reader *>(plugin)) 
		list.append("Reader");

	if (qobject_cast<data::Consumer *>(plugin)) 
		list.append("Consumer");

	if (qobject_cast<plugin::Visualization *>(plugin))
		list.append("Visualization");

	if (qobject_cast<plugin::GUI *>(plugin)) 
		list.append("GUI");

	return list; 
}


//------------------------------[ loadReader ]-----------------------------\\

void Manager::loadReader(data::Reader * reader)
{
	if (!reader) 
		return;

	qDebug()<<"Loaded plugin has data::Reader functionality.";

	// Add the plugin to the list of readers
	this->core->data()->addReader(reader);
}


//-----------------------------[ unloadReader ]----------------------------\\

void Manager::unloadReader(data::Reader * reader)
{
	if (!reader) 
		return;

	qDebug()<<"Unloading plugin that has data::Reader functionality.";

	// Remove the plugin from the list of readers
	this->core->data()->removeReader(reader);
}


//-----------------------------[ loadConsumer ]----------------------------\\

void Manager::loadConsumer(data::Consumer * consumer)
{
	if (!consumer) 
		return;

	qDebug()<<"Loaded plugin has data::Consumer functionality.";

	// Add the plugin to the list of consumers
	this->core->data()->addConsumer(consumer);
}


//----------------------------[ unloadConsumer ]---------------------------\\

void Manager::unloadConsumer(data::Consumer * consumer)
{
	if (!consumer) 
		return;

	qDebug()<<"Unloading plugin that has data::Consumer functionality.";

	// Remove the plugin from the list of consumers
	this->core->data()->removeConsumer(consumer);
}


//--------------------------[ loadVisualization ]--------------------------\\

void Manager::loadVisualization(plugin::Plugin * plugin, plugin::Visualization * vis)
{
	if (!vis) 
		return;

	qDebug()<<"Loaded plugin has Visualization functionality.";

	// Check if the visualization plugin contains a VTK prop
	if (vis->getVtkProp() == NULL)
		return;

	// Add the VTK prop to the renderer
	this->core->gui()->addPluginVtkProp(vis->getVtkProp(), plugin->getName());
}


//-------------------------[ unloadVisualization ]-------------------------\\

void Manager::unloadVisualization(plugin::Visualization * vis)
{
	if (!vis) 
		return;

	qDebug()<<"Unloading plugin with Visualization functionality.";

	// Check if the visualization plugin contains a VTK prop
	Q_ASSERT(vis->getVtkProp() != NULL);

	// Remove the VTK prop from the renderer
	this->core->gui()->removePluginVtkProp(vis->getVtkProp());
}


//-------------------------------[ loadGUI ]-------------------------------\\

void Manager::loadGUI(plugin::Plugin * plugin, plugin::GUI * gui)
{
	if (!gui) 
		return;

	qDebug()<<"Loaded plugin has GUI functionality.";

	// Add the plugin to the list of GUIs
	this->core->gui()->addPluginGui(gui->getGUI(), plugin->getName());
}


//------------------------------[ unloadGUI ]------------------------------\\

void Manager::unloadGUI(plugin::Plugin * plugin, plugin::GUI * gui)
{
	if (!gui) 
		return;

	qDebug()<<"Unloading plugin with GUI functionality.";

	// Remove the plugin from the list of GUIs
	this->core->gui()->removePluginGui(gui->getGUI(), plugin->getName());
}


//-----------------------------[ loadAdvanced ]----------------------------\\

void Manager::loadAdvanced(plugin::AdvancedPlugin * plugin)
{
    if (!plugin) 
		return;

    qDebug()<<"Loaded an advanced plugin.";

	// Give the plugin full control over the core
    plugin->setFullCoreInstance(this->core);
}


//----------------------------[ unloadAdvanced ]---------------------------\\

void Manager::unloadAdvanced(plugin::AdvancedPlugin * plugin)
{
	if (!plugin) 
		return;

	qDebug()<<"Unloading an advanced plugin.";

	// We do not need to do anything here, since advanced plugins aren't added
	// to any specific lists, unlike the other plugin types
}


//----------------------------[ findPluginDir ]----------------------------\\

QString Manager::findPluginDir(QDir dir, QString targetFileName)
{
	// Get a list of all the files in the current directory
	dir.setNameFilters(QStringList());
	dir.setFilter(QDir::Files);
	QStringList files = dir.entryList();

	// Check if the target file name is in the list
	for (int i = 0; i < files.length(); ++i)
	{
		if (files.at(i).contains(targetFileName))
		{
			return dir.absolutePath();
		}
	}
	
	// If we're here, it means that the file was not found, so we move to the
	// subdirectories of the current directory. First, we get a list of all
	// directories.

	dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
	QStringList dirs = dir.entryList();

	// If this list if empty, we could not find the desired file in this branch
	if (dirs.isEmpty())
		return "";

	// Loop through the directories
	for (int i = 0; i < dirs.length(); ++i)
	{
		// Move down into the subdirectory
		dir.cd(dirs.at(i));

		// Search this new directory
		QString newDir = this->findPluginDir(dir, targetFileName);

		// If we found the file, return its directory
		if (!(newDir.isEmpty()))
			return newDir;

		// Otherwise, move back up to the previous directory
		dir.cdUp();
	}

	// None of the subdirectories contained the file
	return "";
}


//-------------------------------[ readAll ]-------------------------------\\

void Manager::readAll(QDir dir)
{
	// Try to find the planes visualization plugin, which is most likely
	// to be included with a release of DTITool.

	QString pluginDir = this->findPluginDir(dir, "PlanesVisPlugin");

	// If found, update the directory to the directory of this plugin
	if (!(pluginDir.isEmpty()))
	{
		pluginDir = QDir::fromNativeSeparators(pluginDir);
		pluginDir += QDir::separator();
		pluginDir = QDir::toNativeSeparators(pluginDir);
		dir.setPath(pluginDir);
	}

	// Loop through all files in the current directory
	foreach (QString filename, dir.entryList(QDir::Files))
	{
		// (Try to) load the current DLL file
		core->plugin()->load(core->plugin()->add(dir.absoluteFilePath(filename)));
	}

	this->removeAllUnloaded();

	// Create a "profile.ini" file with all plugins
//	this->writeProfile(dir);
}


} // namespace plugin


} // namespace bmia
