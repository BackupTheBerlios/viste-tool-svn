/*
 * Manager.h
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
 */


#ifndef bmia_plugin_Manager_h
#define bmia_plugin_Manager_h


/** Includes - Custom Files */

#include "plugin/PluginInterfaces.h"
#include "core/Core.h"
#include "core/UserOutput.h"
#include "data/Manager.h"
#include "gui/MainWindowInterface.h"

/** Includes - Qt */

#include <QPluginLoader>
#include <QFile>
#include <QtDebug>
#include <QString>
#include <QList>
#include <QDir>

/** Includes - VTK */

#include <vtkRenderer.h>


namespace bmia {


namespace plugin {


/** This class manages the loading and unloading of plugins. Loading a plugin is a three-steps process.
	First, a loader is created for the specified filename. Next, the loader opens the plugin file. Finally,
	the plugin is added to one or more lists, depending on its features. For example, Consumer plugins are
	added to the list of consumer plugins in the data manager, while Reader plugins are added to the list
	of available readers. When unloading a plugin, it is removed from these lists, but its loader is 
	kept active, for quick reloading.

	This manager also supports the reading and writing of profile files, which are plain-text files containing
	a list of all plugins that should be loaded on start-up. The user can edit this list (either through the
	tool or manually) to customize the tool and avoid loading plugins that he doesn't need.
 */

class Manager {

	public:
    
		/** Create a new plugin manager instance.
			@param coreInstance		Instance of the DTITool core. */
    
		Manager(Core * coreInstance);

		/** Destroy the plugin manager instance. */
    
		~Manager();

		/** Adds the plugin that can be loaded from the given filename to the list of 
			plugins that can be loaded.
			@param filename		The name of the file that contains the plugin.
			@return				The ID of the plugin that can be loaded, or -1 if the file is not found. */

		int add(QString filename);

		/** Load the specified plugin. If it was already loaded, then it will be unloaded first 
			and then loaded again.
			@return				True if the plugin was loaded successfully, false otherwise. */
    
		bool load(int i);

		/** Returns the number of plugins that have been added (loaded or unloaded), but not removed. */
    
		int getNumberOfPlugins();

		/** Unload the specified plugin.
			@param i			Index of the plugin that will be unloaded. */
    
		void unload(int i);

		/** Return true if the plugin with the given index is loaded, and false otherwise. */
    
		bool isLoaded(int i);

		/** Depending on the value of "loaded", loads or unloads the given plugin.
			@param i			Index of the target plugin.
			@param loaded		Specifies whether the plugin should be loaded or unloaded. */
    
		void setLoaded(int i, bool loaded);

		/** Removes all plugins that are not loaded from the list of plugins. */
     
		void removeAllUnloaded();

		/** Return the filename of the specified plugin.
			@param i			Index of the target plugin. */

		QString getFileName(int i);

		/** Return the filename of the specified plugin, without the directory.
			@param i			Index of the target plugin. */

		QString getShortFileName(int i);

		/** Return the name of the specified plugin. 
			@param i			Index of the target plugin. */
    
		QString getName(int i);

		/** Returns a list of strings describing the interfaces that the specified
			plugin implements, as well as the current version of the plugin. 
			@param i			Index of the target plugin. */

		QStringList getFeatures(int i);

		/** Try to read all plugin files in the directory specified by "dir". This function should
			be executed if "readProfile" fails. This is not preferable, since 1) it will also try
			to load files that are not plugins (like Qt or VTK DLLs), and 2) most users do not need
			all plugins, and loading them all takes up more time and resources. At the end, this
			function will write a default "profile.ini" file containing all loaded plugins. 
			@param dir	Directory containing plugin files. */

		void readAll(QDir dir);

	protected:

		/** Return a pointer to the specified plugin. 
		@param i			Index of the target plugin. */

		Plugin * getPlugin(int i);

		/** Returns true if a plugin file with the given filename was added before, and false otherwise. 
			This can be used to make sure that a plug-in is not loaded multiple times.
			@param filename	Plugin file name. */

		bool isAdded(QString filename);

	private:
    
		/** List of all plugin loaders. */

		QList<QPluginLoader*> pluginLoaders;

		/** Loading functions for the different plugin types. These functions are all called in
			the "load" function. If the generic plugin pointer cannot be cast to the input argument
			of these functions (which happens if the plugin does not have the required functionality),
			these functions immediately return; otherwise, the pointer is added to the corresponding
			list of plugins. For example, if we're adding a Reader plugin (with no additional
			functionality), the only function that will not immediately return is "loadReader", 
			which will instead add the pointer to a list of available readers. */

		void loadReader(data::Reader * reader);
		void loadConsumer(data::Consumer * consumer);
		void loadVisualization(plugin::Plugin * plugin, plugin::Visualization * vis);
		void loadGUI(plugin::Plugin * plugin, plugin::GUI * gui);
		void loadAdvanced(plugin::AdvancedPlugin * plugin);


		/** Unloading functions for the different plugin types. These functions are all called in
			the "unload" function. If the generic plugin pointer cannot be cast to the input argument
			of these functions (which happens if the plugin does not have the required functionality),
			these functions immediately return; otherwise, the pointer is removed from the corresponding
			list of plugins. */

		void unloadReader(data::Reader * reader);
		void unloadConsumer(data::Consumer * consumer);
		void unloadVisualization(plugin::Visualization * vis);
		void unloadGUI(plugin::Plugin * plugin, plugin::GUI * gui);
		void unloadAdvanced(plugin::AdvancedPlugin * plugin);

		/** Pointer to the core object. */
    
		Core * core;

		/** Try to find the directory containing the file called "targetFileName". This filename will
			usually be the first file in the "profile.ini" file. The "dir" input contains the starting
			directory, which is usually the one containing the DTITool executable. Recursively searches
			all subdirectories until it finds the file, in which case it returns the directory containing
			the file (without final slash, and without filename). If the file was not found, the function
			returns an empty string.
			@param dir				Initial directory.
			@param targetFileName	Name of a plugin file (incl. extension). */

		QString findPluginDir(QDir dir, QString targetFileName);

}; // class Manager


} // namespace plugin


} // namespace bmia


#endif // bmia_plugin_Manager_h
