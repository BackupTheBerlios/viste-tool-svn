/*
 * Plugin.h
 *
 * 2009-11-10	Tim Peeters
 * - First version.
 *
 * 2011-01-28	Evert van Aart
 * - Added additional comments.
 *
 */


#ifndef bmia_plugin_Plugin_h
#define bmia_plugin_Plugin_h


/** Includes - Custom Files */

#include "core/SimpleCoreInterface.h"

/** Includes - Qt */

#include <QObject>
#include <QtPlugin>
#include <QString>


namespace bmia {


namespace plugin {

/** This is the base class for the DTITool plugins. All custom plugins should inherit
	from this class using "public plugin::Plugin". The interfaces of a plugin are defined
	by its other optional inheritances:

	- GUI plugins ("public plugin::GUI") should implement a "QWidget * getGUI()" function
	  which returns the Qt widget defining the Graphical User Interface.
    - Reader plugins ("public data::Reader") should implement three functions: 
	  "QStringList getSupportedFileExtensions()", which returns a list of extensions that
	  can be read by this plugin; "QStringList getSupportedFileDescriptions()", which returns
	  a short description for each supported extension; and "void loadDataFromFile(QString 
	  filename)", which performs the actual reading.
    - Consumer plugins ("public data::Consumer") must implement the functions "void 
	  dataSetAdded(data::DataSet * ds)", "void dataSetChanged(data::DataSet * ds)", and 
	  "void dataSetRemoved(data::DataSet * ds)", which allow this plugin to be notified of 
	  the addition of new data sets, and changes to or deletion of existing data sets.
    - Visualization interfaces ("public plugin::Visualization") must implement the function
	  "vtkProp * getVtkProp()", which returns the VTK prop that should be rendered.
    - Advanced plugins ("public plugin::AdvancedPlugin") give the user more control over the
	  core functionality of the tool.
 */


class Plugin : public QObject
{
	Q_OBJECT

	public:

		/** Construct a new plugin with the given name. NOTE: Do NOT access core in the
			constructor, because it usually has not been set-up yet when the plugin is
			created. Use the "init()" function if you need access to the core.
			@param name		Name of the plugin. */
		
		Plugin(QString name);

		/** This function is called by the plugin manager after loading the plugin and 
			setting up the core object that can be used in the plugin. Override in 
			subclasses if needed. */

		virtual void init() 
		{

		};

		/** Destroy this plugin. */
		
		virtual ~Plugin();

		/** Return the name that was given to this plugin when it was constructed. */

		QString getName();

		/** Return the version of this plugin, which can be shown in the "PluginDialog" 
			window. If not implemented in the plugin, the default "Unversioned" will be returned. */

		virtual QString getPluginVersion()
		{
			return "Unversioned";
		}

		/** Set the core instance to use in this plugin. This function is called immediately 
			after constructing a plugin instance and before calling the "init()" function.
			@param inst		Interface for the core. */

		void setCoreInstance(SimpleCoreInterface * inst);

	protected:

		/** Returns the core object of this plugin. Can be called in subclasses to access the core. */
		
		SimpleCoreInterface * core();

		/** The name of this plugin. */

		QString name;

	private:
		
		/** The instance of the core. */

		SimpleCoreInterface * coreInstance;

}; // class Plugin


} // namespace plugin


} // namespace bmia


Q_DECLARE_INTERFACE(bmia::plugin::Plugin, "bmia.plugin.Plugin")


#endif // bmia_plugin_Plugin_h
