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
