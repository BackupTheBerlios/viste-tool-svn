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
 * AdvancedPlugin.h
 *
 * 2010-05-04	Tim Peeters
 * - First version
 *
 */


#ifndef bmia_plugin_AdvancedPlugin_h
#define bmia_plugin_AdvancedPlugin_h


/** Includes */

#include "Plugin.h"


namespace bmia {


class Core;


namespace plugin {

/** Plugin class for plugins that change something fundamental in the tool
	such as the basic user interface, the interaction with the visualization,
	etc. Very few plugins will need to use the subclass "AdvancedPlugin",
	instead of the default "Plugin"!

	Also, the "AdvancedPlugin" and "Core" interfaces can change over time,
	while the other plugin interfaces will be stable, and plugins that do not
	use the subclass "AdvancedPlugin" will not need to be re-implemented 
	when the core system is updated.
*/

class AdvancedPlugin : public Plugin
{
	Q_OBJECT

	public:
	
		/** Construct a new "AdvancedPlugin" instance. NOTE: Do not access the core here,
			since it will not yet have been initialized. Override "init" if you need to
			access the core.
			@param name		Plugin name. */
	
		AdvancedPlugin(QString name);

		/** Destructor */
	
		virtual ~AdvancedPlugin();

		/** Set the full core instance. Called by the plugin manager when the plugin is loaded.
			@param inst		Core instance. */

		void setFullCoreInstance(Core * inst);

	protected:
	
		/** Returns the full core object. */
	
		Core * fullCore();

	private:
	
		/** The instance of the core. Compared to the one available in the regular "Plugin" class, 
			this full core offers more controls. */

		Core * fullCoreInstance;

}; // class AdvancedPlugin


} // namespace plugin


} // namespace bmia


Q_DECLARE_INTERFACE(bmia::plugin::AdvancedPlugin, "bmia.plugin.AdvancedPlugin")


#endif // bmia_plugin_AdvancedPlugin_h
