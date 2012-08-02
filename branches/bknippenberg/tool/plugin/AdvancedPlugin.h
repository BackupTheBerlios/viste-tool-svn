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

	If you think you need this, discuss it with Evert first.

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
