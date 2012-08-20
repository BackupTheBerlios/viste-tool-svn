/*
 * VisualizationInterface.h
 *
 * 2010-02-19	Wiljan van Ravensteijn
 * - First version
 *
 */


#ifndef bmia_plugin_GUIInterfaces_h
#define bmia_plugin_GUIInterfaces_h


class QWidget;


namespace bmia {


namespace plugin {


/** This class implements plugins with GUI functionality. All plugins with 
	this functionality should implement the function "getGUI", which returns the 
	Qt Widget that defines the Graphical User Interface of a plugin.  
*/

class GUI
{
	public:
		
		/** Destructor. */
    
		virtual ~GUI() 
		{

		};

		/** Returns the Qt widget that will be added to the interface. */

		virtual QWidget * getGUI() = 0;

	protected:

	private:

}; // class GUI


} // namespace plugin


} // namespace bmia


Q_DECLARE_INTERFACE(bmia::plugin::GUI, "bmia.plugin.GUI")


#endif // bmia_plugin_GUIInterfaces_h
