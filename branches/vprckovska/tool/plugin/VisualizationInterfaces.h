/*
 * VisualizationInterface.h
 *
 * 2010-02-11	Tim Peeters
 * - First version.
 *
 */


#ifndef bmia_plugin_VisualizationInterfaces_h
#define bmia_plugin_VisualizationInterfaces_h


class vtkProp;


namespace bmia {


namespace plugin {


/** This class implements plugins with visualization functionality. Plugins with
	this functionality should implement the "getVtkProp" function, which returns
	a VTK prop object. This function is called once when the plugin is created. 
	The returned prop is then added to the list of props of the renderer, which 
	will display it on the screen. The visualization should also create and configure
	the mapper that will be used to draw the prop. If you need to draw more than one
	VTK prop with one visualization plugin, you can add them to a "vtkPropAssembly"
	object, and return this assembly in the "getVtkProp" function.
*/

class Visualization
{
	public:
    
		/** Destructor */

		virtual ~Visualization() 
		{

		};

		/** Return a VTK Prop that will be added to the tool's main renderer when 
			this visualization plugin is loaded. The function must always return 
			the same pointer, which may not be NULL. */

		virtual vtkProp * getVtkProp() = 0;

	protected:

	private:

}; // class Visualization


} // namespace plugin


} // namespace bmia


Q_DECLARE_INTERFACE(bmia::plugin::Visualization, "bmia.plugin.Visualization")


#endif // bmia_plugin_VisualizationInterfaces_h
