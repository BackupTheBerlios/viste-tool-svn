/*
 * MainWindowInterface.h
 *
 * 2010-08-05 Tim Peeters
 * - First version
 *
 */


#ifndef bmia_gui_MainWindowInterface_h
#define bmia_gui_MainWindowInterface_h


/** Includes - Qt */

#include <QWidget>
#include <QString>

/** Includes - VTK */

#include <vtkProp.h>

/** Includes - Custom Files */

#include "core/DTIToolSettings.h"


namespace bmia {


namespace gui {


/** Interface for the main window class, used to shield off vertain functions from
	classes that should not have full access to all public functions. 
*/

class MainWindowInterface
{
	public:
	
		/** Add a plugin GUI to the main window.
			@param widget	GUI widget of the plugin. 
			@param name		Plugin name. */
	
		virtual void addPluginGui(QWidget * widget, QString name) = 0;

		/** Remove a plugin GUI from the main window.
			@param widget	GUI widget of the plugin. 
			@param name		Plugin name. */

		virtual void removePluginGui(QWidget * widget, QString name) = 0;

		/** Add a VTK prop to the VTK renderer. 
			@param prop		VTK prop of a visualization plugin.
			@param name		Plugin name. */

		virtual void addPluginVtkProp(vtkProp * prop, QString name) = 0;

		/** Remove a VTK prop from the VTK renderer. 
			@param prop		VTK prop of a visualization plugin. */

		virtual void removePluginVtkProp(vtkProp * prop) = 0;

		/** Apply settings of the DTITool. This is called by the "applySettings"
			function of the core object, which first applies those settings that
			are directly related to the core. This function then applies the settings
			that are related to the main window (e.g., background color, window size). 
			@param settings	DTITool settings. */
	
		virtual void applySettings(DTIToolSettings * settings) = 0;

}; // class MainWindowInterface


} // namespace gui


} // namespace bmia


#endif // bmai_gui_MainWindowInterface_h
