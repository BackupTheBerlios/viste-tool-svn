/*
 * Settings_GenericPageWidget.h
 *
 * 2011-07-13	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_Settings_GenericPageWidget_h
#define bmia_Settings_GenericPageWidget_h


/** Includes - Qt */

#include <QWidget>
#include <QLayout>

/** Includes - Custom Files */

#include "core/DTIToolSettings.h"


namespace bmia {


/** This is a generic Qt widget used for setting pages. All setting pages should
	inherit from this class, and should implement the pure virtual functions
	("getPageName", "initializeControls", and "storeSettings"). A 'settings page'
	is defined as a Qt widget with a number of controls, which can copy the 
	general settings of the DTITool to these controls ("initializeControls"),
	and copy the modified control values back to the settings ("storeSettings"). 
	A boolean value named "settingsModified" is used to keep track of whether
	or not the settings have been modified. 
*/

class Settings_GenericPageWidget : public QWidget
{
	Q_OBJECT

	public:

		/** Constructor. Only initializes "mainLayout" to NULL and "settingsModified"
			to false. */

		Settings_GenericPageWidget();

		/** Destructor. Destroys "mainLayout", which will also destroy all child
			widgets, layouts, and spacers (i.e., you do not have to destroy these
			in your subclass). If you create objects in your subclass that are not
			added to the main layout (although in general, you should not need to),
			you need to destroy these in the destructor of the subclass. */

		~Settings_GenericPageWidget();

		/** Return the name of this page. Should be a hard-coded string. Must be 
			implemented by all subclasses. */

		virtual QString getPageName() = 0;

		/** Copy the current DTITool settings to the GUI controls. Must be implemented
			by all subclasses. Additionally, subclasses should set "settingsModified"
			to false in this function. 
			@param settings		Current DTITool settings. */

		virtual void initializeControls(DTIToolSettings * settings) = 0;

		/** Store the current GUI control values in the settings struct. Must be
			implemented by all subclasses. Return false if no settings were modified,
			true if they were. Implementation should have the following flow: 
			1) If "settingsModified" is false, return false; 2) Otherwise, copy
			GUI values to settings, 3) Set "settingsModified" to false, and 4)
			return true. 
			@param settings		Output DTITool settings. */

		virtual bool storeSettings(DTIToolSettings * settings) = 0;

	protected:

		/** Main layout of this widget. All GUI controls, layouts, and spacers should
			be a child (or grandchild, or even lower) of this layout; that way,
			when the main layout is destroyed, all GUI elements of the widget
			are destroyed with it. */

		QLayout * mainLayout;

		/** Boolean used to keep track of whether or not the settings were modified. */

		bool settingsModified;

	protected slots:

		/** Signal that the settings were modified. In the simplest implementation,
			all GUI controls should be connected to this function (e.g., a spin box
			should connect its "valueChanged" signal to this function, a push button
			its "clicked" signal, and so on). This means that the boolean can only
			go from false to true (i.e., if you can change a setting and then undo
			the change, "settingsModified" will still be true), but since writing
			the settings file is very fast, this does not matter much. */

		void setSettingsModified()
		{
			settingsModified = true;
		}
};


} // namespace bmia


#endif // bmia_Settings_GenericPageWidget_h
