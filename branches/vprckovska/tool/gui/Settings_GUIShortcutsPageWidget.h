/*
 * Settings_GUIShortcutsPageWidget.h
 *
 * 2011-07-14	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_Settings_GUIShortcutsPageWidget_h
#define bmia_Settings_GUIShortcutsPageWidget_h


/** Includes - Custom Files */

#include "Settings_GenericPageWidget.h"
#include "MainWindow.h"

/** Includes - Qt */

#include <QLabel>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QTableWidgetItem>
#include <QComboBox>


namespace bmia {


/** This class represents a page in the settings dialog, containing settings for
	GUI shortcuts. The user can assign up to ten shortcuts ("Ctrl + 0" to "Ctrl + 9")
	to different GUI plugins (i.e., plugins with a GUI component). After setting the
	shortcut, the user can call up the GUI for the specified plugin using the 
	shortcut. Additionally, the user can specify whether the GUI should be shown
	in the top or bottom GUI field; a third options shows the GUI in the top field,
	and clears the bottom GUI (if any).
*/

class Settings_GUIShortcutsPageWidget : Settings_GenericPageWidget
{
	Q_OBJECT

	public:

		/** Constructor. The "MainWindow" pointer is used to access the combo box
			containing the names of all plugins with a GUI. 
			@param mw		Main window of the DTITool. */

		Settings_GUIShortcutsPageWidget(gui::MainWindow * mw);

		/** Destructor */

		~Settings_GUIShortcutsPageWidget();

		/** Return the name of this page. */

		virtual QString getPageName();

		/** Copy current settings to GUI controls. 
			@param settings	Input DTITool settings. */

		virtual void initializeControls(DTIToolSettings * settings);

		/** Copy GUI control values back to the settings; return true if the settings
			were modified, and false otherwise. 
			@param settings	Output DTITool settings. */

		virtual bool storeSettings(DTIToolSettings * settings);

	protected:

		/** Table containing the shortcut settings. */

		QTableWidget * pTable;

}; // class Settings_GUIShortcutsPageWidget


} // namespace bmia


#endif // bmia_Settings_GUIShortcutsPageWidget_h
