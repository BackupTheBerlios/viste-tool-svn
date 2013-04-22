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
 * MainWindow.h
 *
 * 2009-12-22	Tim Peeters
 * - First version
 *
 * 2010-11-29	Evert van Aart
 * - Added support for writing profile file.
 *
 * 2011-01-03	Evert van Aart
 * - Added support for multiple filters (one filter per supported file type).
 * - Added a button for writing and reading the default folder.
 * - Added support for loading multiple files.
 *
 * 2011-02-28	Evert van Aart
 * - User can now change the background color and enable or disable the 
 *   background gradient.
 * - Added support for making screenshots.
 *
 * 2011-03-18	Evert van Aart
 * - Added the Profile Manager, which allows the user to create, modify and delete
 *   settings profile, as well as a dialog for choosing the default profile.
 * - Removed functions for the old settings method (i.e., those related to 
 *   "profile.ini" and "folder.ini", since these options are now all part of
 *   the new "settings.xml" system.
 *
 * 2011-03-25	Evert van Aart
 * - Screenshots can now also be PNGs and BMPs.
 *
 * 2011-06-23	Evert van Aart
 * - Window is now maximized by default.
 *
 * 2011-07-18	Evert van Aart
 * - Added the settings dialog, which can be used to change general settings.
 * - Moved the background color options to this settings dialog.
 * - Added comments, implemented destructor.
 *
 * 2012-03-22	Ralph Brecheisen
 * - Added preprocessor directive to check for OS and conditionally include
 *   gui/ui_MainWindow.h (either with double quotes or parentheses).
 *
 *
 *  * 2013-02-12 Mehmet Yusufoglu
 * - Changed showAbout function, which shows the Help->About window. Reads the data from an xml file. AboutInfo.xml.
 *
 */


#ifndef bmia_gui_MainWindow_h
#define bmia_gui_MainWindow_h


/** Includes - Qt */

#include <QSignalMapper>
#include <QFileDialog>
#include <QToolBar>
#include <QLabel>
#include <QDir>
#include <QColor>
#include <QColorDialog>
#include <QFileDialog>
#include <QTextStream>
#include <QMessageBox>
#include <QSignalMapper>
#include <QtDebug>
#include <QFrame>
#include <QXmlStreamReader>

/** Includes - VTK */

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkJPEGWriter.h>
#include <vtkBMPWriter.h>
#include <vtkImageWriter.h>

/** Includes - Custom Files */

#include "MainWindowInterface.h"
#include "DataDialog.h"
#include "PluginDialog.h"
#include "ProfileManager.h"
#include "DefaultProfileDialog.h"
#include "core/Core.h"
#include "core/UserOutput.h"
#include "core/DTIToolSettings.h"
#include "data/Manager.h"
#include "plugin/Manager.h"

/** Includes - GUI */

#if defined(__unix__) || defined(__APPLE__) || defined(__MACH__)
#include <gui/ui_MainWindow.h>
#else
#include <gui/ui_MainWindow.h>
#endif

namespace bmia {


namespace gui {


/** Main window of the DTITool. It consists of the following components:
	- Main menu, containing common actions like opening files.
	- Toolbar containing buttons for toggling the visibility of visualization plugins.
	- Two GUI fields (top and bottom), each with its own plugin chooser.
	- The main canvas area, where all data is visualized.
	This class takes care of the following actions:
	- Managing the VTK props shown in the main area, and their visibility.
	- Displaying the correct plugin GUIs.
	- Launching dialog windows, including the plugin dialog, the data dialog,
	  the profile manager, the default profile dialog, and the settings dialog.
    - Making screenshots of the main window.
	- Applying DTITool settings to the main window.
*/

class MainWindow : public QMainWindow, public Ui::MainWindow, public MainWindowInterface
{
	Q_OBJECT

	public:

		/** Constructor.
			@param coreInstance	DTITool core.
			@param parent		Parent widget. Since this is the main window, this
								pointer is usually NULL. */

		MainWindow(Core * coreInstance, QWidget * parent = NULL);

		/** Destructor. */

		~MainWindow();

		/** Return the meta-canvas that contains all the subcanvasses. */
	
		vtkMedicalCanvas * getMetaCanvas();

		/** Update the loaded plugins in the plugin dialog. This should be called 
			after a plugin was loaded or unloaded to show the user up-to-date information. */
	
		void refreshPluginList();

		/** Render the VTK widget. */
	
		void vtkRender();

		/** Add a plugin GUI to the main window.
			@param widget	GUI widget of the plugin. 
			@param name		Plugin name. */
	
		void addPluginGui(QWidget * widget, QString name);

		/** Remove a plugin GUI from the main window.
			@param widget	GUI widget of the plugin. 
			@param name		Plugin name. */

		void removePluginGui(QWidget * widget, QString name);

		/** Add a VTK prop to the VTK renderer. 
			@param prop		VTK prop of a visualization plugin.
			@param name		Plugin name. */

		void addPluginVtkProp(vtkProp * prop, QString name);

		/** Remove a VTK prop from the VTK renderer. 
			@param prop		VTK prop of a visualization plugin. */
	
		void removePluginVtkProp(vtkProp * prop);

		/** Apply settings of the DTITool. This is called by the "applySettings"
			function of the core object, which first applies those settings that
			are directly related to the core. This function then applies the settings
			that are related to the main window (e.g., background color, window size). 
			@param settings	DTITool settings. */
	
		void applySettings(DTIToolSettings * settings);

		/** Return a pointer to the (top) plugin combo box. Used for example by
			the settings dialog to get the list of all GUI plugins. */
	
		QComboBox * getPluginComboBox()
		{
			return pluginChooserTop;
		}

	protected:
		
		/** Called by the constructor to connect actions to the slots below. */
	
		void connectActions();

		/** Initialize the VTK render window. */
	
		void initializeVtkRenderWindow();

		/** Return the renderer that is used for the main visualization. */
	
		vtkRenderer * getVtkRenderer();

	signals:
	
		/** Signal emitted whenever one of the visibility buttons in the top 
			toolbar is toggled. Used to show or hide visualization plugins. 
			@param pluginName	Name of the plugin whose visibility was toggled. */
		
		void visToggled(const QString &pluginName);

	protected slots:

		/** Open a data file. Launches a file dialog, and subsequently asks the
			data manager to load the selected file. */
		
		void openData();

		/** List all available data files. Launches the data dialog. */

		void listData();

		/** Close the main window. */

		void quit();

		/** Toggles visibility of the specified visualization plugin.
			@param pluginName	Name of the plugin whose visibility was toggled. */

		void showVis(const QString &pluginName);

		/** List all plugins. Launches the plugin dialog. */

		void listPlugins();

		/** Update the two GUI fields based on the current selection of the top 
			and bottom combo boxes. */
	
		void selectPluginGui();

		/** Shows a message box with information about the DTITool. 
		* Reads an xml file which includes version info and people's names to be included into the acknowledgements list.
		* If there is no xml file, ie. AboutInfo.xml, prints the name, no version, default web page and no acknowledgements. */

		void showAbout();

		/** Change the background color and/or turn the gradient on or off. 
			@param newColor			New background color.
			@param applyGradient	If true, a gradient is applied to the background. */

		void changeBackground(QColor newColor, bool applyGradient);

		/** Make a screenshot and write it to a file. */

		void makeScreenshot();

		/** Launch the profile manager, defined by the "ProfileManager" class. This
			dialog allows the user to edit the profiles, which describe which plugins
			and data files should be loaded on start-up. */

		void launchProfileManager();

		/** Launch a dialog which allows the user to choose a default profile, i.e.,
			the profile that will be used at start-up time. */

		void launchDefaultProfileDialog();

		/** Launch the settings dialog, which allows the user to edit the general 
			settings of the DTITool, like window size and background color. First
			fetches the current settings from the DTITool core, which are used to
			initialize the controls of the settings dialog. After closing the dialog,
			the "settings.xml" file will be re-written if the settings were modified. */

		void launchSettingsDialog();

		/** Function called whenever the uses presses one of the active GUI shortcuts,
			which can be defined through the settings dialog. The GUI shortcuts are
			all of the type "Ctrl+X", where X is between 0 and 9. Select the 
			corresponding plugin GUI in the top or bottom GUI field.
			@param key		Key pressed in combination with "Ctrl". */

		void guiShortcutPressed(int key = 0);

	private:

		/** Renderer of the VTK render window. */

		vtkRenderer * renderer;

		/** Plugin dialog, used to display the currently loaded plugins. */

		PluginDialog * pluginDialog;

		/** Data dialog, used to display the data files that are currently available
			in the data manager. */

		DataDialog * dataDialog;

		/** Core object of the DTITool. */

		Core * core;

		/** Canvas shown in the main window, containing all visualization components. */

		vtkMedicalCanvas * metaCanvas;

		/** Toolbar shown at the top of the main window, containing buttons that
			allow the user to toggle the visibility of visualization plugins. */

		QToolBar * pluginToolbar;

		/** List of all plugin widgets (i.e., plugin GUIs). */
	
		QList<QWidget *> pluginWidgets;

		/** List of plugin names for plugins with a GUI. Kept in sync with the 
			"pluginWidgets" list (i.e., lists have the same size, and elements
			at the same position correspond to each other). */

		QStringList pluginNames;

		/** Array of action pointers used for the GUI shortcuts (i.e., the "Ctrl+X"
			shortcuts that allow the user to quickly open a plugin GUI). */

		QAction * guiShortcutActions[10];

		/** Signal mapper for the GUI shortcuts. Each of the active GUI shortcuts
			is assigned to a "QAction" object, the "toggled" signal of which is
			connected to this signal mapper. When the user presses a shortcut,
			the signal mapper then calls "guiShortcutPressed", using the index
			of the key pressed in combination with "Ctrl" as the input argument. */

		QSignalMapper * guiShortcutMapper;

		/** Information about the GUI shortcuts. For each shortcut, we store the
			name of the target plugin (which will be "None" if the shortcut is
			inactive), and the target position of the GUI (top or bottom). When
			the user presses a shortcut, the function "guiShortcutPressed" uses
			this array to figure out which GUI should be displayed, and where. */

		DTIToolSettings::GUIShortcutInfo guiShortcuts[10];

		/** List of actions used to toggle the visibility of visualization plugins. 
			Shown in the top toolbar as a set of buttons. */

		QList<QAction *> toggleVisActions;

		/** List of VTK props shown in the main window. Contains props off all
			plugins with a visualization component. */

		QList<vtkProp *> vtkProps;

		/** Names of the visualization plugins that supplied VTK props used for
			visualization. Same size and order as "vtkProps". */

		QStringList vtkPropsNames;

		/** Signal mapper for the visibility actions. Maps the actions - which are
			displayed as buttons in the top toolbar - to the name of the corresponding
			plugin, which is then used to find the corresponding VTK prop. */

		QSignalMapper * visibilitySignalMapper;

}; // class MainWindow


} // namespace gui


} // namespace bmia


#endif // bmia_gui_MainWindow_h
