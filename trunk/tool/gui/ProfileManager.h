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
 * ProfileManager.cxx
 *
 * 2011-03-18	Evert van Aart
 * - First version
 *
 * 2011-04-06	Evert van Aart
 * - Fixed a bug that prevented correct removal of plugins.
 *
 * 2011-07-18	Evert van Aart
 * - Added support for writing the general DTITool settings.
 *
 */


#ifndef bmia_ProfileManager_h
#define bmia_ProfileManager_h


/** Includes - Qt */

#include <QDialog>
#include <QLabel>
#include <QComboBox>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFrame>
#include <QLineEdit>
#include <QTableWidget>
#include <QHeaderView>
#include <QListWidget>
#include <QSpacerItem>
#include <QList>
#include <QCheckBox>
#include <QStringList>
#include <QFileDialog>
#include <QMessageBox>
#include <QInputDialog>
#include <QApplication>
#include <QSizePolicy>

/** Includes - Custom Files */

#include "core/DTIToolProfile.h"
#include "core/DTIToolSettings.h"
#include "core/XMLSettingsWriter.h"
#include "plugin/Manager.h"


namespace bmia {


/** This is the Profile Manager, which allows the user to edit the profile settings
	from within the DTITool. These settings include the plugins that should be added
	and (optionally) loaded at start-up; the data files that should be loaded; and
	definitions for the plugin directory (default search directory when adding 
	plugins) and the data directory (default search directory when opening data).
	This dialog is, or should be, modal, meaning that the user cannot control the
	main window while this dialog is active. This means that we do not need to 
	take into account the possibility of the profiles or plugins being modified
	while this dialog is visible. The dialog is created and destroyed in the
	"MainWindow" class, and can be launched by the user through the menu bar.

	Note that the Profile Manager does not check if the plugin files, data files,
	and directories specified by the user actually exist; this is done on start-up
	by the "XMLSettingsReader" class.
*/

class ProfileManager : public QDialog
{
	Q_OBJECT

	public:

		/** Constructor 
			@param rPluginManager	Pointer to the plugin manager. Used in the
									"useCurrentPlugins" function. 
			@param rSettings		General DTITool settings. */

		ProfileManager(plugin::Manager * rPluginManager, DTIToolSettings * rSettings);

		/** Destructor */

		~ProfileManager();

		/** Add the list containing all profiles. 
			@param inList			List of profiles. */

		void addProfileList(QList<DTIToolProfile *> * inList);

	protected slots:

		/** Select a new profile, and copy its settings to the GUI. 
			@param index			Index of the selected profile. */

		void profileToGUI(int index);

		/** Save all profiles to the "settings.xml" file. */

		void save();

		/** Close the dialog. Before closing, check if the user has made any
			changes to the current profile, and if so, ask if he wants to
			save these changes. */

		void close();

		/** Create a new profile. This automatically calls the "save" function. */

		void createProfile();

		/** Rename the current profile. */

		void renameProfile();

		/** Delete the current profile. This automatically calls the "save" function. */
		void deleteProfile();

		/** Remove the data file selected in the "dataList" widget from the 
			current profile. */

		void removeSelectedFile();

		/** Move the file currently selected in the "dataList" widget up one 
			position. Used to control the order in which data files are read. */

		void moveSelectedFileUp();

		/** Add a new data file to be opened on start-up. */

		void addFile();

		/** Add a new plugin to be loaded on start-up. */

		void addPlugin();

		/** Remove the currently selected plugin from the "pluginTable" widget. */

		void removeSelectedPlugin();

		/** Import the current configuration of plugins from the plugin manager
			(i.e., get all plugins that are currently added and/or loaded, and 
			use them to populate the "pluginTable" widget. */

		void useCurrentPlugins();

		/** Set a new plugin directory. */

		void setPluginDir();

		/** Set a new data directory. */

		void setDataDir();

	private:

		QLabel * activeProfileLabel;			/**< Label for the active profile combo box. */
		QComboBox * activeProfileCombo;			/**< Combo box determining the active profile. */
		QHBoxLayout * activeProfileHLayout;		/**< Layout for the active profile combo box and its label. */

		QPushButton * addProfileButton;			/**< Button for creating a new profile. */
		QPushButton * renameProfileButton;		/**< Button for renaming an existing profile. */
		QPushButton * deleteProfileButton;		/**< Button for deleting an existing profile. */
		QHBoxLayout * profileButtonsHLayout;	/**< Layout for the profile control buttons. */

		QFrame * hLine1;						/**< Horizontal separator. */

		QLabel * pluginDirLabel;				/**< Label for the plugin directory. */
		QLineEdit * pluginDirLineEdit;			/**< Line edit containing the plugin directory. */
		QPushButton * pluginDirBrowseButton;	/**< Browse button for settings the plugin directory. */
		QHBoxLayout * pluginDirHLayout;			/**< Layout for the plugin directory. */

		QLabel * dataDirLabel;					/**< Label for the data directory. */
		QLineEdit * dataDirLineEdit;			/**< Line edit containing the data directory. */
		QPushButton * dataDirBrowseButton;		/**< Browse button for setting the data directory. */
		QHBoxLayout * dataDirHLayout;			/**< Layout for the data directory. */
		QFrame * hLine2;						/**< Horizontal separator. */

		QLabel * pluginSectionLabel;			/**< Label for the plugins section. */
		QPushButton * pluginAddButton;			/**< Button for adding a plugin. */
		QPushButton * pluginRemoveButton;		/**< Button for removing the selected plugin. */
		QPushButton * pluginUseCurrentButton;	/**< Button for copying the current plugin configuration. */
		QHBoxLayout * pluginButtonsHLayout;		/**< Layout for the plugin controls. */
		QTableWidget * pluginTable;				/**< Table containing all plugins. */

		QFrame * hLine3;						/**< Horizontal separator. */

		QLabel * dataSectionLabel;				/**< Label for the data section. */
		QPushButton * dataAddButton;			/**< Button for adding a data file. */
		QPushButton * dataRemoveButton;			/**< Button for removing a data file. */
		QPushButton * dataMoveUpButton;			/**< Button for moving a file up one spot. */
		QHBoxLayout * dataButtonsHLayout;		/**< Layout for the data file controls. */
		QListWidget * dataList;					/**< List widget containing all data files. */

		QPushButton * saveButton;				/**< Button for saving all profiles. */
		QPushButton * closeButton;				/**< button for closing the dialog. */
		QSpacerItem * saveSpacer;				/**< Spacer for the "Save" and "Close" buttons. */
		QHBoxLayout * saveHLayout;				/**< Layout for the "Save" and "Close" buttons. */

		QVBoxLayout * mainLayout;				/**< Main layout of the dialog. */

		/** List of all profiles. */

		QList<DTIToolProfile *> * profiles;

		/** General DTITool settings. */

		DTIToolSettings * settings;

		/** Pointer to the current profile. */

		DTIToolProfile * currentProfile;

		/** Current (temporary) profile name. */

		QString currentProfileName;

		/** Full path of the plugin directory. */

		QString fullPluginDir;

		/** Full path of the data directory. */

		QString fullDataDir;

		/** List of full paths for the plugins. */

		QStringList fullPluginFileNames;

		/** List of full paths for the data files. */

		QStringList fullDataFileNames;

		/** Plugin manager pointer, used in the "useCurrentPlugins" function. */

		plugin::Manager * pluginManager;

		/** Add a single plugin to the "pluginTable" widget. 
			@param fileName		Full filename of the plugin file. 
			@param load			Whether or not the second checkbox ("Load") should be checked. */

		void addPluginToTable(QString fileName, bool load);

		/** Delete the checkboxes from one row of the "pluginTable" widget. 
			@param row			Row index. */

		void clearPluginTableRow(int row);

		/** Copy the GUI settings to the current profile. Called by the "save"
			function, right before it writes the profiles to the output. */

		void GUItoProfile();

		/** The Profile Manager keeps track of any changes made to the current 
			profile. If changes have been made, and the user tries to either
			close the dialog or switch to another profile, this function is called.
			It creates a message box asking the user if he wants to save the 
			changes. If the user answers yes, the "save" function is called; if
			he answers no, the changes made are dismissed. */

		void askToSave();

		/** Keeps track of whether or not the current profile has been modified. */

		bool changesMade;

}; // class ProfileManager


} // namespace bmia


#endif // bmia_ProfileManager_h
