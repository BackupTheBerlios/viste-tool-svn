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
 * SettingsDialog.h
 *
 * 2011-07-18	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_SettingsDialog_h
#define bmia_SettingsDialog_h


/** Includes - Qt */

#include <QDialog>
#include <QListWidget>
#include <QSpacerItem>
#include <QLabel>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QWidget>
#include <QPushButton>

/** Includes - Custom Files */

#include "Settings_GenericPageWidget.h"
#include "Settings_GeneralPageWidget.h"
#include "Settings_GUIShortcutsPageWidget.h"
#include "core/DTIToolSettings.h"
#include "core/DTIToolProfile.h"
#include "core/XMLSettingsWriter.h"
#include "MainWindow.h"


namespace bmia {


/** The settings dialog is a dialog window containing the general settings for the
	DTITool. This basic dialog consists of a list of all pages, OK- and Cancel-
	buttons, and an area for the pages itself. These pages are represented by 
	Qt widgets, inherited from "Settings_GenericPageWidget". See the general page
	widget ("Settings_GeneralPageWidgets") for an example page. Each page initializes
	its controls based on the current settings, and can write its modified control
	values back to the settings. This dialog should be used as a modal dialog
	(using the "exec" function), which means that it blocks input to the main
	window while it is active.
*/

class SettingsDialog : public QDialog
{
	Q_OBJECT

	public:

		/** Number of settings pages. */

		static int numberOfPages;

		/** Constructor.
			@param rSettings	Current DTITool settings. 
			@param mw			Main window of the DTITool. */

		SettingsDialog(DTIToolSettings * rSettings, gui::MainWindow * mw);

		/** Destructor. */

		~SettingsDialog();

		/** If true, the settings of one or more pages have been modified. The 
			main window checks this value to see if it should re-apply the 
			settings, and if the "settings.xml" file should be re-written. */

		bool settingsModified;

	protected:

		/** List of all page widgets. */

		Settings_GenericPageWidget ** pages;

		/** Current DTITool settings. */

		DTIToolSettings * settings;

		/** Vertical layout containing the pages and the OK- and Cancel-buttons.
			All page widgets are added to this layout, but only the active page
			is shown, while all other pages are hidden. */

		QVBoxLayout * pageVLayout;

		/** Index of the active page. */

		int currentPage;

	protected slots:

		/** Tell each page to update the DTITool settings based on the current
			control values. Each page reports if it made any changes to the
			settings; if at least one page has modified the settings, the value
			of "settingsModified" is set to true in this function. Called when
			the user closes the dialog with the OK-button. */

		void updateSettings();

		/** Change the active page.
			@param page	Index of the new active page. */

		void changePage(int page);

}; // class SettingsDialog


} // namespace bmia


#endif // bmia_SettingsDialog_h
