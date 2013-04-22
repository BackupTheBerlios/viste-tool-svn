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
 * DefaultProfileDialog.h
 *
 * 2011-03-18	Evert van Aart
 * - First version
 *
 * 2011-07-18	Evert van Aart
 * - Added support for writing the general settings. 
 *
 */


#ifndef bmia_DefaultProfileDialog_h
#define bmia_DefaultProfileDialog_h


/** Includes - Qt */

#include <QDialog>
#include <QList>
#include <QListWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QString>
#include <QMessageBox>

/** Includes - Custom Files */

#include "core/DTIToolProfile.h"
#include "core/DTIToolSettings.h"
#include "core/XMLSettingsWriter.h"


namespace bmia {


/** Simple dialog for selecting the default profile (i.e., the settings that will
	be loaded on start-up). Consists of a list of all plugins, and an OK button.
	The dialog will (or should) be used in modal mode (i.e., as long as it is
	active, the user cannot make changes in the main window), so we do not have to 
	worry about the list of profiles being modified while the dialog is shown.
	Called by the "MainWindow" class.
*/

class DefaultProfileDialog : public QDialog
{
	Q_OBJECT

	public:

		/** Constructor.
			@param rList		List of profiles. 
			@param rSettings	General DTITool settings. */

		DefaultProfileDialog(QList<DTIToolProfile *> * rList, DTIToolSettings * rSettings);

		/** Destructor. */

		~DefaultProfileDialog();

	protected slots:

		/** Write the list of profiles, with new default profile settings, to 
			the "settings.xml" file, and subsequently close the dialog. */

		void save();

	private:

		/** Index of the default profile in the "profiles" list. */

		int defaultProfileID;

		/** List of all current profiles. */

		QList<DTIToolProfile *> * profiles;

		/** General DTITool settings. */

		DTIToolSettings * settings;
		
		/** Widget containing all profile names. */

		QListWidget * profileList;

		/** OK button. */

		QPushButton * okButton;

		/** Horizontal layout of the OK button (for centering). */

		QHBoxLayout * okButtonHLayout;

		/** Main layout of the dialog. */

		QVBoxLayout * mainLayout;


}; // class DefaultProfileDialog


} // namespace


#endif // bmia_DefaultProfileDialog_h
