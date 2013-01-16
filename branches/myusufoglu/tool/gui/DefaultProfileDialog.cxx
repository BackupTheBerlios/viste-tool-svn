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
 * DefaultProfileDialog.cxx
 *
 * 2011-03-18	Evert van Aart
 * - First version
 *
 * 2011-07-18	Evert van Aart
 * - Added support for writing the general settings. 
 *
 */


/** Includes */

#include "DefaultProfileDialog.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

DefaultProfileDialog::DefaultProfileDialog(QList<DTIToolProfile *> * rList, DTIToolSettings * rSettings)
{
	// Store the list pointer
	this->profiles = rList;

	// Store the settings
	this->settings = rSettings;

	this->defaultProfileID = 0;

	// Loop through all profiles
	for (int i = 0; i < this->profiles->size(); ++i)
	{
		DTIToolProfile * currentProfile = this->profiles->at(i);

		// Find the default profile
		if (currentProfile->isDefault)
		{
			this->defaultProfileID = i;
		}

		// Turn off "isDefault" for all profiles
		currentProfile->isDefault = false;
	}

	// Get the default profile
	DTIToolProfile * defaultProfile = this->profiles->at(this->defaultProfileID);

	// Turn "isDefault" back on for the default profile. This is kind of a backwards
	// way of doing this, but the idea is that it will also work in the (unlikely)
	// case that none of the profiles has "isDefault" set to true; in this case,
	// the first profile will be made the default.

	defaultProfile->isDefault = true;

	// Create a list widget
	this->profileList = new QListWidget;

	// Add all profile names to the list
	for (int i = 0; i < this->profiles->size(); ++i)
	{
		DTIToolProfile * currentProfile = this->profiles->at(i);

		this->profileList->addItem(currentProfile->profileName);
	}

	// Select the default profile
	this->profileList->setCurrentRow(this->defaultProfileID);

	// Create the OK button and its layout
	this->okButton = new QPushButton("OK");
	this->okButtonHLayout = new QHBoxLayout;
	this->okButtonHLayout->addWidget(this->okButton, 0, Qt::AlignCenter);

	// Create the main layout
	this->mainLayout = new QVBoxLayout;
	this->mainLayout->addWidget(this->profileList);
	this->mainLayout->addLayout(this->okButtonHLayout);
	this->setLayout(this->mainLayout);

	// Connect the OK button to the "save" function
	connect(this->okButton, SIGNAL(clicked()), this, SLOT(save()));
}


//------------------------------[ Destructor ]-----------------------------\\

DefaultProfileDialog::~DefaultProfileDialog()
{
	// Clear the profile list pointer
	this->profiles = NULL;

	// Delete all widgets and layouts
	delete this->mainLayout;
}


//---------------------------------[ save ]--------------------------------\\

void DefaultProfileDialog::save()
{
	// Get the index of the currently selected profile
	this->defaultProfileID = this->profileList->currentRow();

	// Get the corresponding profile object
	DTIToolProfile * defaultProfile = this->profiles->at(this->defaultProfileID);

	// Loop through all profiles, making sure that "isDefault" is true ONLY for
	// the new default profile (thus avoiding multiple default profiles)

	for (int i = 0; i < this->profiles->size(); ++i)
	{
		DTIToolProfile * currentProfile = this->profiles->at(i);

		if (i == this->defaultProfileID)
		{
			currentProfile->isDefault = true;
		}
		else
		{
			currentProfile->isDefault = false;
		}
	}

	// Create a writer, and write the profiles to "settings.xml"
	XMLSettingsWriter * writer = new XMLSettingsWriter;
	QString settingsPath = defaultProfile->appDir.absoluteFilePath("settings.xml");
	QString err = writer->writeSettings(settingsPath, *(this->profiles), this->settings);

	// Check if anything went wrong
	if (!(err.isEmpty()))
	{
		QMessageBox::warning(this, "Set Default Profile", "Writing the settings file failed with the following error: \n\n" + err);
	}

	// Done, delete the writer and close the dialog
	delete writer;
	this->accept();
}


} // namespace bmia
