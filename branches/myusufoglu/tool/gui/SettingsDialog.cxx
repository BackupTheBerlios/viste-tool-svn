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
 * SettingsDialog.cxx
 *
 * 2011-07-18	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "SettingsDialog.h"


namespace bmia {


// Increment this value when you add a settings page.

int SettingsDialog::numberOfPages = 2;


//-----------------------------[ Constructor ]-----------------------------\\

SettingsDialog::SettingsDialog(DTIToolSettings * rSettings, gui::MainWindow * mw)
{
	// Set the window title
	this->setWindowTitle("Settings");

	// Store the settings pointer
	this->settings = rSettings;

	// Settings have not been modified yet
	this->settingsModified = false;

	// Create the list containing all page names
	QListWidget * pageList = new QListWidget;
		pageList->setAcceptDrops(false);
		pageList->setAlternatingRowColors(false);
		pageList->setResizeMode(QListView::Fixed);
		pageList->setSelectionMode(QAbstractItemView::SingleSelection);
		pageList->setVerticalScrollMode(QAbstractItemView::ScrollPerItem);
		pageList->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
		pageList->setFixedWidth(200);

	// Create an array for the page widget pointers
	this->pages = new Settings_GenericPageWidget*[this->numberOfPages];

	// Create the pages. If you want to create a new page, simply increment
	// "numberOfPages" (at the top of this file), and add the page here.

	this->pages[0] = (Settings_GenericPageWidget *) new Settings_GeneralPageWidget;
	this->pages[1] = (Settings_GenericPageWidget *) new Settings_GUIShortcutsPageWidget(mw);

	// Create the OK- and Cancel-buttons, and add them to a layout
	QPushButton * okButton = new QPushButton("OK");
	QPushButton * cancelButton = new QPushButton("Cancel");
	QSpacerItem * buttonSpacer = new QSpacerItem(10, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
	QHBoxLayout * buttonHLayout = new QHBoxLayout;
		buttonHLayout->addSpacerItem(buttonSpacer);
		buttonHLayout->addWidget(okButton);
		buttonHLayout->addWidget(cancelButton);

	// Create a new vertical layout
	this->pageVLayout = new QVBoxLayout;

	// Add all pages to the layout
	for (int i = 0; i < this->numberOfPages; ++i)
	{
		this->pageVLayout->addWidget(this->pages[i]);

		// Hide all pages, except for the first one
		if (i != 0)
			this->pages[i]->hide();

		// Initialize the controls of each page based on the current settings
		this->pages[i]->initializeControls(this->settings);

		// Add the name of the page to the list
		pageList->addItem(this->pages[i]->getPageName());
	}

	// Add the buttons to the vertical layout
	this->pageVLayout->addLayout(buttonHLayout);

	// We've selected the first page
	pageList->setCurrentRow(0);
	this->currentPage = 0;

	// Create the main layout, containing the page list, the page widgets, and the buttons
	QHBoxLayout * mainLayout = new QHBoxLayout;
		mainLayout->addWidget(pageList);
		mainLayout->addLayout(pageVLayout, 1);

	// Set the main layout of the dialog
	this->setLayout(mainLayout);

	// Connect the signals
	connect(cancelButton,	SIGNAL(clicked()),					this, SLOT(reject())		);
	connect(okButton,		SIGNAL(clicked()),					this, SLOT(updateSettings()));
	connect(pageList,		SIGNAL(currentRowChanged(int)),		this, SLOT(changePage(int))	);
}


//------------------------------[ Destructor ]-----------------------------\\

SettingsDialog::~SettingsDialog()
{
	// Delete the main layout of the dialog. This will also delete all widgets,
	// layouts and spacers that belong to this layout, which includes the 
	// page widgets and all their controls.

	delete this->layout();

	// Delete the array that was used to store the page widget pointers
	delete[] this->pages;
}


//----------------------------[ updateSettings ]---------------------------\\

void SettingsDialog::updateSettings()
{
	this->settingsModified = false;

	// Ask all pages to modify the settings. If a page actually modifies the 
	// settings (because one of its controls has changed), it returns true;
	// if no changes were made, it does not change the settings, and returns false.

	for (int i = 0; i < this->numberOfPages; ++i)
	{
		this->settingsModified |= this->pages[i]->storeSettings(this->settings);
	}

	// Close the dialog
	this->accept();
}


//------------------------------[ changePage ]-----------------------------\\

void SettingsDialog::changePage(int page)
{
	// Hide the previous page
	this->pages[this->currentPage]->hide();

	// Show the new page
	this->pages[page]->show();

	// Store the index of the new page
	this->currentPage = page;
}


} // namespace bmia
