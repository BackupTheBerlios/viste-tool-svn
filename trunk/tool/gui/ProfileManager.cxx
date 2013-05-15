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


#include "ProfileManager.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

ProfileManager::ProfileManager(plugin::Manager * rPluginManager, DTIToolSettings * rSettings)
{
	// Set the window title
	this->setWindowTitle("Profile Manager");

	// Store the pointer to the plugin manager
	this->pluginManager = rPluginManager;

	// Store the DTITool settings pointer
	this->settings = rSettings;

	// No changes made yet
	this->changesMade = false;

	// Combo box containing all profile names
	this->activeProfileLabel = new QLabel("Current Profile");
	this->activeProfileCombo = new QComboBox();
	this->activeProfileCombo->setMinimumWidth(150);
	this->activeProfileHLayout = new QHBoxLayout;
	this->activeProfileHLayout->addWidget(this->activeProfileLabel);
	this->activeProfileHLayout->addWidget(this->activeProfileCombo);

	// Buttons for adding, renaming and deleting profiles
	this->addProfileButton = new QPushButton("Add");
	this->renameProfileButton = new QPushButton("Rename");
	this->deleteProfileButton = new QPushButton("Delete");
	this->profileButtonsHLayout = new QHBoxLayout;
	this->profileButtonsHLayout->addWidget(this->addProfileButton);
	this->profileButtonsHLayout->addWidget(this->renameProfileButton);
	this->profileButtonsHLayout->addWidget(this->deleteProfileButton);

	// Horizontal separator line
	this->hLine1 = new QFrame;
	this->hLine1->setFrameShape(QFrame::HLine);
	this->hLine1->setFrameShadow(QFrame::Sunken);

	// Plugin directory line and browse button
	this->pluginDirLabel = new QLabel("Plugin Directory");
	this->pluginDirLabel->setMinimumWidth(90);
	this->pluginDirLineEdit = new QLineEdit();
	this->pluginDirLineEdit->setReadOnly(true);
	this->pluginDirBrowseButton = new QPushButton("Browse...");
	this->pluginDirHLayout = new QHBoxLayout;
	this->pluginDirHLayout->addWidget(this->pluginDirLabel);
	this->pluginDirHLayout->addWidget(this->pluginDirLineEdit);
	this->pluginDirHLayout->addWidget(this->pluginDirBrowseButton);

	// Data directory line and browse button
	this->dataDirLabel = new QLabel("Data Directory");
	this->dataDirLabel->setMinimumWidth(90);
	this->dataDirLineEdit = new QLineEdit();
	this->dataDirLineEdit->setReadOnly(true);
	this->dataDirBrowseButton = new QPushButton("Browse...");
	this->dataDirHLayout = new QHBoxLayout;
	this->dataDirHLayout->addWidget(this->dataDirLabel);
	this->dataDirHLayout->addWidget(this->dataDirLineEdit);
	this->dataDirHLayout->addWidget(this->dataDirBrowseButton);

	// Horizontal separator line
	this->hLine2 = new QFrame;
	this->hLine2->setFrameShape(QFrame::HLine);
	this->hLine2->setFrameShadow(QFrame::Sunken);

	// Buttons for adding and removing plugins, and fetching the current configuration
	// of plugins from the plugin manager.

	this->pluginSectionLabel = new QLabel("Plugins");
	this->pluginAddButton = new QPushButton("Add");
	this->pluginRemoveButton = new QPushButton("Remove");
	this->pluginUseCurrentButton = new QPushButton("Use Current");
	this->pluginButtonsHLayout = new QHBoxLayout;
	this->pluginButtonsHLayout->addWidget(this->pluginSectionLabel);
	this->pluginButtonsHLayout->addWidget(this->pluginAddButton);
	this->pluginButtonsHLayout->addWidget(this->pluginRemoveButton);
	this->pluginButtonsHLayout->addWidget(this->pluginUseCurrentButton);

	// Table containing two checkboxes per plugin: One determines if a plugin 
	// should be added on start-up, while the second determines whether it
	// should also be loaded (so that it can be used).

	QStringList labels;
	labels.append("  Add  ");
	labels.append("  Load  ");
	labels.append("  File");
	this->pluginTable = new QTableWidget;
	this->pluginTable->setColumnCount(3);
	this->pluginTable->setColumnWidth(0, 50);
	this->pluginTable->setColumnWidth(1, 50);
	this->pluginTable->setColumnWidth(2, this->pluginTable->frameWidth() - 100);
	this->pluginTable->horizontalHeader()->setResizeMode(0, QHeaderView::ResizeToContents);
	this->pluginTable->horizontalHeader()->setResizeMode(1, QHeaderView::ResizeToContents);
	this->pluginTable->horizontalHeader()->setResizeMode(2, QHeaderView::Stretch);
	this->pluginTable->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
	this->pluginTable->setSelectionBehavior(QAbstractItemView::SelectRows);
	this->pluginTable->setShowGrid(false);
	this->pluginTable->verticalHeader()->hide();
	this->pluginTable->setTextElideMode(Qt::ElideMiddle);
	this->pluginTable->setHorizontalHeaderLabels(labels);

	// Horizontal separator line
	this->hLine3 = new QFrame;
	this->hLine3->setFrameShape(QFrame::HLine);
	this->hLine3->setFrameShadow(QFrame::Sunken);

	// Buttons for adding and removing data files, and moving them up in the list
	this->dataSectionLabel = new QLabel("Open Files");
	this->dataAddButton = new QPushButton("Add");
	this->dataRemoveButton = new QPushButton("Remove");
	this->dataMoveUpButton = new QPushButton("Move Up");
	this->dataButtonsHLayout = new QHBoxLayout;
	this->dataButtonsHLayout->addWidget(this->dataSectionLabel);
	this->dataButtonsHLayout->addWidget(this->dataAddButton);
	this->dataButtonsHLayout->addWidget(this->dataRemoveButton);
	this->dataButtonsHLayout->addWidget(this->dataMoveUpButton);

	this->dataList = new QListWidget;
	this->dataList->setMaximumHeight(130);
	this->pluginTable->setMaximumHeight(260);
	// Save and close buttons
	this->saveButton = new QPushButton("Save");
	this->closeButton = new QPushButton("Close");
	this->saveSpacer = new QSpacerItem(100, 10, QSizePolicy::Expanding, QSizePolicy::Minimum);
	this->saveHLayout = new QHBoxLayout;
	this->saveHLayout->addSpacerItem(this->saveSpacer);
	this->saveHLayout->addWidget(this->saveButton);
	this->saveHLayout->addWidget(this->closeButton);

	// Setup the main layout
	this->mainLayout = new QVBoxLayout;
	this->mainLayout->addLayout(this->activeProfileHLayout);
	this->mainLayout->addLayout(this->profileButtonsHLayout);
	this->mainLayout->addWidget(this->hLine1);
	this->mainLayout->addLayout(this->pluginDirHLayout);
	this->mainLayout->addLayout(this->dataDirHLayout);
	this->mainLayout->addWidget(this->hLine2);
	this->mainLayout->addLayout(this->pluginButtonsHLayout);
	this->mainLayout->addWidget(this->pluginTable);
	this->mainLayout->addWidget(this->hLine3);
	this->mainLayout->addLayout(this->dataButtonsHLayout);
	this->mainLayout->addWidget(this->dataList);
	this->mainLayout->addLayout(this->saveHLayout);
	this->setLayout(this->mainLayout);

	// Connect all pushbuttons
	connect(this->saveButton,				SIGNAL(clicked()), this, SLOT(save()));
	connect(this->closeButton,				SIGNAL(clicked()), this, SLOT(close()));
	connect(this->addProfileButton,			SIGNAL(clicked()), this, SLOT(createProfile()));
	connect(this->renameProfileButton,		SIGNAL(clicked()), this, SLOT(renameProfile()));
	connect(this->deleteProfileButton,		SIGNAL(clicked()), this, SLOT(deleteProfile()));
	connect(this->pluginDirBrowseButton,	SIGNAL(clicked()), this, SLOT(setPluginDir()));
	connect(this->dataDirBrowseButton,		SIGNAL(clicked()), this, SLOT(setDataDir()));
	connect(this->pluginUseCurrentButton,	SIGNAL(clicked()), this, SLOT(useCurrentPlugins()));
	connect(this->pluginAddButton,			SIGNAL(clicked()), this, SLOT(addPlugin()));
	connect(this->pluginRemoveButton,		SIGNAL(clicked()), this, SLOT(removeSelectedPlugin()));
	connect(this->dataAddButton,			SIGNAL(clicked()), this, SLOT(addFile()));
	connect(this->dataRemoveButton,			SIGNAL(clicked()), this, SLOT(removeSelectedFile()));
	connect(this->dataMoveUpButton,			SIGNAL(clicked()), this, SLOT(moveSelectedFileUp()));

	// Copy the profile settings to the GUI when a new profile is selected
	connect(this->activeProfileCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(profileToGUI(int)));
}


//------------------------------[ Destructor ]-----------------------------\\

ProfileManager::~ProfileManager()
{
	// Clear pointers
	this->pluginManager		= NULL;
	this->currentProfile	= NULL;
	this->profiles			= NULL;

	// Clear string lists
	this->fullDataFileNames.clear();
	this->fullPluginFileNames.clear();

	// Clear all rows of the plugin table
	for (int i = 0; i < this->pluginTable->rowCount(); ++i)
	{
		this->clearPluginTableRow(i);
	}

	// Delete the main layout. This will also destroy all child widgets and layouts
	delete this->mainLayout;
}


//---------------------------[ addProfileList ]----------------------------\\

void ProfileManager::addProfileList(QList<DTIToolProfile *> * inList)
{
	// Store the list pointer
	this->profiles = inList;

	// Add all profile names to the combo box
	for (int i = 0; i < this->profiles->size(); ++i)
	{
		this->activeProfileCombo->addItem(this->profiles->at(i)->profileName);
	}

	// Can only delete profiles if we have more than one
	this->deleteProfileButton->setEnabled(this->profiles->size() > 1);
}


//-----------------------------[ profileToGUI ]----------------------------\\

void ProfileManager::profileToGUI(int index)
{
	// Check if we need to save the current profile
	this->askToSave();

	// Copy all profile names to the combo box. This ensures that, if a user has
	// renamed a plugin but has chosen not to save this change, the old profile
	// name will be restored to the combo box.

	for (int i = 0; i < this->profiles->size() && i < this->activeProfileCombo->count(); ++i)
	{
		this->activeProfileCombo->setItemText(i, this->profiles->at(i)->profileName);
	}

	if (index < 0 || index >= this->profiles->size())
		return;

	// Get the pointer of the current profile
	this->currentProfile = this->profiles->at(index);

	// Store the profile name
	this->currentProfileName = this->currentProfile->profileName;

	// Clear the path lists
	this->fullPluginFileNames.clear();
	this->fullDataFileNames.clear();

	// Get the directory paths from the profile
	this->fullPluginDir = this->currentProfile->pluginDir.absolutePath();
	this->fullDataDir = this->currentProfile->dataDir.absolutePath();

	// Clear the plugin table
	for (int i = 0; i < this->pluginTable->rowCount(); ++i)
	{
		this->clearPluginTableRow(i);
	}

	this->pluginTable->setRowCount(0);

	// Clear the list of data files
	this->dataList->clear();

	// Set the directory paths (with placeholders)
	this->pluginDirLineEdit->setText(this->currentProfile->addAppPlaceholder(this->fullPluginDir));
	this->dataDirLineEdit->setText(this->currentProfile->addAppPlaceholder(this->fullDataDir));

	// Loop through all plugins of the profile
	for (int i = 0; i < this->currentProfile->pluginLoadList.size(); ++i)
	{
		// Get the plugin file path
		QString fileName = this->currentProfile->pluginLoadList.at(i);

		// Add the path to the list
		this->fullPluginFileNames.append(fileName);

		// Try to add the plugin placeholder to the path; if this fails, also
		// try the application placeholder. This ensures that the "%plugin%"
		// placeholder has priority over the "%app%" placeholder for plugins.

		fileName = this->currentProfile->addPluginPlaceholder(fileName);
		fileName = this->currentProfile->addAppPlaceholder(fileName);

		// Add it to the table, and check both boxes
		this->addPluginToTable(fileName, true);
	}

	// Do the same for the plugins that should be added but not loaded
	for (int i = 0; i < currentProfile->pluginOpenList.size(); ++i)
	{
		QString fileName = this->currentProfile->pluginOpenList.at(i);

		this->fullPluginFileNames.append(fileName);

		fileName = this->currentProfile->addPluginPlaceholder(fileName);
		fileName = this->currentProfile->addAppPlaceholder(fileName);

		// Add it to the table, and check only the first box ("Add")
		this->addPluginToTable(fileName, false);
	}

	// Do something similar for the data files
	for (int i = 0; i < currentProfile->openFileList.size(); ++i)
	{
		this->fullDataFileNames.append(currentProfile->openFileList.at(i));

		QString fileNamePH = currentProfile->addDataPlaceholder(currentProfile->openFileList.at(i));
		fileNamePH = currentProfile->addAppPlaceholder(fileNamePH);

		this->dataList->addItem(fileNamePH);
	}

	// Enable the "Remove" and "Move Up" buttons for data iff there are data files
	this->dataRemoveButton->setEnabled(this->fullDataFileNames.size() > 0);
	this->dataMoveUpButton->setEnabled(this->fullDataFileNames.size() > 0);

	// Enable the "Remove" button for plugins iff there are plugins
	this->pluginRemoveButton->setEnabled(this->fullPluginFileNames.size() > 0);

	// GUI and profiles are now synced up
	this->changesMade = false;
}


//---------------------------[ addPluginToTable ]--------------------------\\

void ProfileManager::addPluginToTable(QString fileName, bool load)
{
	// Create a checkbox, add it (centered) to a horizontal layout, add the layout
	// to a generic widget, and add this widget to the first column. This is a
	// bit of a detour, but as far as I can see, it's the only way to center 
	// the checkbox in the column.

	QCheckBox * newAddCheckbox = new QCheckBox;
	newAddCheckbox->setChecked(true);
	QHBoxLayout * newAddCheckboxHLayout = new QHBoxLayout;
	newAddCheckboxHLayout->addWidget(newAddCheckbox, 0, Qt::AlignCenter);
	newAddCheckboxHLayout->setMargin(0);
	QWidget * newAddCheckboxWidget = new QWidget;
	newAddCheckboxWidget->setLayout(newAddCheckboxHLayout);

	// Do the same for the checkbox in the second column
	QCheckBox * newLoadCheckbox = new QCheckBox;
	newLoadCheckbox->setChecked(load);
	QHBoxLayout * newLoadCheckboxHLayout = new QHBoxLayout;
	newLoadCheckboxHLayout->addWidget(newLoadCheckbox, 0, Qt::AlignCenter);
	newLoadCheckboxHLayout->setMargin(0);
	QWidget * newLoadCheckboxWidget = new QWidget;
	newLoadCheckboxWidget->setLayout(newLoadCheckboxHLayout);

	// Create a table widget item containing the plugin filename
	QTableWidgetItem * newFileNameItem = new QTableWidgetItem(fileName);
	newFileNameItem->setFlags(Qt::ItemIsSelectable | Qt::ItemIsEnabled);

	// Add a row to the table, and select it
	this->pluginTable->setRowCount(this->pluginTable->rowCount() + 1);
	int currentRow = this->pluginTable->rowCount() - 1;

	// Add the contents to the row, and resize the row
	this->pluginTable->setCellWidget(currentRow, 0, newAddCheckboxWidget);
	this->pluginTable->setCellWidget(currentRow, 1, newLoadCheckboxWidget);
	this->pluginTable->setItem(currentRow, 2, newFileNameItem);
	this->pluginTable->resizeRowToContents(currentRow);
}


//--------------------------[ removeSelectedFile ]-------------------------\\

void ProfileManager::removeSelectedFile()
{
	// Check if we've selected a file
	if (this->dataList->currentRow() < 0 || this->dataList->currentRow() > this->dataList->count())
	{
		return;
	}

	// Remove the file from the list and from the GUI
	int currentRow = this->dataList->currentRow();
	this->dataList->takeItem(currentRow);
	this->fullDataFileNames.removeAt(currentRow);

	// Disable the buttons if there are no more files
	if (this->dataList->count() <= 0)
	{
		this->dataRemoveButton->setEnabled(false);
		this->dataMoveUpButton->setEnabled(false);
	}

	// Current profile has been modified
	this->changesMade = true;
}


//--------------------------[ moveSelectedFileUp ]-------------------------\\

void ProfileManager::moveSelectedFileUp()
{
	// Check if we've selected a file
	if (this->dataList->currentRow() <= 0 || this->dataList->currentRow() > this->dataList->count())
	{
		return;
	}

	// Move the file up one spot, both in the list of filenames and in the GUI
	int currentRow = this->dataList->currentRow();
	QListWidgetItem * listItem = this->dataList->takeItem(currentRow);
	this->dataList->insertItem(currentRow - 1, listItem);
	this->dataList->setCurrentRow(currentRow - 1);
	this->fullDataFileNames.move(currentRow, currentRow - 1);

	// Current profile has been modified
	this->changesMade = true;
}


//-------------------------------[ addFile ]-------------------------------\\

void ProfileManager::addFile()
{
	// Use a file dialog to get a filename
	QString newFileName = QFileDialog::getOpenFileName(this, "Open data file...", this->fullDataDir);

	// Check if the filename is valid
	if (newFileName.isNull() || newFileName.isEmpty())
	{
		return;
	}

	// Check if the file has already been added
	if (this->fullDataFileNames.contains(newFileName))
	{
		QMessageBox::warning(this, "Profile manager", "File has already been added!");
		return;
	}

	// Add the full file path to the list
	this->fullDataFileNames.append(newFileName);

	// Add the data- or app-placeholder, if possible
	newFileName = this->currentProfile->addPlaceholder(newFileName, this->fullDataDir, "%data%");
	newFileName = this->currentProfile->addAppPlaceholder(newFileName);

	// Add the shortened path to the GUI
	this->dataList->addItem(newFileName);

	// Enable the buttons
	this->dataRemoveButton->setEnabled(true);
	this->dataMoveUpButton->setEnabled(true);

	// Current profile has been modified
	this->changesMade = true;
}


//------------------------------[ addPlugin ]------------------------------\\

void ProfileManager::addPlugin()
{
	QString newFileName = QFileDialog::getOpenFileName(this, "Open plugin file...", this->fullPluginDir);

	if (newFileName.isNull() || newFileName.isEmpty())
	{
		return;
	}

	if (this->fullPluginFileNames.contains(newFileName))
	{
		QMessageBox::warning(this, "Profile manager", "Plugin has already been added!");
		return;
	}

	this->fullPluginFileNames.append(newFileName);

	newFileName = this->currentProfile->addPlaceholder(newFileName, this->fullPluginDir, "%plugin%");
	newFileName = this->currentProfile->addAppPlaceholder(newFileName);

	// Add the shortened path to the table
	this->addPluginToTable(newFileName, true);
	this->pluginTable->scrollToBottom();

	// Enable the button
	this->pluginRemoveButton->setEnabled(true);

	this->changesMade = true;
}


//-------------------------[ removeSelectedPlugin ]------------------------\\

void ProfileManager::removeSelectedPlugin()
{
	if (this->pluginTable->currentRow() < 0 || this->pluginTable->currentRow() > this->pluginTable->rowCount())
	{
		return;
	}

	int currentRow = this->pluginTable->currentRow();

	// Clear the selected row of the table
	this->clearPluginTableRow(currentRow);

	// Remove the row, and remove the entry from the list of filenames
	this->fullPluginFileNames.removeAt(currentRow);
	this->pluginTable->removeRow(currentRow);

	// Disable the "Remove" button if there are no more plugins
	if (this->pluginTable->rowCount() <= 0)
	{
		this->pluginRemoveButton->setEnabled(false);
	}

	this->changesMade = true;
}


//----------------------------[ setPluginDir ]-----------------------------\\

void ProfileManager::setPluginDir()
{
	// Use a file dialog to get a directory
	QString newDir = QFileDialog::getExistingDirectory(this, "Open plugin directory...", this->currentProfile->appDir.absolutePath());

	// Check if the directory path is valid
	if (newDir.isNull() || newDir.isEmpty())
	{
		return;
	}

	// Store the path
	this->fullPluginDir = newDir;

	// Add the application placeholder, if possible
	newDir = this->currentProfile->addAppPlaceholder(newDir);

	// Copy the shortened path to the line edit
	this->pluginDirLineEdit->setText(newDir);

	// For each table row, re-add the placeholders
	for (int i = 0; i < this->pluginTable->rowCount(); ++i)
	{
		QTableWidgetItem * currentItem = this->pluginTable->item(i, 2);
		QString fullPath = this->fullPluginFileNames.at(i);
		QString placeholderPath = this->currentProfile->addPlaceholder(fullPath, this->fullPluginDir, "%plugin%");
		placeholderPath = this->currentProfile->addAppPlaceholder(placeholderPath);
		currentItem->setText(placeholderPath);
	}

	this->changesMade = true;
}


//------------------------------[ setDataDir ]-----------------------------\\

void ProfileManager::setDataDir()
{
	// Similar to "setPluginDir"

	QString newDir = QFileDialog::getExistingDirectory(this, "Open data directory...", this->currentProfile->appDir.absolutePath());

	if (newDir.isNull() || newDir.isEmpty())
	{
		return;
	}

	this->fullDataDir = newDir;

	newDir = this->currentProfile->addAppPlaceholder(newDir);

	this->dataDirLineEdit->setText(newDir);

	for (int i = 0; i < this->dataList->count(); ++i)
	{
		QString fullPath = this->fullDataFileNames.at(i);
		QString placeholderPath = this->currentProfile->addPlaceholder(fullPath, this->fullDataDir, "%data%");
		placeholderPath = this->currentProfile->addAppPlaceholder(placeholderPath);
		this->dataList->takeItem(i);
		this->dataList->insertItem(i, placeholderPath);
	}

	this->changesMade = true;
}


//-------------------------[ clearPluginTableRow ]-------------------------\\

void ProfileManager::clearPluginTableRow(int row)
{
	if (row < 0 || row >= this->pluginTable->rowCount())
		return;

	// Get the widget of the first column and delete it
	QWidget * cellWidget = this->pluginTable->cellWidget(row, 0);
	delete cellWidget;
	this->pluginTable->setCellWidget(row, 0, NULL);

	// Get the widget of the second column and delete it
	cellWidget = this->pluginTable->cellWidget(row, 1);
	delete cellWidget;
	this->pluginTable->setCellWidget(row, 1, NULL);
}


//--------------------------[ useCurrentPlugins ]--------------------------\\

void ProfileManager::useCurrentPlugins()
{
	// Check if the pluginManager has been set
	if (this->pluginManager == NULL)
		return;

	// Clear the table
	for (int i = 0; i < this->pluginTable->rowCount(); ++i)
	{
		this->clearPluginTableRow(i);
	}

	this->pluginTable->setRowCount(0);

	// Clear the list of plugin filenames
	this->fullPluginFileNames.clear();

	// Loop through all plugins of the plugin manager
	for (int i = 0; i < this->pluginManager->getNumberOfPlugins(); ++i)
	{
		// Get the path of the plugin file
		QString fullPluginPath = this->pluginManager->getFileName(i);

		// Get whether or not the plugin is currently loaded
		bool isLoaded = this->pluginManager->isLoaded(i);

		// Add the plugin path to the list
		this->fullPluginFileNames.append(fullPluginPath);

		// Add the plugin- or application-placeholder
		QString placeholderPath = this->currentProfile->addPlaceholder(fullPluginPath, fullPluginDir, "%plugin%");
		placeholderPath = this->currentProfile->addAppPlaceholder(placeholderPath);

		// Add the new plugin to the table
		this->addPluginToTable(placeholderPath, isLoaded);
	}

	// Enable the "Remove" button if there are plugins in the table
	this->pluginRemoveButton->setEnabled(this->pluginTable->rowCount() > 0);

	this->changesMade = true;
}


//----------------------------[ GUItoProfile ]-----------------------------\\

void ProfileManager::GUItoProfile()
{
	// Store the name, which is kept in the combo box
	this->currentProfile->profileName = this->currentProfileName;

	// Create "QDir" directory handles from the directory paths
	this->currentProfile->pluginDir = QDir(this->fullPluginDir);
	this->currentProfile->dataDir   = QDir(this->fullDataDir);

	// Clear the lists of plugins and data files
	this->currentProfile->pluginLoadList.clear();
	this->currentProfile->pluginOpenList.clear();
	this->currentProfile->openFileList.clear();

	// Loop through all rows of the plugin table
	for (int i = 0; i < this->pluginTable->rowCount(); ++i)
	{
		// Get the checkbox of the first column
		QWidget * cellWidget = this->pluginTable->cellWidget(i, 0);
		QCheckBox * cellCheckBox = (QCheckBox *) cellWidget->children().at(1);

		// Do nothing if this checkbox ("Add") is unchecked
		if (cellCheckBox->isChecked() == false)
			continue;

		// Get the checkbox of the second column
		cellWidget = this->pluginTable->cellWidget(i, 1);
		cellCheckBox = (QCheckBox *) cellWidget->children().at(1);

		// Depending on the state of the second checkbox, add the plugin either
		// to the list of plugins that should be added and loaded on start-up,
		// or to the list of plugins that should only be added, and not loaded.

		if (cellCheckBox->isChecked())
		{
			this->currentProfile->pluginLoadList.append(this->fullPluginFileNames.at(i));
		}
		else
		{
			this->currentProfile->pluginOpenList.append(this->fullPluginFileNames.at(i));
		}
	}

	// Add all data files to the profile list
	for (int i = 0; i < this->fullDataFileNames.size(); ++i)
	{
		this->currentProfile->openFileList.append(this->fullDataFileNames.at(i));
	}
}


//---------------------------------[ save ]--------------------------------\\

void ProfileManager::save()
{
	// copy the GUI settings to the current profile
	this->GUItoProfile();
	
	// Create a writer, and write the "settings.xml" file
	XMLSettingsWriter * writer = new XMLSettingsWriter;
	QString settingsPath = this->currentProfile->appDir.absoluteFilePath("settings.xml");
	QString err = writer->writeSettings(settingsPath, *(this->profiles), this->settings);

	// Check if everything went right
	if (!(err.isEmpty()))
	{
		QMessageBox::warning(this, "Profile Manager", "Writing the settings file failed with the following error: \n\n" + err);
	}

	delete writer;

	// Changes have been saved!
	this->changesMade = false;
}


//--------------------------------[ close ]--------------------------------\\

void ProfileManager::close()
{
	// First ask if the user wants to save, and then close the dialog
	this->askToSave();
	this->accept();
}


//------------------------------[ askToSave ]------------------------------\\

void ProfileManager::askToSave()
{
	// Only ask if the user has modified the current profile
	if (this->changesMade)
	{
		// Ask if we should save the profiles
		QMessageBox::StandardButton answer = QMessageBox::question(this, 
			"Save profiles?", 
			"Save the changes made to the current profile?", 
			QMessageBox::Yes | QMessageBox::No, 
			QMessageBox::Yes);

		// If so, save the "settings.xml" file
		if (answer == QMessageBox::Yes)
		{
			this->save();
		}

		// Regardless of what the user answered, we set "changedMade" to false;
		// if the user chose to save the profiles, the current profile and the
		// GUI settings are the same; if the user chose not to save the profiles,
		// s/he consents to dismissing the changes made, so we can safely overwrite
		// the GUI settings or close the window. This prevents multiple pop-ups
		// of the above question window.

		this->changesMade = false;
	}

}


//----------------------------[ createProfile ]----------------------------\\

void ProfileManager::createProfile()
{
	// First check if we need to save the current profile
	this->askToSave();

	bool ok = false;

	// Ask the user for a new name
	QString newName = QInputDialog::getText(this, "New profile", "Enter name", QLineEdit::Normal, "New profile", &ok);

	// Check if the name's okay and if the user clicked the OK button
	if (newName.isEmpty() || newName.isNull() || ok == false)
	{
		return;
	}

	// Create a new profile with the desired name
	DTIToolProfile * newProfile = new DTIToolProfile;
	newProfile->profileName = newName;
	newProfile->isDefault = false;
	
	// Set all directories to the application directory
	QDir appDir = QDir(qApp->applicationDirPath());
	newProfile->appDir	  = appDir;
	newProfile->dataDir   = appDir;
	newProfile->pluginDir = appDir;

	// Add the profile to the list
	this->profiles->append(newProfile);

	// Add it to the combo box, and select it
	this->activeProfileCombo->addItem(newName);
	this->activeProfileCombo->setCurrentIndex(this->activeProfileCombo->count() - 1);

	// Import the current configurations of plugins from the plugin manager
	this->useCurrentPlugins();

	// Save the "settings.xml" file now
	this->save();

	this->deleteProfileButton->setEnabled(true);
}


//----------------------------[ renameProfile ]----------------------------\\

void ProfileManager::renameProfile()
{
	if (this->currentProfile == NULL)
		return;

	bool ok = false;

	// Get a new name for the current profile
	QString newName = QInputDialog::getText(this, "New profile name", "Enter name", QLineEdit::Normal, this->currentProfile->profileName, &ok);

	if (newName.isEmpty() || newName.isNull() || ok == false)
	{
		return;
	}

	// Copy the new name to the combo box. The profile itself will not be changed
	// until the user decides to save the current profile.

	this->activeProfileCombo->setItemText(this->activeProfileCombo->currentIndex(), newName);
	this->currentProfileName = newName;

	this->changesMade = true;
}


//----------------------------[ deleteProfile ]----------------------------\\

void ProfileManager::deleteProfile()
{
	if (this->currentProfile == NULL)
		return;

	// Make sure that the user is sure
	QMessageBox::StandardButton answer = QMessageBox::question(this, 
		"Delete profile?", 
		"Really delete the current profile?", 
		QMessageBox::Yes | QMessageBox::No, 
		QMessageBox::Yes);

	if (answer == QMessageBox::Yes)
	{
		// Prevent "askToSave" from triggering
		this->changesMade = false;

		// Delete the profile
		delete this->currentProfile;
		this->currentProfile = NULL;

		// Remove it from the list and from the GUI
		this->profiles->removeAt(this->activeProfileCombo->currentIndex());
		this->activeProfileCombo->removeItem(this->activeProfileCombo->currentIndex());

		// Cannot delete the last profile
		if (this->profiles->size() <= 1)
		{
			this->deleteProfileButton->setEnabled(false);
		}

		// Save the "settings.xml" file now
		this->save();
	}
}


} // namespace bmia
