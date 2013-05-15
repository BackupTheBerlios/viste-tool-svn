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
 * Settings_GUIShortcutsPageWidget.cxx
 *
 * 2011-07-14	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "Settings_GUIShortcutsPageWidget.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

Settings_GUIShortcutsPageWidget::Settings_GUIShortcutsPageWidget(gui::MainWindow * mw)
{
	// Get the combo box containing all GUI plugins (and starting with "None")
	QComboBox * mwPluginCombo = mw->getPluginComboBox();

	// Create the text shown at the top of the page
	QString topTextString = QString("The following keyboard shortcuts can be used ");
		topTextString += "to quickly access the GUI of the specified plugins. The GUI will ";
		topTextString += "be shown in the top or bottom GUI field. 'Top (Exclusive)' means ";
		topTextString += "that the GUI will be shown in the top GUI field, and that the bottom ";
		topTextString += "GUI field will be cleared. Note that you can only define shortcuts ";
		topTextString += "for GUI plugins that are currently loaded.";

	QLabel * topText = new QLabel(topTextString);
		topText->setWordWrap(true);

	// Column labels for the table
	QStringList labels;
		labels.append(" Shortcut ");
		labels.append(" Plugin");
		labels.append(" Location");

	// Create and setup a table for the shortcuts
	QTableWidget * table = new QTableWidget;
		table->setRowCount(10);
		table->setColumnCount(3);
		table->horizontalHeader()->setResizeMode(0, QHeaderView::ResizeToContents);
		table->horizontalHeader()->setResizeMode(1, QHeaderView::Stretch);
		table->horizontalHeader()->setResizeMode(2, QHeaderView::ResizeToContents);
		table->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft);
		table->setSelectionBehavior(QAbstractItemView::SelectItems);
		table->setShowGrid(false);
		table->verticalHeader()->hide();
		table->setTextElideMode(Qt::ElideMiddle);
		table->setHorizontalHeaderLabels(labels);

	// Loop through all ten rows of the the table
	for (int i = 0; i < 10; ++i)
	{
		// Generate the shortcut string
		QString shortcutString = QString(" Ctrl + ") + QString::number(i) + " ";

		// Add the string to the table as an item
		QTableWidgetItem * shortcutStringItem = new QTableWidgetItem(shortcutString);
		table->setItem(i, 0, shortcutStringItem);

		// Create a new combo box
		QComboBox * pluginCombo = new QComboBox;

		// Add all loaded GUI plugins to the new combo box
		for (int j = 0; j < mwPluginCombo->count(); ++j)
		{
			pluginCombo->addItem(mwPluginCombo->itemText(j));
		}

		// Add the combo box to the table
		table->setCellWidget(i, 1, pluginCombo);

		// Settings have been changed if the user changes one of these combo boxes
		connect(pluginCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setSettingsModified()));

		// Create a combo box for the GUI position
		QComboBox * positionCombo = new QComboBox;
			positionCombo->addItem("Top");
			positionCombo->addItem("Top (Exclusive)");
			positionCombo->addItem("Bottom");

		// Add the position combo box to the table
		table->setCellWidget(i, 2, positionCombo);

		// Settings have been changed if the user changes one of these combo boxes
		connect(positionCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setSettingsModified()));
	}

	// Automatically set the height of the rows and the table itself
	table->resizeRowsToContents();
	table->setFixedHeight(table->rowHeight(0) * 10 + table->horizontalHeader()->height() + 2);
	table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

	// Generate the text shown at the bottom of the page
	QString bottomTextString = QString("NOTE: On Apple systems, the Command key ");
		bottomTextString += "is used instead of the Control key.";

	QLabel * bottomText = new QLabel(bottomTextString);
		bottomText->setWordWrap(true);

	// Add all items to a vertical layout, with a spacer below them
	QSpacerItem * pageSpacer = new QSpacerItem(10, 10, QSizePolicy::Minimum, QSizePolicy::Expanding);
	QVBoxLayout * pageLayout = new QVBoxLayout;
		pageLayout->addWidget(topText);
		pageLayout->addWidget(table);
		pageLayout->addWidget(bottomText);
		pageLayout->addSpacerItem(pageSpacer);

	// Set this vertical layout as the main layout of the widget
	this->mainLayout = (QLayout *) pageLayout;
	this->setLayout(this->mainLayout);

	// Store a pointer to the table
	this->pTable = table;
}


//------------------------------[ Destructor ]-----------------------------\\

Settings_GUIShortcutsPageWidget::~Settings_GUIShortcutsPageWidget()
{
	// Nothing to do here; the main layout is destroyed in the parent class
}


//-----------------------------[ getPageName ]-----------------------------\\

QString Settings_GUIShortcutsPageWidget::getPageName()
{
	return "GUI Shortcuts";
}


//--------------------------[ initializeControls ]-------------------------\\

void Settings_GUIShortcutsPageWidget::initializeControls(DTIToolSettings * settings)
{
	// New settings, so they have not yet been modified
	this->settingsModified = false;

	// Loop through all rows of the table
	for (int i = 0; i < 10; ++i)
	{
		// Get the GUI position stored in the settings
		DTIToolSettings::GUIPosition position = settings->guiShortcuts[i].position;

		// Get the combo box of the GUI position from the table, and set its index
		QComboBox * currentPositionCombo = (QComboBox *) this->pTable->cellWidget(i, 2);
		currentPositionCombo->setCurrentIndex((int) position);

		// Get the name of the target plugin
		QString pluginName = settings->guiShortcuts[i].plugin;

		// Get the combo box containing the plugin names from the table
		QComboBox * currentPluginCombo = (QComboBox *) this->pTable->cellWidget(i, 1);

		// Try to find the plugin in the combo box
		int pluginIndex = currentPluginCombo->findText(pluginName);

		// If we cannot find it...
		if (pluginIndex == -1)
		{
			// ...try to find the same name, but with the "(Plugin not loaded)" extension
			pluginIndex = currentPluginCombo->findText(pluginName + " (Plugin not loaded)");

			// If we still cannot find it...
			if (pluginIndex == -1)
			{
				// ...add the name to all ten combo boxes with the "(Plugin not loaded)" extension...
				for (int j = 0; j < 10; ++j)
				{
					QComboBox * tempPluginCombo = (QComboBox *) this->pTable->cellWidget(j, 1);
					tempPluginCombo->addItem(pluginName + " (Plugin not loaded)");
				}

				// ...and select it in the current combo box
				currentPluginCombo->setCurrentIndex(currentPluginCombo->count() - 1);
				continue;
			}
		}

		// If we did find the plugin, select it in the combo box
		currentPluginCombo->setCurrentIndex(pluginIndex);

	} // for [all ten rows]
}


//----------------------------[ storeSettings ]----------------------------\\

bool Settings_GUIShortcutsPageWidget::storeSettings(DTIToolSettings * settings)
{
	// Do nothing if no settings were modified
	if (this->settingsModified == false)
		return false;

	// Loop through all ten rows of the table
	for (int i = 0; i < 10; ++i)
	{
		// Get the combo box containing the plugin names from the table
		QComboBox * currentPluginCombo = (QComboBox *) this->pTable->cellWidget(i, 1);

		// Get the name of the selected plugin
		QString pluginName = currentPluginCombo->currentText();

		// If this name has the "(Plugin not loaded)" extension, remove it
		if (pluginName.endsWith(" (Plugin not loaded)"))
		{
			pluginName = pluginName.left(pluginName.length() - 20);
		}

		// Store the name
		settings->guiShortcuts[i].plugin = pluginName;

		// Get the combo box of the GUI position from the table
		QComboBox * currentPositionCombo = (QComboBox *) this->pTable->cellWidget(i, 2);

		// Store the position
		settings->guiShortcuts[i].position = (DTIToolSettings::GUIPosition) currentPositionCombo->currentIndex();
	}

	// Modified settings have been stored, so the controls are now up-to-date
	this->settingsModified = false;

	// Return true to signal that we've changed the settings
	return true;
}


} // namespace bmia
