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
 * PluginDialog.cxx
 *
 * 2010-02-11	Tim Peeters
 * - First version.
 *
 * 2011-01-28	Evert van Aart
 * - Added comments, minor code refactoring.
 *
 * 2011-07-21	Evert van Aart
 * - Added destructor.
 *
 */


/** Includes */

#include "PluginDialog.h"


namespace bmia {


namespace gui {


//-----------------------------[ Constructor ]-----------------------------\\

PluginDialog::PluginDialog(plugin::Manager * pmanager, QWidget * parent) :
								QDialog(parent),
								treeWidget(new QTreeWidget),
								cleanButton(new QPushButton("Clean")),
								addButton(new QPushButton("Add"))
{
	// Do nothing if we don't have a plugin manager
	if (!pmanager)
	{
		return;
	}

	// Store the plugin manager
	this->manager = pmanager;

	// Setup the tree widget
    this->treeWidget->setAlternatingRowColors(false);
    this->treeWidget->setAnimated(true);
    this->treeWidget->setSelectionMode(QAbstractItemView::NoSelection);
    this->treeWidget->setColumnCount(1);
    this->treeWidget->header()->hide();

	// Connect the buttons to their respective functions
	connect(cleanButton, SIGNAL(clicked()), this, SLOT(clean()));
    connect(addButton,	 SIGNAL(clicked()), this, SLOT(add()));

	// Setup the layout of the dialog window
    QVBoxLayout * mainLayout	= new QVBoxLayout;
    QHBoxLayout * buttonLayout	= new QHBoxLayout;
    mainLayout->addWidget(this->treeWidget);
    buttonLayout->addStretch(0);
    buttonLayout->addWidget(cleanButton);
    buttonLayout->addWidget(addButton);
    buttonLayout->addStretch(0);
    mainLayout->addLayout(buttonLayout);
    this->setLayout(mainLayout);

	// Set the window title
    this->setWindowTitle(tr("Plugin List"));

	// Set the plugin directory to an empty string
	this->pluginDir = "";
}


//------------------------------[ Destructor ]-----------------------------\\

PluginDialog::~PluginDialog()
{
	// Clear the tree widget
	if (this->treeWidget)
		this->treeWidget->clear();

	// Delete the main layout of the dialog
	if (this->layout())
		delete this->layout();
}


//--------------------------------[ clean ]--------------------------------\\

void PluginDialog::clean()
{
	// Disconnect the signal for changed items
	disconnect(this->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem *, int)), this, SLOT(onItemChanged(QTreeWidgetItem *, int)));

	// Clean the entire tree widget
	this->treeWidget->clear();

	// Tell the manager to remove unloaded plugins
	this->manager->removeAllUnloaded();

	// Rebuild the tree widget
    for (int i = 0; i < this->manager->getNumberOfPlugins(); ++i)
	{
		this->populateTreeWidget(i);
	}

	// Reconnect the signal
	connect(this->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(onItemChanged(QTreeWidgetItem*, int)));
}


//-----------------------------[ rebuildList ]-----------------------------\\

void PluginDialog::rebuildList()
{
	// Disconnect the signal for changed items
	disconnect(this->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem *, int)), this, SLOT(onItemChanged(QTreeWidgetItem *, int)));

	// Clean the entire tree widget
	this->treeWidget->clear();

	// Rebuild the tree widget
	for (int i = 0; i < this->manager->getNumberOfPlugins(); ++i)
	{
		this->populateTreeWidget(i);
	}

	// Reconnect the signal
	connect(this->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(onItemChanged(QTreeWidgetItem*, int)));
}


//---------------------------------[ add ]---------------------------------\\

void PluginDialog::add()
{
	// Get the filename using a file dialog
	QString filename = QFileDialog::getOpenFileName(	this,
														"Load Plugin",
														this->pluginDir,
														"*.so *.dylib *.dll");

	// Check if the filename is correct
	if (filename.isNull() || filename.isEmpty()) 
		return;

	// Try to add the plugin to the manager
	int pluginId = this->manager->add(filename);

	// Check if the plugin was added successfully 
	if (pluginId < 0) 
		return;

	// Load the plugin and add it to the tree widget
	if (this->manager->load(pluginId))
	{
		this->populateTreeWidget(pluginId);
	}
}


//--------------------------[ populateTreeWidget ]-------------------------\\

void PluginDialog::populateTreeWidget(int i)
{
	// Disconnect the signal for changed items
	disconnect(this->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem *, int)), this, SLOT(onItemChanged(QTreeWidgetItem *, int)));

	// Create a new item for the tree widget
	QTreeWidgetItem * pluginItem = new QTreeWidgetItem(this->treeWidget);
	pluginItem->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsSelectable | Qt::ItemIsEnabled);

	// Check if the plugin has been loaded
	if (!this->manager->isLoaded(i))
	{ 
		// If not, uncheck the item
		pluginItem->setCheckState(0, Qt::Unchecked);
		pluginItem->setText(0, this->manager->getShortFileName(i));
	} 
	else
	{ 
		// If the plugin is loaded, check the item
		pluginItem->setCheckState(0, Qt::Checked);
		pluginItem->setText(0, this->manager->getName(i));

		pluginItem->setToolTip(0, this->manager->getFileName(i));

		// Get the features of the plugin
		QStringList features = this->manager->getFeatures(i);

		// Add each feature to the tree item
		foreach (QString feature, features)
		{
			QTreeWidgetItem * interfaceItem = new QTreeWidgetItem(pluginItem);
			interfaceItem->setText(0, feature);
		}
	}

	// Use a bold font for the plugin names
	QFont boldFont = pluginItem->font(0);
	boldFont.setBold(true);
	pluginItem->setFont(0, boldFont);

	// Reconnect the signal
	connect(this->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(onItemChanged(QTreeWidgetItem*, int)));
}


//----------------------------[ onItemChanged ]----------------------------\\

void PluginDialog::onItemChanged(QTreeWidgetItem * item, int column)
{
	// Get the number of items in the tree
    int n = this->treeWidget->topLevelItemCount();

	// The number of tree items should always match the number of plugins
    assert(n == this->manager->getNumberOfPlugins());

	// Get the index of the changed item
    int i = this->treeWidget->indexOfTopLevelItem(item);

	// Load or unload the plugin
    this->manager->setLoaded(i, item->checkState(0) == Qt::Checked);
}


} // namespace bmia


} // namespace gui
