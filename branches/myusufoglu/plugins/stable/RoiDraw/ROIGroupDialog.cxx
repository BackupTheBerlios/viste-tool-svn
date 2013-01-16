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
 * ROIGroupDialog.h
 *
 * 2011-02-16	Evert van Aart
 * - First version. 
 * 
 */


/** Includes */

#include "ROIGroupDialog.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

ROIGroupDialog::ROIGroupDialog()
{
	// Create widgets for choosing the group name
	this->step1Label = new QLabel("Step 1: Choose a name for the group.");
	this->step1LineEdit = new QLineEdit();

	// Create the table widget and its label, which will contain all input ROIs
	this->step2Label = new QLabel("Step 2: Select the input ROIs.");
	this->step2TableWidget = new QTableWidget();

	// Configure the table widget. The first column is empty, and used only for
	// spacing purposes. We also hide the headers and the grid, and we allow the
	// table widget to take up all available space.

	this->step2TableWidget->setColumnCount(2);
	this->step2TableWidget->setColumnWidth(0, 10);
	this->step2TableWidget->setShowGrid(false);
	this->step2TableWidget->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
	this->step2TableWidget->horizontalHeader()->hide();
	this->step2TableWidget->verticalHeader()->hide();
	this->step2TableWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	// Create widgets for step 3 (deleting input ROIs).
	this->step3Label = new QLabel("Step 3: Choose whether or not to delete the input ROIs.");
	this->step3Checkbox = new QCheckBox("Delete input ROIs");
	this->step3Checkbox->setChecked(true);

	// Create the "OK" and "Cancel" buttons
	this->okButton = new QPushButton("OK");
	this->cancelButton = new QPushButton("Cancel");
	this->buttonLayout = new QHBoxLayout;
	this->buttonLayout->addWidget(this->okButton);
	this->buttonLayout->addWidget(this->cancelButton);

	// Add everything to the main layout
	this->mainLayout = new QVBoxLayout;
	this->mainLayout->addWidget(this->step1Label);
	this->mainLayout->addWidget(this->step1LineEdit);
	this->mainLayout->addWidget(this->step2Label);
	this->mainLayout->addWidget(this->step2TableWidget);
	this->mainLayout->addWidget(this->step3Label);
	this->mainLayout->addWidget(this->step3Checkbox);
	this->mainLayout->addLayout(this->buttonLayout);

	// Set the layout of the dialog window
	this->setLayout(this->mainLayout);

	// Connect the two buttons
	connect(this->okButton, SIGNAL(clicked()), this, SLOT(accept()));
	connect(this->cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
}


//------------------------------[ Destructor ]-----------------------------\\

ROIGroupDialog::~ROIGroupDialog()
{
	// Delete all checkboxes
	for (int i = 0; i < this->step2TableWidget->rowCount(); ++i)
	{
		QWidget * currentCheckbox = this->step2TableWidget->cellWidget(i, 1);
		delete ((QCheckBox *) currentCheckbox);
		this->step2TableWidget->setCellWidget(i, 1, NULL);
	}

	// Delete the widgets and layouts
	delete this->step1Label;
	delete this->step1LineEdit;
	delete this->step2Label;
	delete this->step2TableWidget;
	delete this->step3Label;
	delete this->step3Checkbox;
	delete this->okButton;
	delete this->cancelButton;
	delete this->buttonLayout;
	delete this->mainLayout;
}


//----------------------------[ setDefaultName ]---------------------------\\

void ROIGroupDialog::setDefaultName(QString defaultName)
{
	if (!(this->step1LineEdit))
		return;

	// Copy the default name to the line edit widget
	this->step1LineEdit->setText(defaultName);
}


//-----------------------------[ getGroupName ]----------------------------\\

QString ROIGroupDialog::getGroupName()
{
	if (!(this->step1LineEdit))
		return "ERROR: Line Edit not set!";

	// Get the text of the line edit widget
	return this->step1LineEdit->text();
}


//--------------------------[ getDeleteInputROIs ]-------------------------\\

bool ROIGroupDialog::getDeleteInputROIs()
{
	if (!(this->step3Checkbox))
		return false;

	// Get the state of the checkbox
	return this->step3Checkbox->isChecked();
}


//------------------------------[ addROIName ]-----------------------------\\

void ROIGroupDialog::addROIName(QString newName)
{
	// Add a row to the table widget
	this->step2TableWidget->setRowCount(this->step2TableWidget->rowCount() + 1);

	// Add a checkbox with the ROI name to the new row
	QCheckBox * newCheckbox = new QCheckBox(newName);
	this->step2TableWidget->setCellWidget(this->step2TableWidget->rowCount() - 1, 1, newCheckbox);

	// Resize the row and the second column to fit the contents
	this->step2TableWidget->resizeColumnToContents(1);
	this->step2TableWidget->resizeRowToContents(this->step2TableWidget->rowCount() - 1);
}


//---------------------------[ getSelectedROIs ]---------------------------\\

QList<int> ROIGroupDialog::getSelectedROIs()
{
	// Create an empty list
	QList<int> selectedROIs;

	// Loop through all rows of the table widget
	for (int rowID = 0; rowID < this->step2TableWidget->rowCount(); ++rowID)
	{
		// Get the checkbox of the current row
		QCheckBox * currentCheckbox = (QCheckBox *) this->step2TableWidget->cellWidget(rowID, 1);

		// If the checkbox is checked, add the current row ID to the list
		if (currentCheckbox->isChecked())
		{
			selectedROIs.append(rowID);
		}
	}

	return selectedROIs;
}


} // namespace bmia
