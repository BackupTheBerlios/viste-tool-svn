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

/**
 * FiberOutputPlugin.cxx
 *
 * 2010-12-21	Evert van Aart
 * - First version.
 *
 */


/** Includes */

#include "FiberOutputPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberOutputPlugin::FiberOutputPlugin() : Plugin("Fiber Output")
{
	// Create a new Qt widget
	this->widget = new QWidget();

	// Create a new GUI form
	this->ui = new Ui::FiberOutputForm();

	// Setup the GUI
	this->ui->setupUi(this->widget);

	// Enable/disable and show/hide controls
	this->enableControls();

	connect(this->ui->OutputTXTButton, SIGNAL(clicked()), this, SLOT(writeTXT()));	
	connect(this->ui->OutputXMLButton, SIGNAL(clicked()), this, SLOT(writeXML()));	
	connect(this->ui->DataSourceROIRadio, SIGNAL(clicked()), this, SLOT(enableControls()));	
	connect(this->ui->DataSourceFibersRadio, SIGNAL(clicked()), this, SLOT(enableControls()));	
	connect(this->ui->ROIList, SIGNAL(itemSelectionChanged()), this, SLOT(enableControls()));
	connect(this->ui->MeasuresList, SIGNAL(itemSelectionChanged()), this, SLOT(enableControls()));
}


//------------------------------[ Destructor ]-----------------------------\\

FiberOutputPlugin::~FiberOutputPlugin()
{
	delete this->widget;
	this->widget = NULL;

	// Clear the data set lists
	this->dtiImageDataSets.clear();
	this->eigenImageDataSets.clear();
	this->measureImageDataSets.clear();
	this->fiberDataSets.clear();
	this->seedDataSets.clear();
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * FiberOutputPlugin::getGUI()
{
	// Return the Qt widget
	return this->widget;
}


//----------------------------[ enableControls ]---------------------------\\

void FiberOutputPlugin::enableControls()
{
	// If the tensor image and/or eigensystem image has not been set, we disable everything
	bool imagesSet = (this->ui->DTIDataCombo->count() > 0) && (this->ui->EigenDataCombo->count() > 0);

	this->ui->DataSourceGroup->setEnabled(imagesSet);
	this->ui->OutputOptionsGroup->setEnabled(imagesSet);
	this->ui->OutputOptionsEigenCheck->setEnabled(imagesSet);
	this->ui->OutputOptionsMeanVarCheck->setEnabled(imagesSet);
	this->ui->OutputOptionsPerVoxelCheck->setEnabled(imagesSet);
	this->ui->OutputOptionsTensorCheck->setEnabled(imagesSet);
	this->ui->MeasuresLabel->setEnabled(imagesSet);
	this->ui->MeasuresList->setEnabled(imagesSet);

	// Check if fibers have been enabled
	bool fibersEnable = imagesSet && this->ui->DataSourceFibersRadio->isChecked();

	this->ui->OutputOptionsVolumeCheck->setEnabled(fibersEnable);
	this->ui->OutputOptionsLengthCheck->setEnabled(fibersEnable);
	this->ui->FibersCombo->setVisible(fibersEnable);
	this->ui->FibersLabel->setVisible(fibersEnable);

	// Check if ROIs have been enabled
	bool ROIEnable = imagesSet && this->ui->DataSourceROIRadio->isChecked();

	this->ui->ROILabel->setVisible(ROIEnable);
	this->ui->ROIList->setVisible(ROIEnable);

	// Check if we can run the output filter
	bool dataSourceSelected =	(fibersEnable && this->ui->FibersCombo->currentIndex()       >= 0) ||
								(ROIEnable    && this->ui->ROIList->selectedItems().size()   >  0);

	this->ui->OutputTXTButton->setEnabled(dataSourceSelected);
	this->ui->OutputXMLButton->setEnabled(dataSourceSelected);
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void FiberOutputPlugin::dataSetAdded(data::DataSet * ds)
{
	// Check if the input data set exists
	if (!ds)
		return;

	// Get the kind of the data set
	QString kind = ds->getKind();

	// For each type, add the data set name to the corresponding GUI
	// control, and add the data set pointer to the right list. We check
	// whether or not the data sets are correct (e.g., all necessary pointers
	// have been set) inside the "FiberOutput" classes.

	// DTI tensors
	if (kind == "DTI")
	{
		this->ui->DTIDataCombo->addItem(ds->getName());
		this->dtiImageDataSets.append(ds);
	}

	// Eigenvectors and -values
	else if (kind == "eigen")
	{
		this->ui->EigenDataCombo->addItem(ds->getName());
		this->eigenImageDataSets.append(ds);
	}

	// Scalar measures
	else if (kind == "scalar volume")
	{
		this->ui->MeasuresList->addItem(ds->getName());
		this->measureImageDataSets.append(ds);
	}

	// Seed points
	else if (kind == "seed points")
	{
		this->ui->ROIList->addItem(ds->getName());
		this->seedDataSets.append(ds);
	}

	// Fibers
	else if (kind == "fibers")
	{
		this->ui->FibersCombo->addItem(ds->getName());
		this->fiberDataSets.append(ds);
	}

	// Data set kind not supported; ignore it
	else
	{
		return;
	}

	// Disable/enable controls based on new settings
	this->enableControls();
}


//----------------------------[ dataSetChanged ]---------------------------\\

void FiberOutputPlugin::dataSetChanged(data::DataSet * ds)
{
	// Since the data sets aren't actually used until the user clicks one of the
	// output buttons, all we have to do here is update the data set name.

	// Check if the input data set exists
	if (!ds)
		return;

	// Get the kind of the data set
	QString kind = ds->getKind();

	// DTI tensors
	if (kind == "DTI")
	{
		if(!(this->dtiImageDataSets.contains(ds)))
			return;

		int index = this->dtiImageDataSets.indexOf(ds);
		this->ui->DTIDataCombo->setItemText(index, ds->getName());
	}

	// Eigenvectors and -values
	else if (kind == "eigen")
	{
		if(!(this->eigenImageDataSets.contains(ds)))
			return;

		int index = this->eigenImageDataSets.indexOf(ds);
		this->ui->EigenDataCombo->setItemText(index, ds->getName());
	}

	// Scalar measures
	else if (kind == "scalar volume")
	{
		if(!(this->measureImageDataSets.contains(ds)))
			return;

		int index = this->measureImageDataSets.indexOf(ds);
		this->ui->MeasuresList->item(index)->setText(ds->getName());
	}

	// Seed points
	else if (kind == "seed points")
	{
		if(!(this->seedDataSets.contains(ds)))
			return;

		int index = this->seedDataSets.indexOf(ds);
		this->ui->ROIList->item(index)->setText(ds->getName());
	}

	// Fibers
	else if (kind == "fibers")
	{
		if(!(this->fiberDataSets.contains(ds)))
			return;

		int index = this->fiberDataSets.indexOf(ds);
		this->ui->FibersCombo->setItemText(index, ds->getName());
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void FiberOutputPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Check if the input data set exists
	if (!ds)
		return;

	// Get the kind of the data set
	QString kind = ds->getKind();

	// DTI tensors
	if (kind == "DTI")
	{
		if(!(this->dtiImageDataSets.contains(ds)))
			return;

		int index = this->dtiImageDataSets.indexOf(ds);
		this->ui->DTIDataCombo->removeItem(index);
		this->dtiImageDataSets.removeAt(index);
	}

	// Eigenvectors and -values
	else if (kind == "eigen")
	{
		if(!(this->eigenImageDataSets.contains(ds)))
			return;

		int index = this->eigenImageDataSets.indexOf(ds);
		this->ui->EigenDataCombo->removeItem(index);
		this->eigenImageDataSets.removeAt(index);
	}

	// Scalar measures
	else if (kind == "scalar volume")
	{
		if(!(this->measureImageDataSets.contains(ds)))
			return;

		int index = this->measureImageDataSets.indexOf(ds);
		this->ui->MeasuresList->takeItem(index);
		this->measureImageDataSets.removeAt(index);
	}

	// Seed points
	else if (kind == "seed points")
	{
		if(!(this->seedDataSets.contains(ds)))
			return;

		int index = this->seedDataSets.indexOf(ds);
		this->ui->ROIList->takeItem(index);
		this->seedDataSets.removeAt(index);
	}

	// Fibers
	else if (kind == "fibers")
	{
		if(!(this->fiberDataSets.contains(ds)))
			return;

		int index = this->fiberDataSets.indexOf(ds);
		this->ui->FibersCombo->removeItem(index);
		this->fiberDataSets.removeAt(index);
	}
}


//-------------------------------[ writeTXT ]------------------------------\\

void FiberOutputPlugin::writeTXT()
{
	// Create a dialog to get the filename
	QString fileName = QFileDialog::getSaveFileName(NULL, tr("Save Output File"), "", tr("Text files (*.txt)"));

	// Do nothing if the filename is empty
	if (fileName.isEmpty())
	{
		return;
	}

	// Create new output class, use it to write the output, and delete it
	FiberOutput * out = (FiberOutput *) new FiberOutputTXT;
	this->writeOutput(out, fileName);
	delete out;
}


//-------------------------------[ writeXML ]------------------------------\\

void FiberOutputPlugin::writeXML()
{
	// Create a dialog to get the filename
	QString fileName = QFileDialog::getSaveFileName(NULL, tr("Save Output File"), "", tr("XML files (*.xml)"));

	// Do nothing if the filename is empty
	if (fileName.isEmpty())
	{
		return;
	}

	// Create new output class, use it to write the output, and delete it
	FiberOutput * out = (FiberOutput *) new FiberOutputXML;
	this->writeOutput(out, fileName);
	delete out;
}


//-----------------------------[ writeOutput ]-----------------------------\\

void FiberOutputPlugin::writeOutput(FiberOutput * out, QString fileName)
{
	// Set the DTI Image
	int selectedDTIImageIndex = this->ui->DTIDataCombo->currentIndex();
	data::DataSet * selectedDTIImage = this->dtiImageDataSets.at(selectedDTIImageIndex);
	out->setTensorImage(selectedDTIImage->getVtkObject(), selectedDTIImage->getName().toStdString());

	// Set the Eigensystem Image
	int selectedEigenImageIndex = this->ui->EigenDataCombo->currentIndex();
	data::DataSet * selectedEigenImage = this->eigenImageDataSets.at(selectedEigenImageIndex);
	out->setEigenImage(selectedEigenImage->getVtkObject(), selectedEigenImage->getName().toStdString());

	// Set the data source (ROIs or fibers)
	FiberOutput::DataSourceType dataSource;

	if (this->ui->DataSourceROIRadio->isChecked())
	{
		dataSource = FiberOutput::DS_ROI;
	}
	else
	{
		dataSource = FiberOutput::DS_Fibers;
	}

	// Add seed point data sets if we have selected ROIs as the data source
	if (dataSource == FiberOutput::DS_ROI)
	{
		// Get the list of selected ROIs
		QList<QListWidgetItem *> selectedROIs = this->ui->ROIList->selectedItems();

		// Loop through the list
		for (int i = 0; i < selectedROIs.size(); ++i)
		{
			// For each selected ROI, get its index
			QListWidgetItem * currentROI = selectedROIs.at(i);
			int currentROIIndex = this->ui->ROIList->row(currentROI);

			// Using the index, get the data set pointer, and add it to the output object
			data::DataSet * currentROIDS = this->seedDataSets.at(currentROIIndex);
			out->addSeedPoints(currentROIDS->getVtkObject(), currentROIDS->getName().toStdString());
		}
	}

	// Add fiber data sets if we have selected fibers as the data source
	else
	{
		int currentFibersIndex = this->ui->FibersCombo->currentIndex();
		data::DataSet * currentFibers = this->fiberDataSets.at(currentFibersIndex);
		out->addFibers(currentFibers->getVtkObject(), currentFibers->getName().toStdString());
	}

	// Get the list of selected measures
	QList<QListWidgetItem *> selectedMeasures = this->ui->MeasuresList->selectedItems();

	// Loop through the list
	for (int i = 0; i < selectedMeasures.size(); ++i)
	{
		// For each selected measure, get its index
		QListWidgetItem * currentMeasure = selectedMeasures.at(i);
		int currentMeasureIndex = this->ui->MeasuresList->row(currentMeasure);

		// Using the index, get the data set pointer, and add it to the output object
		data::DataSet * currentMeasureDS = this->measureImageDataSets.at(currentMeasureIndex);
		currentMeasureDS->getVtkImageData()->Update();
		out->addScalarImage(currentMeasureDS->getVtkObject(), this->getShortMeasureName(currentMeasureDS->getName()));
	}

	// Set the optional outputs
	out->setOutputTensor(this->ui->OutputOptionsTensorCheck->isChecked());
	out->setOutputEigenvector(this->ui->OutputOptionsEigenCheck->isChecked());
	out->setOutputFiberLength(this->ui->OutputOptionsLengthCheck->isChecked() && dataSource == FiberOutput::DS_Fibers);
	out->setOutputFiberVolume(this->ui->OutputOptionsVolumeCheck->isChecked() && dataSource == FiberOutput::DS_Fibers);

	// Write the data
	std::string outMessage = out->saveData(	fileName.toLatin1().data(), 
											dataSource, 
											this->ui->OutputOptionsPerVoxelCheck->isChecked(), 
											this->ui->OutputOptionsMeanVarCheck->isChecked());

	// Display error messages if something went wrong
	if (outMessage.length() > 0)
	{
		QString fullText = "The FiberOutput object returned the following error:\n";
		fullText.append(outMessage.c_str());
		QMessageBox::warning(NULL, "Fiber Output Plugin", fullText);
	}
}


//-------------------------[ getShortMeasureName ]-------------------------\\

std::string FiberOutputPlugin::getShortMeasureName(QString longName)
{
	// Find the last space
	int lastSpace = longName.lastIndexOf(" ");

	// Return the input if it contains no spaces
	if (lastSpace == -1)
	{
		return longName.toStdString();
	}

	// Get the characters after the last string
	QString shortName = longName.right(longName.length() - lastSpace - 1);

	// Return this short name if it is not empty
	if (!(shortName.isEmpty()))
	{
		return shortName.toStdString();
	}
	// Otherwise, return the original name
	else
	{
		return longName.toStdString();
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libFiberOutputPlugin, bmia::FiberOutputPlugin)
