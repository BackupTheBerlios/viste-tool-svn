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
 * HARDIConverterPlugin.cxx
 *
 * 2011-08-04	Evert van Aart
 * - Version 1.0.0.
 * - First version. Currently, only SH-to-DSF conversions are supported.
 *
 */


/** Includes */

#include "HARDIConverterPlugin.h"


namespace bmia {



//-----------------------------[ Constructor ]-----------------------------\\

HARDIConverterPlugin::HARDIConverterPlugin() : Plugin("HARDI Converter")
{
	// Initialization is done in the "init" function.
}


//---------------------------------[ init ]--------------------------------\\

void HARDIConverterPlugin::init()
{
	// Create the GUI of the widget
	this->widget = new QWidget();
	this->ui = new Ui::HARDIConverterForm();
	this->ui->setupUi(this->widget);

	// Enable the controls
	this->enableControls();

	// Connect GUI controls to slot functions
	connect(this->ui->inputDataCombo,		SIGNAL(currentIndexChanged(int)), this, SLOT(loadDataInfo(int)));
	connect(this->ui->outputKindDSFRadio,	SIGNAL(clicked()), this, SLOT(enableControls()));
	connect(this->ui->applyButton,			SIGNAL(clicked()), this, SLOT(applyConversion()));
}


//------------------------------[ Destructor ]-----------------------------\\

HARDIConverterPlugin::~HARDIConverterPlugin()
{
	// Delete the GUI
	delete this->widget;

	// Delete all data set pointers
	for (QList<dataSetInfo>::iterator i = this->dataList.begin(); i != this->dataList.end(); ++i)
	{
		dataSetInfo info = (*i);
		info.outDSs.clear();
	}

	this->dataList.clear();
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * HARDIConverterPlugin::getGUI()
{
	// Return the GUI widget
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void HARDIConverterPlugin::dataSetAdded(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions
	if (ds->getKind() == "discrete sphere")
	{
		// Check if the data set contains an image
		vtkImageData * image = ds->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		// Check if the point data contains a spherical directions array
		vtkDoubleArray * anglesArray = vtkDoubleArray::SafeDownCast(imagePD->GetArray("Spherical Directions"));

		if (!anglesArray)
			return;

		// Create a new information object
		dataSetInfo newInfo;
		newInfo.inDS = ds;

		// Add the data set to the list and the GUI
		this->dataList.append(newInfo);
		this->ui->inputDataCombo->addItem(ds->getName());

		// If this was the first data set, select it now
		if (this->ui->inputDataCombo->count() == 1 && this->ui->inputDataCombo->currentIndex() != 0)
			this->ui->inputDataCombo->setCurrentIndex(0);
	}

	// Spherical Harmonics
	else if (ds->getKind() == "spherical harmonics")
	{
		// Check if the data set contains an image
		vtkImageData * image = ds->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		// Check if the point data contains a spherical directions array
		vtkDataArray * shArray = imagePD->GetScalars();

		if (!shArray)
			return;

		// Create a new information object
		dataSetInfo newInfo;
		newInfo.inDS = ds;

		// Add the data set to the list and the GUI
		this->dataList.append(newInfo);
		this->ui->inputDataCombo->addItem(ds->getName());

		// If this was the first data set, select it now
		if (this->ui->inputDataCombo->count() == 1 && this->ui->inputDataCombo->currentIndex() != 0)
			this->ui->inputDataCombo->setCurrentIndex(0);
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void HARDIConverterPlugin::dataSetChanged(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions and Spherical Harmonics
	if (ds->getKind() == "discrete sphere" || ds->getKind() == "spherical harmonics")
	{
		int dataIndex = 0;

		// Loop through all input data sets
		for (QList<dataSetInfo>::iterator i = this->dataList.begin(); i != this->dataList.end(); ++i, ++dataIndex)
		{
			dataSetInfo info = (*i);

			// If we've found the data set, update its name
			if (ds == info.inDS)
				this->ui->outputDataCombo->setItemText(dataIndex, ds->getName());
		}

		// Load the new settings to the GUI
		this->loadDataInfo(this->ui->inputDataCombo->currentIndex());
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void HARDIConverterPlugin::dataSetRemoved(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions and Spherical Harmonics
	if (ds->getKind() == "discrete sphere" || ds->getKind() == "spherical harmonics")
	{
		int dataIndex = 0;

		// Loop through all input data sets
		for (QList<dataSetInfo>::iterator i = this->dataList.begin(); i != this->dataList.end(); ++i, ++dataIndex)
		{
			dataSetInfo info = (*i);

			// If this was the input data set...
			if (ds == info.inDS)
			{
				// ...simply remove it from the list and the GUI
				info.outDSs.clear();
				this->dataList.removeAt(dataIndex);
				this->ui->inputDataCombo->removeItem(dataIndex);
			}
			else
			{
				// Otherwise, check if we've removed one of the output data sets
				int outIndex = info.outDSs.indexOf(ds);

				// If so, remove it from the list
				if (outIndex != -1)
				{
					info.outDSs.removeAt(outIndex);
					this->dataList.replace(dataIndex, info);
				}
			}

		}

		// Load the new settings to the GUI
		this->loadDataInfo(this->ui->inputDataCombo->currentIndex());
	}
}


//-------------------------[ renameOutputDataSet ]-------------------------\\

void HARDIConverterPlugin::renameOutputDataSet()
{
	// Get the information of the selected input data set
	int dsIndex = this->ui->inputDataCombo->currentIndex();

	if (dsIndex < 0 || dsIndex >= this->dataList.size())
		return;

	dataSetInfo info = this->dataList.at(dsIndex);

	if (this->ui->outputDataCombo->count() <= 0 || this->ui->outputDataCombo->currentIndex() < 0)
		return;

	bool ok;

	// Ask the user for a new data set name
	QString newName = QInputDialog::getText(this->getGUI(), tr("HARDI Converter"), 
		tr("Enter new data set name..."), QLineEdit::Normal, this->ui->outputDataCombo->currentText(), &ok);

	// If the user selected a valid name, update the data set
	if (ok && !newName.isEmpty())
	{
		info.outDSs[this->ui->outputDataCombo->currentIndex()]->setName(newName);
		this->core()->data()->dataSetChanged(info.outDSs[this->ui->outputDataCombo->currentIndex()]);
	}
}


//----------------------------[ enableControls ]---------------------------\\

void HARDIConverterPlugin::enableControls()
{
	// Check if we've selected an input data set
	if (this->ui->inputDataCombo->currentIndex() >= 0 && this->ui->inputDataCombo->currentIndex() < this->dataList.size())
	{
		// Enable the three groups
		this->ui->inputGroup->setEnabled(true);
		this->ui->convGroup->setEnabled(true);
		this->ui->outputGroup->setEnabled(true);

		// Get the input data set
		data::DataSet * inDS = this->dataList[this->ui->inputDataCombo->currentIndex()].inDS;

		// Output data kind cannot be the same as the input data kind
		if (inDS->getKind() == "discrete sphere")
			this->ui->outputKindDSFRadio->setEnabled(false);

		bool isEnabledDSF = this->ui->outputKindDSFRadio->isEnabled();

		bool isCheckedDSF = this->ui->outputKindDSFRadio->isChecked();

		// Note: After adding other conversions, you should include some code here
		// that checks whether the currently checked radio button is enabled. If not,
		// it should check one of the other (enabled) radio buttons.

		// Show or hide tessellation controls
		this->ui->convTessLabel->setVisible(isCheckedDSF);
		this->ui->convTessSpin->setVisible(isCheckedDSF);

		// Only enable the radio button for overwriting if we've got available output data sets
		this->ui->outputOverwriteRadio->setEnabled(this->ui->outputDataCombo->count() > 0);

		// If the overwrite radio button is disable, check the "New Data Set" radio button
		if (this->ui->outputOverwriteRadio->isEnabled() == false)
			this->ui->outputNewRadio->setChecked(true);

		// Only enable the renaming button and the output data combo box if we've 
		// selected an output data set.

		bool enableOutputControls = this->ui->outputOverwriteRadio->isEnabled() &&
									this->ui->outputDataCombo->currentIndex() >= 0;

		this->ui->outputRenameButton->setEnabled(enableOutputControls);
		this->ui->outputDataCombo->setEnabled(enableOutputControls);
		this->ui->outputDataLabel->setEnabled(enableOutputControls);

		// Check whether we can apply the conversion
		bool enableApply = isCheckedDSF && isEnabledDSF;

		if (this->ui->outputOverwriteRadio->isChecked())
			enableApply &= enableOutputControls;

		this->ui->applyButton->setEnabled(enableApply);
	}

	// If no input data set has been selected, simply disable everything
	else
	{
		this->ui->inputGroup->setEnabled(false);
		this->ui->convGroup->setEnabled(false);
		this->ui->outputGroup->setEnabled(false);
		this->ui->applyButton->setEnabled(false);
	}
}


//-----------------------------[ loadDataInfo ]----------------------------\\

void HARDIConverterPlugin::loadDataInfo(int index)
{
	// Get the information of the selected input data set
	if (index < 0 || index >= this->dataList.size())
	{
		this->ui->inputTypeVarLabel->setText("<None Selected>");
		return;
	}

	dataSetInfo info = this->dataList.at(index);

	if (info.inDS->getKind() == "discrete sphere")
		this->ui->inputTypeVarLabel->setText("Discrete Sphere Function");
	else if (info.inDS->getKind() == "spherical harmonics")
		this->ui->inputTypeVarLabel->setText("Spherical Harmonics");
	else
		this->ui->inputTypeVarLabel->setText("Unknown Type");

	// Clear the combo box of output data sets
	this->ui->outputDataCombo->clear();

	// Add all output data sets to the combo box
	for (QList<data::DataSet *>::iterator i = info.outDSs.begin(); i != info.outDSs.end(); ++i)
	{
		this->ui->outputDataCombo->addItem((*i)->getName());
	}

	this->ui->outputDataCombo->setCurrentIndex(info.outDSs.isEmpty() ? -1 : 0);

	// Enable or disable controls based on the new settings
	this->enableControls();
}


//---------------------------[ applyConversion ]---------------------------\\

void HARDIConverterPlugin::applyConversion()
{
	// Determine the input type
	QString typeAString = this->dataList[this->ui->inputDataCombo->currentIndex()].inDS->getKind();

	DataType typeA = DT_Unknown;

	if (typeAString == "discrete sphere")		typeA = DT_DSF;
	if (typeAString == "spherical harmonics")	typeA = DT_SH;

	// Determine the output type
	DataType typeB = DT_Unknown;

	if (this->ui->outputKindDSFRadio->isChecked())		typeB = DT_DSF;

	// Call the correct conversion function
	if (typeA == DT_SH && typeB == DT_DSF)
	{
		this->convertSHtoDSF();
	}
	else
	{
		// This should never happen
		this->core()->out()->showMessage("Selected conversion is not supported!", "HARDI Converter");
	}
}


//----------------------------[ convertSHtoDSF ]---------------------------\\

void HARDIConverterPlugin::convertSHtoDSF()
{
	// Get the input data set and image
	data::DataSet * inDS = this->dataList[this->ui->inputDataCombo->currentIndex()].inDS;
	vtkImageData * inImage = inDS->getVtkImageData();

	// Create the conversion filter
	vtkSH2DSFFilter * filter = vtkSH2DSFFilter::New();
	this->core()->out()->createProgressBarForAlgorithm(filter, "HARDI Converter");
	filter->setTessOrder(this->ui->convTessSpin->value());
	filter->SetInput(inImage);

	// Run the filter
	filter->Update();

	// Add the result to the data manager
	this->createOutput(filter->GetOutput(), inDS->getName() + " [DSF]", "discrete sphere");

	// Done!
	this->core()->out()->deleteProgressBarForAlgorithm(filter);
	filter->Delete();
}


//-----------------------------[ createOutput ]----------------------------\\

void HARDIConverterPlugin::createOutput(vtkImageData * outImage, QString name, QString type)
{
	// Get the information of the selected input data set
	dataSetInfo info = this->dataList[this->ui->inputDataCombo->currentIndex()];

	// Either create a new data set...
	if (this->ui->outputNewRadio->isChecked())
	{
		data::DataSet * newDS = new data::DataSet(name, type, vtkObject::SafeDownCast(outImage));
		newDS->getAttributes()->copyTransformationMatrix(info.inDS);
		this->core()->data()->addDataSet(newDS);
	}

	// ...or overwrite an existing one
	else
	{
		int outIndex = this->ui->outputDataCombo->currentIndex();

		data::DataSet * outDS = info.outDSs[outIndex];
		outDS->updateData(vtkObject::SafeDownCast(outImage));
		this->core()->data()->dataSetChanged(outDS);
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libHARDIConverterPlugin, bmia::HARDIConverterPlugin)
