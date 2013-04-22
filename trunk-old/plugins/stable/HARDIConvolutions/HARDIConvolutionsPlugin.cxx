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
 * HARDIConvolutionsPlugin.cxx
 *
 * 2011-07-22	Evert van Aart
 * - Version 1.0.0.
 * - First version
 *
 * 2011-08-05	Evert van Aart
 * - Version 1.0.1.
 * - Fixed error in the computation of the Duits kernels.
 * - Fixed computation of unit vectors from spherical angles.
 * 
 */


/** Includes */

#include "HARDIConvolutionsPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

HARDIConvolutionsPlugin::HARDIConvolutionsPlugin() : Plugin("HARDI Convolutions")
{

}


//---------------------------------[ init ]--------------------------------\\

void HARDIConvolutionsPlugin::init()
{
	// Create the GUI of the widget
	this->widget = new QWidget();
	this->ui = new Ui::HARDIConvolutionsForm();
	this->ui->setupUi(this->widget);

	// Enable the controls
	this->enableControls();

	// Connect the combo boxes
	connect(this->ui->genTypeCombo,			SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()			));
	connect(this->ui->dataCombo,			SIGNAL(currentIndexChanged(int)), this, SLOT(loadDataInfo(int)			));
	connect(this->ui->outputOverwriteCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(selectOutputDataSet(int)	));

	// Connect the radio- and check boxes
	connect(this->ui->outputOverwriteRadio, SIGNAL(clicked()), this, SLOT(changeOutputMethod()	));
	connect(this->ui->outputNewRadio,		SIGNAL(clicked()), this, SLOT(changeOutputMethod()	));
	connect(this->ui->genButton,			SIGNAL(clicked()), this, SLOT(generateKernels()		));
	connect(this->ui->kigLoadButton,		SIGNAL(clicked()), this, SLOT(loadKIGFile()			));
	connect(this->ui->outputRenameButton,	SIGNAL(clicked()), this, SLOT(renameOutputDataSet()	));
	connect(this->ui->applyButton,			SIGNAL(clicked()), this, SLOT(applyConvolution()	));
	connect(this->ui->thresholdAbsRadio,	SIGNAL(clicked()), this, SLOT(enableControls()		));
	connect(this->ui->thresholdPercRadio,	SIGNAL(clicked()), this, SLOT(enableControls()		));
}


//------------------------------[ Destructor ]-----------------------------\\

HARDIConvolutionsPlugin::~HARDIConvolutionsPlugin()
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

	// Delete any existing Kernel Image Groups
	for (QList<QStringList *>::iterator i = this->kernelImageGroups.begin(); i != this->kernelImageGroups.end(); ++i)
	{
		(*i)->clear();
		delete (*i);
	}

	this->kernelImageGroups.clear();
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * HARDIConvolutionsPlugin::getGUI()
{
	// Return the GUI widget
	return this->widget;
}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList HARDIConvolutionsPlugin::getSupportedFileExtensions()
{
	QStringList list;
	list.push_back("kig");
	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList HARDIConvolutionsPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("Kernel Image Groups");
	return list;
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void HARDIConvolutionsPlugin::loadDataFromFile(QString filename)
{
	// Create a file handle with the specified filename
	QFile kigFile(filename);

	// Try to open the file
	if (!kigFile.open(QIODevice::ReadOnly))
	{
		this->core()->out()->showMessage("Could not open Kernel Image Group file!", "HARDI Convolutions");
		return;
	}

	// Create a text stream
	QTextStream in(&kigFile);

	// Create a new string list for this Kernel Image Group
	QStringList * newKIG = new QStringList;

	// Decompose the file name into a base name and a path
	QString baseName, absolutePath;
	this->decomposeFileName(filename, baseName, absolutePath);

	// Check if this base name already exists
	int targetIndex = this->ui->kigCombo->findText(baseName);

	// Read all lines
	while (!(in.atEnd()))
	{
		QString kigFileName = in.readLine();
		kigFileName = kigFileName.simplified();

		// Add the full, absolute file path to the list
		if (!(kigFileName.isEmpty()))
			newKIG->append(absolutePath + kigFileName);
	}

	// If this KIG does not yet exist, add it to the list and the GUI
	if (targetIndex == -1)
	{
		this->kernelImageGroups.append(newKIG);
		this->ui->kigCombo->addItem(baseName);
	}

	// Otherwise, first delete the old one, and add the new one to the list
	else
	{
		QStringList * oldKIG = this->kernelImageGroups.at(targetIndex);
		delete oldKIG;
		this->kernelImageGroups.replace(targetIndex, newKIG);
	}

	// Close the file
	kigFile.close();

	// Enable or disable the controls based on the new settings
	this->enableControls();
}


//-----------------------------[ loadKIGFile ]-----------------------------\\

void HARDIConvolutionsPlugin::loadKIGFile()
{
	// Ask the user for a file name
	QString kigFileName = QFileDialog::getOpenFileName(this->getGUI(), "Choose kernel group name...", 
				this->core()->getDataDirectory().absolutePath(), tr("Kernel Image Groups (*.kig)"));
	
	// If the user selected a valid file, load it now
	if (kigFileName.isEmpty() == false && kigFileName.isNull() == false)
		this->loadDataFromFile(kigFileName);
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void HARDIConvolutionsPlugin::dataSetAdded(data::DataSet * ds)
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
		newInfo.selectedOutputDS = NULL;

		// Add the data set to the list and the GUI
		this->dataList.append(newInfo);
		this->ui->dataCombo->addItem(ds->getName());
		
		// If this was the first data set, select it now
		if (this->ui->dataCombo->count() == 1 && this->ui->dataCombo->currentIndex() != 0)
			this->ui->dataCombo->setCurrentIndex(0);
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void HARDIConvolutionsPlugin::dataSetChanged(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions
	if (ds->getKind() == "discrete sphere")
	{
		int dataIndex = 0;

		// Loop through all input data sets
		for (QList<dataSetInfo>::iterator i = this->dataList.begin(); i != this->dataList.end(); ++i, ++dataIndex)
		{
			dataSetInfo info = (*i);

			// If we've found the data set, update its name
			if (ds == info.inDS)
				this->ui->dataCombo->setItemText(dataIndex, ds->getName());
		}

		// Load the new settings to the GUI
		this->loadDataInfo(this->ui->dataCombo->currentIndex());
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void HARDIConvolutionsPlugin::dataSetRemoved(data::DataSet * ds)
{
	if (!ds)
		return;

	// Discrete Sphere Functions
	if (ds->getKind() == "discrete sphere")
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
				this->ui->dataCombo->removeItem(dataIndex);
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
		this->loadDataInfo(this->ui->dataCombo->currentIndex());
	}
}


//-------------------------[ renameOutputDataSet ]-------------------------\\

void HARDIConvolutionsPlugin::renameOutputDataSet()
{
	// Get the information of the selected input data set
	int dsIndex = this->ui->dataCombo->currentIndex();

	if (dsIndex < 0 || dsIndex >= this->dataList.size())
		return;

	dataSetInfo info = this->dataList.at(dsIndex);

	if (info.selectedOutputDS == NULL || this->ui->outputOverwriteCombo->currentIndex() < 0)
		return;

	if (info.outDSs.indexOf(info.selectedOutputDS) != this->ui->outputOverwriteCombo->currentIndex())
		return;

	bool ok;

	// Ask the user for a new data set name
	QString newName = QInputDialog::getText(this->getGUI(), tr("HARDI Convolutions"), 
		tr("Enter new data set name..."), QLineEdit::Normal, info.selectedOutputDS->getName(), &ok);

	// If the user selected a valid name, update the data set
	if (ok && !newName.isEmpty())
	{
		info.selectedOutputDS->setName(newName);
		this->core()->data()->dataSetChanged(info.selectedOutputDS);
	}
}


//-------------------------[ selectOutputDataSet ]-------------------------\\

void HARDIConvolutionsPlugin::selectOutputDataSet(int index)
{
	// Get the information of the selected input data set
	int dsIndex = this->ui->dataCombo->currentIndex();

	if (dsIndex < 0 || dsIndex >= this->dataList.size())
		return;

	dataSetInfo info = this->dataList.at(dsIndex);

	if (index < 0 || index >= info.outDSs.size())
		return;

	// Update the selected data set pointer
	info.selectedOutputDS = info.outDSs.at(index);
	
	// Update the list
	this->dataList.replace(dsIndex, info);
}


//--------------------------[ changeOutputMethod ]-------------------------\\

void HARDIConvolutionsPlugin::changeOutputMethod()
{
	// Get the information of the selected input data set
	int dsIndex = this->ui->dataCombo->currentIndex();

	if (dsIndex < 0 || dsIndex >= this->dataList.size())
		return;

	dataSetInfo info = this->dataList.at(dsIndex);

	int outIndex = this->ui->outputOverwriteCombo->currentIndex();
	if (outIndex < 0 || outIndex >= info.outDSs.size())
		return;

	// Either clear the selected data set pointer...
	if (this->ui->outputNewRadio->isChecked())
		info.selectedOutputDS = NULL;
	// ...or update it to an existing data set
	else 
		info.selectedOutputDS = info.outDSs.at(outIndex);

	// Update the list
	this->dataList.replace(dsIndex, info);
}


//-----------------------------[ loadDataInfo ]----------------------------\\

void HARDIConvolutionsPlugin::loadDataInfo(int index)
{
	// Get the information of the selected input data set
	if (index < 0 || index >= this->dataList.size())
		return;

	dataSetInfo info = this->dataList.at(index);

	// Clear the combo box of output data sets
	this->ui->outputOverwriteCombo->clear();

	// If necessary, force-select the "New Data Set" radio button
	if (info.selectedOutputDS == NULL || !(info.outDSs.contains(info.selectedOutputDS)))
		this->ui->outputNewRadio->setChecked(true);

	// Add all output data sets to the combo box
	for (QList<data::DataSet *>::iterator i = info.outDSs.begin(); i != info.outDSs.end(); ++i)
	{
		this->ui->outputOverwriteCombo->addItem((*i)->getName());
	}

	// Select the previously selected output data set
	this->ui->outputOverwriteCombo->setCurrentIndex(info.outDSs.indexOf(info.selectedOutputDS));

	// Enable or disable controls based on the new settings
	this->enableControls();
}


//----------------------------[ enableControls ]---------------------------\\

void HARDIConvolutionsPlugin::enableControls()
{
	int typeIndex = this->ui->genTypeCombo->currentIndex();

	// Show only controls for the Duits kernels
	if (typeIndex == KernelGenerator::KT_Duits)
	{
		this->ui->paraD33Label->show();
		this->ui->paraD33Spin->show();
		this->ui->paraD44Label->show();
		this->ui->paraD44Spin->show();
		this->ui->paraTLabel->show();
		this->ui->paraTSpin->show();

		this->ui->paraSigmaLabel->hide();
		this->ui->paraSigmaSpin->hide();
		this->ui->paraKappaLabel->hide();
		this->ui->paraKappaSpin->hide();
	}
	// Show only controls for the Barmpoutis kernels
	else if (typeIndex == KernelGenerator::KT_Barmpoutis)
	{
		this->ui->paraD33Label->hide();
		this->ui->paraD33Spin->hide();
		this->ui->paraD44Label->hide();
		this->ui->paraD44Spin->hide();
		this->ui->paraTLabel->hide();
		this->ui->paraTSpin->hide();

		this->ui->paraSigmaLabel->show();
		this->ui->paraSigmaSpin->show();
		this->ui->paraKappaLabel->show();
		this->ui->paraKappaSpin->show();
	}

	// Disable the generate kernels button if there are no input data sets
	this->ui->genButton->setEnabled(this->ui->dataCombo->currentIndex() >= 0);

	// Enable or disable the masking threshold controls
	this->ui->thresholdPercSpin->setEnabled(this->ui->thresholdPercRadio->isChecked());
	this->ui->thresholdAbsSpin->setEnabled(this->ui->thresholdAbsRadio->isChecked());

	// Disable the Kernel Image Group controls if there are no groups
	bool enableKIGControls = this->ui->kigCombo->currentIndex() >= 0;
	this->ui->kigLabel->setEnabled(enableKIGControls);
	this->ui->kigCombo->setEnabled(enableKIGControls);
	this->ui->kernelSourceNIfTIRadio->setEnabled(enableKIGControls);

	// If one of the kernel source radio buttons is disabled, select the other one
	if (this->ui->kernelSourceNIfTIRadio->isEnabled() && !(this->ui->kernelSourceGenRadio->isEnabled()))
		this->ui->kernelSourceNIfTIRadio->setChecked(true);

	if (this->ui->kernelSourceGenRadio->isEnabled() && !(this->ui->kernelSourceNIfTIRadio->isEnabled()))
		this->ui->kernelSourceGenRadio->setChecked(true);

	// Only enable controls for the output data set if there are available data sets
	bool enableOutputListControls = this->ui->outputOverwriteCombo->currentIndex() >= 0;
	this->ui->outputOverwriteRadio->setEnabled(enableOutputListControls);
	this->ui->outputOverwriteLabel->setEnabled(enableOutputListControls);
	this->ui->outputOverwriteCombo->setEnabled(enableOutputListControls);
	this->ui->outputRenameButton->setEnabled(enableOutputListControls);
		
	// If one of the output method radio buttons is disabled, select the other one
	if (this->ui->outputNewRadio->isEnabled() && !(this->ui->outputOverwriteRadio->isEnabled()))
		this->ui->outputNewRadio->setChecked(true);

	if (this->ui->outputOverwriteRadio->isEnabled() && !(this->ui->outputNewRadio->isEnabled()))
		this->ui->outputOverwriteRadio->setChecked(true);

	// If we're directly generating kernels, only enable the apply button if the generate button is enabled
	if (this->ui->kernelSourceGenRadio->isEnabled() && this->ui->kernelSourceGenRadio->isChecked())
		this->ui->applyButton->setEnabled(this->ui->genButton->isEnabled());

	// If we're loading kernels from NIfTI files, only enable the apply button if we've got Kernel Image Groups
	else if (this->ui->kernelSourceNIfTIRadio->isEnabled() && this->ui->kernelSourceNIfTIRadio->isChecked())
		this->ui->applyButton->setEnabled(enableKIGControls);

	// If neither condition holds, we cannot apply convolution
	else
		this->ui->applyButton->setEnabled(false);
}


//--------------------------[ decomposeFileName ]--------------------------\\

void HARDIConvolutionsPlugin::decomposeFileName(QString inFN, QString & baseName, QString & absolutePath)
{
	// Set both parts equal to the input
	baseName = inFN;
	absolutePath = inFN;

	// Remove the extension
	if (baseName.endsWith(".kig"))
	{
		baseName = baseName.left(baseName.length() - QString(".kig").length());
	}

	// Find the last forward or backward slash
	int lastFSlash = baseName.lastIndexOf("/");
	int lastBSlash = baseName.lastIndexOf("\\");
	int lastSlash = (lastFSlash > lastBSlash) ? lastFSlash : lastBSlash;

	// If we found a slash, split the input at that point
	if (lastSlash >= 0)
	{
		baseName = baseName.right(baseName.length() - lastSlash - 1);
		absolutePath = absolutePath.left(lastSlash + 1);
	}
	// Otherwise, we've got a relative path
	else
	{
		absolutePath = "";
	}
}


//----------------------------[ setupGenerator ]---------------------------\\

void HARDIConvolutionsPlugin::setupGenerator(KernelGenerator * generator)
{
	// Set GUI options
	generator->SetKernelType((KernelGenerator::KernelType) this->ui->genTypeCombo->currentIndex());
	generator->SetNormalizeKernels(this->ui->genNormalizeCheck->isChecked());
	generator->SetD33(this->ui->paraD33Spin->value());
	generator->SetD44(this->ui->paraD44Spin->value());
	generator->SetT(this->ui->paraTSpin->value());
	generator->SetSigma(this->ui->paraSigmaSpin->value());
	generator->SetKappa(this->ui->paraKappaSpin->value());

	int kernelDims[6];

	// Select the kernel extents
	switch (this->ui->kernelSizeCombo->currentIndex())
	{
		// 1 x 1 x 1
		case 0:
			kernelDims[0] =  0;		kernelDims[1] = 0;
			kernelDims[2] =  0;		kernelDims[3] = 0;
			kernelDims[4] =  0;		kernelDims[5] = 0;
			break;

		// 3 x 3 x 3
		case 1:
			kernelDims[0] = -1;		kernelDims[1] = 1;
			kernelDims[2] = -1;		kernelDims[3] = 1;
			kernelDims[4] = -1;		kernelDims[5] = 1;
			break;

		// 5 x 5 x 5
		case 2:
			kernelDims[0] = -2;		kernelDims[1] = 2;
			kernelDims[2] = -2;		kernelDims[3] = 2;
			kernelDims[4] = -2;		kernelDims[5] = 2;
			break;

		// 7 x 7 x 7
		case 3:
			kernelDims[0] = -3;		kernelDims[1] = 3;
			kernelDims[2] = -3;		kernelDims[3] = 3;
			kernelDims[4] = -3;		kernelDims[5] = 3;
			break;

		// 9 x 9 x 9
		case 4:
			kernelDims[0] = -4;		kernelDims[1] = 4;
			kernelDims[2] = -4;		kernelDims[3] = 4;
			kernelDims[4] = -4;		kernelDims[5] = 4;
			break;

		// This should never happen
		default:
			kernelDims[0] = -1;		kernelDims[1] = 1;
			kernelDims[2] = -1;		kernelDims[3] = 1;
			kernelDims[4] = -1;		kernelDims[5] = 1;
			break;
	}

	generator->SetExtent(kernelDims);

	// Get the input image and its angles array
	int inDataIndex = this->ui->dataCombo->currentIndex();
	vtkImageData * inImage = (this->dataList[inDataIndex].inDS)->getVtkImageData();
	vtkDataArray * anglesArray = inImage->GetPointData()->GetArray("Spherical Directions");

	int numberOfAngles = anglesArray->GetNumberOfTuples();

	std::vector<double *> * directions = new std::vector<double *>;

	// Loop through all angles
	for (int i = 0; i < numberOfAngles; ++i)
	{
		double * v = new double[3];

		// Get the two angles (azimuth and zenith)
		double * angles = anglesArray->GetTuple2(i);

		// Compute the 3D coordinates for these angles on the unit sphere
		v[0] = sinf(angles[0]) * cosf(angles[1]);
		v[1] = sinf(angles[0]) * sinf(angles[1]);
		v[2] = cosf(angles[0]);

		// Add the 3D vector to the list
		directions->push_back(v);
	}

	generator->SetDirections(directions);

	// Add the spacing of the input image to the generator
	double spacing[3];
	inImage->GetSpacing(spacing);
	generator->SetSpacing(spacing);

	// If available, add the transformation matrix of the input data set to the generator
	vtkObject * obj = NULL;
	this->dataList[inDataIndex].inDS->getAttributes()->getAttribute("transformation matrix", obj);

	if (obj)
		generator->SetTransformationMatrix(vtkMatrix4x4::SafeDownCast(obj));
}


//---------------------------[ generateKernels ]---------------------------\\

void HARDIConvolutionsPlugin::generateKernels()
{
	// Get the index of the input data set
	int inDataIndex = this->ui->dataCombo->currentIndex();

	if (inDataIndex < 0 || inDataIndex >= this->ui->dataCombo->count())
	{
		this->core()->out()->showMessage("No input selected!", "HARDI Convolutions");
		return;
	}

	// Ask the user for an output file name 
	QString outFileName = QFileDialog::getSaveFileName(this->getGUI(), "Choose kernel group name...", 
		this->core()->getDataDirectory().absolutePath(), tr("Kernel Image Groups (*.kig)"));

	if (outFileName.isNull() || outFileName.isEmpty())
	{
		return;
	}

	// Create a file handle with the specified filename
	QFile outFile(outFileName);

	// Try to open the file
	if (!outFile.open(QIODevice::WriteOnly))
	{
		this->core()->out()->showMessage("Could not open output file!", "HARDI Convolutions");
		return;
	}

	// Decompose the full file path into a path and a base name
	QString baseName = "";
	QString absolutePath = "";
	this->decomposeFileName(outFileName, baseName, absolutePath);

	// Create and setup the kernel generator
	KernelGenerator * generator = KernelGenerator::New();
	this->setupGenerator(generator);
	generator->setFileNameAndPath(absolutePath, baseName);

	// Run the generator
	bool generateSuccess = generator->BuildKernelFamily();

	if (generateSuccess)
	{
		// Create a text stream
		QTextStream out(&outFile);

		QStringList * fileNames = generator->GetKernelImageFileNames();

		// Write all file names to the output ".kig" file
		for (int i = 0; i < fileNames->size(); ++i)
		{
			QString currentFileName = fileNames->at(i);
			out << currentFileName << "\n";
		}

		out.flush();

		// Delete the file names, and immediately load the file we just created.
		// This will ensure that the file names have the correct full path,
		// and that everything is added to the GUI correctly.

		delete fileNames;
		this->loadDataFromFile(outFileName);
	}

	// Close the output file
	outFile.close();
	
	generator->Delete();
}


//---------------------------[ applyConvolution ]--------------------------\\

void HARDIConvolutionsPlugin::applyConvolution()
{
	// Get the index of the input data set
	int inDataIndex = this->ui->dataCombo->currentIndex();

	if (inDataIndex < 0 || inDataIndex >= this->ui->dataCombo->count())
	{
		this->core()->out()->showMessage("No input selected!", "HARDI Convolutions");
		return;
	}

	// Get the information for this input data set
	dataSetInfo info = this->dataList.at(inDataIndex);

	// Get the input image
	vtkImageData * inImage = (this->dataList[inDataIndex].inDS)->getVtkImageData();

	// Create the convolution filter, and create a progress bar for this filter
	vtkHARDIConvolutionFilter * filter = vtkHARDIConvolutionFilter::New();
	this->core()->out()->createProgressBarForAlgorithm(filter, "HARDI Convolutions");

	// Set the options of the filter
	if (this->ui->thresholdAbsRadio->isChecked())
		filter->setThreshold(this->ui->thresholdAbsSpin->value(), false);
	else
		filter->setThreshold((double) this->ui->thresholdPercSpin->value() / 100.0, true);

	filter->setNumberOfIterations(this->ui->iterSpin->value());
	filter->SetInput(inImage);

	KernelGenerator * generator = NULL;

	// Either create and setup a kernel generator...
	if (	this->ui->kernelSourceGenRadio->isChecked() &&
			this->ui->kernelSourceGenRadio->isEnabled() )
	{
		generator = KernelGenerator::New();
		this->setupGenerator(generator);
		filter->setGenerator(generator);
	}
	// ...or set the list of NIfTI file name
	else if (	this->ui->kernelSourceNIfTIRadio->isChecked()	&& 
				this->ui->kernelSourceNIfTIRadio->isEnabled()	&&
				this->ui->kigCombo->currentIndex() >= 0			&& 
				this->ui->kigCombo->currentIndex() < this->kernelImageGroups.size() )
	{
		QStringList * selectedKIG = this->kernelImageGroups.at(this->ui->kigCombo->currentIndex());
		filter->setNIfTIFileNames(selectedKIG);
	}

	// Run the filter
	filter->Update();

	vtkImageData * outImage = filter->GetOutput();

	if (!(outImage))
		return;

	// Create a new data set
	if (this->ui->outputNewRadio->isChecked())
	{
		QString outName = (this->dataList[inDataIndex].inDS)->getName() + " [CONV]";
		data::DataSet * newDS = new data::DataSet(outName, "discrete sphere", outImage);
		info.outDSs.append(newDS);
		this->dataList.replace(inDataIndex, info);

		// Select the new data set 
		if (info.selectedOutputDS == NULL)
		{
			info.selectedOutputDS = newDS;
			this->dataList.replace(inDataIndex, info);
			this->ui->outputOverwriteCombo->setCurrentIndex(0);
		}

		// Copy the transformation matrix, and add the data set to the data manager
		newDS->getAttributes()->copyTransformationMatrix(this->dataList[inDataIndex].inDS);
		this->core()->data()->addDataSet(newDS);
		this->loadDataInfo(inDataIndex);
	}
	// Otherwise, update the existing data set
	else
	{
		info.selectedOutputDS->updateData(outImage);
		info.selectedOutputDS->getAttributes()->copyTransformationMatrix(this->dataList[inDataIndex].inDS);
		this->core()->data()->dataSetChanged(info.selectedOutputDS);
	}

	// Delete temporary objects
	this->core()->out()->deleteProgressBarForAlgorithm(filter);
	filter->Delete();

	if (generator)
		generator->Delete();
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libHARDIConvolutionsPlugin, bmia::HARDIConvolutionsPlugin)
