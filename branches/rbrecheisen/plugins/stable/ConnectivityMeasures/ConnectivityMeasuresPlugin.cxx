/*
 * ConnectivityMeasuresPlugin.cxx
 *
 * 2011-05-11	Evert van Aart
 * - Version 1.0.0.
 * - First version.
 *
 * 2011-06-06	Evert van Aart
 * - Version 1.1.0.
 * - Increased stability.
 * - Added an option for applying the ranking measure value to each fiber point
 *   (thus getting a single color for each fiber).
 *
 * 2011-08-22	Evert van Aart
 * - Version 1.1.1.
 * - Fixed a crash in the ranking filter.
 * - Added progress bars for all filters.
 *
 */


/** Includes */

#include "ConnectivityMeasuresPlugin.h"
#include "vtkFiberRankingFilter.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

ConnectivityMeasuresPlugin::ConnectivityMeasuresPlugin() : Plugin("Connectivity Measures")
{
	// Create the GUI of the widget
	this->widget = new QWidget();
	this->ui = new Ui::ConnectivityMeasuresForm();
	this->ui->setupUi(this->widget);

	// Connect the controls
	this->connectControls(true);

	// Set pointer to NULL
	this->ignoreDataSet = NULL;
}


//---------------------------------[ init ]--------------------------------\\

void ConnectivityMeasuresPlugin::init()
{

}


//------------------------------[ Destructor ]-----------------------------\\

ConnectivityMeasuresPlugin::~ConnectivityMeasuresPlugin()
{
	// Unload the GUI
	delete this->widget;

	// Clear the list of DTI images
	this->dtiImages.clear();

	// Loop through the fiber information list
	for (QList<FiberInformation>::iterator i = this->infoList.begin(); i != this->infoList.end(); ++i)
	{
		// Get the current fiber information
		FiberInformation currentInfo = (*i);

		// Delete the output data set
		if (currentInfo.outDS)
			this->core()->data()->removeDataSet(currentInfo.outDS);

		// Delete the connectivity measure filter
		if (currentInfo.cmFilter)
		{
			this->core()->out()->deleteProgressBarForAlgorithm(currentInfo.cmFilter);
			currentInfo.cmFilter->Delete();
		}

		// Delete the ranking filter
		if (currentInfo.rankingFilter)
		{
			this->core()->out()->deleteProgressBarForAlgorithm(currentInfo.rankingFilter);
			currentInfo.rankingFilter->Delete();
		}
	}

	// Clear the list of fiber information
	this->infoList.clear();
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * ConnectivityMeasuresPlugin::getGUI()
{
	// Return the GUI widget
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void ConnectivityMeasuresPlugin::dataSetAdded(data::DataSet * ds)
{
	if (!ds)
		return;

	// Fibers
	if (ds->getKind() == "fibers")
	{
		if (ds->getVtkPolyData() == NULL)
			return;

		// If the data set has the "hasCM" attribute, it was generated by this
		// plugin, and we do not want to add it to the input data sets.

		if (ds->getAttributes()->hasIntAttribute("hasCM"))
			return;

		// Create a new information object
		FiberInformation newInfo;

		// Use the currently selected auxiliary image
		int auxImageIndex = this->ui->measureDTIImageCombo->currentIndex();

		if (auxImageIndex < 0 || auxImageIndex >= this->dtiImages.size())
			newInfo.auxImage = NULL;
		else
			newInfo.auxImage = this->dtiImages.at(auxImageIndex);

		// Initialize the rest of the fiber information object
		newInfo.doNormalize		= true;
		newInfo.inDS			= ds;
		newInfo.measure			= CM_GeodesicConnectionStrength;
		newInfo.numberOfFibers	= 1;
		newInfo.outDS			= NULL;
		newInfo.percentage		= 10;
		newInfo.rankBy			= RM_FiberEnd;
		newInfo.ranking			= RO_AllFibers;
		newInfo.useSingleValue	= true;
		newInfo.cmFilter		= NULL;
		newInfo.rankingFilter	= NULL;

		// Add the data set to the list and to the GUI
		this->infoList.append(newInfo);
		this->ui->inputCombo->addItem(ds->getName());
		return;
	}

	// DTI tensor volume
	if (ds->getKind() == "DTI")
	{
		if (!(ds->getVtkImageData()))
			return;

		// Add the image to the list and to the GUI
		this->dtiImages.append(ds);
		this->ui->measureDTIImageCombo->addItem(ds->getName());

		// If this is the first DTI image, select it
		if (this->ui->measureDTIImageCombo->count() == 1)
		{
			this->ui->measureDTIImageCombo->setCurrentIndex(0);

			// Check if there are any fiber sets that do not yet have an auxiliary 
			// image set, and if so, use the new DTI data set. 

			for (QList<FiberInformation>::iterator i = this->infoList.begin(); i != this->infoList.end(); ++i)
			{
				if ((*i).auxImage == NULL)
					(*i).auxImage = ds;
			}

			// Update the GUI
			this->settingsToGUI(this->ui->inputCombo->currentIndex());
		}

		return;
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void ConnectivityMeasuresPlugin::dataSetChanged(data::DataSet * ds)
{
	if (!ds)
		return;

	// Fibers
	if (ds->getKind() == "fibers")
	{
		// If we've just changed this data set in the "update" function, 
		// we ignore it here (all we changed was the "isVisible" attribute).

		if (this->ignoreDataSet == ds)
		{
			this->ignoreDataSet = NULL;
			return;
		}

		// Check if the data set exists
		int dsIndex = this->findInputDataSet(ds);

		if (dsIndex == -1)
			return;

		// Update the data set name in the GUI
		this->ui->inputCombo->setItemText(dsIndex, ds->getName());

		// Get the fiber information structure for this data set
		FiberInformation currentInfo = this->infoList.at(dsIndex);
		
		// If we've already got a CM filter for this data...
		if (currentInfo.cmFilter)
		{
			// ...update its input...
			currentInfo.cmFilter->SetInput(ds->getVtkPolyData());
			currentInfo.cmFilter->Modified();

			// ...and render the scene
			this->core()->render();
		}

		return;
	}

	// DTI tensor volume
	if (ds->getKind() == "DTI")
	{
		// Get the index of the DTI data set
		int dsIndex = this->dtiImages.indexOf(ds);

		if (dsIndex == -1)
			return;

		// Update the data set name
		this->ui->measureDTIImageCombo->setItemText(dsIndex, ds->getName());

		return;
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void ConnectivityMeasuresPlugin::dataSetRemoved(data::DataSet * ds)
{
	if (!ds)
		return;

	// Fibers
	if (ds->getKind() == "fibers")
	{
		// Check if the data set exists
		int dsIndex = this->findInputDataSet(ds);

		if (dsIndex == -1)
			return;

		// Get the current fiber information object
		FiberInformation currentInfo = this->infoList.at(dsIndex);

		// Remove the output data set
		if (currentInfo.outDS)
			this->core()->data()->removeDataSet(currentInfo.outDS);

		// Delete the connectivity measure filter
		if (currentInfo.cmFilter)
		{
			this->core()->out()->deleteProgressBarForAlgorithm(currentInfo.cmFilter);
			currentInfo.cmFilter->Delete();
		}

		// Remove the ranking filter
		if (currentInfo.rankingFilter)
		{
			this->core()->out()->deleteProgressBarForAlgorithm(currentInfo.rankingFilter);
			currentInfo.rankingFilter->Delete();
		}

		// Remove the data set from the list and from the GUI
		this->infoList.removeAt(dsIndex);
		this->ui->inputCombo->removeItem(dsIndex);

		return;
	}

	// DTI tensor volume
	if (ds->getKind() == "DTI")
	{
		// Get the index of the DTI data set
		int dsIndex = this->dtiImages.indexOf(ds);

		if (dsIndex == -1)
			return;

		int index = 0;

		// Loop through all input fiber data sets
		for (QList<FiberInformation>::iterator i = this->infoList.begin(); i != this->infoList.end(); ++i, ++index)
		{
			// Get the information for the current fiber set
			FiberInformation currentInfo = (*i);

			// Check if this fiber data set used the DTI image
			if (currentInfo.auxImage == ds)
			{
				// If so, set the image pointer to NULL
				currentInfo.auxImage = NULL;

				// Replace the information in the list
				this->infoList.replace(index, currentInfo);

				// If this is the currently selected fiber data set, update the GUI
				if (this->ui->inputCombo->currentIndex() == index)
					this->settingsToGUI(index);
			}
		}

		// Remove the DTI image from the list and from the GUI
		this->dtiImages.removeAt(dsIndex);
		this->ui->measureDTIImageCombo->removeItem(dsIndex);
	}
}


//---------------------------[ findInputDataSet ]--------------------------\\

int ConnectivityMeasuresPlugin::findInputDataSet(data::DataSet * ds)
{
	int index = 0;

	// Loop through all input fiber data sets
	for (QList<FiberInformation>::iterator i = this->infoList.begin(); i != this->infoList.end(); ++i, ++index)
	{
		// Return the index if we've found the target data set
		if ((*i).inDS == ds)
			return index;
	}

	return -1;
}


//--------------------------[ changeInputFibers ]--------------------------\\

void ConnectivityMeasuresPlugin::changeInputFibers(int index)
{
	// Disconnect controls
	this->connectControls(false);

	// Load settings from new fibers
	this->settingsToGUI(index);

	// Enable/disable controls
	this->enableControls();

	// Reconnect controls
	this->connectControls(true);
}


//----------------------------[ settingsToGUI ]----------------------------\\

void ConnectivityMeasuresPlugin::settingsToGUI(int index)
{
	// Check if the index is within the correct range
	if (index < 0 || index >= this->infoList.size())
		return;

	// /get the information for the current set of fibers
	FiberInformation currentInfo = this->infoList.at(index);

	// If we've already got an output data set, copy its name to the line edit box
	if (currentInfo.outDS)
		this->ui->outputLineEdit->setText(currentInfo.outDS->getName());

	// Otherwise, use the input data set name, appended with "[CM]"
	else
		this->ui->outputLineEdit->setText(currentInfo.inDS->getName() + " [CM]");

	// Select the measure
	this->ui->measureCombo->setCurrentIndex((int) currentInfo.measure);
	
	// Select the auxiliary image
	int auxImageIndex = this->dtiImages.indexOf(currentInfo.auxImage);
	this->ui->measureDTIImageCombo->setCurrentIndex(auxImageIndex);

	// Set the rest of the options
	this->ui->normalizeCheck->setChecked(currentInfo.doNormalize);
	this->ui->rankByEndRadio->setChecked(currentInfo.rankBy == RM_FiberEnd);
	this->ui->rankByAverageRadio->setChecked(currentInfo.rankBy == RM_Average);
	this->ui->rankOutCombo->setCurrentIndex((int) currentInfo.ranking);
	this->ui->rankPercSlide->setValue(currentInfo.percentage);
	this->ui->rankPercSpin->setValue(currentInfo.percentage);
	this->ui->rankNumberSpin->setValue(currentInfo.numberOfFibers);
	this->ui->singleValueCheck->setChecked(currentInfo.useSingleValue);
}


//----------------------------[ enableControls ]---------------------------\\

void ConnectivityMeasuresPlugin::enableControls()
{
	// We disable everything if no data set has been selected
	bool globalEnable = (this->ui->inputCombo->currentIndex() >= 0);

	// Enable or disable controls based on the global enable value
	this->ui->outputLabel->setEnabled(globalEnable);
	this->ui->outputLineEdit->setEnabled(globalEnable);
	this->ui->measureLabel->setEnabled(globalEnable);
	this->ui->measureCombo->setEnabled(globalEnable);
	this->ui->normalizeCheck->setEnabled(globalEnable);
	this->ui->rankByLabel->setEnabled(globalEnable);
	this->ui->rankByAverageRadio->setEnabled(globalEnable);
	this->ui->rankByEndRadio->setEnabled(globalEnable);
	this->ui->rankOutLabel->setEnabled(globalEnable);
	this->ui->rankOutCombo->setEnabled(globalEnable);
	this->ui->singleValueCheck->setEnabled(globalEnable);
	this->ui->updateButton->setEnabled(globalEnable);

	// Only enable the options for auxiliary images if we've selected a measure 
	// that requires an auxiliary image.

	bool enableDTIImages = globalEnable && 
		((ConnectivityMeasure) this->ui->measureCombo->currentIndex() == CM_GeodesicConnectionStrength);

	this->ui->measureDTIImageLabel->setEnabled(enableDTIImages);
	this->ui->measureDTIImageCombo->setEnabled(enableDTIImages);

	// Enable the options for the number of fibers...
	bool enableNumberOfFibers = globalEnable && 
		((RankingOutput) this->ui->rankOutCombo->currentIndex() == RO_BestNumber);

	this->ui->rankNumberLabel->setEnabled(enableNumberOfFibers);
	this->ui->rankNumberSpin->setEnabled(enableNumberOfFibers);

	// ...or for the percentage of fibers, depending on which option was selected
	bool enablePercentage = globalEnable && 
		((RankingOutput) this->ui->rankOutCombo->currentIndex() == RO_BestPercentage);

	this->ui->rankPercLabel->setEnabled(enablePercentage);
	this->ui->rankPercSpin->setEnabled(enablePercentage);
	this->ui->rankPercSlide->setEnabled(enablePercentage);
}


//---------------------------[ connectControls ]---------------------------\\

void ConnectivityMeasuresPlugin::connectControls(bool doConnect)
{
	if (doConnect)
	{
		connect(this->ui->inputCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(changeInputFibers(int)	));
		connect(this->ui->measureCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(enableControls()			));
		connect(this->ui->rankOutCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(enableControls()			));
		connect(this->ui->updateButton,		SIGNAL(clicked()),					this, SLOT(update()					));
	}
	else
	{
		disconnect(this->ui->inputCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(changeInputFibers(int)	));
		disconnect(this->ui->measureCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(enableControls()			));
		disconnect(this->ui->rankOutCombo,	SIGNAL(currentIndexChanged(int)),	this, SLOT(enableControls()			));
		disconnect(this->ui->updateButton,	SIGNAL(clicked()),					this, SLOT(update()					));
	}
}


//--------------------------------[ update ]-------------------------------\\

void ConnectivityMeasuresPlugin::update()
{
	if (this->ui->inputCombo->currentIndex() < 0 || this->ui->inputCombo->currentIndex() >= this->infoList.size())
		return;

	// Get the information for the current set of fibers
	FiberInformation currentInfo = this->infoList.at(this->ui->inputCombo->currentIndex());

	// Get the index of the auxiliary image
	int auxImageIndex = this->ui->measureDTIImageCombo->currentIndex();

	// If the index is out of range, set the pointer to NULL
	if (auxImageIndex < 0 || auxImageIndex >= this->infoList.size())
		currentInfo.auxImage = NULL;
	// Otherwise, use the data set pointer at the target index
	else
		currentInfo.auxImage = this->dtiImages.at(auxImageIndex);

	// The auxiliary image is only required for some measures
	bool auxImageIsRequired = (ConnectivityMeasure) this->ui->measureCombo->currentIndex() == CM_GeodesicConnectionStrength;

	// If we need an auxiliary image, but we do not have it, display an error
	if (auxImageIsRequired && currentInfo.auxImage == NULL)
	{
		this->core()->out()->showMessage("Cannot compute Connectivity Measure: Input image has not been set.", "Connectivity Measures");
		return;
	}
		
	// Copy options from the GUI to the fiber information object
	currentInfo.doNormalize		= this->ui->normalizeCheck->isChecked();
	currentInfo.measure			= (ConnectivityMeasure) this->ui->measureCombo->currentIndex();
	currentInfo.rankBy			= (this->ui->rankByEndRadio->isChecked()) ? RM_FiberEnd : RM_Average;
	currentInfo.numberOfFibers	= this->ui->rankNumberSpin->value();
	currentInfo.percentage		= this->ui->rankPercSpin->value();
	currentInfo.ranking			= (RankingOutput) this->ui->rankOutCombo->currentIndex();
	currentInfo.useSingleValue	= this->ui->singleValueCheck->isChecked();

	// If the line edit is empty...
	if (this->ui->outputLineEdit->text().isEmpty())
	{
		// ...fill it using either the output data set name...
		if (currentInfo.outDS)
			this->ui->outputLineEdit->setText(currentInfo.outDS->getName());
		// ...or the input data set name appended with "[CM]"
		else
			this->ui->outputLineEdit->setText(currentInfo.inDS->getName() + " [CM]");
	}

	// Delete existing connectivity measure filter
	if (currentInfo.cmFilter)
	{
		this->core()->out()->deleteProgressBarForAlgorithm(currentInfo.cmFilter);
		currentInfo.cmFilter->Delete();
	}

	// One case for each connectivity measure
	switch(currentInfo.measure)
	{
		// Geodesic Connection Strength
		case CM_GeodesicConnectionStrength:
			currentInfo.cmFilter = vtkGeodesicConnectionStrengthFilter::New();
			break;

		// Unknown measure
		default:
			this->core()->out()->showMessage("Invalid measure.", "Connectivity Measures");
			return;
	}

	// Create a progress bar for the filter
	this->core()->out()->createProgressBarForAlgorithm(currentInfo.cmFilter, "Connectivity Measures");

	// Configure and run the CM filter
	currentInfo.cmFilter->SetInput(currentInfo.inDS->getVtkPolyData());
	currentInfo.cmFilter->setNormalize(currentInfo.doNormalize);
	currentInfo.cmFilter->setAuxImage(currentInfo.auxImage->getVtkImageData());
	currentInfo.cmFilter->Update();

	// Get the output of the filter
	vtkPolyData * currentOutput = currentInfo.cmFilter->GetOutput();

	// Delete existing ranking filter
	if (currentInfo.rankingFilter)
	{
		this->core()->out()->deleteProgressBarForAlgorithm(currentInfo.rankingFilter);
		currentInfo.rankingFilter->Delete();
	}

	// Create, configure and run the ranking filter
	currentInfo.rankingFilter = vtkFiberRankingFilter::New();
	currentInfo.rankingFilter->setMeasure(currentInfo.rankBy);
	currentInfo.rankingFilter->setNumberOfFibers(currentInfo.numberOfFibers);
	currentInfo.rankingFilter->setOutputMethod(currentInfo.ranking);
	currentInfo.rankingFilter->setPercentage(currentInfo.percentage);
	currentInfo.rankingFilter->setUseSingleValue(currentInfo.useSingleValue);
	currentInfo.rankingFilter->SetInput(currentOutput);
	currentInfo.rankingFilter->Update();

	// Create a progress bar for the ranking filter
	this->core()->out()->createProgressBarForAlgorithm(currentInfo.rankingFilter, "Connectivity Measures");

	// Get the output of the ranking filter
	currentOutput = currentInfo.rankingFilter->GetOutput();

	// If we've already got an output data set, update it
	if (currentInfo.outDS)
	{
		currentInfo.outDS->updateData(currentOutput);
		currentInfo.outDS->setName(this->ui->outputLineEdit->text());

		// Fibers should be visible, and the visualization pipeline should be updated
		currentInfo.outDS->getAttributes()->addAttribute("isVisible", 1.0);
		currentInfo.outDS->getAttributes()->addAttribute("updatePipeline", 1.0);

		// Copy the transformation matrix to the output
		currentInfo.outDS->getAttributes()->copyTransformationMatrix(currentInfo.inDS);

		this->core()->data()->dataSetChanged(currentInfo.outDS);
	}

	// Otherwise, create a new data set
	else
	{
		currentInfo.outDS = new data::DataSet(this->ui->outputLineEdit->text(), "fibers", currentOutput);

		// Fibers should be visible, and the visualization pipeline should be updated
		currentInfo.outDS->getAttributes()->addAttribute("isVisible", 1.0);
		currentInfo.outDS->getAttributes()->addAttribute("updatePipeline", 1.0);

		// We add this attribute to make sure that output data sets are not added to the input data sets
		currentInfo.outDS->getAttributes()->addAttribute("hasCM", 1);

		// Copy the transformation matrix to the output
		currentInfo.outDS->getAttributes()->copyTransformationMatrix(currentInfo.inDS);

		this->core()->data()->addDataSet(currentInfo.outDS);
	}

	// Ensure that the "dataSetChanged" function does not actually do anything
	// when we 'change' the input data set (since we only add one attribute).

	this->ignoreDataSet = currentInfo.inDS;

	// Hide the input data set
	currentInfo.inDS->getAttributes()->addAttribute("isVisible", -1.0);
	this->core()->data()->dataSetChanged(currentInfo.inDS);

	// Update the information list
	this->infoList.replace(this->ui->inputCombo->currentIndex(), currentInfo);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libConnectivityMeasuresPlugin, bmia::ConnectivityMeasuresPlugin)
