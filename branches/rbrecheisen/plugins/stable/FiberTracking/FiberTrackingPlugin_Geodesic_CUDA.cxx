/*
 * FiberTrackingPlugin_Geodesic_CUDA.cxx
 *
 * 2011-05-17	Evert van Aart
 * - First Version. Created to reduce the size of "fiberTrackingPlugin.cxx", as well
 *   as to make a clear distinction between the supported fiber tracking methods.
 *
 * 2011-07-06	Evert van Aart
 * - First version for the CUDA-enabled version. Small diferrences in the setup
 *   of the GUI and the input parameters of the filter.
 *
 */


/** Includes */

#include "FiberTrackingPlugin.h"
#include "vtkFiberTrackingGeodesicFilter_CUDA.h"


namespace bmia {


//-------------------------[ setupGUIForGeodesics ]------------------------\\

void FiberTrackingPlugin::setupGUIForGeodesics()
{
	// Do nothing if the geodesic GUI already exists
	if (this->geodesicGUI)
		return;

	// Clear the toolbox (to remove controls for other methods)
	this->clearToolbox();

	// Enable all controls that may have been disabled for other methods
	this->enableAllControls();

	// Create a new geodesic GUI information object
	this->geodesicGUI = new geodesicGUIElements;

	// Initialize structure to NULL
	memset(this->geodesicGUI, 0, sizeof(geodesicGUIElements));

	// Additional Angles Options - Pattern
	QLabel * anglesPatternLabel = new QLabel("Pattern   ");
		anglesPatternLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	QComboBox * anglesPatternCombo = new QComboBox;
		anglesPatternCombo->addItem("Cone around MEV");
		anglesPatternCombo->addItem("Simple Sphere");
		anglesPatternCombo->addItem("Icosahedron");
		anglesPatternCombo->setCurrentIndex(0);
	QHBoxLayout * anglesPatternHLayout = new QHBoxLayout;
		anglesPatternHLayout->addWidget(anglesPatternLabel);
		anglesPatternHLayout->addWidget(anglesPatternCombo);

	// Additional Angles Options - Cone
	QLabel * anglesConeNumberLabel = new QLabel("Number of additional angles");
	QSpinBox * anglesConeNumberSpin = new QSpinBox;
		anglesConeNumberSpin->setMinimum(1);
		anglesConeNumberSpin->setMaximum(512);
		anglesConeNumberSpin->setSingleStep(1);
		anglesConeNumberSpin->setValue(8);
	QLabel * anglesConeWidthLabel = new QLabel("Cone width");
	QDoubleSpinBox * anglesConeWidthSpin = new QDoubleSpinBox;
		anglesConeWidthSpin->setMinimum(0.0);
		anglesConeWidthSpin->setMaximum(1.0);
		anglesConeWidthSpin->setSingleStep(0.01);
		anglesConeWidthSpin->setValue(0.1);
		anglesConeWidthSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

	// Additional Angles Options - Simple Sphere
	QLabel * anglesSphereNumberPLabel = new QLabel("Number of angles (phi)");
	QSpinBox * anglesSphereNumberPSpin = new QSpinBox;
		anglesSphereNumberPSpin->setMinimum(1);
		anglesSphereNumberPSpin->setMaximum(512);
		anglesSphereNumberPSpin->setSingleStep(1);
		anglesSphereNumberPSpin->setValue(8);
	QLabel * anglesSphereNumberTLabel = new QLabel("Number of angles (theta)");
	QSpinBox * anglesSphereNumberTSpin = new QSpinBox;
		anglesSphereNumberTSpin->setMinimum(1);
		anglesSphereNumberTSpin->setMaximum(512);
		anglesSphereNumberTSpin->setSingleStep(1);
		anglesSphereNumberTSpin->setValue(8);
		anglesSphereNumberTSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

	// Additional Angles Options - Icosahedron
	QLabel * anglesIcoTessLabel = new QLabel("Tess. Order");
	QComboBox * anglesIcoTessCombo = new QComboBox;
		anglesIcoTessCombo->addItem("1 (12 dirs/seed)");
		anglesIcoTessCombo->addItem("2 (42 dirs/seed)");
		anglesIcoTessCombo->addItem("3 (162 dirs/seed)");
		anglesIcoTessCombo->addItem("4 (642 dirs/seed)");
		anglesIcoTessCombo->addItem("5 (2562 dirs/seed)");
		anglesIcoTessCombo->addItem("6 (10242 dirs/seed)");
		anglesIcoTessCombo->setCurrentIndex(2);

	// Additional Angles Options - Grid Layout
	QGridLayout * angleOptionsGLayout = new QGridLayout;
		angleOptionsGLayout->addWidget(anglesConeNumberLabel,		0, 0);
		angleOptionsGLayout->addWidget(anglesConeNumberSpin,		0, 1);
		angleOptionsGLayout->addWidget(anglesConeWidthLabel,		1, 0);
		angleOptionsGLayout->addWidget(anglesConeWidthSpin,			1, 1);
		angleOptionsGLayout->addWidget(anglesSphereNumberPLabel,	2, 0);
		angleOptionsGLayout->addWidget(anglesSphereNumberPSpin,		2, 1);
		angleOptionsGLayout->addWidget(anglesSphereNumberTLabel,	3, 0);
		angleOptionsGLayout->addWidget(anglesSphereNumberTSpin,		3, 1);
		angleOptionsGLayout->addWidget(anglesIcoTessLabel,			4, 0);
		angleOptionsGLayout->addWidget(anglesIcoTessCombo,			4, 1);
		angleOptionsGLayout->itemAtPosition(2, 0)->widget()->setVisible(false);
		angleOptionsGLayout->itemAtPosition(2, 1)->widget()->setVisible(false);
		angleOptionsGLayout->itemAtPosition(3, 0)->widget()->setVisible(false);
		angleOptionsGLayout->itemAtPosition(3, 1)->widget()->setVisible(false);
		angleOptionsGLayout->itemAtPosition(4, 0)->widget()->setVisible(false);
		angleOptionsGLayout->itemAtPosition(4, 1)->widget()->setVisible(false);

	// Group for all the additional angles options
	QVBoxLayout * anglesGroupVLayout = new QVBoxLayout;
		anglesGroupVLayout->addLayout(anglesPatternHLayout);
		anglesGroupVLayout->addLayout(angleOptionsGLayout);
	QGroupBox * anglesGroup = new QGroupBox("Additional Shooting Angles");
		anglesGroup->setCheckable(true);
		anglesGroup->setChecked(false);
		anglesGroup->setLayout(anglesGroupVLayout);

	// Optional stopping criteria
	QCheckBox * stoppingLengthCheck = new QCheckBox("Maximum Fiber Length");
	QCheckBox * stoppingAngleCheck = new QCheckBox("Maximum Fiber Angles");
	QCheckBox * stoppingScalarCheck = new QCheckBox("Scalar Threshold");
	QVBoxLayout * stoppingGroupVLayout = new QVBoxLayout;
		stoppingGroupVLayout->addWidget(stoppingLengthCheck);
		stoppingGroupVLayout->addWidget(stoppingAngleCheck);
		stoppingGroupVLayout->addWidget(stoppingScalarCheck);
	QGroupBox * stoppingGroup = new QGroupBox("Optional Stopping Criteria");
		stoppingGroup->setLayout(stoppingGroupVLayout);

	// ODE Solver options
	QLabel * odeLabel = new QLabel("ODE Solver   ");
		odeLabel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		odeLabel->setEnabled(false);
	QComboBox * odeCombo = new QComboBox;
		odeCombo->addItem("Euler");
		odeCombo->addItem("RK2 (Heun's Method)");
		odeCombo->addItem("RK2 (Mid-Point Method)");
		odeCombo->addItem("RK4");
		odeCombo->setCurrentIndex(1);
		odeCombo->setEnabled(false);
	QHBoxLayout * odeHLayout = new QHBoxLayout;
		odeHLayout->addWidget(odeLabel);
		odeHLayout->addWidget(odeCombo);

	// Spacer for the tracking options page
	QSpacerItem * trackingSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

	// Main vertical layout for the tracking options
	QVBoxLayout * trackingVLayout = new QVBoxLayout;
		trackingVLayout->addWidget(anglesGroup);
		trackingVLayout->addWidget(stoppingGroup);
		trackingVLayout->addLayout(odeHLayout);
		trackingVLayout->addSpacerItem(trackingSpacer);

	// Add the new controls to a widget
	this->geodesicGUI->geodesicTrackingWidget = new QWidget;
	this->geodesicGUI->geodesicTrackingWidget->setLayout(trackingVLayout);

	// Add the widget as a new page of the tool
	this->ui->fiberTrackingToolbox->addItem(this->geodesicGUI->geodesicTrackingWidget, "Additional Tracking Options");

	// Pre-Processing Options
	QCheckBox * ppEnableCheck = new QCheckBox("Enable Tensor Pre-Processing");
		ppEnableCheck->setChecked(true);
	QLabel * ppGainLabel = new QLabel("Tensor Gain");
	QSpinBox * ppGainSpin = new QSpinBox;
		ppGainSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		ppGainSpin->setMinimum(1);
		ppGainSpin->setMaximum(9999);
		ppGainSpin->setSingleStep(50);
		ppGainSpin->setValue(1);
	QHBoxLayout * ppGainHLayout = new QHBoxLayout;
		ppGainHLayout->addWidget(ppGainLabel);
		ppGainHLayout->addWidget(ppGainSpin);
	QLabel * ppSharpenLabel = new QLabel("Sharpening");
	QComboBox * ppSharpenCombo = new QComboBox;
		ppSharpenCombo->addItem("None");
		ppSharpenCombo->addItem("Tensor Exponentiation");
		ppSharpenCombo->addItem("Division by Trace");
		ppSharpenCombo->addItem("Exponentiation + Division");
	QHBoxLayout * ppSharpenHLayout = new QHBoxLayout;
		ppSharpenHLayout->addWidget(ppSharpenLabel);
		ppSharpenHLayout->addWidget(ppSharpenCombo);
	QLabel * ppThresholdLabel = new QLabel("Scalar Threshold for Sharpening");
		ppThresholdLabel->setEnabled(false);
	QDoubleSpinBox * ppThresholdSpin = new QDoubleSpinBox;
		ppThresholdSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
		ppThresholdSpin->setMinimum(0.0);
		ppThresholdSpin->setMaximum(1.0);
		ppThresholdSpin->setSingleStep(0.01);
		ppThresholdSpin->setValue(0.1);
		ppThresholdSpin->setEnabled(false);
	QLabel * ppExpLabel = new QLabel("Tensor Exponent");
		ppExpLabel->setEnabled(false);
	QSpinBox * ppExpSpin = new QSpinBox;
		ppExpSpin->setMinimum(1);
		ppExpSpin->setMaximum(10);
		ppExpSpin->setSingleStep(1);
		ppExpSpin->setValue(2);
		ppExpSpin->setEnabled(false);
	QGridLayout * ppSharpenGLayout = new QGridLayout;
		ppSharpenGLayout->addWidget(ppThresholdLabel, 0, 0);
		ppSharpenGLayout->addWidget(ppThresholdSpin, 0, 1);
		ppSharpenGLayout->addWidget(ppExpLabel, 1, 0);
		ppSharpenGLayout->addWidget(ppExpSpin, 1, 1);

	// Main spacer for the pre-processing page
	QSpacerItem * ppSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

	// Main vertical layout for the pre-processing options
	QVBoxLayout * ppVLayout = new QVBoxLayout;
		ppVLayout->addWidget(ppEnableCheck);
		ppVLayout->addLayout(ppGainHLayout);
		ppVLayout->addLayout(ppSharpenHLayout);
		ppVLayout->addLayout(ppSharpenGLayout);
		ppVLayout->addSpacerItem(ppSpacer);

	// Add the new controls to a widget
	this->geodesicGUI->geodesicPPWidget = new QWidget;
	this->geodesicGUI->geodesicPPWidget->setLayout(ppVLayout);

	// Add the widget as a new page of the tool
	this->ui->fiberTrackingToolbox->addItem(this->geodesicGUI->geodesicPPWidget, "Preprocessing Options");

	// CUDA Acceleration
	QLabel * cudaLabel = new QLabel("CUDA Enabled");

	// Create a spacer and a main layout for the performance page
	QSpacerItem * perfSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);
	QVBoxLayout * perfMainLayout = new QVBoxLayout;
		perfMainLayout->addWidget(cudaLabel);
		perfMainLayout->addSpacerItem(perfSpacer);

	// Add the new controls to a widget
	this->geodesicGUI->geodesicPerformanceWidget = new QWidget;
	this->geodesicGUI->geodesicPerformanceWidget->setLayout(perfMainLayout);

	// Add the widget as a new page of the tool
	this->ui->fiberTrackingToolbox->addItem(this->geodesicGUI->geodesicPerformanceWidget, "Performance");

	// Store the pointers to relevant controls
	this->geodesicGUI->aaGroup				= anglesGroup;
	this->geodesicGUI->aaPatternCombo		= anglesPatternCombo;
	this->geodesicGUI->aaConeNumberSpin		= anglesConeNumberSpin;
	this->geodesicGUI->aaConeWidthSpin		= anglesConeWidthSpin;
	this->geodesicGUI->aaSpherePSpin		= anglesSphereNumberPSpin;
	this->geodesicGUI->aaSphereTSpin		= anglesSphereNumberTSpin;
	this->geodesicGUI->aaIcoTessOrderCombo	= anglesIcoTessCombo;
	this->geodesicGUI->aaGLayout			= angleOptionsGLayout;
	this->geodesicGUI->stopScalarCheck		= stoppingScalarCheck;
	this->geodesicGUI->stopLengthCheck		= stoppingLengthCheck;
	this->geodesicGUI->stopAngleCheck		= stoppingAngleCheck;
	this->geodesicGUI->odeCombo				= odeCombo;
	this->geodesicGUI->ppEnableCheck		= ppEnableCheck;
	this->geodesicGUI->ppSharpenCombo		= ppSharpenCombo;
	this->geodesicGUI->ppGainSpin			= ppGainSpin;
	this->geodesicGUI->ppThresholdSpin		= ppThresholdSpin;
	this->geodesicGUI->ppExponentSpin		= ppExpSpin;
	this->geodesicGUI->ppSharpenGLayout		= ppSharpenGLayout;
	this->geodesicGUI->perfARadio			= NULL;
	this->geodesicGUI->perfBRadio			= NULL;
	this->geodesicGUI->perfCRadio			= NULL;

	// Disable or enable controls when preprocessing is turned off or on
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppGainLabel,		SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppGainSpin,		SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppSharpenLabel,	SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppSharpenCombo,	SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppThresholdLabel, SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppThresholdSpin,	SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppExpLabel,		SLOT(setEnabled(bool)));
	connect(ppEnableCheck, SIGNAL(toggled(bool)), ppExpSpin,		SLOT(setEnabled(bool)));

	// Update the GUI when one of the following controls is changed
	connect(ppEnableCheck,		SIGNAL(toggled(bool)),				this, SLOT(updateGeodesicGUI()));
	connect(anglesPatternCombo, SIGNAL(currentIndexChanged(int)),	this, SLOT(updateGeodesicGUI()));
	connect(ppSharpenCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(updateGeodesicGUI()));

	// Select the "Additional Tracking Options" page for now
	this->ui->fiberTrackingToolbox->setCurrentIndex(1);
}


//--------------------------[ updateGeodesicGUI ]--------------------------\\

void FiberTrackingPlugin::updateGeodesicGUI()
{
	// We can't do anything if we don't have a geodesic GUI
	if (!(this->geodesicGUI))
		return;

	// Booleans determining which of the additional angle controls should be showed
	bool showCone   = this->geodesicGUI->aaPatternCombo->currentIndex() == (int) vtkFiberTrackingGeodesicFilter::AAP_Cone;
	bool showSphere = this->geodesicGUI->aaPatternCombo->currentIndex() == (int) vtkFiberTrackingGeodesicFilter::AAP_SimpleSphere;
	bool showIco	= this->geodesicGUI->aaPatternCombo->currentIndex() == (int) vtkFiberTrackingGeodesicFilter::AAP_Icosahedron;

	// Show only the relevant additional angle controls
	this->geodesicGUI->aaGLayout->itemAtPosition(0, 0)->widget()->setVisible(showCone);
	this->geodesicGUI->aaGLayout->itemAtPosition(0, 1)->widget()->setVisible(showCone);
	this->geodesicGUI->aaGLayout->itemAtPosition(1, 0)->widget()->setVisible(showCone);
	this->geodesicGUI->aaGLayout->itemAtPosition(1, 1)->widget()->setVisible(showCone);
	this->geodesicGUI->aaGLayout->itemAtPosition(2, 0)->widget()->setVisible(showSphere);
	this->geodesicGUI->aaGLayout->itemAtPosition(2, 1)->widget()->setVisible(showSphere);
	this->geodesicGUI->aaGLayout->itemAtPosition(3, 0)->widget()->setVisible(showSphere);
	this->geodesicGUI->aaGLayout->itemAtPosition(3, 1)->widget()->setVisible(showSphere);
	this->geodesicGUI->aaGLayout->itemAtPosition(4, 0)->widget()->setVisible(showIco);
	this->geodesicGUI->aaGLayout->itemAtPosition(4, 1)->widget()->setVisible(showIco);

	// Enable the sharpening threshold controls if we apply sharpening AND preprocessing
	bool enableThreshold = (this->geodesicGUI->ppSharpenCombo->currentIndex() != (int) geodesicPreProcessor::SM_None) &&
							this->geodesicGUI->ppEnableCheck->isChecked();

	this->geodesicGUI->ppSharpenGLayout->itemAtPosition(0, 0)->widget()->setEnabled(enableThreshold);
	this->geodesicGUI->ppSharpenGLayout->itemAtPosition(0, 1)->widget()->setEnabled(enableThreshold);

	// Enable the exponent controls if we use exponents for sharpening AND preprocessing is enabled
	bool enableExponent = this->geodesicGUI->ppEnableCheck->isChecked() &&
							((this->geodesicGUI->ppSharpenCombo->currentIndex() == (int) geodesicPreProcessor::SM_Exponent) ||
							 (this->geodesicGUI->ppSharpenCombo->currentIndex() == (int) geodesicPreProcessor::SM_TraceDivAndExp));

	this->geodesicGUI->ppSharpenGLayout->itemAtPosition(1, 0)->widget()->setEnabled(enableExponent);
	this->geodesicGUI->ppSharpenGLayout->itemAtPosition(1, 1)->widget()->setEnabled(enableExponent);
}


//-----------------------[ doGeodesicFiberTracking ]-----------------------\\

void FiberTrackingPlugin::doGeodesicFiberTracking(vtkImageData * dtiImageData, vtkImageData * aiImageData)
{
	// Do nothing if the geodesic GUI does not exist
	if (!(this->geodesicGUI))
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "Geodesic Fiber Tracking GUI not available.");
		return;
	}

	// Get the geodesic GUI elements
	geodesicGUIElements * gGUI = this->geodesicGUI;

	// Get a list of the selected seed point sets
	QList<QListWidgetItem *> selectedSeeds = this->ui->seedList->selectedItems();

	// Check if the list contains items
	if (selectedSeeds.isEmpty())
	{
		QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", "No seed point sets selected!", 
			QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("Fiber Tracking Plugin: No seed point sets selected!");
		return;
	}

	// Create an iterator for the selected seed point list
	QList<QListWidgetItem *>::iterator selectedSeedsIter;

	// Loop through all selected seed point sets
	for (	selectedSeedsIter  = selectedSeeds.begin(); 
			selectedSeedsIter != selectedSeeds.end(); 
			++selectedSeedsIter								)
	{
		// Get the name of the current set
		QString selectedSeedName = (*selectedSeedsIter)->text();

		// Create an iterator for the list of stored seed point data sets
		QList<data::DataSet *>::iterator seedListIter;

		// Loop through all data sets in the list, and find the one with the same name as the
		// selected item in the GUI. This is probably not very efficient, but since there 
		// usually aren't a lot of seed point sets, this does not significantly influence
		// the running time.

		for (	seedListIter  = this->seedList.begin(); 
				seedListIter != this->seedList.end();	
				++seedListIter								)
		{
			// Get the name of the seed point data set.
			QString seedName = (*seedListIter)->getName();

			// Compare the name to that of the selected GUI item
			if (selectedSeedName == seedName)
				break;
		}

		// Check if the selected item was found in the list of seed point data sets
		if (seedListIter == this->seedList.end())
		{
			QString errorMessage = "Fiber Tracking Plugin: Could not find data set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", errorMessage, 
									QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Get the data from the data set, cast to a "vtkUnstructuredGrid" pointer
		vtkUnstructuredGrid * seedUG = (vtkUnstructuredGrid *) (*seedListIter)->getVtkObject();

		// Check if the data exists
		if (!seedUG)
		{
			QString errorMessage = "Fiber Tracking Plugin: No data defined for seed point set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", errorMessage, 
									QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Get the point array from the data set
		vtkPoints * seedPoints = seedUG->GetPoints();

		// Check if the point array exists
		if (!seedPoints)
		{
			QString errorMessage = "Fiber Tracking Plugin: No data defined for seed point set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", errorMessage, 
									QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Check the seed point set contains points
		if (seedPoints->GetNumberOfPoints() == 0)
		{
			QString errorMessage = "Fiber Tracking Plugin: No seed points in seed point set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "Fiber Tracking Plugin", errorMessage, 
									QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Create the fiber tracking filter
		vtkFiberTrackingGeodesicFilter * geodesicFiberTrackingFilter = vtkFiberTrackingGeodesicFilter::New();

		// Show progress of the filter
		this->core()->out()->createProgressBarForAlgorithm(geodesicFiberTrackingFilter, "Fiber Tracking", "Tracking geodesic fibers...");

		// Set the input images of the filter
		geodesicFiberTrackingFilter->SetInput(dtiImageData);
		geodesicFiberTrackingFilter->SetAnisotropyIndexImage(aiImageData);

		// Set the user variables from the GUI
		geodesicFiberTrackingFilter->SetIntegrationStepLength((float) this->ui->parametersStepSizeSpinner->value());
		geodesicFiberTrackingFilter->SetMaximumPropagationDistance((float) this->ui->parametersMaxLengthSpinner->value());
		geodesicFiberTrackingFilter->SetMinimumFiberSize((float) this->ui->parametersMinLengthSpinner->value());
		geodesicFiberTrackingFilter->SetMinScalarThreshold((float) this->ui->parametersMinAISpin->value());
		geodesicFiberTrackingFilter->SetMaxScalarThreshold((float) this->ui->parametersMaxAISpin->value());
		geodesicFiberTrackingFilter->SetStopDegrees((float) this->ui->parametersAngleSpinner->value());

		// Set the parameters unique to the geodesic fiber-tracking algorithm
		geodesicFiberTrackingFilter->setAdditionalAnglesOptions(gGUI->aaGroup->isChecked(), 
			(vtkFiberTrackingGeodesicFilter::AdditionalAnglesPattern) gGUI->aaPatternCombo->currentIndex());
		geodesicFiberTrackingFilter->setAAConeOptions(gGUI->aaConeNumberSpin->value(), gGUI->aaConeWidthSpin->value());
		geodesicFiberTrackingFilter->setAASphereOptions(gGUI->aaSpherePSpin->value(), gGUI->aaSphereTSpin->value());
		geodesicFiberTrackingFilter->setAAIcoTessOrder(gGUI->aaIcoTessOrderCombo->currentIndex() + 1);
		geodesicFiberTrackingFilter->setPreProcessingOptions(gGUI->ppEnableCheck->isChecked(),
			gGUI->ppSharpenCombo->currentIndex(),
			gGUI->ppGainSpin->value(),
			gGUI->ppThresholdSpin->value(),
			gGUI->ppExponentSpin->value());
		geodesicFiberTrackingFilter->setUseStopScalar(gGUI->stopScalarCheck->isChecked());
		geodesicFiberTrackingFilter->setUseStopLength(gGUI->stopLengthCheck->isChecked());
		geodesicFiberTrackingFilter->setUseStopAngle(gGUI->stopAngleCheck->isChecked());
		geodesicFiberTrackingFilter->setODESolver((vtkFiberTrackingGeodesicFilter::ODESolver) gGUI->odeCombo->currentIndex());

		// Set the current seed point set as the input of the filter
		geodesicFiberTrackingFilter->SetSeedPoints((vtkDataSet *) seedUG);

		// Set name of Region of Interest
		geodesicFiberTrackingFilter->setROIName((*seedListIter)->getName());

		// Update the filter
		geodesicFiberTrackingFilter->Update();

		// Get the resulting fibers
		vtkPolyData * outFibers = geodesicFiberTrackingFilter->GetOutput();

		// Name the fibers after the input image and the seed point set
		QString fiberName = this->dtiDataList.at(this->ui->DTIDataCombo->currentIndex())->getName()
			+ " - " + (*seedListIter)->getName();

		// Add the fibers to the output
		this->addFibersToDataManager(outFibers, fiberName, TM_Geodesic, (*seedListIter));

		// Set the source of the output fibers to NULL. This will stop the tracking filter
		// from updating whenever the seed points change, which is assumed to
		// be undesirable behavior.

		outFibers->SetSource(NULL);

		// Stop showing filter progress
		this->core()->out()->deleteProgressBarForAlgorithm(geodesicFiberTrackingFilter);

		// Delete the tracking filter
		geodesicFiberTrackingFilter->Delete();
	}	
}


} // namespace bmia

