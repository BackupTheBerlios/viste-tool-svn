/*
 * FiberTrackingPlugin.cxx
 *
 * 2011-05-17	Evert van Aart
 * - First Version. Created to reduce the size of "fiberTrackingPlugin.cxx", as well
 *   as to make a clear distinction between the supported fiber tracking methods.
 *
 */


/** Includes */

#include "FiberTrackingPlugin.h"


namespace bmia {


//---------------------------[ setupGUIForWVS ]----------------------------\\

void FiberTrackingPlugin::setupGUIForWVS()
{
	// Do nothing if the WVS GUI controls already exist
	if (this->wvsGUI)
		return;

	// Clear the toolbox (to remove controls for other methods)
	this->clearToolbox();

	// Enable all controls that may have been disabled for other methods
	this->enableAllControls();

	// Create a structure for the WVS GUI elements
	this->wvsGUI = new wvsGUIElements;

	// Seed distance label and spin box
	QLabel * seedDistanceLabel = new QLabel("Seed Distance (mm)");
	QDoubleSpinBox * seedDistanceSpin = new QDoubleSpinBox;
		seedDistanceSpin->setMinimum(0.1);
		seedDistanceSpin->setMaximum(50);
		seedDistanceSpin->setSingleStep(1.0);
		seedDistanceSpin->setValue(5.0);
		seedDistanceSpin->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
	QHBoxLayout * seedDistanceHLayout = new QHBoxLayout;
		seedDistanceHLayout->addWidget(seedDistanceLabel);
		seedDistanceHLayout->addWidget(seedDistanceSpin);

	// Minimum distance label, spin box, and slider
	QLabel * minimumDistanceLabel = new QLabel("Minimum Fiber Distance (%)");
	QSpinBox * minimumDistanceSpin  = new QSpinBox;
		minimumDistanceSpin->setMinimum(0);
		minimumDistanceSpin->setMaximum(100);
		minimumDistanceSpin->setSingleStep(1);
		minimumDistanceSpin->setValue(50);
	QSlider * minimumDistanceSlide = new QSlider(Qt::Horizontal);
		minimumDistanceSlide->setMinimum(0);
		minimumDistanceSlide->setMaximum(100);
		minimumDistanceSlide->setSingleStep(1);
		minimumDistanceSlide->setValue(50);
	QHBoxLayout * minimumDistanceHLayout = new QHBoxLayout;
		minimumDistanceHLayout->addWidget(minimumDistanceSpin);
		minimumDistanceHLayout->addWidget(minimumDistanceSlide);

	// Connect the slider to the spin and vice versa
	connect(minimumDistanceSpin,  SIGNAL(valueChanged(int)), minimumDistanceSlide, SLOT(setValue(int)));
	connect(minimumDistanceSlide, SIGNAL(valueChanged(int)), minimumDistanceSpin,  SLOT(setValue(int)));

	// Create a spacer for the WVS page
	QSpacerItem * mainSpacer = new QSpacerItem(20, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);

	// Add all items to a vertical layout
	QVBoxLayout * VLayout = new QVBoxLayout;
		VLayout->addLayout(seedDistanceHLayout);
		VLayout->addWidget(minimumDistanceLabel);
		VLayout->addLayout(minimumDistanceHLayout);
		VLayout->addSpacerItem(mainSpacer);

	// Add the new controls to a widget
	this->wvsGUI->wvsWidget = new QWidget;
	this->wvsGUI->wvsWidget->setLayout(VLayout);

	// Store pointers to the active controls for easy access
	this->wvsGUI->wvsSeedDistanceSpin = seedDistanceSpin;
	this->wvsGUI->wvsMinDistanceSpin  = minimumDistanceSpin;
	this->wvsGUI->wvsMinDistanceSlide = minimumDistanceSlide;

	// Add the widget as a new page of the tool
	int tabIndex = this->ui->fiberTrackingToolbox->addItem(this->wvsGUI->wvsWidget, "Whole Volume Seeding Options");

	// Select the new page
	this->ui->fiberTrackingToolbox->setCurrentIndex(tabIndex);

	// Disable the seed list, since we do not need it
	this->ui->seedList->setEnabled(false);
}


//--------------------------[ doWVSFiberTracking ]-------------------------\\

void FiberTrackingPlugin::doWVSFiberTracking(vtkImageData * dtiImageData, vtkImageData * aiImageData)
{
	// Do nothing if the WVS GUI does not exist. This should never happen, since
	// we create the GUI when switching to the WVS method.

	if (!(this->wvsGUI))
		return;

	// Create the Whole Volume Seeding filter
	vtkFiberTrackingWVSFilter * wvsFilter = vtkFiberTrackingWVSFilter::New();

	// Create a progress bar for this filter
	this->core()->out()->createProgressBarForAlgorithm(wvsFilter, "Fiber Tracking");

	// Set the input images of the filter
	wvsFilter->SetInput(dtiImageData);
	wvsFilter->SetAnisotropyIndexImage(aiImageData);

	// Set the user variables from the GUI
	wvsFilter->SetIntegrationStepLength((float) this->ui->parametersStepSizeSpinner->value());
	wvsFilter->SetMaximumPropagationDistance((float) this->ui->parametersMaxLengthSpinner->value());
	wvsFilter->SetMinimumFiberSize((float) this->ui->parametersMinLengthSpinner->value());
	wvsFilter->SetMinScalarThreshold((float) this->ui->parametersMinAISpin->value());
	wvsFilter->SetMaxScalarThreshold((float) this->ui->parametersMaxAISpin->value());
	wvsFilter->SetStopDegrees((float) this->ui->parametersAngleSpinner->value());
	wvsFilter->SetMinDistancePercentage(this->wvsGUI->wvsMinDistanceSpin->value() / 100.0f);
	wvsFilter->SetSeedDistance(this->wvsGUI->wvsSeedDistanceSpin->value());

	// Update the filter
	wvsFilter->Update();

	// Get the resulting fibers
	vtkPolyData * outFibers = wvsFilter->GetOutput();

	// Name the fibers after the input image and the seed point set
	QString fiberName = dtiDataList.at(this->ui->DTIDataCombo->currentIndex())->getName() + " - Whole Volume Seeding";

	// Add the fibers to the output
	this->addFibersToDataManager(outFibers, fiberName, TM_WVS, NULL);

	// Set the source of the output fibers to NULL. This will stop the tracking filter
	// from updating when its input changes, which is assumed to be undesirable behavior.

	outFibers->SetSource(NULL);

	// Delete the progress bar
	this->core()->out()->deleteProgressBarForAlgorithm(wvsFilter);

	// Delete the filter
	wvsFilter->Delete();
}


} // namespace bmia
