/*
 * HARDIFiberTrackingPlugin_Deterministic.cxx
 *
 * 2011-10-14	Anna Vilanova
 * - First Version. Created to reduce the size of "HARDIfiberTrackingPlugin.cxx", as well
 *   as to make a clear distinction between the supported fiber tracking methods.
 *
 */


/** Includes */

#include "HARDIFiberTrackingPlugin.h"
#include "vtkHARDIFiberTrackingFilter.h"


namespace bmia {


//------------------------[ setupGUIForHARDIDeterministic ]-----------------------\\


void HARDIFiberTrackingPlugin::setupGUIForHARDIDeterministic()
{
	// Remove all tabs of the toolbox except for the first one (tracking parameters)
	this->clearToolbox();

	// Enabled controls that may have disabled for other tracking methods
	this->enableAllControls();
}


//----------------------[ doStreamlineFiberTracking ]----------------------\\

void HARDIFiberTrackingPlugin::doDeterministicFiberTracking(vtkImageData * HARDIimageData, vtkImageData * aiImageData)
{
	// Get a list of the selected seed point sets
	QList<QListWidgetItem *> selectedSeeds = this->ui->seedList->selectedItems();

	// Check if the list contains items
	if (selectedSeeds.isEmpty())
	{
		QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", "No seed point sets selected!", 
			QMessageBox::Ok, QMessageBox::Ok);
		this->core()->out()->logMessage("HARDI Fiber Tracking Plugin: No seed point sets selected!");
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
			QString errorMessage = "HARDI Fiber Tracking Plugin: Could not find data set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", errorMessage, 
				QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Get the data from the data set, cast to a "vtkUnstructuredGrid" pointer
		vtkUnstructuredGrid * seedUG = (vtkUnstructuredGrid *) (*seedListIter)->getVtkObject();

		// Check if the data exists
		if (!seedUG)
		{
			QString errorMessage = "HARDI Fiber Tracking Plugin: No data defined for seed point set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", errorMessage, 
				QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Get the point array from the data set
		vtkPoints * seedPoints = seedUG->GetPoints();

		// Check if the point array exists
		if (!seedPoints)
		{
			QString errorMessage = "HARDI Fiber Tracking Plugin: No data defined for seed point set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", errorMessage, 
				QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Check the seed point set contains points
		if (seedPoints->GetNumberOfPoints() == 0)
		{
			QString errorMessage = "HARDI Fiber Tracking Plugin: No seed points in seed point set '" + (*seedListIter)->getName() + "'";
			QMessageBox::warning(this->getGUI(), "HARDI Fiber Tracking Plugin", errorMessage, 
				QMessageBox::Ok, QMessageBox::Ok);
			this->core()->out()->logMessage(errorMessage);
			continue;
		}

		// Create the fiber tracking filter
		HARDIFiberTrackingFilter = vtkHARDIFiberTrackingFilter::New();

		// Show progress of the filter
		this->core()->out()->createProgressBarForAlgorithm(HARDIFiberTrackingFilter, "Fiber Tracking");

		// Set the input images of the filter
		HARDIFiberTrackingFilter->SetInput(HARDIimageData);
		HARDIFiberTrackingFilter->SetAnisotropyIndexImage(aiImageData);

		// Set the user variables from the GUI
		HARDIFiberTrackingFilter->SetIntegrationStepLength((float) this->ui->parametersStepSizeSpinner->value());
		HARDIFiberTrackingFilter->SetMaximumPropagationDistance((float) this->ui->parametersMaxLengthSpinner->value());
		HARDIFiberTrackingFilter->SetMinimumFiberSize((float) this->ui->parametersMinLengthSpinner->value());
		HARDIFiberTrackingFilter->SetMinScalarThreshold((float) this->ui->parametersMinAISpin->value());
		HARDIFiberTrackingFilter->SetMaxScalarThreshold((float) this->ui->parametersMaxAISpin->value());
		HARDIFiberTrackingFilter->SetStopDegrees((float) this->ui->parametersAngleSpinner->value());
		HARDIFiberTrackingFilter->SetIterations((unsigned int) this->ui->iterationSpinner->value());
		HARDIFiberTrackingFilter->SetCleanMaxima((bool) this->ui->cleanBox->isChecked());
		HARDIFiberTrackingFilter->SetTreshold((float) this->ui->tresholdSpinner->value());
		HARDIFiberTrackingFilter->SetTesselationOrder((unsigned int) this->ui->tesselationSpinner->value());

		// Set the current seed point set as the input of the filter
		HARDIFiberTrackingFilter->SetSeedPoints((vtkDataSet *) seedUG);

		// Set name of Region of Interest
		HARDIFiberTrackingFilter->setROIName((*seedListIter)->getName());

		// Update the filter
		HARDIFiberTrackingFilter->Update();

		// Get the resulting fibers
		vtkPolyData * outFibers = HARDIFiberTrackingFilter->GetOutput();

		// Name the fibers after the input image and the seed point set
		QString fiberName = this->HARDIDataList.at(this->ui->HARDIDataCombo->currentIndex())->getName()
			+ " - " + (*seedListIter)->getName();

		// Add the fibers to the output
		this->addFibersToDataManager(outFibers, fiberName, TM_Deterministic, (*seedListIter));

		// Set the source of the output fibers to NULL. This will stop the tracking filter
		// from updating whenever the seed points change, which is assumed to
		// be undesirable behavior.

		outFibers->SetSource(NULL);

		// Stop showing filter progress
		this->core()->out()->deleteProgressBarForAlgorithm(HARDIFiberTrackingFilter);

		// Delete the tracking filter
		HARDIFiberTrackingFilter->Delete(); 
	}	
}


} // namespace bmia
