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


//------------------------[ setupGUIForStreamlines ]-----------------------\\

void FiberTrackingPlugin::setupGUIForStreamlines()
{
	// Remove all tabs of the toolbox except for the first one (tracking parameters)
	this->clearToolbox();

	// Enabled controls that may have disabled for other tracking methods
	this->enableAllControls();
}


//----------------------[ doStreamlineFiberTracking ]----------------------\\

void FiberTrackingPlugin::doStreamlineFiberTracking(vtkImageData * dtiImageData, vtkImageData * aiImageData)
{
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
		dtiFiberTrackingFilter = vtkFiberTrackingFilter::New();

		// Show progress of the filter
		this->core()->out()->createProgressBarForAlgorithm(dtiFiberTrackingFilter, "Fiber Tracking");

		// Set the input images of the filter
		dtiFiberTrackingFilter->SetInput(dtiImageData);
		dtiFiberTrackingFilter->SetAnisotropyIndexImage(aiImageData);

		// Set the user variables from the GUI
		dtiFiberTrackingFilter->SetIntegrationStepLength((float) this->ui->parametersStepSizeSpinner->value());
		dtiFiberTrackingFilter->SetMaximumPropagationDistance((float) this->ui->parametersMaxLengthSpinner->value());
		dtiFiberTrackingFilter->SetMinimumFiberSize((float) this->ui->parametersMinLengthSpinner->value());
		dtiFiberTrackingFilter->SetMinScalarThreshold((float) this->ui->parametersMinAISpin->value());
		dtiFiberTrackingFilter->SetMaxScalarThreshold((float) this->ui->parametersMaxAISpin->value());
		dtiFiberTrackingFilter->SetStopDegrees((float) this->ui->parametersAngleSpinner->value());

		// Set the current seed point set as the input of the filter
		dtiFiberTrackingFilter->SetSeedPoints((vtkDataSet *) seedUG);

		// Set name of Region of Interest
		dtiFiberTrackingFilter->setROIName((*seedListIter)->getName());

		// Update the filter
		dtiFiberTrackingFilter->Update();

		// Get the resulting fibers
		vtkPolyData * outFibers = dtiFiberTrackingFilter->GetOutput();

		// Name the fibers after the input image and the seed point set
		QString fiberName = this->dtiDataList.at(this->ui->DTIDataCombo->currentIndex())->getName()
			+ " - " + (*seedListIter)->getName();

		// Add the fibers to the output
		this->addFibersToDataManager(outFibers, fiberName, TM_Streamlines, (*seedListIter));

		// Set the source of the output fibers to NULL. This will stop the tracking filter
		// from updating whenever the seed points change, which is assumed to
		// be undesirable behavior.

		outFibers->SetSource(NULL);

		// Stop showing filter progress
		this->core()->out()->deleteProgressBarForAlgorithm(dtiFiberTrackingFilter);

		// Delete the tracking filter
		dtiFiberTrackingFilter->Delete();
	}	
}


} // namespace bmia
