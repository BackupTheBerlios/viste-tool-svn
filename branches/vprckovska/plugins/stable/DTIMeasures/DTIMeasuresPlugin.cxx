/*
 * DTIMeasuresPlugin.cxx
 *
 * 2010-03-09	Tim Peeters
 * - First version
 *
 * 2011-01-20	Evert van Aart
 * - Added support for transformation matrices.
 * - Anisotropy images are now computed on demand.
 *
 * 2011-03-10	Evert van Aart
 * - Version 1.0.0.
 * - Increased stability when changing or removing data sets.
 * - Added additional comments.
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.2.
 * - Improved attribute handling.
 *
 */


/** Includes */

#include "DTIMeasuresPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

DTIMeasuresPlugin::DTIMeasuresPlugin() : plugin::Plugin("DTI Measures")
{

}


//------------------------------[ Destructor ]-----------------------------\\

DTIMeasuresPlugin::~DTIMeasuresPlugin()
{
	// Loop through all the AI filters
	for (int i = 0; i < this->aiFilters.size(); ++i)
	{
		// Delete the progress bar for the current filter
		this->core()->out()->deleteProgressBarForAlgorithm(this->aiFilters[i]);
	}

	// Clear the list of filters
	this->aiFilters.clear();

	// Loop through all input DTI images
	for (int imageID = 0; imageID < this->images.size(); ++imageID)
	{
		// Get the output information for the current DTI image
		OutputInformation currentImage = this->images.at(imageID);

		// Delete the eigensystem image from the manager
		if (currentImage.eigenOutput)
		{
			this->core()->data()->removeDataSet(currentImage.eigenOutput);
		}

		// Copy the list of AI outputs
		QList<data::DataSet *> aiListCopy = currentImage.aiOutputs;

		// Loop through all AI images
		for (int aiID = 0; aiID < aiListCopy.size(); ++aiID)
		{
			// Get the current AI image and delete it
			data::DataSet * currentAIImage = aiListCopy.at(aiID);

			if (currentAIImage)
			{
				this->core()->data()->removeDataSet(currentAIImage);
			}
		}
	}

	// Clear the list
	this->images.clear();
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void DTIMeasuresPlugin::dataSetAdded(data::DataSet * ds)
{
	if (!ds)
		return;

	// We're only interested in DTI images that have not yet been added
	int existingImageID;
	if (ds->getKind() != "DTI" && this->findInputImage(ds, existingImageID) == false)
		return;

	// Get the DTI image
	vtkImageData * dtiImage = ds->getVtkImageData();

	if (!dtiImage)
		return;

	// Create a new output information structure
	OutputInformation newInfo;
	newInfo.input = ds;

	// Get the name of the data set
	QString baseName = ds->getName();
	this->core()->out()->logMessage("Adding anisotropyFilter measures for DTI data set " + baseName);

	// Create a new filter for computing the eigensystem and run it
	vtkTensorToEigensystemFilter * eigenFilter = vtkTensorToEigensystemFilter::New();
	this->core()->out()->createProgressBarForAlgorithm(eigenFilter, "DTI Measures");
	eigenFilter->SetInput(dtiImage);
	eigenFilter->Update();
	this->core()->out()->deleteProgressBarForAlgorithm(eigenFilter);

	// Check if the eigensystem output was computed successfully
	if (!(eigenFilter->GetOutput()))
	{
		this->core()->out()->showMessage("Computation of eigensystem failed!");
		eigenFilter->Delete();
		return;
	}

	// Create a new data set for the eigensystem image
	data::DataSet * eigenDs = new data::DataSet(baseName, "eigen", 
												eigenFilter->GetOutput());

	// Copy the transformation matrix of the DTI image to the eigensystem image
	eigenDs->getAttributes()->copyTransformationMatrix(ds);

	// Add the eigensystem data set to the data manager
	this->core()->data()->addDataSet(eigenDs);

	// Store the eigensystem pointer
	newInfo.eigenOutput = eigenDs;
    
	// Filter for computing the anisotropyFilter
	vtkEigenvaluesToAnisotropyFilter * anisotropyFilter = NULL;

	// Loop through all supported AI measures
	for (int aiID = 0; aiID < AnisotropyMeasures::numberOfMeasures; ++aiID)
	{
		// Create a new filter for the current measure
		anisotropyFilter = vtkEigenvaluesToAnisotropyFilter::New();
		anisotropyFilter->SetMeasure(aiID);
		anisotropyFilter->SetInput(eigenFilter->GetOutput());
	
		// Check if the AI measure was computed successfully
		if (!(anisotropyFilter->GetOutput()))
		{
			QString numString;
			numString.setNum(aiID);
			this->core()->out()->showMessage("Computation failed for AI measure #" + numString);
			continue;
		}
	
		data::DataSet * anistropyDS = new data::DataSet(baseName + " " + QString(AnisotropyMeasures::GetShortName(aiID)),
			"scalar volume",
			anisotropyFilter->GetOutput());
	
		// "Delete" the filter. Due to reference counting, the filter is not 
		// actually deleted, but kept in memory connected to its output.

		anisotropyFilter->Delete(); 

		// Add a progress bar for the AI filter
		this->core()->out()->createProgressBarForAlgorithm(anisotropyFilter, "DTI Measures",
			"Computing scalar measure '" + QString(AnisotropyMeasures::GetLongName(aiID)) + "'");

		// Add the filter to the list, for future reference
		this->aiFilters.append(anisotropyFilter);

		// Copy the transformation matrix of the DTI image to the anisotropyFilter image
		anistropyDS->getAttributes()->copyTransformationMatrix(ds);

		// Add the AI data set to the data manager
		this->core()->data()->addDataSet(anistropyDS);

		// Add the data set pointer to the list
		newInfo.aiOutputs.append(anistropyDS);

	} // for [imageID]

    eigenFilter->Delete();
    eigenFilter = NULL;

    // Filter for computing scalar invariant measures
    vtkTensorToInvariantFilter * invariantFilter = NULL;

    // Loop through all supported invariant measures
    for(int invariantID = 0; invariantID < Invariants::numberOfMeasures; ++invariantID)
    {
        // Create new filter for the current invariant measure
        invariantFilter = vtkTensorToInvariantFilter::New();
        invariantFilter->SetInvariant(invariantID);
        invariantFilter->SetInput(dtiImage);

        // Check if invariant was computed correctly
        if(!(invariantFilter->GetOutput()))
        {
            QString numString;
            numString.setNum(invariantID);
            this->core()->out()->showMessage("Computation failed for invariant measure #" + numString);
            continue;
        }

        data::DataSet * invariantDS = new data::DataSet(baseName + " " + QString(Invariants::GetShortName(invariantID)),
            "scalar volume", invariantFilter->GetOutput());

        // "Delete" the filter. Due to reference counting, the filter is not
        // actually deleted, but kept in memory connected to its output.

        invariantFilter->Delete();

        // Add a progress bar for the AI filter
        this->core()->out()->createProgressBarForAlgorithm(invariantFilter, "DTI Measures",
            "Computing scalar measure '" + QString(Invariants::GetLongName(invariantID)) + "'");

        // Add the filter to the list, for future reference
        this->invariantFilters.append(invariantFilter);

        // Copy the transformation matrix of the DTI image to the anisotropyFilter image
        invariantDS->getAttributes()->copyTransformationMatrix(ds);

        // Add the AI data set to the data manager
        this->core()->data()->addDataSet(invariantDS);

        // Add the data set pointer to the list
        newInfo.invariantOutputs.append(invariantDS);
    }

	// Add the output information to the list
	this->images.append(newInfo);
}


//----------------------------[ dataSetChanged ]---------------------------\\

void DTIMeasuresPlugin::dataSetChanged(data::DataSet * ds)
{
    int imageID;

	// When a DTI image is externally modified, we simply remove and re-add it
	if (ds->getKind() == "DTI" && this->findInputImage(ds, imageID))
	{
		this->dataSetRemoved(ds);
		this->dataSetAdded(ds);
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void DTIMeasuresPlugin::dataSetRemoved(data::DataSet * ds)
{
	int imageID;
	int aiID;

	// When a DTI image is removed, we remove all images generated from it
	if (ds->getKind() == "DTI" && this->findInputImage(ds, imageID))
	{
		// Get the output information for the current DTI image
		OutputInformation currentImage = this->images.at(imageID);

		// Delete the eigensystem image from the manager
		if (currentImage.eigenOutput)
		{
			this->core()->data()->removeDataSet(currentImage.eigenOutput);
		}

		// Copy the list of AI outputs
		QList<data::DataSet *> aiListCopy = currentImage.aiOutputs;

		// Loop through all AI images
		for (int aiID = 0; aiID < aiListCopy.size(); ++aiID)
		{
			// Get the current AI image and delete it
			data::DataSet * currentAIImage = aiListCopy.at(aiID);

			if (currentAIImage)
			{
				this->core()->data()->removeDataSet(currentAIImage);
			}
		}

		// Finally, delete the entry for the DTI image itself
		this->images.removeAt(imageID);
	}
	// Remove the pointer to the eigensystem data
	else if (ds->getKind() == "eigen" && this->findEigenImage(ds, imageID))
	{
		this->images[imageID].eigenOutput = NULL;

		// We assume that, when the user deletes an eigensystem image (manually),
		// he does not necessarily want to throw away the AI images computed
		// from this image. Therefore, we now set the source of the related AI
		// images to NULL, essentially disconnecting these images from the
		// eigensystem image.

		OutputInformation currentImage = this->images.at(imageID);
		for (int i = 0; i < currentImage.aiOutputs.size(); ++i)
		{
			vtkImageData * currentAIImage = currentImage.aiOutputs.at(i)->getVtkImageData();
			currentAIImage->SetSource(NULL);
		}
	}
	// Remove the pointer to the AI image
	else if (ds->getKind() == "scalar volume" && this->findAIImage(ds, imageID, aiID))
	{
		OutputInformation currentImage = this->images.at(imageID);
		currentImage.aiOutputs.removeAt(aiID);
		this->images[imageID] = currentImage;
	}
}


//----------------------------[ findInputImage ]---------------------------\\

bool DTIMeasuresPlugin::findInputImage(data::DataSet * ds, int& imageID)
{
	// Check if the list is empty
	if (this->images.isEmpty())
		return false;

	// Loop through all elements of the list
	for (int i = 0; i < this->images.size(); ++i)
	{
		// Check if the pointer matches
		if (this->images.at(i).input == ds)
		{
			imageID = i;
			return true;
		}
	}

	// No match found!
	return false;
}


//----------------------------[ findEigenImage ]---------------------------\\

bool DTIMeasuresPlugin::findEigenImage(data::DataSet * ds, int& imageID)
{
	// Check if the list is empty
	if (this->images.isEmpty())
		return false;

	// Loop through all elements of the list
	for (int i = 0; i < this->images.size(); ++i)
	{
		// Check if the pointer matches
		if (this->images.at(i).eigenOutput == ds)
		{
			imageID = i;
			return true;
		}
	}

	// No match found!
	return false;
}


//-----------------------------[ findAIImage ]-----------------------------\\

bool DTIMeasuresPlugin::findAIImage(data::DataSet * ds, int& imageID, int& aiID)
{
	// Check if the list is empty
	if (this->images.isEmpty())
		return false;

	// Loop through all elements of the list
	for (int i = 0; i < this->images.size(); ++i)
	{
		OutputInformation currentImage = this->images.at(i);

		// Loop through all AI images
		for (int j = 0; j < currentImage.aiOutputs.size(); ++j)
		{
			// Check if the pointer matches
			if (currentImage.aiOutputs.at(j) == ds)
			{
				imageID = i;
				aiID = j;
				return true;
			}
		}
	}

	// No match found!
	return false;
}

//-----------------------------[ findInvariantImage ]-----------------------------\\

bool DTIMeasuresPlugin::findInvariantImage(data::DataSet * ds, int& imageID, int& invariantID)
{
    // Check if list is empty
    if(this->images.isEmpty())
        return false;

    // Loop through all elements of the list
    for(int i = 0; i < this->images.size(); ++i)
    {
        OutputInformation currentImage = this->images.at(i);

        // Loop through all invariant images
        for(int j = 0; j < currentImage.invariantOutputs.size(); ++j)
        {
            // Check if pointer matches
            if(currentImage.invariantOutputs.at(j) == ds)
            {
                imageID = i;
                invariantID = j;
                return true;
            }
        }
    }

    // No match found!
    return false;
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libDTIMeasuresPlugin, bmia::DTIMeasuresPlugin)
