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
 * FiberFilterPlugin.cxx
 *
 * 2010-10-04	Tim Peeters
 * - First version
 *
 * 2010-11-09	Evert van Aart
 * - Implemented filtering functionality
 *
 * 2011-01-24	Evert van Aart
 * - Added support for transformation matrices
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.0.
 * - Improved attribute handling.
 *
 */


/** Includes */

#include "FiberFilterPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberFilterPlugin::FiberFilterPlugin() : plugin::Plugin("Fiber Filter")
{
    qDebug() << "Constructing FiberFilterPlugin object.";
}


//---------------------------------[ init ]--------------------------------\\

void FiberFilterPlugin::init()
{
    // Write text output for debugging
    qDebug() << "FiberFilterPlugin::init()";

    // Create the Qt widgets
    this->tabWidget = new QTabWidget();

    for (int i = 1; i <= FiberFilterPlugin::NUM_TABS; i++)
	{
		FiberFilterWidget * widget = new FiberFilterWidget(this, i);
		this->filterWidgets.append(widget);
		this->tabWidget->addTab(widget, QString::number(i));
	}

	// Set the first item of the ROI- and fiberlists to NULL, since
	// each combo box in the widget will also start with "None" or "Off".

	this->ROIList.append(NULL);
	this->FiberList.append(NULL);

	// Allocate room for the list of output data set pointers
	this->outputDataSets = (data::DataSet **) malloc(FiberFilterPlugin::NUM_TABS * sizeof(data::DataSet *));

	// Initialize all pointers to NULL
	for (int i = 0; i < FiberFilterPlugin::NUM_TABS; ++i)
	{
		outputDataSets[i] = NULL;
	}
}


//------------------------------[ Destructor ]-----------------------------\\

FiberFilterPlugin::~FiberFilterPlugin()
{
	// Delete all filter widgets
	for (int i = 0; i < FiberFilterPlugin::NUM_TABS; ++i)
	{
		delete (this->filterWidgets.at(i));
	}

	// Clear the list
	this->filterWidgets.clear();

	// Delete the tab widget
    delete this->tabWidget; 
	this->tabWidget = NULL;

	// Free the allocated space for the output data set pointers
	free(this->outputDataSets);
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * FiberFilterPlugin::getGUI()
{
	// Return the main tab widget
    return this->tabWidget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void FiberFilterPlugin::dataSetAdded(data::DataSet * ds)
{
	// Assert the data set pointer
    Q_ASSERT(ds);

	// Get the type of the data set
    QString kind = ds->getKind();

	// Fiber data sets
    if (kind == "fibers")
	{
		// Check if the data set contains polydata
		if (!(ds->getVtkPolyData()))
			return;

		// Add the data set to the list of fibers
		this->FiberList.append(ds);

		// Add the fibers to each filter widget
		for (int i = 0; i < FiberFilterPlugin::NUM_TABS; i++)
	    {
			this->filterWidgets.at(i)->fibersAdded(ds);
	    }
	}

	// Regions of Interest
    else if (kind == "regionOfInterest")
	{
		// Check if the data set contains polydata
		if (!(ds->getVtkPolyData()))
			return;

		// Add the new ROI to the list of ROIs
		this->ROIList.append(ds);

		// Add the ROI to each fiber widget
		for (int i = 0; i < FiberFilterPlugin::NUM_TABS; i++)
	    {
			this->filterWidgets.at(i)->roiAdded(ds);
	    }
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void FiberFilterPlugin::dataSetChanged(data::DataSet* ds)
{
	// Assert the data set pointer
    Q_ASSERT(ds);

	// Get the type of the data set
    QString kind = ds->getKind();

	// Fiber data sets
    if (kind == "fibers")
	{
		// Check if the data set has been added to this plugin
		if (!(this->FiberList.contains(ds)))
			return;

		// Get the index of the fiber data set
		int fiberIndex = this->FiberList.indexOf(ds);

		// Change the fiber settings in each filter widget
		for (int i = 0; i < FiberFilterPlugin::NUM_TABS; i++)
	    {
			this->filterWidgets.at(i)->fibersChanged(fiberIndex, ds->getName());
	    }
	} 

	// Regions if Interest
    else if (kind == "regionOfInterest")
	{
		// Check if the data set has been added to this plugin
		if (!(this->ROIList.contains(ds)))
			return;

		// Get the index of the ROI data set
		int roiIndex = this->ROIList.indexOf(ds);

		// Change the ROI settings in each filter widget
		for (int i = 0; i < FiberFilterPlugin::NUM_TABS; i++)
	    {
			this->filterWidgets.at(i)->roiChanged(roiIndex, ds->getName());
	    } 
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void FiberFilterPlugin::dataSetRemoved(data::DataSet* ds)
{
	// Assert the data set pointer
    Q_ASSERT(ds);

	// Get the type of the data set
	QString kind = ds->getKind();

	// Fiber data sets
	if (kind == "fibers")
	{
		// Check if the data set has been added to this plugin
		if (!(this->FiberList.contains(ds)))
			return;

		// Get the index of the fiber data set
		int fiberIndex = this->FiberList.indexOf(ds);

		// Remove the fibers from each filter widget
		for (int i = 0; i < FiberFilterPlugin::NUM_TABS; i++)
	    {
			this->filterWidgets.at(i)->fibersRemoved(fiberIndex);
	    }

		// Remove the data set from the list
		this->FiberList.removeAt(fiberIndex);
	}

	// Regions of Interest
    else if (kind == "regionOfInterest")
	{
		// Check if the data set has been added to this plugin
		if (!(this->ROIList.contains(ds)))
			return;

		// Get the index of the ROI data set
		int roiIndex = this->ROIList.indexOf(ds);

		// Remove the ROI from each filter widget
		for (int i = 0; i < FiberFilterPlugin::NUM_TABS; i++)
	    {
			this->filterWidgets.at(i)->roiRemoved(roiIndex);
	    } 

		// Remove the data set from the list		
		this->ROIList.removeAt(roiIndex);
	}
}


//--------------------------------[ getROI ]-------------------------------\\

vtkPolyData * FiberFilterPlugin::getROI(int i)
{
	// Check if the index is within range
	if (i < 0 || i >= this->ROIList.size())
		return NULL;

	// Get the data set pointer
	data::DataSet * currentDS = this->ROIList.at(i);

	// Check if the pointer has been set
	if (!currentDS)
		return NULL;

	// Return the polydata of the data set
	return (currentDS->getVtkPolyData());
}


//------------------------------[ getFibers ]------------------------------\\

vtkPolyData * FiberFilterPlugin::getFibers(int i)
{
	// Check if the index is within range
	if (i < 0 || i >= this->FiberList.size())
		return NULL;

	// Get the data set pointer
	data::DataSet * currentDS = this->FiberList.at(i);

	// Check if the pointer has been set
	if (!currentDS)
		return NULL;

	// Return the polydata of the data set
	return (currentDS->getVtkPolyData());
}


//-----------------------[ getTransformationMatrix ]-----------------------\\

vtkMatrix4x4 * FiberFilterPlugin::getTransformationMatrix(int i)
{
	// VTK object stored as attribute
	vtkObject * object;

	// Transformation matrix
	vtkMatrix4x4 * m = NULL;

	// Get the input fiber data set
	data::DataSet * ds = this->FiberList.at(i);

	// Check if the fiber set contains a "transformation matrix" attribute
	if (!(ds->getAttributes()->getAttribute("transformation matrix", object)))
		return m;

	// Convert the VTK object to a 4x4 matrix and return it
	return vtkMatrix4x4::SafeDownCast(object);
}


//------------------------[ addFibersToDataManager ]-----------------------\\

bool FiberFilterPlugin::addFibersToDataManager(vtkPolyData * out, QString name, int filterID, vtkMatrix4x4 * m)
{
	// Check whether the current fiber widget has already created a data set.
	if (this->outputDataSets[filterID] == NULL)
	{
		// Do nothing if no lines were generated
		if (out->GetNumberOfLines() == 0)
		{
			this->core()->out()->logMessage("FiberFilterPlugin: Output fiber set " + name + " does not contain any fibers.");
			return false;
		}

		// Create a new data set
		data::DataSet * ds = new data::DataSet(name, "fibers", (vtkObject *) out);

		if (m)
		{
			// Add the transformation matrix to the output
			vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
			mCopy->DeepCopy(m);
			ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(mCopy));
		}

		// Add the data set to the data manager
		this->core()->data()->addDataSet(ds);

		// Store the pointer in the list of output data sets
		this->outputDataSets[filterID] = ds;
	}
	else
	{
		// Get the existing output data set
		data::DataSet * ds = this->outputDataSets[filterID];

		// Delete old fiber data set
		if (out->GetNumberOfLines() == 0)
		{
			this->core()->out()->logMessage("FiberFilterPlugin: Output fiber set " + name + " does not contain any fibers.");
			this->outputDataSets[filterID] = NULL;
			this->core()->data()->removeDataSet(ds);

			return false;
		}

		// Update the data object of the data set
		ds->updateData((vtkObject *) out);

		// Update the name of the data set
		ds->setName(name);		

		// Set "updatePipeline" to 1.0, to signal the visualization plugin that it should re-execute its pipeline.
		ds->getAttributes()->addAttribute("updatePipeline", 1.0);

		// Set "isVisible" to 1.0, to make sure that the fibers are visible
		ds->getAttributes()->addAttribute("isVisible", 1.0);

		if (m)
		{
			// Add the transformation matrix to the output
			vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
			mCopy->DeepCopy(m);
			vtkObject * object = NULL;

			ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(mCopy));
		}

		// Signal the data manager that the data set has changed
		this->core()->data()->dataSetChanged(ds);
	}

	return true;
}


void FiberFilterPlugin::hideInputFibers(int index)
{
	// Check if the index is within range
	if (index < 0 || index >= this->FiberList.size())
		return;

	// Get the data set pointer
	data::DataSet * ds = this->FiberList.at(index);

	// Check if the pointer has been set
	if (!ds)
		return;

	// Set the attribute "isVisible" to -1.0
	ds->getAttributes()->addAttribute("isVisible", -1.0);

	// Signal the data manager that the data set has changed
	this->core()->data()->dataSetChanged(ds);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libFiberFilterPlugin, bmia::FiberFilterPlugin)
