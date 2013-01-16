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
 * ClusteringPlugin.h
 *
 * 2010-10-21	Evert van Aart
 * - First Version.
 *
 * 2011-02-02	Evert van Aart
 * - Implemented "dataSetChanged" and "dataSetRemoved".
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.0.
 * - Improved attribute handling.
 *
 */


/** Includes */

#include "ClusteringPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

ClusteringPlugin::ClusteringPlugin() : plugin::Plugin("Clustering")
{
	// Create a new Qt widget
    this->widget = new QWidget();

	// Create a new GUI form
    this->ui = new Ui::ClusteringForm();
	
	// Setup the GUI
    this->ui->setupUi(this->widget);

	// Set default variable values
	this->ignoreFiberDSChanged = false;

	// Loop through all initial output cluster names, defined in the GUI
	for (int i = 0; i < this->ui->outputCombo->count(); ++i)
	{
		// Add a new output cluster information object to the list, with the 
		// name equal to the text in the combo box, the color equal to white,
		// and automatic coloring on by default.

		outputClusterInformation newOutputInfo;
		newOutputInfo.name = this->ui->outputCombo->itemText(i);
		newOutputInfo.color = QColor(255, 255, 255);	
		newOutputInfo.useAutoColor = true;
		this->outputInfoList.append(newOutputInfo);
	}

	// Connect GUI signals to slot functions
	connect(this->ui->tableWidget, SIGNAL(itemSelectionChanged ()), this, SLOT(colorInputFibers()));
	connect(this->ui->outputAddButton, SIGNAL(clicked()), this, SLOT(addOutputCluster()));
	connect(this->ui->outputRenameButton, SIGNAL(clicked()), this, SLOT(renameOutputCluster()));
	connect(this->ui->outputRemoveButton, SIGNAL(clicked()), this, SLOT(removeOutputCluster()));
	connect(this->ui->updateButton, SIGNAL(clicked()), this, SLOT(updateOutputClusters()));
	connect(this->ui->fiberCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(changeFibers()));
	connect(this->ui->cluCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(createClusterList()));
	connect(this->ui->outputCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(outputClusterChanged()));
	connect(this->ui->outputColorAutoRadio, SIGNAL(toggled(bool)), this, SLOT(setColorAutoOrManual()));
	connect(this->ui->outputColorManualRadio, SIGNAL(toggled(bool)), this, SLOT(setColorAutoOrManual()));
	connect(this->ui->outputColorPickerButton, SIGNAL(clicked()), this, SLOT(setManualColor()));
	connect(this->ui->showInputButton, SIGNAL(clicked()), this, SLOT(showInputHideOutput()));
	connect(this->ui->showOutputButton, SIGNAL(clicked()), this, SLOT(showOutputHideInput()));
	connect(this->ui->saveButton, SIGNAL(clicked()), this, SLOT(writeSettings()));
	connect(this->ui->loadButton, SIGNAL(clicked()), this, SLOT(readSettings()));

	// Initialize the color chart
	this->initColorChart();
}


//------------------------------[ Destructor ]-----------------------------\\

ClusteringPlugin::~ClusteringPlugin()
{
	// Remove all items of the table widget in the GUI
	this->clearTable();

	// Clear the color chart
	this->colorChart.clearColors();

	// Clear the lists used in this class
	this->outputInfoList.clear();
	this->fiberList.clear();
	this->clusterList.clear();
	this->addedDataSets.clear();

	// Delete the GUI components
	delete (this->widget);
	delete (this->ui);
}


//--------------------------------[ GetGUI ]-------------------------------\\

QWidget * ClusteringPlugin::getGUI()
{
	// Return the Qt widget
    return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void ClusteringPlugin::dataSetAdded(data::DataSet * ds)
{
	// Fiber sets
	if (ds->getKind() == "fibers")
	{
		// Check if the data set contains poly data
		if (!(ds->getVtkPolyData()))
			return;

		// Add new data set to the list of data sets
		this->fiberList.append(ds);

		// Create new item in the combo box
		this->ui->fiberCombo->addItem(ds->getName());
	}
	// Clustering information
	else if (ds->getKind() == "clusters")
	{
		// Check if the data set contains data
		if (!ds->getVtkObject())
			return;

		// Add new data set to the list of data sets
		this->clusterList.append(ds);

		// Create new item in the combo box
		this->ui->cluCombo->addItem(ds->getName());

		// If this was the first item, create the input cluster list
		if (this->ui->cluCombo->count() == 1)
			this->createClusterList();
	}
}


//-----------------------------[ changeFibers ]----------------------------\\

void ClusteringPlugin::changeFibers()
{
	// Color the new fibers
	this->colorInputFibers();
}


//--------------------------[ createClusterList ]--------------------------\\

void ClusteringPlugin::createClusterList()
{
	// Get index of clustering information object
	int clusterIndex = this->ui->cluCombo->currentIndex();

	// Check if the index is in the range of the combo box
	if (clusterIndex < 0 || clusterIndex > this->ui->cluCombo->count())
		return;

	// Get the current data set pointer from the list of saved data sets
	data::DataSet * currentDS = this->clusterList.at(clusterIndex);

	// Check if the data set exists
	if (!currentDS)
		return;

	// Get the actual data from the data set
	vtkStructuredPoints * clusterData = vtkStructuredPoints::SafeDownCast(currentDS->getVtkObject());

	// Check if the data exists
	if (!clusterData)
		return;
	
	// Get the number of points in the clustering information
	vtkIdType numberOfPoints = clusterData->GetNumberOfPoints();

	// Compute the number of unique cluster IDs in the input
	this->numberOfInputClusters = this->getNumberOfInputClusters(clusterData);

	// If something went wrong in the "getNumberOfInputClusters" function,
	// it returns "-1", and we now exit this function.

	if (this->numberOfInputClusters == -1)
		return;

	// Clear the table widget in the GUI.
	this->clearTable();

	// Set row count, force column width, and block resizing of columns and rows
	this->ui->tableWidget->setRowCount(this->numberOfInputClusters);
	this->ui->tableWidget->setColumnWidth(0, 25);
	this->ui->tableWidget->setColumnWidth(1, 170);
	this->ui->tableWidget->horizontalHeader()->setResizeMode(QHeaderView::Custom);
	this->ui->tableWidget->verticalHeader()->setResizeMode(QHeaderView::Custom);

	// Set the labels of the columns 
	QStringList columnLabels;
	columnLabels.append(" ");
	columnLabels.append("Output Cluster");
	this->ui->tableWidget->setHorizontalHeaderLabels(columnLabels);

	// Loop through all input clusters
	for (int cluId = 0; cluId < this->numberOfInputClusters; ++cluId)
	{
		// Create a new combo box, set it to "<None>"
		QComboBox * newCombo = new QComboBox;
		newCombo->addItem("<None>");

		// Add all existing output clusters to the new combo box
		for (int i = 0; i < this->ui->outputCombo->count(); ++i)
		{
			newCombo->addItem(this->ui->outputCombo->itemText(i));
		}

		// Set the combo box as the cell widget in the current row
		this->ui->tableWidget->setCellWidget(cluId, 1, newCombo);

		// Fix row height
		this->ui->tableWidget->setRowHeight(cluId, 20);

		// Get cluster color from the chart
		QColor newColor = this->colorChart.getColor(cluId % this->colorChart.getNumberOfColors());

		// Set new item to the first column of the current row to enable 
		// cell coloring, using the new color from the chart.

		QTableWidgetItem * newItem = new QTableWidgetItem;
		newItem->setBackgroundColor(newColor);
		newItem->setForeground(QBrush(Qt::SolidPattern));
		this->ui->tableWidget->setItem(cluId, 0, newItem);
	}

	// Color the fibers using the new input cluster data
	this->colorInputFibers();
}


//------------------------------[ clearTable ]-----------------------------\\

void ClusteringPlugin::clearTable()
{
	// Loop through all rows in the table widget
	for (int i = 0; i < this->ui->tableWidget->rowCount(); ++i)
	{
		// Get the combo box in the second column of the current row
		QComboBox * currentCombo = (QComboBox *) this->ui->tableWidget->cellWidget(i, 1);

		// Delete existing combo boxes
		if (currentCombo)
			delete currentCombo;

		// Get the item of the first column of the current row (colored cell)
		QTableWidgetItem * currentItem = this->ui->tableWidget->item(i, 0);

		// Delete existing table items
		if (currentItem)
			delete currentItem;
	}

	// Remove all rows and columns
	this->ui->tableWidget->clear();
}


//----------------------------[ dataSetChanged ]---------------------------\\

void ClusteringPlugin::dataSetChanged(data::DataSet * ds)
{
	// Fibers
	if (ds->getKind() == "fibers" && this->fiberList.contains(ds))
	{
		// Ignore changes in the fiber data set if the "dataSetChanged"
		// function was called by the "colorInputFibers" function; this
		// effectively prevents infinite loops.

		if (this->ignoreFiberDSChanged)
		{
			this->ignoreFiberDSChanged = false;
			return;
		}

		// Get the fiber index
		int dsIndex = this->fiberList.indexOf(ds);

		// Change the name in the GUI
		this->ui->fiberCombo->setItemText(dsIndex, ds->getName());

		// Update the fibers
		this->changeFibers();

		return;
	}

	// Clustering information
	if (ds->getKind() == "clusters" && this->clusterList.contains(ds))
	{
		// Get the clustering index
		int dsIndex = this->clusterList.indexOf(ds);

		// Change the name in the GUI
		this->ui->cluCombo->setItemText(dsIndex, ds->getName());

		// Update the cluster list
		this->createClusterList();

		return;
	}
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void ClusteringPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Fibers
	if (ds->getKind() == "fibers" && this->fiberList.contains(ds))
	{
		// Get the fiber index
		int dsIndex = this->fiberList.indexOf(ds);

		// Remove the data set from the list
		this->fiberList.removeAt(dsIndex);

		// Remove the item from the GUI
		this->ui->fiberCombo->removeItem(dsIndex);

		return;
	}

	// Clustering information
	if (ds->getKind() == "clusters" && this->clusterList.contains(ds))
	{
		// Get the fiber index
		int dsIndex = this->clusterList.indexOf(ds);

		// Remove the data set from the list
		this->clusterList.removeAt(dsIndex);

		// Remove the item from the GUI
		this->ui->cluCombo->removeItem(dsIndex);

		return;
	}
}


//-------------------------[ outputClusterExists ]-------------------------\\

bool ClusteringPlugin::outputClusterExists(QString in)
{
	// <None> always exists
	if (in == "<None>")
		return true;

	// Loop through all existing output clusters
	for (int i = 0; i < this->ui->outputCombo->count(); ++i)
	{
		// Return "true" if the current name matches the input
		if (this->ui->outputCombo->itemText(i) == in)
			return true;
	}

	// No match found
	return false;
}


//--------------------------[ getNextDefaultName ]-------------------------\\

QString ClusteringPlugin::getNextDefaultName()
{
	// Try at most one million default names
	for (int cluId = 1; cluId < 1000000; ++cluId)
	{
		// Create the default string using the current loop index
		QString tempName;
		tempName.sprintf("Output Cluster %d", cluId);

		// If the new default name does not yet exist, return it
		if (!(this->outputClusterExists(tempName)))
			return tempName;
	}

	// This only happens when the user has manually added one million
	// output clusters... doesn't seem very likely.

	return (QString) "ERROR: Too many output clusters!";
}


//---------------------------[ addOutputCluster ]--------------------------\\

void ClusteringPlugin::addOutputCluster()
{
	// Is the new name valid?
	bool validName = false;

	// Did the user click "OK"?
	bool ok = false;

	// New output cluster name
	QString newName;

	// Repeat until the user has entered a valid name
	while(!validName)
	{
		// Ask for a new name, using a default name as default text
		newName = QInputDialog::getText(NULL, "Clustering Plugin", "Enter new output cluster name:", 
											QLineEdit::Normal, this->getNextDefaultName(), &ok);

		// If the user pressed "OK"...
		if (ok)
		{
			// If the new name does not yet exist, is not empty, and does not exceed 
			// a character limit, we say that it is valid.
			if (!(this->outputClusterExists(newName)) && !(newName.isEmpty()) && !(newName.length() > 100))
				validName = true;
			// Otherwise, display an error message, and repeat the loop
			else
				QMessageBox::warning(NULL, "Clustering Plugin", "Invalid name!");
		}
		// If the user pressed "Cancel", we exit the function here
		else
		{
			return;
		}
	}

	// Add new output to the Output Cluster combo box
	this->ui->outputCombo->addItem(newName);
	this->outputClusterChanged();

	// Add a new output cluster information object to the list, with the 
	// name equal to the text in the combo box, the color equal to white,
	// and automatic coloring on by default.

	outputClusterInformation newOutputInfo;
	newOutputInfo.name = newName;
	newOutputInfo.color = QColor(255, 255, 255);
	newOutputInfo.useAutoColor = true;
	this->outputInfoList.append(newOutputInfo);

	// Loop through all rows of the table widget
	for (int cluId = 0; cluId < this->numberOfInputClusters; ++cluId)
	{
		// Get the combo box in the second column of the current row
		QComboBox * currentCombo = (QComboBox * )this->ui->tableWidget->cellWidget(cluId, 1);

		// Check if the combo box exists
		if (!currentCombo)
			continue;

		// Add the new name to the combo box
		currentCombo->addItem(newName);
	}
}


//-------------------------[ renameOutputCluster ]-------------------------\\

void ClusteringPlugin::renameOutputCluster()
{
	// Is the new name valid?
	bool validName = false;

	// Did the user click "OK"?
	bool ok = false;

	// New output cluster name
	QString newName;

	// Get the index of the current output cluster
	int currentClusterIndex = this->ui->outputCombo->currentIndex();

	// Check if the index is within range
	if (currentClusterIndex < 0 || currentClusterIndex > this->ui->outputCombo->count())
		return;

	// Get the current cluster name
	QString currentName = this->ui->outputCombo->itemText(currentClusterIndex);

	// Repeat until the user has entered a valid name
	while(!validName)
	{
		// Ask for a new name, usingthe current name as default text
		newName = QInputDialog::getText(NULL, "Clustering Plugin", "Enter new output cluster name:", 
											QLineEdit::Normal, currentName, &ok);

		// If the user pressed "OK"...
		if (ok)
		{
			// If the new name does not yet exist, is not empty, and does not exceed 
			// a character limit, we say that it is valid.
			if (!(this->outputClusterExists(newName)) && !(newName.isEmpty()) && !(newName.length() > 100))
			{
				validName = true;
			}
			else
			{
				// Do nothing if the new name is the same as the old one
				if (newName == currentName)
					return;
			
				// Otherwise, display an error message, and repeat the loop
				QMessageBox::warning(NULL, "Clustering Plugin", "Invalid name!");
			}
		}
		// If the user pressed "Cancel", we exit the function here
		else
		{
			return;
		}
	}

	// Change the text in the combo box
	this->ui->outputCombo->setItemText(currentClusterIndex, newName);

	// Get the current output cluster information, change its name, and re-add it to the list
	outputClusterInformation currentOutputInfo = this->outputInfoList.at(currentClusterIndex);
	currentOutputInfo.name = newName;
	this->outputInfoList.replace(currentClusterIndex, currentOutputInfo);

	// Loop through all rows of the table widget
	for (int cluId = 0; cluId < this->numberOfInputClusters; ++cluId)
	{
		// Get the combo box in the second column of the current row
		QComboBox * currentCombo = (QComboBox * )this->ui->tableWidget->cellWidget(cluId, 1);

		// Check if the combo box exists
		if (!currentCombo)
			continue;

		// Change the text of the current combo box
		currentCombo->setItemText(currentClusterIndex + 1, newName);
	}
}


//--------------------------[ removeOutputCluster ]-------------------------\\

void ClusteringPlugin::removeOutputCluster()
{
	// Get the index of the current output cluster
	int currentClusterIndex = this->ui->outputCombo->currentIndex();

	// Check if the index is within range
	if (currentClusterIndex < 0 || currentClusterIndex > this->ui->outputCombo->count())
		return;

	// Remove item from the information list, and from the combo box
	this->outputInfoList.removeAt(currentClusterIndex);
	this->ui->outputCombo->removeItem(currentClusterIndex);

	// Trigger update of the GUI
	this->outputClusterChanged();

	// Loop through all rows of the table widget
	for (int cluId = 0; cluId < this->numberOfInputClusters; ++cluId)
	{
		// Get the combo box in the second column of the current row
		QComboBox * currentCombo = (QComboBox * )this->ui->tableWidget->cellWidget(cluId, 1);

		// Check if the combo box exists
		if (!currentCombo)
			continue;

		// If the deleted item is the selected item of this combo box, change
		// the selection to the first item ("<None>").

		if (currentCombo->currentIndex() == currentClusterIndex + 1)
		{
			currentCombo->setCurrentIndex(0);
		}

		// Remove the item from this combo box
		currentCombo->removeItem(currentClusterIndex + 1);
	}
}


//-----------------------[ getNumberOfInputClusters ]----------------------\\

int ClusteringPlugin::getNumberOfInputClusters(vtkStructuredPoints * in)
{
	// Get the number of points in the clustering information data, which should
	// be equal to the number of lines in the input fiber set.

	int numberOfLines = in->GetNumberOfPoints();

	// Check if the clustering information contains point data
	if (!(in->GetPointData()))
		return -1;

	// Get the scalars from the input
	vtkDataArray * clusteringScalars = in->GetPointData()->GetScalars();

	// Check if the scalars exist
	if (!(clusteringScalars))
		return -1;

	// Create a list of cluster IDs
	QList<double> clusterIdList;

	// Loop through all points in the clustering information
	for (vtkIdType ptId = 0; ptId < numberOfLines; ++ptId)
	{
		// Get the cluster ID (as a double)
		double c = clusteringScalars->GetTuple1(ptId);

		// If the cluster ID is already in the list, do nothing
		if (clusterIdList.contains(c))
			continue;

		// Otherwise, add it to the list
		clusterIdList.append(c);
	}

	// Return the size of the ID list, which equals the number of IDs
	return clusterIdList.size();
}


//---------------------------[ colorInputFibers ]--------------------------\\

void ClusteringPlugin::colorInputFibers()
{
	// Get the index of the selected fiber set
	int currentFiberIndex = this->ui->fiberCombo->currentIndex();

	// Check if the index is within range
	if (currentFiberIndex < 0 || currentFiberIndex >= this->ui->fiberCombo->count())
		return;

	// get the data set of the selected fibers
	data::DataSet * currentFiberDS = this->fiberList.at(currentFiberIndex);

	// Check if the data set exists
	if (!currentFiberDS)
		return;

	// Get the poly data object of the fibers
	vtkPolyData * currentFibers = currentFiberDS->getVtkPolyData();

	// Check if the poly data object exists
	if (!currentFibers)
		return;

	// Get the index of the clustering information object
	int currentClusterIndex = this->ui->cluCombo->currentIndex();

	// Check if the cluster index is within range
	if (currentClusterIndex < 0 || currentClusterIndex >= this->ui->cluCombo->count())
		return;

	// Get the data set of the clustering information
	data::DataSet * currentClusterDS = this->clusterList.at(currentClusterIndex);

	// Check if the clustering data set exists
	if (!currentClusterDS)
		return;

	// Get the structured points data of the clustering information
	vtkStructuredPoints * currentCluster = vtkStructuredPoints::SafeDownCast(currentClusterDS->getVtkObject());

	// Check if the points exist
	if (!currentCluster)
		return;

	// Get the scalar array containing the input cluster IDs
	vtkDataArray * currentClusterScalars = currentCluster->GetPointData()->GetScalars();

	// Check if the scalar array exists
	if (!(currentClusterScalars))
		return;

	// Check if the number of points in the clustering information is equal to
	// the number of fiber lines in the fiber set. 

	if (currentFibers->GetNumberOfLines() != currentCluster->GetNumberOfPoints())
		return;

	// Create the output scalar array, which consists of three unsigned characters
	// (RGB values) for each input fiber.

	vtkUnsignedCharArray * newScalars = vtkUnsignedCharArray::New();
	newScalars->SetNumberOfComponents(3);
	newScalars->SetNumberOfTuples(currentFibers->GetNumberOfLines());

	// Get the cell data of the fibers, and set its scalar array to the newly created
	// unsigned character array. This will add RGB values to each cell (fiber).

	vtkCellData * currentCellData = currentFibers->GetCellData();
	currentCellData->SetScalars(newScalars);
	newScalars->Delete();

	// Get the input lines
	vtkCellArray * currentLines = currentFibers->GetLines();

	// Initialize traversal of the input fibers
	currentLines->InitTraversal();

	// Output fiber color
	unsigned char outputRGB[3];

	// Get the index of the selected input cluster in the table widget
	int selectedCluster = this->ui->tableWidget->currentRow();

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < currentFibers->GetNumberOfLines(); ++lineId)
	{
		// Get the input cluster ID for the current line
		int clusterId = (int) currentClusterScalars->GetTuple1(lineId);

		// Use the color chart to get the input color
		QColor currentColor = this->colorChart.getColor(clusterId % this->colorChart.getNumberOfColors());

		// If this line belongs to the selected cluster, use the "lighter" function
		// to make the input color lighter, to show that it is selected.

		if (clusterId == selectedCluster)
		{
			currentColor = this->colorChart.lighter(currentColor, 0.5);
		}

		// Copy the input color to the output
		outputRGB[0] = currentColor.red();
		outputRGB[1] = currentColor.green();
		outputRGB[2] = currentColor.blue();

		// Set the output color in the scalar array
		newScalars->SetTupleValue(lineId, outputRGB);
	}

	// Temporary attribute value
	double attribute;

	// Set "updatePipeline" to 1.0, to signal the visualization plugin that it should re-execute the pipeline.
	currentFiberDS->getAttributes()->addAttribute("updatePipeline", 1.0);

	// Don't trigger the "dataSetChanged" function of this class
	this->ignoreFiberDSChanged = true;

	// Tell the core that the input fiber set has changed
	this->core()->data()->dataSetChanged(currentFiberDS);
}


//-------------------------[ updateOutputClusters ]------------------------\\

void ClusteringPlugin::updateOutputClusters()
{
	// Get pointers of the input fiber set and the clustering information, in
	// the same way as in the "colorInputFibers" function. 

	int currentFiberIndex = this->ui->fiberCombo->currentIndex();

	if (currentFiberIndex < 0 || currentFiberIndex >= this->ui->fiberCombo->count())
		return;

	data::DataSet * currentFiberDS = this->fiberList.at(currentFiberIndex);

	if (!currentFiberDS)
		return;

	vtkPolyData * currentFibers = currentFiberDS->getVtkPolyData();

	if (!currentFibers)
		return;

	int currentClusterIndex = this->ui->cluCombo->currentIndex();

	if (currentClusterIndex < 0 || currentClusterIndex >= this->ui->cluCombo->count())
		return;

	data::DataSet * currentClusterDS = this->clusterList.at(currentClusterIndex);

	if (!currentClusterDS)
		return;

	vtkStructuredPoints * currentCluster = vtkStructuredPoints::SafeDownCast(currentClusterDS->getVtkObject());

	if (!currentCluster)
		return;

	vtkDataArray * currentClusterScalars = currentCluster->GetPointData()->GetScalars();

	if (!(currentClusterScalars))
		return;

	// Check if the number of points in the clustering information is equal to
	// the number of fiber lines in the fiber set. 

	if (currentFibers->GetNumberOfLines() != currentCluster->GetNumberOfPoints())
		return;

	// Create an iterator for the list of existing output clusters
	QList<data::DataSet *>::iterator oldDSIter;

	// Delete all existing clusters
	for (oldDSIter = this->addedDataSets.begin(); oldDSIter != this->addedDataSets.end(); ++oldDSIter)
	{
		this->core()->data()->removeDataSet(*oldDSIter);
	}

	// Clear the list of existing clusters
	this->addedDataSets.clear();

	// Get the number of output clusters
	int numberOfOutputClusters = this->ui->outputCombo->count();

	// Allocate arrays for the output pointers
	vtkPolyData **          newFibers   = (vtkPolyData **)          malloc(numberOfOutputClusters * sizeof(vtkPolyData *         ));
	vtkPoints **            newPoints   = (vtkPoints **)            malloc(numberOfOutputClusters * sizeof(vtkPoints *           ));
	vtkCellArray **         newLines    = (vtkCellArray **)         malloc(numberOfOutputClusters * sizeof(vtkCellArray *        ));
	vtkUnsignedCharArray ** newScalars  = (vtkUnsignedCharArray **) malloc(numberOfOutputClusters * sizeof(vtkUnsignedCharArray *));
	vtkCellData **          newCellData = (vtkCellData **)          malloc(numberOfOutputClusters * sizeof(vtkCellData *         ));

	// For each output cluster...
	for (int i = 0; i < numberOfOutputClusters; ++i)
	{
		// Create a new poly data object
		newFibers[i] = vtkPolyData::New();
		vtkPointData * newPD = newFibers[i]->GetPointData();

		// Create arrays for lines (fibers) and points
		newPoints[i] = vtkPoints::New();
		newLines[i] = vtkCellArray::New();
		newFibers[i]->SetPoints(newPoints[i]);
		newFibers[i]->SetLines(newLines[i]);
		newPoints[i]->Delete();
		newLines[i]->Delete();

		// Create a scalar array for the output RGB values
		newScalars[i] = vtkUnsignedCharArray::New();
		newScalars[i]->SetNumberOfComponents(3);
		newScalars[i]->Allocate(1000);

		// Get the cell data of the output fiber set
		newCellData[i] = newFibers[i]->GetCellData();
	}

	// Get the input fiber lines, and initialize their traversal
	vtkCellArray * currentLines = currentFibers->GetLines();
	currentLines->InitTraversal();

	// List of point IDs of the current fiber
	vtkIdType * pointList;

	// Number of points in the current fiber
	vtkIdType numberOfPoints;

	// List of point IDs in the output fiber
	vtkIdList * newPointList = vtkIdList::New();

	// Output color
	unsigned char outputRGB[3];

	// Loop through all input fibers
	for (vtkIdType lineId = 0; lineId < currentFibers->GetNumberOfLines(); ++lineId)
	{
		// Get point IDs and number of point of the current fiber
		currentLines->GetNextCell(numberOfPoints, pointList);

		// Get the input cluster ID of the current fiber
		int inputClusterId = (int) currentClusterScalars->GetTuple1(lineId);

		// Get the combo box belonging to the input cluster
		QComboBox * currentComboBox = (QComboBox *) this->ui->tableWidget->cellWidget(inputClusterId, 1);

		// Check if the combo box exists
		if (!currentComboBox)
			continue;

		// Get the output cluster index for this input cluster
		int outputClusterId = currentComboBox->currentIndex() - 1;

		// Check if the output cluster ID is within range
		if (outputClusterId < 0 || outputClusterId >= this->ui->outputCombo->count())
			continue;

		// Loop through all points in the current fiber
		for (vtkIdType ptId = 0; ptId < numberOfPoints; ++ptId)
		{
			// Temporary point coordinates
			double tempX[3];

			// Get current fiber point
			currentFibers->GetPoint(pointList[ptId], tempX);

			// Write point to output
			vtkIdType newPointId = newPoints[outputClusterId]->InsertNextPoint(tempX);

			// Write new point ID to output list
			newPointList->InsertNextId(newPointId);
		}

		// Save output list as a new fiber
		vtkIdType newLineId = newLines[outputClusterId]->InsertNextCell(newPointList);

		// Reset the point list
		newPointList->Reset();

		// Color of the output fiber
		QColor currentOutputColor;
		
		// Get output information for the current output cluster
		outputClusterInformation currentOutputInfo = this->outputInfoList.at(outputClusterId);

		// Either get the output color from the color chart...
		if (currentOutputInfo.useAutoColor)
		{
			currentOutputColor = this->colorChart.getColor(outputClusterId % this->colorChart.getNumberOfColors());
		}
		// ... or use the manually defined color
		else
		{
			currentOutputColor = currentOutputInfo.color;
		}

		// Copy color to the output array
		outputRGB[0] = (unsigned char) currentOutputColor.red();
		outputRGB[1] = (unsigned char) currentOutputColor.green();
		outputRGB[2] = (unsigned char) currentOutputColor.blue();

		// Save color in new scalar array
		newScalars[outputClusterId]->InsertTupleValue(newLineId, outputRGB);
	}

	// We're done with the point list, delete it
	newPointList->Delete();

	// For each of the output clusters...
	for (int i = 0; i < numberOfOutputClusters; ++i)
	{
		// Check if lines have been added (otherwise, do nothing)
		if (newFibers[i]->GetNumberOfCells() > 0)
		{
			// Set output RGB values to the cell data
			newCellData[i]->SetScalars(newScalars[i]);

			// Generate output name
			QString dsName = this->ui->fiberCombo->currentText() + " - " + this->ui->outputCombo->itemText(i);

			// Create a new data set for this cluster, and add it to the data manager
			data::DataSet * newDS = new data::DataSet(dsName, "fibers", (vtkObject *) newFibers[i]);

			// Check if the input fiber set contains a transformation matrix
			newDS->getAttributes()->copyTransformationMatrix(currentFiberDS);

			this->core()->data()->addDataSet(newDS);

			// Add new data set to the list of existing data sets
			this->addedDataSets.append(newDS);
		}

		// Remove the scalar array
		newScalars[i]->Delete();
	}

	// Don't show the input fibers
	this->showInput(false);

	// Free the allocated pointer arrays
	free(newFibers);
	free(newPoints);
	free(newLines);
	free(newScalars);
	free(newCellData);

}


//-------------------------[ outputClusterChanged ]------------------------\\

void ClusteringPlugin::outputClusterChanged()
{
	// Get the current output cluster index
	int currentClusterIndex = this->ui->outputCombo->currentIndex();

	// Check if the index is within range
	if (currentClusterIndex < 0 || currentClusterIndex > this->ui->outputCombo->count())
		return;

	// Get the information of the selected output cluster
	outputClusterInformation currentOutputInfo = this->outputInfoList.at(currentClusterIndex);

	// Update the GUI based on the "useAutoColor" value
	this->ui->outputColorAutoRadio->setChecked(currentOutputInfo.useAutoColor);
	this->ui->outputColorManualRadio->setChecked(!(currentOutputInfo.useAutoColor));
	this->ui->outputColorPickerButton->setEnabled(!(currentOutputInfo.useAutoColor));
}

//-------------------------[ setColorAutoOrManual ]------------------------\\

void ClusteringPlugin::setColorAutoOrManual()
{
	// Get the current output cluster index
	int currentClusterIndex = this->ui->outputCombo->currentIndex();

	// Check if the index is within range
	if (currentClusterIndex < 0 || currentClusterIndex > this->ui->outputCombo->count())
		return;

	// Get the information of the selected output cluster
	outputClusterInformation currentOutputInfo = this->outputInfoList.at(currentClusterIndex);

	// Set the "useAutoColor" value based on the GUI state
	currentOutputInfo.useAutoColor = this->ui->outputColorAutoRadio->isChecked();

	// Replace the updated information object in the list
	this->outputInfoList.replace(currentClusterIndex, currentOutputInfo);

	// Update the GUI based on the new information
	this->outputClusterChanged();
}


//----------------------------[ setManualColor ]---------------------------\\

void ClusteringPlugin::setManualColor()
{
	// Get the current output cluster index
	int currentClusterIndex = this->ui->outputCombo->currentIndex();

	// Check if the index is within range
	if (currentClusterIndex < 0 || currentClusterIndex > this->ui->outputCombo->count())
		return;

	// Get the information of the selected output cluster
	outputClusterInformation currentOutputInfo = this->outputInfoList.at(currentClusterIndex);

	// Use a color dialog to get a new color
	QColor newColor = QColorDialog::getColor(currentOutputInfo.color, 0);

	// If the new color is valid, copy it to the information object,
	// and replace the updated information in the list.
    if (newColor.isValid())
	{
		currentOutputInfo.color = newColor;
		this->outputInfoList.replace(currentClusterIndex, currentOutputInfo);
	}
}

//------------------------------[ showInput ]------------------------------\\

void ClusteringPlugin::showInput(bool show)
{
	// Get the input fiber data set
	int currentFiberIndex = this->ui->fiberCombo->currentIndex();

	if (currentFiberIndex < 0 || currentFiberIndex >= this->ui->fiberCombo->count())
		return;

	data::DataSet * currentFiberDS = this->fiberList.at(currentFiberIndex);

	if (!currentFiberDS)
		return;
	
	double visibility = (show) ? (1.0) : (-1.0);

	// Turn the visibility on or off
	currentFiberDS->getAttributes()->addAttribute("isVisible", visibility);

	// We do not need to re-execute the visualization pipeline
	currentFiberDS->getAttributes()->addAttribute("updatePipeline", 0.0);

	// Tell the data manager that we modified the data set
	this->core()->data()->dataSetChanged(currentFiberDS);
}


//------------------------------[ showOutput ]-----------------------------\\

void ClusteringPlugin::showOutput(bool show)
{
	double visibility = (show) ? (1.0) : (-1.0);
	double attribute;

	// Loop through all added output clusters
	for (int i = 0; i < this->addedDataSets.size(); ++i)
	{
		// Get the output data set
		data::DataSet * ds = this->addedDataSets.at(i);

		// Turn the visibility on or off
		ds->getAttributes()->addAttribute("isVisible", visibility);

		// We do not need to re-execute the visualization pipeline
		ds->getAttributes()->addAttribute("updatePipeline", 0.0);

		// Tell the data manager that we modified the data set
		this->core()->data()->dataSetChanged(ds);
	}
}


//-------------------------[ showInputHideOutput ]-------------------------\\

void ClusteringPlugin::showInputHideOutput()
{
	this->showInput(true);
	this->showOutput(false);
}


//-------------------------[ showOutputHideInput ]-------------------------\\

void ClusteringPlugin::showOutputHideInput()
{
	this->showInput(false);
	this->showOutput(true);
}


//----------------------------[ writeSettings ]----------------------------\\

void ClusteringPlugin::writeSettings()
{
	// Clustering information needs to be selected
	if (this->ui->cluCombo->currentIndex() == -1)
	{
		QMessageBox::warning(NULL, "Clustering Plugin", "Please load clustering information first!");
		return;
	}

	// Write the settings to a file
	ClusteringSettingsIO::writeOutputFile(this->ui->tableWidget);
}


//-----------------------------[ readSettings ]----------------------------\\

void ClusteringPlugin::readSettings()
{
	// Clustering information needs to be selected
	if (this->ui->cluCombo->currentIndex() == -1)
	{
		QMessageBox::warning(NULL, "Clustering Plugin", "Please load clustering information first!");
		return;
	}

	// Create the reader
	ClusteringSettingsIO * io = new ClusteringSettingsIO;

	// Try to open the settings file
	if (io->openFileForReading())
	{
		// Check if all cluster IDs in the .bun file are in the same range as the
		// input IDs in the selected clustering file.

		if (!(io->checkClusteringIDs(this->ui->tableWidget->rowCount())))
		{
			QMessageBox::warning(NULL, "Clustering Plugin", "Illegal input cluster IDs in .bun file!");
			return;
		}

		// Add new output cluster names to the combo boxes
		io->populateOutputClusters(this);

		// Set the combo boxes in the table to the stored settings
		io->setOutputClusters(this->ui->tableWidget);
	}

	// Delete the reader
	delete io;
}


//-----------------------[ addOutputClusterFromFile ]----------------------\\

void ClusteringPlugin::addOutputClusterFromFile(QString name)
{
	// Do nothing if the name already exists
	if (this->outputClusterExists(name))
		return;

	// Add new output to the Output Cluster combo box
	this->ui->outputCombo->addItem(name);
	this->outputClusterChanged();

	// Add a new output cluster information object to the list, with the 
	// name equal to the text in the combo box, the color equal to white,
	// and automatic coloring on by default.

	outputClusterInformation newOutputInfo;
	newOutputInfo.name = name;
	newOutputInfo.color = QColor(255, 255, 255);
	newOutputInfo.useAutoColor = true;
	this->outputInfoList.append(newOutputInfo);

	// Loop through all rows of the table widget
	for (int cluId = 0; cluId < this->numberOfInputClusters; ++cluId)
	{
		// Get the combo box in the second column of the current row
		QComboBox * currentCombo = (QComboBox * )this->ui->tableWidget->cellWidget(cluId, 1);

		// Check if the combo box exists
		if (!currentCombo)
			continue;

		// Add the new name to the combo box
		currentCombo->addItem(name);
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libClusteringPlugin, bmia::ClusteringPlugin)
