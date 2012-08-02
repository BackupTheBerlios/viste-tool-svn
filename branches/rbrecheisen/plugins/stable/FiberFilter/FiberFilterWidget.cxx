/*
 * FiberFilterWidget.cxx
 *
 * 2010-10-05	Tim Peeters
 * - First version
 *
 * 2010-11-08	Evert van Aart
 * - Added filtering functionality
 *
 * 2010-11-22	Evert van Aart
 * - Fixed output name formatting
 *
 */


/** Includes */

#include "FiberFilterWidget.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberFilterWidget::FiberFilterWidget(FiberFilterPlugin * ffp, int rFilterID)
{
	// Assert the plugin pointer
    Q_ASSERT(ffp);

	// Store the plugin pointer
    this->plugin = ffp;

	// Store the filter ID
	this->filterID = rFilterID;

	// Setup the GUI
    this->setupUi(this);

	// Add default items to all combo boxes
    this->roiBox1->addItem("Off");
    this->roiBox2->addItem("Off");
    this->roiBox3->addItem("Off");
    this->roiBox4->addItem("Off");
    this->roiBox5->addItem("Off");
    this->inputBox->addItem("None");

	// Connect update button to "update" function
    connect(this->updateButton, SIGNAL(clicked()), this, SLOT(update()));

	// Enable or disable certain GUI controls
	connect(this->roiBox1,   SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()));
	connect(this->roiBox2,   SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()));
	connect(this->roiBox3,   SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()));
	connect(this->roiBox4,   SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()));
	connect(this->roiBox5,   SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()));
	connect(this->inputBox,  SIGNAL(currentIndexChanged(int)), this, SLOT(enableControls()));
	connect(this->nameEdit,  SIGNAL(textChanged(QString)),     this, SLOT(enableControls()));
	connect(this->notCheck1, SIGNAL(clicked()),                this, SLOT(enableControls()));
	connect(this->notCheck2, SIGNAL(clicked()),                this, SLOT(enableControls()));
	connect(this->notCheck3, SIGNAL(clicked()),                this, SLOT(enableControls()));
	connect(this->notCheck4, SIGNAL(clicked()),                this, SLOT(enableControls()));
	connect(this->notCheck5, SIGNAL(clicked()),                this, SLOT(enableControls()));

	// Connect other slot function to GUI signals
	connect(this->inputBox, SIGNAL(currentIndexChanged(int)), this, SLOT(setOutputName()));
	connect(this->nameEdit, SIGNAL(textChanged(QString)), this, SLOT(nameChanged()));

	// At first, no name has been set yet, so we use the default output name
	this->outputNameModified = false;
}


//------------------------------[ Destructor ]-----------------------------\\

FiberFilterWidget::~FiberFilterWidget()
{

}


//--------------------------------[ update ]-------------------------------\\

void FiberFilterWidget::update()
{
	// Get the fiber data set from the plugin
	vtkPolyData * fibers = this->plugin->getFibers(this->inputBox->currentIndex());

	// Get the input transformation matrix
	vtkMatrix4x4 * m = this->plugin->getTransformationMatrix(this->inputBox->currentIndex());

	// Check if the fibers exist
	if (!fibers)
		return;

	// Create a new filter
	vtk2DROIFiberFilter * fiberFilter = vtk2DROIFiberFilter::New();

	// Set the input of the filters
	fiberFilter->SetInput(fibers);

	// Only cut the fibers off at the first and last ROI if the option
	// has been checked, and if this checkbox is currently enabled.
	fiberFilter->SetCutFibersAtROI(this->cutFibersCheck->isChecked() && this->cutFibersCheck->isEnabled());

	// Polydata object representing the ROI
	vtkPolyData * roi;

	// Is this a "NOT"-type ROI?
	bool bNOT;

	// For all five ROIs, get the options/pointers from the GUI 
	// and add them to the filter.

	roi = this->plugin->getROI(this->roiBox1->currentIndex());
	bNOT = this->notCheck1->isChecked();
	fiberFilter->addROI(roi, bNOT);

	roi = this->plugin->getROI(this->roiBox2->currentIndex());
	bNOT = this->notCheck2->isChecked();
	fiberFilter->addROI(roi, bNOT);

	roi = this->plugin->getROI(this->roiBox3->currentIndex());
	bNOT = this->notCheck3->isChecked();
	fiberFilter->addROI(roi, bNOT);

	roi = this->plugin->getROI(this->roiBox4->currentIndex());
	bNOT = this->notCheck4->isChecked();
	fiberFilter->addROI(roi, bNOT);

	roi = this->plugin->getROI(this->roiBox5->currentIndex());
	bNOT = this->notCheck5->isChecked();
	fiberFilter->addROI(roi, bNOT);

	// Update the filter to filter the fibers
	fiberFilter->Update();

	// Get the output fibers
	vtkPolyData * output = fiberFilter->GetOutput();

	// Add this fiber data set to the output
	bool dataSetWasAdded = this->plugin->addFibersToDataManager(output, this->nameEdit->text(), this->filterID, m);

	// Hide the input fibers of the filter
	if (dataSetWasAdded)
		this->plugin->hideInputFibers(this->inputBox->currentIndex());

	// Remove the filter
	fiberFilter->Delete();
}


//-----------------------------[ fibersAdded ]-----------------------------\\

void FiberFilterWidget::fibersAdded(data::DataSet* ds)
{
	// Assert data set and type
    Q_ASSERT(ds);
    Q_ASSERT(ds->getKind() == "fibers");

	// Add the name of the data set to the "Input" combo box
	this->inputBox->addItem(ds->getName());
}


//----------------------------[ fibersChanged ]----------------------------\\

void FiberFilterWidget::fibersChanged(int index, QString newName)
{
	// Check if the index is within range
	if (index < 0 || index >= this->inputBox->count())
		return;

	// Replace the fiber name
	this->inputBox->setItemText(index, newName);

	// If the renamed item has been selected, and the user has not (yet)
	// modified the output name, we change the output name based on the
	// "newName" string.

	if (this->inputBox->currentIndex() == index && this->outputNameModified == false)
		this->setOutputName();
}


//----------------------------[ fibersRemoved ]----------------------------\\

void FiberFilterWidget::fibersRemoved(int index)
{
	// Check if the index is within range
	if (index < 0 || index >= this->inputBox->count())
		return;

	// If the fiber set being deleted was currently selected,
	// we change the index of the input combo box to zero ("None")
	if (this->inputBox->currentIndex() == index)	
	{
		this->inputBox->setCurrentIndex(0);

		// Rename the output if needed
		this->setOutputName();
	}

	// Delete the item from the combo box
	this->inputBox->removeItem(index);
}


//-------------------------------[ roiAdded ]------------------------------\\

void FiberFilterWidget::roiAdded(data::DataSet* ds)
{
	// Assert data set and type
    Q_ASSERT(ds);
    Q_ASSERT(ds->getKind() == "regionOfInterest");

	// Add the name of the ROI to all five ROI combo boxes
    this->roiBox1->addItem(ds->getName());
    this->roiBox2->addItem(ds->getName());
    this->roiBox3->addItem(ds->getName());
    this->roiBox4->addItem(ds->getName());
    this->roiBox5->addItem(ds->getName());
}


//------------------------------[ roiChanged ]-----------------------------\\

void FiberFilterWidget::roiChanged(int index, QString newName)
{
	// Check if the index is within range
	if (index < 0 || index >= this->roiBox1->count())
		return;

	// Replace the ROI name in all five ROI combo boxes.
	this->roiBox1->setItemText(index, newName);
	this->roiBox2->setItemText(index, newName);
	this->roiBox3->setItemText(index, newName);
	this->roiBox4->setItemText(index, newName);
	this->roiBox5->setItemText(index, newName);
}


//------------------------------[ roiRemoved ]-----------------------------\\

void FiberFilterWidget::roiRemoved(int index)
{
   	// Check if the index is within range
	if (index < 0 || index >= this->roiBox1->count())
		return;

	// Check if the ROI being deleted has been selected in one
	// of the ROI combo boxes; if so, reset that box to "Off".

	if (this->roiBox1->currentIndex() == index)		this->roiBox1->setCurrentIndex(0);
	if (this->roiBox2->currentIndex() == index)		this->roiBox2->setCurrentIndex(0);
	if (this->roiBox3->currentIndex() == index)		this->roiBox3->setCurrentIndex(0);
	if (this->roiBox4->currentIndex() == index)		this->roiBox4->setCurrentIndex(0);
	if (this->roiBox5->currentIndex() == index)		this->roiBox5->setCurrentIndex(0);

	// Remove the item from all combo boxes
	this->roiBox1->removeItem(index);
	this->roiBox1->removeItem(index);
	this->roiBox1->removeItem(index);
	this->roiBox1->removeItem(index);
	this->roiBox1->removeItem(index);

	// Enable/disable GUI controls based on the new situation
	this->enableControls();
}


//----------------------------[ enableControls ]---------------------------\\

void FiberFilterWidget::enableControls()
{
	// Everything is disabled if no input has been selected
	bool globalEnable = this->inputBox->currentIndex() != 0;

	// The update button is also disabled if no output name has been set
	bool updateEnable = !(this->nameEdit->text().isEmpty());

	// Enable/disable controls based on "globalEnable"
	this->filterROIsLabel->setEnabled(globalEnable);
	this->roiBox1->setEnabled(globalEnable);
	this->roiBox2->setEnabled(globalEnable);
	this->roiBox3->setEnabled(globalEnable);
	this->roiBox4->setEnabled(globalEnable);
	this->roiBox5->setEnabled(globalEnable);
	this->nameLabel->setEnabled(globalEnable);
	this->nameEdit->setEnabled(globalEnable);

	// Only enable "NOT" checkboxes if the corresponding ROI has been set
	this->notCheck1->setEnabled(globalEnable && this->roiBox1->currentIndex() != 0);
	this->notCheck2->setEnabled(globalEnable && this->roiBox2->currentIndex() != 0);
	this->notCheck3->setEnabled(globalEnable && this->roiBox3->currentIndex() != 0);
	this->notCheck4->setEnabled(globalEnable && this->roiBox4->currentIndex() != 0);
	this->notCheck5->setEnabled(globalEnable && this->roiBox5->currentIndex() != 0);

	// Compute the number of ROIs for which "NOT" is false
	int numberOfROIs = 0;
	
	if (this->roiBox1->currentIndex() != 0 && !(this->notCheck1->isChecked()))		numberOfROIs++;
	if (this->roiBox2->currentIndex() != 0 && !(this->notCheck1->isChecked()))		numberOfROIs++;
	if (this->roiBox3->currentIndex() != 0 && !(this->notCheck1->isChecked()))		numberOfROIs++;
	if (this->roiBox4->currentIndex() != 0 && !(this->notCheck1->isChecked()))		numberOfROIs++;
	if (this->roiBox5->currentIndex() != 0 && !(this->notCheck1->isChecked()))		numberOfROIs++;

	// If we have at least two non-"NOT" ROIs, we can cut the fibers at the extreme ROIs
	this->cutFibersCheck->setEnabled(globalEnable && numberOfROIs > 1);

	// Now compute the number of ROIs that have been set
	numberOfROIs = 0;

	if (this->roiBox1->currentIndex() != 0)		numberOfROIs++;
	if (this->roiBox2->currentIndex() != 0)		numberOfROIs++;
	if (this->roiBox3->currentIndex() != 0)		numberOfROIs++;
	if (this->roiBox4->currentIndex() != 0)		numberOfROIs++;
	if (this->roiBox5->currentIndex() != 0)		numberOfROIs++;

	// Update the button if and only if 1) the input has been set, 2) an output name
	// has been set, and 3) the number of ROIs is more than zero.
	this->updateButton->setEnabled(globalEnable && updateEnable && numberOfROIs > 0);
}


//----------------------------[ setOutputName ]----------------------------\\

void FiberFilterWidget::setOutputName()
{
	// Only change the name if the user has not yet modified the output name himself
	if (this->outputNameModified == false)
	{
		// Disconnect the "enableControls" function from the "nameEdit" to avoid
		// unnecessary triggering of "enableControls".
		disconnect(this->nameEdit, SIGNAL(textChanged(QString)), this, SLOT(enableControls()));
		
		// Check if the input has been set to "None"
		if (this->inputBox->currentIndex() != 0)
		{
			// Set the new output name, which is the input fiber name + " - Filter X"
			QString newName = this->inputBox->currentText() + " - Filter " + QString::number(this->filterID);
			this->nameEdit->setText(newName);
		}
		else
		{
			// Clear the line edit dialog
			this->nameEdit->setText("");
		}

		// Reconnect the line edit dialog to the "enableControls" function
		connect(this->nameEdit, SIGNAL(textChanged(QString)), this, SLOT(enableControls()));
	}
}


//-----------------------------[ nameChanged ]-----------------------------\\

void FiberFilterWidget::nameChanged()
{
	// If the line edit is empty, we (re-)enable automatic output name selection
	if (this->nameEdit->text().isEmpty())
		this->outputNameModified = false;
	// Otherwise, it means that the user has changed the text, and we disable
	// automatic changing of the output name.
	else
		this->outputNameModified = true;
}


} // namespace bmia
