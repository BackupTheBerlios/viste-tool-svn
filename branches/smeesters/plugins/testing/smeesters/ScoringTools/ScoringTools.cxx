#include "ScoringTools.h"

#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

namespace bmia
{

///
///      INITIALIZATION
///

//------------------------[ Plugin constructor ]-----------------------\\

ScoringTools::ScoringTools() : plugin::AdvancedPlugin("ScoringTools")
{
    this->widget = NULL;
    this->form   = NULL;
}

//------------------------[ Plugin destructor ]-----------------------\\

ScoringTools::~ScoringTools()
{
    delete this->widget;
    delete this->form;
}

//------------------------[ Initialization ]-----------------------\\

void ScoringTools::init()
{
    this->widget = new QWidget();
    this->form = new Ui::ScoringToolsForm();
    this->form->setupUi(this->widget);

    // Link events in the GUI to function calls
    this->connectAll();
    this->assembly = vtkPropAssembly::New();

    // default selected fiber (none)
    selectedFiberDataset = -1;
}

///
///      DATA I/O
///

//------------------------[ Dataset added ]-----------------------\\

void ScoringTools::dataSetAdded(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // Load fiber dataset
    if (kind == "fibers")
	{
	    // Check if fiber has polydata
	    if (d->getVtkPolyData() == NULL)
			return;

        // Create new fiber struct
        SortedFibers* sortedFibers = new SortedFibers;

        // Initialize struct
        sortedFibers->ds = d;
        sortedFibers->ds_processed = NULL;
		sortedFibers->userSelectedLine = 0;
		sortedFibers->selectedScalarType = 0;

        // Add the new data set to the list of currently available fiber sets
        this->sortedFibersList.append(sortedFibers);

        // Add to UI combobox for distance measurements to fibers
        this->form->fibersCombo->addItem(d->getName());

        // If first fiber set, select by default
        if(this->sortedFibersList.count() == 1)
            SelectFiberDataSet(0);
	}
}

//------------------------[ Dataset changed ]-----------------------\\

void ScoringTools::dataSetChanged(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // to-do

}

//------------------------[ Dataset removed ]-----------------------\\

void ScoringTools::dataSetRemoved(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // Remove fiber dataset
    if (kind == "fibers")
	{
	    // Check if the data set exists
		int dsIndex = this->FindInputDataSet(d);

        // Does not exist, return
		if (dsIndex == -1)
			return;

        // Remove from UI combobox for selection of overlay
        this->form->fibersCombo->removeItem(dsIndex);

        // Remove from collection
        this->sortedFibersList.removeAt(dsIndex);
	}
}

int ScoringTools::FindInputDataSet(data::DataSet * ds)
{
	int index = 0;

	// Loop through all input fiber data sets
	for (QList<SortedFibers*>::iterator i = this->sortedFibersList.begin(); i != this->sortedFibersList.end(); ++i, ++index)
	{
		// Return the index if we've found the target data set
		if ((*i)->ds == ds)
			return index;
	}

	return -1;
}

///
///      PROCESSING
///

void ScoringTools::SelectFiberDataSet(int index)
{
    // set selected fiber index
    this->selectedFiberDataset = index - 1;

    // no fiber selected
    if(index == -1)
        return;

    // Clear scalar type list
    for(int i = this->form->scalarTypeCombo->count()-1; i>=0; i--)
    {
        this->form->scalarTypeCombo->removeItem(i);
    }

    // Add new scalar types to list
    SortedFibers* sortedFibers = this->sortedFibersList.at(index);
    vtkPolyData * polydata = sortedFibers->ds->getVtkPolyData();

    // Get number of scalar types
    sortedFibers->numberOfScalarTypes = polydata->GetPointData()->GetNumberOfArrays();

    // Fill scalar list with names
    for(int i = 0; i < sortedFibers->numberOfScalarTypes; i++)
    {
        this->form->scalarTypeCombo->addItem(polydata->GetPointData()->GetArray(i)->GetName());
    }

    // Select the standard scalar
    SelectScalarType(sortedFibers->selectedScalarType);
}

void ScoringTools::SelectScalarType(int index)
{
    // return if fiber is none
    if(selectedFiberDataset == -1)
        return;

    // Get selected scalar type data
    SortedFibers* sortedFibers = this->sortedFibersList.at(this->selectedFiberDataset);

    // Check if amount of scalar types is larger than 0
    if(sortedFibers->numberOfScalarTypes == 0)
        return;

    // Get scalar data
    vtkPolyData * polydata = sortedFibers->ds->getVtkPolyData();
    vtkDoubleArray* scalarData = static_cast<vtkDoubleArray*>(polydata->GetPointData()->GetArray(index));
    //scalarData->Print(std::cout);
    return;
    // Set average value slider ranges
    double scalarRange[2];
    scalarData->GetValueRange(scalarRange);
    this->form->averageValueSlider->setRange(scalarRange[0]*100,scalarRange[1]*100);
    this->form->averageValueSpinBox->setRange(scalarRange[0],scalarRange[1]);

    //std::cout << "Min: " << scalarRange[0] << " " << scalarRange[1] << std::endl;

    polydata->GetPointData()->SetActiveScalars(polydata->GetPointData()->GetArray(index)->GetName());
    this->core()->data()->dataSetChanged(sortedFibers->ds);

    // Update selected scalar type
    sortedFibers->selectedScalarType = index;
}

void ScoringTools::ComputeFibers()
{
    // return if fiber is none
    if(selectedFiberDataset == -1)
        return;

    // critera
    double averageScoreRange[2] = {this->form->averageValueSpinBox->value(),999};

    // Get polydata of original fibers
    SortedFibers* sortedFibers = this->sortedFibersList.at(this->selectedFiberDataset);
    vtkPolyData * polydata = sortedFibers->ds->getVtkPolyData();

    // Perform fiber selection filter
    vtkFiberSelectionFilter* selectionFilter = vtkFiberSelectionFilter::New();
	selectionFilter->SetInput(polydata);
	selectionFilter->SetAverageScoreRange(averageScoreRange);
	selectionFilter->Update();
	vtkPolyData* outputPoly = selectionFilter->GetOutput();

	// Create a progress bar for the ranking filter
	this->core()->out()->createProgressBarForAlgorithm(selectionFilter, "Fiber selection");

    // Construst vIST/e dataset
    data::DataSet* ds = sortedFibers->ds_processed;
	if (ds != NULL)
	{
		ds->updateData(outputPoly);
		ds->setName("test");

		// Fibers should be visible, and the visualization pipeline should be updated
		ds->getAttributes()->addAttribute("isVisible", 1.0);
		ds->getAttributes()->addAttribute("updatePipeline", 1.0);

		// Copy the transformation matrix to the output
		ds->getAttributes()->copyTransformationMatrix(sortedFibers->ds);

		this->core()->data()->dataSetChanged(ds);
	}

	// Otherwise, create a new data set
	else
	{
		ds = new data::DataSet("test", "fibers", outputPoly);

		// Fibers should be visible, and the visualization pipeline should be updated
		ds->getAttributes()->addAttribute("isVisible", 1.0);
		ds->getAttributes()->addAttribute("updatePipeline", 1.0);

		// We add this attribute to make sure that output data sets are not added to the input data sets
		ds->getAttributes()->addAttribute("hasCM", 1);

		// Copy the transformation matrix to the output
		ds->getAttributes()->copyTransformationMatrix(sortedFibers->ds);

		this->core()->data()->addDataSet(ds);
	}

    // Update ds locally
	sortedFibers->ds_processed = ds;

	// Hide the input data set
	sortedFibers->ds->getAttributes()->addAttribute("isVisible", -1.0);
	this->core()->data()->dataSetChanged(sortedFibers->ds);
}

///
///     GUI CALLBACKS
///

void ScoringTools::fibersComboChanged(int index)
{
    SelectFiberDataSet(index);
}

void ScoringTools::scalarTypeComboChanged(int index)
{
    SelectScalarType(index);
}

void ScoringTools::averageValueSliderChanged(int value)
{
    averageValueSpinBoxChanged((double)(value/100.0));
}

void ScoringTools::averageValueSpinBoxChanged(double value)
{
    this->form->averageValueSlider->blockSignals(true);
    this->form->averageValueSpinBox->blockSignals(true);

    this->form->averageValueSlider->setValue(value*100);
    this->form->averageValueSpinBox->setValue(value);

    this->form->averageValueSlider->blockSignals(false);
    this->form->averageValueSpinBox->blockSignals(false);
}

void ScoringTools::updateButtonClicked()
{
    ComputeFibers();
}

///
///      GUI CONTROLS
///

//------------------------[ Connect Qt elements ]-----------------------\\

void ScoringTools::connectAll()
{
    connect(this->form->fibersCombo,SIGNAL(currentIndexChanged(int)),this,SLOT(fibersComboChanged(int)));
    connect(this->form->scalarTypeCombo,SIGNAL(currentIndexChanged(int)),this,SLOT(scalarTypeComboChanged(int)));
    connect(this->form->averageValueSlider,SIGNAL(valueChanged(int)),this,SLOT(averageValueSliderChanged(int)));
    connect(this->form->averageValueSpinBox,SIGNAL(valueChanged(double)),this,SLOT(averageValueSpinBoxChanged(double)));
    connect(this->form->updateButton,SIGNAL(clicked()),this,SLOT(updateButtonClicked()));
}

//------------------------[ Disconnect Qt elements ]-----------------------\\

void ScoringTools::disconnectAll()
{

}



///
///     vISTe communication
///

//-----------[ Returns visualization component as VTK object ]---------------\\
//
vtkProp * ScoringTools::getVtkProp()
{
    return this->assembly;
}

//-----------------[ Returns GUI component as Qt widget ]---------------\\
//
QWidget * ScoringTools::getGUI()
{
    return this->widget;
}

}

Q_EXPORT_PLUGIN2(libScoringTools, bmia::ScoringTools)
