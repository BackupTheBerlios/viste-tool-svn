#include "ScoringMeasures.h"

#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

#define SLIDER_SUBSTEPS 100

namespace bmia
{

///
///      INITIALIZATION
///

//------------------------[ Plugin constructor ]-----------------------\\

ScoringMeasures::ScoringMeasures() : plugin::AdvancedPlugin("ScoringMeasures")
{
    this->widget = NULL;
    this->form   = NULL;
}

//------------------------[ Plugin destructor ]-----------------------\\

ScoringMeasures::~ScoringMeasures()
{
    delete this->widget;
    delete this->form;
}

//------------------------[ Initialization ]-----------------------\\

void ScoringMeasures::init()
{
    this->widget = new QWidget();
    this->form = new Ui::ScoringMeasuresForm();
    this->form->setupUi(this->widget);

    // Link events in the GUI to function calls
    this->connectAll();
    this->assembly = vtkPropAssembly::New();

    // default selected fiber (none)
    selectedFiberDataset = -1;

    // disable GUI by default
    DisableGUI();
}

///
///      DATA I/O
///

//------------------------[ Dataset added ]-----------------------\\

void ScoringMeasures::dataSetAdded(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // Load fiber dataset
    if (kind == "fibers")
	{
        // Check if dataset is already added
        if(FindInputDataSet(d) != -1)
            return;

	    // Check if fiber has polydata
	    if (d->getVtkPolyData() == NULL)
			return;

        // Check if the fiber is created by this plugin
        if (d->getAttributes()->hasIntAttribute("isSM"))
			return;

        // Create new fiber struct
        SortedFibers* sortedFibers = new SortedFibers;

        // Initialize struct
        sortedFibers->ds = d;
        sortedFibers->ds_processed = NULL;
		sortedFibers->outputFiberDataName = d->getName().append("_[SM]");
		sortedFibers->processed = false;

		// Create parameter settings struct
		ParameterSettings* ps = new ParameterSettings;
		ps->useGlyphData = true;
		sortedFibers->ps = ps;

        // Add the new data set to the list of currently available fiber sets
        this->sortedFibersList.append(sortedFibers);

        // Add to UI combobox for distance measurements to fibers
        this->form->fibersCombo->addItem(d->getName());

        // If first fiber set, select by default
        if(this->sortedFibersList.count() == 1)
            SelectFiberDataSet(0);
	}

	// Discrete sphere functions
	else if (kind == "discrete sphere")
	{
	    // Check if dataset is already added
	    if(this->glyphDataSets.contains(d))
            return;

		// Check if the data set contains an image
		vtkImageData * image = d->getVtkImageData();

		if (!image)
			return;

		// Check if the image contains point data
		vtkPointData * imagePD = image->GetPointData();

		if (!imagePD)
			return;

		// Check if the point data contains a spherical directions array
		if (!(imagePD->GetArray("Spherical Directions")))
			return;

		// We can use this data set, so add it to the list and the GUI
		this->glyphDataSets.append(d);
		this->form->glyphDataCombo->addItem(d->getName());
	}
}

//------------------------[ Dataset changed ]-----------------------\\

void ScoringMeasures::dataSetChanged(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // to-do

}

//------------------------[ Dataset removed ]-----------------------\\

void ScoringMeasures::dataSetRemoved(data::DataSet * d)
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

        // Select 'none' in combobox
        this->form->fibersCombo->setCurrentIndex(0);
        this->SelectFiberDataSet(0);

        // Remove from UI combobox for selection of overlay
        this->form->fibersCombo->removeItem(dsIndex+1);

        // Clean up struct
        SortedFibers* sortedFibers = this->sortedFibersList.at(dsIndex);
        sortedFibers->ds = NULL;
        sortedFibers->ds_processed = NULL;

        // Remove from collection
        this->sortedFibersList.removeAt(dsIndex);
	}
}

int ScoringMeasures::FindInputDataSet(data::DataSet * ds)
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

void ScoringMeasures::SelectFiberDataSet(int index)
{
    // set selected fiber index
    this->selectedFiberDataset = index - 1;

    // no fiber selected
    if(this->selectedFiberDataset == -1)
    {
        DisableGUI();
        return;
    }

    // Set output data name
    SortedFibers* sortedFibers = this->sortedFibersList.at(this->selectedFiberDataset);
    this->form->outputLineEdit->setText(sortedFibers->outputFiberDataName);

    // Update GUI
    UpdateGUI();

    // Enable GUI
    EnableGUI();
}

void ScoringMeasures::SelectGlyphDataSet(int index)
{
    // return if fiber is none
    if(this->selectedFiberDataset == -1)
        return;

    // Get selected scalar type data
    SortedFibers* sortedFibers = this->sortedFibersList.at(this->selectedFiberDataset);

     // Update selected scalar type
    sortedFibers->selectedGlyphData = index;
}

void ScoringMeasures::ComputeScore()
{

}

///
///     GUI CALLBACKS
///

void ScoringMeasures::EnableGUI()
{
    if(GetSortedFibers()->ps->useGlyphData)
    {
        this->form->lambdaSlider->setEnabled(true);
        this->form->lambdaSpinBox->setEnabled(true);
        this->form->lambdaLabel->setEnabled(true);
        this->form->lambdaTopLabel->setEnabled(true);
    }
    else
    {
        this->form->lambdaSlider->setEnabled(false);
        this->form->lambdaSpinBox->setEnabled(false);
        this->form->lambdaLabel->setEnabled(false);
        this->form->lambdaTopLabel->setEnabled(false);
    }
    this->form->dataDependentGroupBox->setEnabled(true);
    this->form->dataIndependentGroupBox->setEnabled(true);
    this->form->updateButton->setEnabled(true);
    this->form->outputLineEdit->setEnabled(true);
}

void ScoringMeasures::DisableGUI()
{
    this->form->dataDependentGroupBox->setEnabled(false);
    this->form->dataIndependentGroupBox->setEnabled(false);
    this->form->updateButton->setEnabled(false);
    this->form->outputLineEdit->setEnabled(false);
}

void ScoringMeasures::BlockSignals()
{

}

void ScoringMeasures::AllowSignals()
{

}

void ScoringMeasures::UpdateGUI()
{
    // block signal propagation
    BlockSignals();



    // re-enable signals
    AllowSignals();
}

SortedFibers* ScoringMeasures::GetSortedFibers()
{
    SortedFibers* sortedFibers = this->sortedFibersList.at(this->selectedFiberDataset);
    return sortedFibers;
}

///
///     SLOTS
///

void ScoringMeasures::fibersComboChanged(int index)
{
    SelectFiberDataSet(index);
}

void ScoringMeasures::glyphDataComboChanged(int index)
{
    SelectGlyphDataSet(index);
}

void ScoringMeasures::updateButtonClicked()
{
    ComputeScore();
}

void ScoringMeasures::usedInScoringCheckBoxChanged(bool checked)
{
    GetSortedFibers()->ps->useGlyphData = checked;
    EnableGUI();
}

///
///      GUI CONTROLS
///

//------------------------[ Connect Qt elements ]-----------------------\\

void ScoringMeasures::connectAll()
{
    connect(this->form->fibersCombo,SIGNAL(currentIndexChanged(int)),this,SLOT(fibersComboChanged(int)));
    connect(this->form->glyphDataCombo,SIGNAL(currentIndexChanged(int)),this,SLOT(glyphDataComboChanged(int)));
    connect(this->form->updateButton,SIGNAL(clicked()),this,SLOT(updateButtonClicked()));
    connect(this->form->usedInScoringCheckBox,SIGNAL(toggled(bool)),this,SLOT(usedInScoringCheckBoxChanged(bool)));
}

//------------------------[ Disconnect Qt elements ]-----------------------\\

void ScoringMeasures::disconnectAll()
{

}



///
///     vISTe communication
///

//-----------[ Returns visualization component as VTK object ]---------------\\
//
vtkProp * ScoringMeasures::getVtkProp()
{
    return this->assembly;
}

//-----------------[ Returns GUI component as Qt widget ]---------------\\
//
QWidget * ScoringMeasures::getGUI()
{
    return this->widget;
}

}

Q_EXPORT_PLUGIN2(libScoringMeasures, bmia::ScoringMeasures)
