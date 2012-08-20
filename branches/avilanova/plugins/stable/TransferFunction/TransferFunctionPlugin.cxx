/*
 * TransferFunctionPlugin.cxx
 *
 * 2010-02-22	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-01-04	Evert van Aart
 * - Refactored code, added comments.
 * - Changed the GUI.
 *
 * 2011-03-28	Evert van Aart
 * - Version 1.0.0.
 * - Prevented divide-by-zero errors for scalar images with zero range. 
 *
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.1.
 * - When saving transfer functions, the plugin now automatically selects the 
 *   data directory defined in the default profile. 
 *
 */


/** Includes */

#include "TransferFunctionPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

TransferFunctionPlugin::TransferFunctionPlugin() : plugin::Plugin("Transfer Function")
{
	// Create a new widget for the GUI
	this->qWidget = new QWidget(0, Qt::Widget);

	// Create the GUI
	this->ui = new Ui::TransferFunctionForm;

	// Setup the GUI
	ui->setupUi(this->qWidget);

	// Connect GUI signals to "slots" functions in this class
	connect(this->ui->pushButtonSave,               SIGNAL(clicked()),                 this, SLOT(save()));
	connect(this->ui->pushButtonAdopt,              SIGNAL(clicked()),                 this, SLOT(adoptRange()));
	connect(this->ui->widgetTransferFunctionEditor, SIGNAL(transferFunctionChanged()), this, SLOT(transferFunctionChanged()));
	connect(this->ui->comboBoxTransferFunction,     SIGNAL(currentIndexChanged(int)),  this, SLOT(setCurrentColormap(int)));
	connect(this->ui->comboBoxDataset,              SIGNAL(currentIndexChanged(int)),  this, SLOT(setCurrentDataset(int)));
	connect(this->ui->doubleSpinBoxMinRange,        SIGNAL(valueChanged(double)),      this, SLOT(setRange()));
	connect(this->ui->doubleSpinBoxMaxRange,        SIGNAL(valueChanged(double)),      this, SLOT(setRange()));
	connect(this->ui->checkBoxFlattening,           SIGNAL(toggled(bool)),             this, SLOT(flatteningToggled(bool)));
	connect(this->ui->pushButtonNew,               SIGNAL(clicked()),                  this, SLOT(addNew()));
	connect(this->ui->pushButtonAddPWF,            SIGNAL(clicked()),                  this, SLOT(addPiecewisefunction()));
}


//------------------------------[ Destructor ]-----------------------------\\

TransferFunctionPlugin::~TransferFunctionPlugin()
{
	// Delete the GUI
	delete this->qWidget;

	// Clear the lists of pointers
	this->compatibleDataSets.clear();
	this->compatibleTransferFunctions.clear();
	this->transferFunctions.clear();
	this->piecewiseFunctions.clear();
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * TransferFunctionPlugin::getGUI()
{
	return this->qWidget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void TransferFunctionPlugin::dataSetAdded(data::DataSet * ds)
{
	// Scalar volumes
	if (ds->getKind() == "scalar volume")
	{
		// Add the data set to the list of scalar volumes
		this->compatibleDataSets.append(ds);

		// Add the data set to the GUI
		this->ui->comboBoxDataset->addItem(ds->getName());
	}

	// Transfer functions
	else if (ds->getKind() == "transfer function")
	{
		// VTK object containing the piecewise function
		vtkObject * cpf;

		// Add the transfer function pointer to the list
		this->compatibleTransferFunctions.append(ds);

		// Get the pointer to the VTK object, and add it to the list
		this->transferFunctions.append(vtkColorTransferFunction::SafeDownCast(ds->getVtkObject()));
	  
		// Try to get the piecewise function from the attributes
		if (ds->getAttributes()->getAttribute("piecewise function", cpf))
		{
			// If successful, add it to the list
			this->piecewiseFunctions.append(vtkPiecewiseFunction::SafeDownCast(cpf));
		}
		else
		{
			// Otherwise, add a "NULL" to the list
			this->piecewiseFunctions.append(NULL);
		}

		// Add the new data set to the GUI, and select it
		this->ui->comboBoxTransferFunction->addItem(ds->getName());
		this->ui->comboBoxTransferFunction->setCurrentIndex(this->ui->comboBoxTransferFunction->count() - 1);
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void TransferFunctionPlugin::dataSetChanged(data::DataSet * ds)
{
	// TODO: Implement this function
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void TransferFunctionPlugin::dataSetRemoved(data::DataSet * ds)
{
	// Scalar volumes
	if (ds->getKind() == "scalar volume")
	{
		// Get the index of the data set
		int index = this->compatibleDataSets.indexOf(ds);
	  
		// If the data set has been added to this plugin, remove it from the list
		// and from the combo box in the GUI.

		if (index != -1)
		{
			this->compatibleDataSets.removeAt(index);
			this->ui->comboBoxDataset->removeItem(index);
		}
	}

	// Transfers functions
	else if (ds->getKind() == "transfer function")
	{
		// Get the index of the transfer function
		int index = this->compatibleTransferFunctions.indexOf(ds);
	  
		// If the transfer function has been added to this plugin...
		if (index != -1)
		{
			// ...remove pointers and GUI elements
			this->compatibleTransferFunctions.removeAt(index);
			this->transferFunctions.removeAt(index);
			this->piecewiseFunctions.removeAt(index);
			this->ui->comboBoxTransferFunction->removeItem(index);
		}
	}
}


//-------------------------[ saveTransferFunction ]------------------------\\

bool TransferFunctionPlugin::saveTransferFunction(vtkColorTransferFunction * pTf, vtkPiecewiseFunction * pPf)
{
	// Output file name
	QString fileName;

	QDir dataDir = this->core()->getDataDirectory();

	// Use a file dialog to get the output file name
	fileName = QFileDialog::getSaveFileName(this->qWidget, "Save Transfer Function", dataDir.absolutePath(), "Transfer Functions (*.tf)");

	// Do nothing if no file has been selected
	if (fileName.isEmpty())
		return false;

	// Create a new file information object
	QFileInfo info(fileName);

	// Append the extension if necessary
	if (info.suffix() != "tf")
	{
		fileName += ".tf";
	}

	// Create a Qt file handler for the output file
	QFile file(fileName);

	// Values per node
	double node[6];

	// Single line of the output
	QString line;

	// Try to open the file
	if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
		return false;

	// Write the minimum and maximum of the range to the first two lines
	file.write(QString::number(this->ui->doubleSpinBoxMinRange->value()).toAscii() + "\n");
	file.write(QString::number(this->ui->doubleSpinBoxMaxRange->value()).toAscii() + "\n");

	// Write whether or not we have a piecewise function
	if (pPf != NULL)
	{
		file.write("True\n");
	}
	else
	{
		file.write("False\n");
	}

	// Loop through all points in the transfer function
	for(int i = 0; i < pTf->GetSize(); ++i)
	{
		// Reset the output line
		line = "";

		// Copy the point values to the "node" array
		pTf->GetNodeValue(i, node);

		// Write the node values to a string
		line += QString::number(node[0]) + " ";
		line += QString::number(node[1]) + " ";
		line += QString::number(node[2]) + " ";
		line += QString::number(node[3]) + " ";

		if (pPf != NULL)
		{
			line += QString::number(pPf->GetValue(node[0])) + " ";
		}

		line += "\n";

		// Write the line to the output
		file.write(line.toAscii());
	}

	// Close the output file
	file.close();
	return true;
}


//---------------------------------[ save ]--------------------------------\\

void TransferFunctionPlugin::save()
{
	// Get the current transfer function index
	int index = this->ui->comboBoxTransferFunction->currentIndex();

	// Do nothing if the index is out of bounds
	if (index < 0 || index >= this->ui->comboBoxTransferFunction->count())
		return;

	// Try to save the transfer function
	if (!saveTransferFunction(this->transferFunctions.at(index), this->piecewiseFunctions.at(index)))
	{
		this->core()->out()->logMessage("The transfer function could not be saved");
	}
}

 
//-----------------------[ transferFunctionChanged ]-----------------------\\

void TransferFunctionPlugin::transferFunctionChanged()
{
	// Repaint the transfer function widget
	this->ui->widgetColormap->repaint();

	// Get the current transfer function index
	int index = this->ui->comboBoxTransferFunction->currentIndex();

	// Do nothing if the index is out of bounds
	if (index < 0 || index >= this->ui->comboBoxTransferFunction->count())
		return;

	// Tell the data manager that the data set has changed
	this->core()->data()->dataSetChanged(this->compatibleTransferFunctions.at(index));
	this->core()->render();
}


//------------------------------[ adoptRange ]-----------------------------\\

void TransferFunctionPlugin::adoptRange()
{
	// Minimum and maximum of the range
	double range[2];

	// Get the current image data index
	int index = this->ui->comboBoxDataset->currentIndex();

	// Do nothing if the index is out of bounds
	if (index < 0 || index >= this->ui->comboBoxDataset->count())
		return;

	// Get the range of the selected image data
	this->compatibleDataSets.at(index)->getVtkImageData()->Update();
	this->compatibleDataSets.at(index)->getVtkImageData()->GetScalarRange(range);

	// Update the GUI
	this->ui->doubleSpinBoxMinRange->setValue(range[0]);
	this->ui->doubleSpinBoxMaxRange->setValue(range[1]);
	this->ui->widgetTransferFunctionEditor->setIntensityRange(range[0], range[1]);
	this->ui->widgetColormap->setIntensityRange(range[0], range[1]);
}


//--------------------------[ setCurrentColormap ]-------------------------\\

void TransferFunctionPlugin::setCurrentColormap(int index)
{
	// Current transfer function, piecewise function, and scalar range
	vtkColorTransferFunction * pTf;
	vtkPiecewiseFunction * pPf;
	double minRange;
	double maxRange;

	// If no transfer function has been set...
	if (index == -1)
	{
		// ...set the transfer and piecewise function pointers to NULL...
		this->ui->widgetColormap->setTransferFunction(NULL);
		this->ui->widgetColormap->setPiecewiseFunction(NULL);
		this->ui->widgetTransferFunctionEditor->setTransferFunction(NULL);
		this->ui->widgetTransferFunctionEditor->setPiecewiseFunction(NULL);

		// ...and exit the function
		return;
	}

	// Get the transfer function and piecewise function
	pTf = this->transferFunctions.at(index);
	pPf = this->piecewiseFunctions.at(index);

	// Get the minimum and maximum of the scalar range
	this->compatibleTransferFunctions.at(index)->getAttributes()->getAttribute("minRange", minRange);
	this->compatibleTransferFunctions.at(index)->getAttributes()->getAttribute("maxRange", maxRange);

	// Copy the new settings to the GUI
	this->ui->doubleSpinBoxMinRange->setValue(minRange);
	this->ui->doubleSpinBoxMaxRange->setValue(maxRange);
	this->ui->widgetTransferFunctionEditor->setIntensityRange(minRange, maxRange);
	this->ui->widgetTransferFunctionEditor->setTransferFunction(pTf);
	this->ui->widgetTransferFunctionEditor->setPiecewiseFunction(pPf);
	this->ui->widgetColormap->setIntensityRange(minRange, maxRange);
	this->ui->widgetColormap->setTransferFunction(pTf);
	this->ui->widgetColormap->setPiecewiseFunction(pPf);

	// Enable the PWF button if this transfer function does not yet have a piecewise function
	this->ui->pushButtonAddPWF->setEnabled(pPf == NULL);
}


//--------------------------[ setCurrentDataset ]--------------------------\\

void TransferFunctionPlugin::setCurrentDataset(int index)
{
	// Minimum and maximum of scalar range
	double range[2];

	// Selected image data
	vtkImageData * image;

	// Do nothing if the index is out of bounds
	if (index < 0 || index >= this->ui->comboBoxDataset->count())
		return;

	// Get the range of the selected image data
	image = this->compatibleDataSets.at(index)->getVtkImageData();
	image->Update();
	image->GetScalarRange(range);

	// Copy the new settings to the GUI
	this->ui->doubleSpinBoxMinRange->setValue(range[0]);
	this->ui->doubleSpinBoxMaxRange->setValue(range[1]);
	this->ui->widgetTransferFunctionEditor->setIntensityRange(range[0], range[1]);
	this->ui->widgetTransferFunctionEditor->setDataSet(image, this->ui->checkBoxFlattening->isChecked());
	this->ui->widgetColormap->setIntensityRange(range[0], range[1]);
	this->ui->widgetTransferFunctionEditor->repaint();
}


//-------------------------------[ setRange ]------------------------------\\

void TransferFunctionPlugin::setRange()
{
	// Get the minimum and maximum from the spin boxes in the GUI
	double minRange = this->ui->doubleSpinBoxMinRange->value();
	double maxRange = this->ui->doubleSpinBoxMaxRange->value();

	// Maximum should always be higher than minimum
	if (minRange >= maxRange)
		return;

	// Copy the range to the widgets
	this->ui->widgetTransferFunctionEditor->setIntensityRange(minRange, maxRange);
	this->ui->widgetColormap->setIntensityRange(minRange, maxRange);
}


//--------------------------[ flatteningToggled ]--------------------------\\

void TransferFunctionPlugin::flatteningToggled(bool checked)
{
	// Get the index of the selected image data set
	int index = this->ui->comboBoxDataset->currentIndex();

	// Do nothing if the index is out of bounds
	if (index < 0 || index >= this->ui->comboBoxDataset->count())
		return;

	// Get the selected image data
	vtkImageData * image = this->compatibleDataSets.at(index)->getVtkImageData();
	image->Update();

	// Copy settings to the editor widget
	this->ui->widgetTransferFunctionEditor->setDataSet(image, checked);
}


//--------------------------------[ addNew ]-------------------------------\\

void TransferFunctionPlugin::addNew()
{
	// Used to determine whether "Ok" was pressed
	bool ok;

	// Use an input dialog to get a name for the new transfer function
	QString text = QInputDialog::getText(	this->qWidget, tr("New Transfer Function"),
											tr("Transfer Function Name"), QLineEdit::Normal,
											"Transfer Function", &ok);

	// Do nothing if the name is not valid
	if (!ok || text.isEmpty())
		return;

	// Loop through the available transfer functions
	for (int i = 0; i < this->compatibleTransferFunctions.length(); ++i)
	{
		// Check if the new name exists
		if (this->compatibleTransferFunctions.at(i)->getName() == text)
		{
			// If so, append a "1", and try again
			text += "1";
			i = 0;
		}
	}

	// Create a new transfer function
	vtkColorTransferFunction * pTf = vtkColorTransferFunction::New();

	// Set default transfer function points
	pTf->AddRGBPoint(this->ui->doubleSpinBoxMinRange->value(), 0.0, 0.0, 0.0);
	pTf->AddRGBPoint(this->ui->doubleSpinBoxMaxRange->value(), 1.0, 1.0, 1.0);

	// Create a new data set, and add the default scalar range to its attributes
	data::DataSet * ds = new data::DataSet(text, "transfer function", pTf);
	ds->getAttributes()->addAttribute("minRange",this->ui->doubleSpinBoxMinRange->value());
	ds->getAttributes()->addAttribute("maxRange",this->ui->doubleSpinBoxMaxRange->value());

	// Add the new transfer function to the data manager. This will also call "dataSetAdded"
	// of this plugin, which will add the new data set to the lists and to the GUI.

	this->core()->data()->addDataSet(ds);
}


//-------------------------[ addPiecewisefunction ]------------------------\\

void TransferFunctionPlugin::addPiecewisefunction()
{
	// Piecewise function pointer
	vtkObject * pPf = NULL;

	// Get the index of the current transfer function
	int index = this->ui->comboBoxTransferFunction->currentIndex();
    
	// Do nothing if the index is out of bounds
	if (index < 0 || index >= this->ui->comboBoxTransferFunction->count())
		return;

	// Do nothing if the selected data set already contains a piecewise function
	if (this->compatibleTransferFunctions.at(index)->getAttributes()->getAttribute("piecewise function", pPf))
		return;

	// Create a new piecewise function
	vtkPiecewiseFunction * newPf = vtkPiecewiseFunction::New();

	// Loop through all points in the current transfer function
	for(int i = 0; i < this->transferFunctions.at(index)->GetSize(); ++i)
	{
		// Data of the current point
		double node[6];

		// Get the values for the current point
		this->transferFunctions.at(index)->GetNodeValue(i, node);

		// Add the point to the piecewise function
		newPf->AddPoint(node[0], 0.5);
	}

	// Add the piecewise function object to the attributes of the data set
	this->compatibleTransferFunctions.at(index)->getAttributes()->addAttribute("piecewise function", newPf);

	// Select the current transfer function
	this->ui->comboBoxTransferFunction->setCurrentIndex(index);

	// Add the piecewise function pointer to the list
	this->piecewiseFunctions[index] = newPf;

	// Update the widgets
	this->ui->widgetColormap->setPiecewiseFunction(newPf);
	this->ui->widgetTransferFunctionEditor->setPiecewiseFunction(newPf);

	// Tell the data manager that the data set has changed
	this->core()->data()->dataSetChanged(this->compatibleTransferFunctions.at(index));
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libbmia_TransferFunctionPlugin, bmia::TransferFunctionPlugin)
