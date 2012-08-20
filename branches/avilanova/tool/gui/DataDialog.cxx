/*
/*
 * DataDialog.cxx
 *
 * 2010-03-08	Tim Peeters
 * - First version.
 *
 * 2010-08-03	Tim Peeters
 * - Implement "data::Consumer functions".
 * - Replace "assert" by "Q_ASSERT".
 *
 * 2011-02-09	Evert van Aart
 * - Fixed range displaying for images.
 * 
 * 2011-03-14	Evert van Aart
 * - Added data set size to the dialog.
 * - Instead of completely rebuilding the tree widget every time any data set is
 *   added, modified, or deleted, we now only update the relevant item.
 *
 * 2011-05-13	Evert van Aart
 * - Modified attribute handling.
 *
 * 2011-07-21	Evert van Aart
 * - Added a destructor.
 *
 */


#include "DataDialog.h"


namespace bmia {


using namespace data;


namespace gui {


//-----------------------------[ Constructor ]-----------------------------\\

DataDialog::DataDialog(Manager * dManager, QWidget * parent) :
											QDialog(parent),
											treeWidget(new QTreeWidget),
											closeButton(new QPushButton("Close"))
{
	// Store the pointer to the data manager
	Q_ASSERT(dManager);
	this->manager = dManager;

	// Setup the tree widgets
	this->treeWidget->setColumnCount(3);
	this->treeWidget->setColumnWidth(0, 300);
	this->treeWidget->setColumnWidth(1, 10);
	this->treeWidget->setColumnWidth(2, 75);
	this->treeWidget->setAlternatingRowColors(false);
	this->treeWidget->setAnimated(true);
	this->treeWidget->header()->hide();
	this->treeWidget->header()->setStretchLastSection(false);

	this->setMinimumWidth(430);

	// Connect the close button to the "close" function
	connect(closeButton, SIGNAL(clicked()), this, SLOT(close()));

	// Create the main layout and the layout for the buttons
	QVBoxLayout * mainLayout   = new QVBoxLayout;
	QHBoxLayout * buttonLayout = new QHBoxLayout;

	// Add the tree widget to the main layout
	mainLayout->addWidget(this->treeWidget);

	// Setup the button layout
	buttonLayout->addStretch(0);
	buttonLayout->addWidget(closeButton);
	buttonLayout->addStretch(0);

	// Add the button layout to the main layout
	mainLayout->addLayout(buttonLayout);

	// Set the layout
	this->setLayout(mainLayout);

	this->setWindowTitle(tr("List of available data sets"));

	// Add self to the data manager as a consumer
	this->manager->addConsumer(this);
}


//------------------------------[ Destructor ]-----------------------------\\

DataDialog::~DataDialog()
{
	// Clear the list of data sets
	this->dataSets.clear();

	// Clear the tree widget
	if (this->treeWidget)
		this->treeWidget->clear();

	// Delete the main layout of the dialog
	if (this->layout())
		delete this->layout();
}


//--------------------------------[ update ]-------------------------------\\

void DataDialog::update()
{
	// Remove everything from the tree widget
	this->treeWidget->clear();

	// Get all data sets from the manager
    QList<DataSet *> dataSets = this->manager->listAllDataSets();

	// Loop through the data sets
    for (int i = 0; i < dataSets.size(); ++i)
	{
		if (!dataSets[i])
			continue;
		
		// Add the data set to the tree widget
		this->populateTreeWidget(dataSets[i]);
	}
}


//--------------------------[ populateTreeWidget ]-------------------------\\

void DataDialog::populateTreeWidget(DataSet * ds)
{
	// Create a new item from the data set
	QTreeWidgetItem * newItem = this->createWidgetItem(ds);

	// Append it to the tree widget
	this->treeWidget->addTopLevelItem(newItem);

	// Recompute the size of the window
	this->treeWidget->resizeColumnToContents(0);
	this->treeWidget->resizeColumnToContents(1);

	if (this->treeWidget->columnWidth(0) < 300)
		this->treeWidget->setColumnWidth(0, 300);
	if (this->treeWidget->columnWidth(2) <  60)
		this->treeWidget->setColumnWidth(2, 60);

	this->setFixedWidth(this->treeWidget->columnWidth(0) + this->treeWidget->columnWidth(2) + 50);
}


//---------------------------[ createWidgetItem ]--------------------------\\

QTreeWidgetItem * DataDialog::createWidgetItem(bmia::data::DataSet *ds)
{
	// Create a new item
	QTreeWidgetItem * dsItem = new QTreeWidgetItem;

	// Set the text of the data set
	dsItem->setText(0, ds->getName() + " : " + ds->getKind());

	// Get the VTK image data, polydata, and object. If the VTK data in the data
	// set is not of the specified type, the pointer will be NULL.

	vtkImageData * image   = ds->getVtkImageData();
	vtkPolyData * polyData = ds->getVtkPolyData();
	vtkObject * obj        = ds->getVtkObject();

	vtkPointSet * pointSet = NULL;

	// Cast the VTK object to a point set pointer
	if (obj) 
	{
		pointSet = vtkPointSet::SafeDownCast(obj);
	}

	// Estimated data size
	unsigned long dataSize = 0;

	// True if we've got a VTK data object for which the function "GetActualMemorySize"
	// is available, false otherwise.

	bool dataSizeAvailable = false;

	// VTK image data
	if (image)
	{
		// Print data set type and dimensionality
		this->addSubItem(dsItem, "Type: Image (" + QString::number(image->GetDataDimension()) + "D)");

		// Get and print image dimensions
		int dims[3]; 
		image->GetDimensions(dims);
		this->addSubItem(dsItem, "Dimensions: " +	QString::number(dims[0]) + " x " + 
			QString::number(dims[1]) + " x " + 
			QString::number(dims[2]));

		// Get and print scalar range
		double range[2];
		image->GetScalarRange(range);
		this->addSubItem(dsItem, "Range: " + QString::number(range[0]) + ", " + QString::number(range[1]));

		// Print scalar type
		this->addSubItem(dsItem, "Scalar Type: " + QString(image->GetScalarTypeAsString()));

		// Print the number of scalar components
		this->addSubItem(dsItem, "Components: " + QString::number(image->GetNumberOfScalarComponents()));

		// Get and print the voxel spacing
		double spacing[3]; 
		image->GetSpacing(spacing);
		this->addSubItem(dsItem, "Spacing: " +	QString::number(spacing[0]) + ", " + 
			QString::number(spacing[1]) + ", " + 
			QString::number(spacing[2]));

		// Get the memory size
		dataSize = image->GetActualMemorySize();
		dataSizeAvailable = true;

	}  // if [image data]

	// VTK polydata
	// Note: No "else" here on purpose. Might have both image- and polydata.

	if (polyData) 
	{
		// Get and print information about the polydata
		this->addSubItem(dsItem, "Type: PolyData");
		this->addSubItem(dsItem, "Number of Verts: "  + QString::number(polyData->GetNumberOfVerts()));
		this->addSubItem(dsItem, "Number of Lines: "  + QString::number(polyData->GetNumberOfLines()));
		this->addSubItem(dsItem, "Number of Polys: "  + QString::number(polyData->GetNumberOfPolys()));
		this->addSubItem(dsItem, "Number of Strips: " + QString::number(polyData->GetNumberOfStrips()));

		// Get the memory size
		dataSize += polyData->GetActualMemorySize();
		dataSizeAvailable = true;

	} // if [polyData]

	// VTK point set
	else if (pointSet)
	{
		// Print the number of points in the set
		this->addSubItem(dsItem, "Type: Point Set");
		this->addSubItem(dsItem, "Number of Points: " + QString::number(pointSet->GetNumberOfPoints()));

		// Get the memory size
		dataSize = pointSet->GetActualMemorySize();
		dataSizeAvailable = true;
	}

	// Print the data size in the second column
	if (dataSizeAvailable)
	{
		dsItem->setText(2, QString::number(dataSize) + "kB");
	}
	else
	{
		dsItem->setText(2, "N/A");
	}

	// Align the data size on the right
	dsItem->setTextAlignment(2, Qt::AlignRight);

	// Get the attributes of the data set
	Attributes * attr = ds->getAttributes();
	Q_ASSERT(attr);

	// Print all integer attributes
	QHash<QString, int> * intHash = attr->getIntAttributes();

	if (intHash)
	{
		for (QHash<QString, int>::const_iterator i = intHash->constBegin(); i != intHash->constEnd(); ++i)
		{
			this->addSubItem(dsItem, i.key() + " = " + QString::number(i.value()));
		}
	}

	// Print all double attributes
	QHash<QString, double> * doubleHash = attr->getDoubleAttributes();

	if (doubleHash)
	{
		for (QHash<QString, double>::const_iterator i = doubleHash->constBegin(); i != doubleHash->constEnd(); ++i)
		{
			this->addSubItem(dsItem, i.key() + " = " + QString::number(i.value()));
		}
	}

	// Print all vector attribute names
	QHash<QString, QList<double> > * vectorHash = attr->getVectorDoubleAttributes();

	if (vectorHash)
	{
		for (QHash<QString, QList<double> >::const_iterator i = vectorHash->constBegin(); i != vectorHash->constEnd(); ++i)
		{
			this->addSubItem(dsItem, "Vector: " + i.key());
		}
	}

	// Print all VTK attribute names
	QHash<QString, vtkObject *> * vtkHash = attr->getVtkAttributes();

	if (vtkHash)
	{
		for (QHash<QString, vtkObject *>::const_iterator i = vtkHash->constBegin(); i != vtkHash->constEnd(); ++i)
		{
			this->addSubItem(dsItem, "VTK Object: " + i.key());
		}
	}

	// Return the new widget item
	return dsItem;
}


//------------------------------[ addSubItem ]-----------------------------\\

void DataDialog::addSubItem(QTreeWidgetItem * parentItem, QString itemText)
{
	// Create a new tree widget item
    QTreeWidgetItem * subItem = new QTreeWidgetItem(parentItem);

	// Set the required text
    subItem->setText(0, itemText);
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void DataDialog::dataSetAdded(DataSet * ds)
{
	if (!ds)
		return;

	// Check if the data set has already been added (should never happen)
	if (this->dataSets.contains(ds))
		this->dataSetChanged(ds);

	// Add the data set to the list
	this->dataSets.append(ds);

	// Add it to the tree widget
	this->populateTreeWidget(ds);
}


//----------------------------[ dataSetChanged ]---------------------------\\

void DataDialog::dataSetChanged(DataSet * ds)
{
	if (!ds)
		return;

	// Check if the data set has been added before
	if (!(this->dataSets.contains(ds)))
		return;

	// Get the index of the data set
	int dsIndex = this->dataSets.indexOf(ds);

	// Remove the corresponding item from the tree widget, and delete it
	QTreeWidgetItem * currentItem = this->treeWidget->takeTopLevelItem(dsIndex);
	delete currentItem;

	// Create a new item for the data set
	QTreeWidgetItem * newItem = this->createWidgetItem(ds);

	// Insert the new item at the position of the old one
	this->treeWidget->insertTopLevelItem(dsIndex, newItem);

	// Recompute the size of the window
	this->treeWidget->resizeColumnToContents(0);
	this->treeWidget->resizeColumnToContents(1);

	if (this->treeWidget->columnWidth(0) < 300)
		this->treeWidget->setColumnWidth(0, 300);
	if (this->treeWidget->columnWidth(2) <  60)
		this->treeWidget->setColumnWidth(2, 60);

	this->setFixedWidth(this->treeWidget->columnWidth(0) + this->treeWidget->columnWidth(2) + 50);
}


//----------------------------[ dataSetRemoved ]---------------------------\\

void DataDialog::dataSetRemoved(DataSet * ds)
{
	if (!ds)
		return;

	// Check if the data set has been added before
	if (!(this->dataSets.contains(ds)))
		return;

	// Get the index of the data set
	int dsIndex = this->dataSets.indexOf(ds);

	// Remove the corresponding item from the tree widget, and delete it
	QTreeWidgetItem * currentItem = this->treeWidget->takeTopLevelItem(dsIndex);
	delete currentItem;

	// Remove the data set from the list
	this->dataSets.removeAt(dsIndex);
}


//--------------------------------[ close ]--------------------------------\\

void DataDialog::close()
{
	// Simply hide the dialog window
    this->hide();
}


} // namespace gui


} // namespace bmia
