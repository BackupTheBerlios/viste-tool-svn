#include "DistanceMeasures.h"

#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New()

namespace bmia
{

///
///      INITIALIZATION
///

//------------------------[ Plugin constructor ]-----------------------\\

DistanceMeasures::DistanceMeasures() : plugin::AdvancedPlugin("DistanceMeasures")
{
    this->widget = NULL;
    this->form   = NULL;
}

//------------------------[ Plugin destructor ]-----------------------\\

DistanceMeasures::~DistanceMeasures()
{
    delete this->widget;
    delete this->form;
}

//------------------------[ Initialization ]-----------------------\\

void DistanceMeasures::init()
{
    this->widget = new QWidget();
    this->form = new Ui::DistanceMeasuresForm();
    this->form->setupUi(this->widget);

    // Link events in the GUI to function calls
    this->connectAll();
    this->assembly = vtkPropAssembly::New();
}

///
///      DATA I/O
///

//------------------------[ Dataset added ]-----------------------\\

void DistanceMeasures::dataSetAdded(data::DataSet * d)
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
		sortedFibers->userSelectedLine = 0;

        // Add the new data set to the list of currently available fiber sets
        this->sortedFibersList.append(sortedFibers);

        // Add to UI combobox for distance measurements to fibers
        this->form->comboBoxFiberData->addItem(d->getName());
	}
}

//------------------------[ Dataset changed ]-----------------------\\

void DistanceMeasures::dataSetChanged(data::DataSet * d)
{
    // Assert the data set pointer (should never be NULL)
    Q_ASSERT(d);

	// Get the kind of the data set
    QString kind = d->getKind();

    // to-do

}

//------------------------[ Dataset removed ]-----------------------\\

void DistanceMeasures::dataSetRemoved(data::DataSet * d)
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
        this->form->comboBoxFiberData->removeItem(dsIndex);

        // Remove from collection
        this->sortedFibersList.removeAt(dsIndex);
	}
}

int DistanceMeasures::FindInputDataSet(data::DataSet * ds)
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

//------------------------[ Create text labels ]-----------------------\\

vtkActor2D* DistanceMeasures::GenerateLabels(vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkStringArray> labels)
{
	VTK_CREATE(vtkPolyData, polydata);
	polydata->SetPoints(points);

	VTK_CREATE(vtkVertexGlyphFilter, glyphFilter);
	glyphFilter->SetInputConnection(polydata->GetProducerPort());
	glyphFilter->Update();

	// Add label array.
	glyphFilter->GetOutput()->GetPointData()->AddArray(labels);

	// Create a mapper and actor for the points.
	VTK_CREATE(vtkPolyDataMapper,pointMapper);
	pointMapper->SetInputConnection(glyphFilter->GetOutputPort());

	VTK_CREATE(vtkActor, pointActor);
	pointActor->SetMapper(pointMapper);

	// Generate the label hierarchy.
	VTK_CREATE(vtkPointSetToLabelHierarchy, pointSetToLabelHierarchyFilter);
	pointSetToLabelHierarchyFilter->SetInputConnection(
	glyphFilter->GetOutputPort());
	pointSetToLabelHierarchyFilter->SetLabelArrayName("labels");
	pointSetToLabelHierarchyFilter->Update();

	// Create a mapper and actor for the labels.
	VTK_CREATE(vtkLabelPlacementMapper, labelMapper);
	labelMapper->SetInputConnection(pointSetToLabelHierarchyFilter->GetOutputPort());
	labelMapper->UseDepthBufferOff();
	labelMapper->PlaceAllLabelsOn();

	vtkActor2D* labelActor = vtkActor2D::New();
	labelActor->SetMapper(labelMapper);
	labelActor->SetLayerNumber(5);

	return labelActor;
}

///
///     GUI CALLBACKS
///

void DistanceMeasures::UpdateCoordinates()
{
    //PlanesVisPlugin* planesPlugin = static_cast<PlanesVisPlugin*>(this->fullCore()->plugin()->getPlugin(this->fullCore()->plugin()->indexOf("Planes")));

    //this->slicePosX =
}

void DistanceMeasures::setMeasuredPoint(int id)
{
    this->UpdateCoordinates();

    /*MeasuredPoint* point = this->measuredPointList.at(id);

    if(!point->set)
    {
        this->assembly->AddPart(point->sphere);
    }

    point->set = true;

    point->x = this->clickedPoint[0];
    point->y = this->clickedPoint[1];
    point->z = this->clickedPoint[2];

    point->sphere->SetPosition(point->x,point->y,point->z);

    if(id == 0)
    {
        this->form->inputXPointA->setValue(point->x);
        this->form->inputYPointA->setValue(point->y);
        this->form->inputZPointA->setValue(point->z);
    }
    else
    {
        this->form->inputXPointB->setValue(point->x);
        this->form->inputYPointB->setValue(point->y);
        this->form->inputZPointB->setValue(point->z);
    }

    calculateDistance();

    this->core()->render();*/
}

void DistanceMeasures::calculateDistance()
{
    /*MeasuredPoint* pointA = this->measuredPointList.at(0);
    MeasuredPoint* pointB = this->measuredPointList.at(1);

    if(!pointA->set || !pointB->set)
        return;

    double distance =   sqrt( (pointA->x - pointB->x)*(pointA->x - pointB->x) +
                        (pointA->y - pointB->y)*(pointA->y - pointB->y) +
                        (pointA->z - pointB->z)*(pointA->z - pointB->z) );
    QString labeltext = QString("Measured distance: %1 mm").arg(distance,0,'f',2);
	QString labeltext_short = QString("%1 mm").arg(distance,0,'f',2);
    this->form->measuredDistanceLabel->setText(labeltext);

    // renew line
    vtkSmartPointer<vtkLineSource> lineSource =
        vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(pointA->x,pointA->y,pointA->z);
    lineSource->SetPoint2(pointB->x,pointB->y,pointB->z);
    vtkSmartPointer<vtkPolyDataMapper> lineMapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    lineMapper->SetInputConnection(lineSource->GetOutputPort());

    vtkActor* lineActor =
        vtkActor::New();
    lineActor->SetMapper(lineMapper);
    lineActor->GetProperty()->SetLineStipplePattern(0xFF00);

    this->assembly->AddPart(lineActor);
    if(measuredLine != NULL)
        this->assembly->RemovePart(measuredLine);
    measuredLine = lineActor;



	measuredLabelPoints = vtkPoints::New();
	measuredLabelStrings = vtkStringArray::New();

	measuredLabelStrings->SetName("labels");
	measuredLabelPoints->InsertNextPoint(pointA->x,pointA->y,pointA->z+5);
	measuredLabelStrings->InsertNextValue(this->form->lineEditNamePointA->text().toLocal8Bit().constData());

	measuredLabelPoints->InsertNextPoint(pointB->x,pointB->y,pointB->z+5);
	measuredLabelStrings->InsertNextValue(this->form->lineEditNamePointB->text().toLocal8Bit().constData());

	measuredLabelPoints->InsertNextPoint(pointA->x + (pointB->x-pointA->x)/2.0,
							pointA->y + (pointB->y-pointA->y)/2.0,
							pointA->z + (pointB->z-pointA->z)/2.0+5);
	measuredLabelStrings->InsertNextValue(labeltext_short.toLocal8Bit().constData());

    if(measuredLabels != NULL)
        this->assembly->RemovePart(measuredLabels);

	measuredLabels = this->GenerateLabels(measuredLabelPoints,measuredLabelStrings);
	this->assembly->AddPart(measuredLabels);


    if (this->currentElectrodesColor.isValid())
    {
        measuredLine->GetProperty()->SetColor(this->currentElectrodesColor.redF(), this->currentElectrodesColor.greenF(), this->currentElectrodesColor.blueF());
        pointA->sphere->GetProperty()->SetColor(this->currentElectrodesColor.redF(), this->currentElectrodesColor.greenF(), this->currentElectrodesColor.blueF());
        pointB->sphere->GetProperty()->SetColor(this->currentElectrodesColor.redF(), this->currentElectrodesColor.greenF(), this->currentElectrodesColor.blueF());
    }*/
}

///
///      GUI CONTROLS
///

//------------------------[ Connect Qt elements ]-----------------------\\

void DistanceMeasures::connectAll()
{

}

//------------------------[ Disconnect Qt elements ]-----------------------\\

void DistanceMeasures::disconnectAll()
{

}

//------------------------[ Set point A ]-----------------------\\

void DistanceMeasures::buttonSetPointAClicked()
{
    setMeasuredPoint(0);
}

//------------------------[ Set point B ]-----------------------\\

void DistanceMeasures::buttonSetPointBClicked()
{
    setMeasuredPoint(1);
}

///
///     vISTe communication
///

//-----------[ Returns visualization component as VTK object ]---------------\\
//
vtkProp * DistanceMeasures::getVtkProp()
{
    return this->assembly;
}

//-----------------[ Returns GUI component as Qt widget ]---------------\\
//
QWidget * DistanceMeasures::getGUI()
{
    return this->widget;
}

}

Q_EXPORT_PLUGIN2(libDistanceMeasures, bmia::DistanceMeasures)
