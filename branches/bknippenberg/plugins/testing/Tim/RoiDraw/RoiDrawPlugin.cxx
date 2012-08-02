/*
 * RoiDrawPlugin.cxx
 *
 * 2010-11-02	Tim Peeters
 * - First version
 *
 * 2010-12-14	Evert van Aart
 * - Hotfix solution for the bug in Windows, which causes errors when
 *   switching windows. This is an improvised solution, a better 
 *   solution should be made in the near future.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for saving ROIs. Handles cannot be saved at the moment,
 *   and a loaded ROI should be added to the drawing window (which is not
 *   the case at the moment), but this will do for now.
 *
 */

#include "RoiDrawPlugin.h"
#include "ui_rois.h"
#include "data/DataSet.h"
#include "RoiDrawDialog.h"
#include "Helpers/TransformationMatrixIO.h"

#include <QDebug>
#include <QFileDialog>

#include <vtkActor.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropAssembly.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>

namespace bmia {

RoiDrawPlugin::RoiDrawPlugin() : Plugin("ROI Edit")
{
    this->widget = new QWidget();
    this->ui = new Ui::RoiForm();
    this->ui->setupUi(this->widget);

    this->dialog = new RoiDrawDialog(this);

    this->assembly = vtkPropAssembly::New();

    Q_ASSERT(this->roiData.size() == 0);
    Q_ASSERT(this->volumeData.size() == 0);

    this->selectedRoi = -1; // no ROI selected

    // leave the option for seeding from voxels disabled until
    // there are data sets that can be chosen here.
    this->ui->seedVoxelsRadio->setEnabled(false);
    this->ui->seedVoxelsComboBox->setEnabled(false);
    this->ui->renameButton->setEnabled(false);
    this->ui->deleteButton->setEnabled(false);
    this->ui->drawButton->setEnabled(false);

    connect(this->ui->roiComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(selectRoi(int)));
    connect(this->ui->drawButton, SIGNAL(clicked()), this, SLOT(draw()));
    connect(this->ui->newButton, SIGNAL(clicked()), this, SLOT(newRoi()));
    connect(this->ui->renameButton, SIGNAL(clicked()), this, SLOT(rename()));
    connect(this->ui->deleteButton, SIGNAL(clicked()), this, SLOT(deleteRoi()));
	connect(this->ui->saveROIButton, SIGNAL(clicked()), this, SLOT(saveROI()));

    connect(this->ui->noSeedingRadio, SIGNAL(clicked()), this, SLOT(noSeeding()));
    connect(this->ui->seedDistanceRadio, SIGNAL(clicked()), this, SLOT(distanceSeeding()));
    connect(this->ui->seedVoxelsRadio, SIGNAL(clicked()), this, SLOT(voxelSeeding()));
    connect(this->ui->seedDistanceSpinBox, SIGNAL(valueChanged(double)), this, SLOT(updateSeedDistance()));
    connect(this->ui->seedVoxelsComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateSeedVoxels()));
}

void RoiDrawPlugin::init()
{
    // Nothing to do (yet?)
}

RoiDrawPlugin::~RoiDrawPlugin()
{
    // TODO: close the dialog to avoid segfault!? (Shouldn't that be automatic?)
    this->dialog->hide();
    delete this->dialog; this->dialog = NULL;
    delete this->widget; this->widget = NULL;
    this->assembly->Delete(); this->assembly = NULL;
}

QWidget* RoiDrawPlugin::getGUI()
{
    return this->widget;
}

vtkProp* RoiDrawPlugin::getVtkProp()
{
    return this->assembly;
}

void RoiDrawPlugin::dataSetAdded(data::DataSet* ds)
{
    Q_ASSERT(ds);
    if (ds->getKind() == "regionOfInterest")
	{
	this->roiData.append(ds);

	vtkPolyData* polydata = ds->getVtkPolyData();
	Q_ASSERT(polydata);

	// Build pipeline for rendering the ROI in the 3D view
	vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
	mapper->ScalarVisibilityOff();
	mapper->SetInput(polydata);
	vtkActor* actor = vtkActor::New();
	actor->SetMapper(mapper);
	mapper->Delete(); mapper = NULL;
	actor->GetProperty()->SetColor(0.0, 1.0, 0.0);

	vtkObject * obj = NULL;
	if (ds->getAttributes()->getAttribute("transformation matrix", obj))
	{
		actor->SetUserMatrix(vtkMatrix4x4::SafeDownCast(obj));
	}

	this->assembly->AddPart(actor);
	actor->Delete(); actor = NULL;

	// add the new data set at the end of the combobox.
	this->ui->roiComboBox->addItem(ds->getName());

	// Select the newly added ROI:
	this->ui->roiComboBox->setCurrentIndex(this->ui->roiComboBox->count()-1);

	if (this->roiData.size() == 1)
	    {
	    this->ui->renameButton->setEnabled(true);
	    this->ui->deleteButton->setEnabled(true);
	    this->ui->drawButton->setEnabled(true);
		this->ui->noSeedingRadio->setEnabled(true);
		this->ui->seedDistanceRadio->setEnabled(true);
	    } // if
	}
      else if (ds->getKind() == "DTI")
	{
	this->volumeData.append(ds);
	this->ui->seedVoxelsComboBox->addItem(ds->getName());
	} // add volume
    else if (ds->getKind() == "sliceActor")
	{
	this->dialog->addSliceData(ds);
	} // add slice actor to the ROI draw window

    if ((this->roiData.size() > 0) && (this->volumeData.size() > 0))
	{
	this->ui->seedVoxelsRadio->setEnabled(true);
	this->ui->seedVoxelsComboBox->setEnabled(true);
	}
}

void RoiDrawPlugin::dataSetChanged(data::DataSet* ds)
{
    Q_ASSERT(ds);
    if (ds->getKind() == "sliceActor")
	{
	this->dialog->sliceDataChanged(ds);
	}
    else if (ds->getKind() == "regionOfInterest")
	{ // this data set must have been added before.
	Q_ASSERT(ds->getVtkPolyData());

	int data_index = this->roiData.indexOf(ds);
	Q_ASSERT(data_index >= 0);
	Q_ASSERT(data_index < this->roiData.size());

	// in case the name was updated:
	this->ui->roiComboBox->setItemText(data_index, ds->getName());

	// in case the polydata was updated:
	vtkActor* actor = this->getRoiActor(data_index);
	Q_ASSERT(actor);
	vtkPolyDataMapper* mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
	Q_ASSERT(mapper);
	mapper->SetInput(ds->getVtkPolyData());

	vtkObject * obj = NULL;
	if (ds->getAttributes()->getAttribute("transformation matrix", obj))
	{
		actor->SetUserMatrix(vtkMatrix4x4::SafeDownCast(obj));
	}

	this->dialog->turnPickingOff();
	this->core()->render();
	this->dialog->makeCurrent();
	}
}

void RoiDrawPlugin::dataSetRemoved(data::DataSet* ds)
{
    Q_ASSERT(ds);
    if (ds->getKind() == "regionOfInterest")
	{ // this data set must have been added before.
	this->selectRoi(-1);

	int data_index = this->roiData.indexOf(ds);

	Q_ASSERT(data_index >= 0);
	Q_ASSERT(data_index < this->roiData.size());

	vtkActor* actor = this->getRoiActor(data_index);
	Q_ASSERT(actor);
	this->assembly->RemovePart(actor);
	actor = NULL;

	this->roiData.removeAt(data_index);

	this->ui->roiComboBox->removeItem(data_index);
	// automatically changes selected item and updates.
	
	vtkPropCollection* collection = this->assembly->GetParts();
	Q_ASSERT(collection);
    	Q_ASSERT(collection->GetNumberOfItems() == this->ui->roiComboBox->count());
	}
    // TODO: ds->getKind() == "DTI" or sliceActor.

    if (this->roiData.size() == 0)
	{ // no data. disable options
	this->dialog->hide();
	this->ui->seedVoxelsRadio->setEnabled(false);
	this->ui->seedVoxelsComboBox->setEnabled(false);
	this->ui->renameButton->setEnabled(false);
	this->ui->deleteButton->setEnabled(false);
	this->ui->drawButton->setEnabled(false);
	this->ui->nameEdit->setText("");
	}
}

void RoiDrawPlugin::selectRoi(int index)
{
    Q_ASSERT(index >= -1);
    Q_ASSERT(index < this->ui->roiComboBox->count());

    // deselect the current ROI
    if (this->selectedRoi != -1)
	{ // there is a selected ROI. Deselect it.
	vtkActor* actor = this->getRoiActor(this->selectedRoi);
	actor->GetProperty()->SetColor(0.0, 1.0, 0.0);
	this->selectedRoi = -1;
	}

    if (index == -1) return;

    // no ROI is selected
   
    this->selectedRoi = index;
    vtkActor* actor = this->getRoiActor(this->selectedRoi);
    actor->GetProperty()->SetColor(1.0, 1.0, 0.0);
    Q_ASSERT(this->getSelectedRoi());
    this->ui->nameEdit->setText(this->getSelectedRoi()->getName());
    actor = NULL;

    data::DataSet* ds = this->roiData.at(selectedRoi);
    Q_ASSERT(ds);
    data::Attributes* attr = ds->getAttributes();
    Q_ASSERT(attr);

    // Set the seeding options.
    disconnect(this->ui->noSeedingRadio, SIGNAL(clicked()), this, SLOT(noSeeding()));
    disconnect(this->ui->seedDistanceRadio, SIGNAL(clicked()), this, SLOT(distanceSeeding()));
    disconnect(this->ui->seedVoxelsRadio, SIGNAL(clicked()), this, SLOT(voxelSeeding()));
    disconnect(this->ui->seedDistanceSpinBox, SIGNAL(valueChanged(double)), this, SLOT(updateSeedDistance()));
    disconnect(this->ui->seedVoxelsComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateSeedVoxels()));

    double seedDistance = 0.0;
    vtkObject* voxelData = NULL;
    if (attr->getAttribute("Seed distance", seedDistance))
	{
	this->ui->seedDistanceSpinBox->setValue(seedDistance);
	this->ui->seedDistanceRadio->setChecked(true);	
	} // there
    else if (attr->getAttribute("Seed voxels", voxelData))
	{
	Q_ASSERT(voxelData);
	vtkImageData* image = vtkImageData::SafeDownCast(voxelData);
	Q_ASSERT(image);
	this->ui->seedVoxelsRadio->setChecked(true);
	for (int i=0; i < this->volumeData.size(); i++)
	    {
	    Q_ASSERT(this->volumeData.at(i));
	    if (image == this->volumeData.at(i)->getVtkImageData())
		{
		this->ui->seedVoxelsComboBox->setCurrentIndex(i);
		} // if
	    } // for
	}
    else
	{
	this->ui->noSeedingRadio->setChecked(true);
	}
    
    // TODO: connect
    connect(this->ui->noSeedingRadio, SIGNAL(clicked()), this, SLOT(noSeeding()));
    connect(this->ui->seedDistanceRadio, SIGNAL(clicked()), this, SLOT(distanceSeeding()));
    connect(this->ui->seedVoxelsRadio, SIGNAL(clicked()), this, SLOT(voxelSeeding()));
    connect(this->ui->seedDistanceSpinBox, SIGNAL(valueChanged(double)), this, SLOT(updateSeedDistance()));
    connect(this->ui->seedVoxelsComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(updateSeedVoxels()));


	this->dialog->turnPickingOff();
	this->core()->render();
	this->dialog->makeCurrent();
}

void RoiDrawPlugin::newRoi()
{
    // Create a new ROI dataset
    int numRois = this->roiData.size();
    QString newdataname = QString("new ROI %1").arg(numRois+1);

    vtkPolyData* polydata = vtkPolyData::New(); // emtpy ROI
    data::DataSet* ds = new data::DataSet(newdataname, "regionOfInterest", polydata);
    polydata->Delete(); polydata = NULL;

    // Add the new ROI dataset to the data manager:
    this->core()->data()->addDataSet(ds);
    // That's it. After adding the data set, dataSetAdded(ds) is automatically
    // called by the data manager, which takes care of the rest of the stuff.
}

vtkActor* RoiDrawPlugin::getRoiActor(int index)
{
    vtkPropCollection* collection = this->assembly->GetParts();
    Q_ASSERT(collection);

    Q_ASSERT(collection->GetNumberOfItems() == this->ui->roiComboBox->count());

    Q_ASSERT(index < collection->GetNumberOfItems());
    Q_ASSERT(index >= 0);

    vtkObject* obj = collection->GetItemAsObject(index);
    Q_ASSERT(obj);
    vtkActor* actor = vtkActor::SafeDownCast(obj);
    Q_ASSERT(actor);

    return actor;
}

void RoiDrawPlugin::draw()
{
    this->dialog->show();
}

void RoiDrawPlugin::updateROI(vtkPolyData* newdata, vtkMatrix4x4 * m)
{
    if (this->selectedRoi == -1) return;

    data::DataSet* ds = this->getSelectedRoi();
    Q_ASSERT(ds);

	vtkObject * obj = NULL;

	if (m)
	{
		vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
		mCopy->DeepCopy(m);

		if (ds->getAttributes()->getAttribute("transformation matrix", obj))
		{
			ds->getAttributes()->changeAttribute("transformation matrix", vtkObject::SafeDownCast(mCopy));
		}
		else
		{
			ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(mCopy));
		}
	}

	ds->updateData(newdata);
    this->core()->data()->dataSetChanged(ds);
}

void RoiDrawPlugin::noSeeding(bool updateDS)
{
    qDebug()<<"No seeding.";
    data::DataSet* ds = this->getSelectedRoi();
    if (!ds) return;

    double seedDist = 0.0;
    if (ds->getAttributes()->getAttribute("Seed distance", seedDist))
	{
	ds->getAttributes()->removeAttribute("Seed distance", seedDist);
	}

    vtkObject* voxels = NULL;
    if (ds->getAttributes()->getAttribute("Seed voxels", voxels))
	{
	ds->getAttributes()->removeAttribute("Seed voxels", voxels);
	}
    voxels = NULL;
 
	if (updateDS)
	{
		this->core()->data()->dataSetChanged(ds);
	}
}

void RoiDrawPlugin::distanceSeeding()
{
    this->noSeeding(false);
    this->updateSeedDistance();
}

void RoiDrawPlugin::voxelSeeding()
{
    this->noSeeding(false);
    this->updateSeedVoxels();
}

void RoiDrawPlugin::updateSeedDistance()
{
    if (!this->ui->seedDistanceRadio->isChecked()) return;

    // seeding type is distance seeding.
    double dist = this->ui->seedDistanceSpinBox->value();

    data::DataSet* ds = this->getSelectedRoi();
    if (!ds) return;

    double oldDist = 0.0;
    if (!ds->getAttributes()->getAttribute("Seed distance", oldDist))
	{
	qDebug()<<"adding seed distance attribute with value "<<dist;
	ds->getAttributes()->addAttribute("Seed distance", dist);
	}
    else // attribute "Seed distance" exists
	{
	if (dist == oldDist)
	    {
	    return; // nothing changed.
	    } // if
	else
	    {
	    qDebug()<<"changing seed distance attribute to value "<<dist;
	    ds->getAttributes()->changeAttribute("Seed distance", dist);
	    } // else
	} // else

    this->core()->data()->dataSetChanged(ds);
}

void RoiDrawPlugin::updateSeedVoxels()
{
    if (!this->ui->seedVoxelsRadio->isChecked()) return;

    // seeding type is seed voxels
    int imageDataIndex = this->ui->seedVoxelsComboBox->currentIndex();
    Q_ASSERT(imageDataIndex != -1);

    data::DataSet* imageDs = NULL;
    data::DataSet* roiDs = this->getSelectedRoi();
    if (!roiDs) return;

    vtkObject* voxels = NULL;
    if (roiDs->getAttributes()->getAttribute("Seed voxels", voxels))
	{
	roiDs->getAttributes()->removeAttribute("Seed voxels", voxels);
	}

    imageDs = this->volumeData.at(imageDataIndex);
    Q_ASSERT(imageDs);
    voxels = imageDs->getVtkImageData();
    Q_ASSERT(voxels);

    roiDs->getAttributes()->addAttribute("Seed voxels", voxels);
    this->core()->data()->dataSetChanged(roiDs);
}

data::DataSet* RoiDrawPlugin::getSelectedRoi()
{
    if (this->selectedRoi == -1) return NULL;
    data::DataSet* ds = this->roiData.at(this->selectedRoi);

    Q_ASSERT(ds);
    return ds;
}

void RoiDrawPlugin::deleteRoi()
{
    data::DataSet* ds = getSelectedRoi();
    if (!ds) return;
    this->core()->data()->removeDataSet(ds);
}


void RoiDrawPlugin::saveROI()
{
	int roiIndex = this->ui->roiComboBox->currentIndex();

	if (roiIndex < 0 || roiIndex >= this->roiData.size())
		return;

	// Open a file dialog to get a filename
	QString fileName = QFileDialog::getSaveFileName(NULL, "Write ROI", "", " Region of Interest (*.pol)");
	
	// Check if the filename is correct
	if (fileName.isEmpty())
		return;

	// Convert the QString to a character array
	QByteArray ba = fileName.toAscii();
	char * fileNameChar = ba.data();

	// Get the polydata object containing the fibers
	vtkPolyData * output = this->roiData.at(roiIndex)->getVtkPolyData();

	// Check if the fibers exist
	if (!output)
		return;

	// Create a polydata writer
	vtkPolyDataWriter * writer = vtkPolyDataWriter::New();

	// Configure the writer
	writer->SetFileName(fileNameChar);
	writer->SetInput(output);
	writer->SetFileTypeToASCII();

	// Write output file
	writer->Write();

	// Delete the writer
	writer->Delete();

	vtkObject * attObject;

	// Check if the ROI contains a transformation matrix
	if (this->roiData.at(roiIndex)->getAttributes()->getAttribute("transformation matrix", attObject))
	{
		std::string err = "";

		// If so, write the matrix to a ".tfm" file
		bool success = TransformationMatrixIO::writeMatrix(std::string(fileNameChar), vtkMatrix4x4::SafeDownCast(attObject), err);

		// Display error messages if necessary
		if (!success)
		{
			this->core()->out()->showMessage(QString(err.c_str()));
		}
	}
}

void RoiDrawPlugin::rename()
{
    data::DataSet* ds = getSelectedRoi();
    if (!ds) return;

    QString newName = this->ui->nameEdit->text();
    ds->setName(newName);
    this->core()->data()->dataSetChanged(ds);
}

} // namespace bmia
Q_EXPORT_PLUGIN2(libRoiDrawPlugin, bmia::RoiDrawPlugin)
