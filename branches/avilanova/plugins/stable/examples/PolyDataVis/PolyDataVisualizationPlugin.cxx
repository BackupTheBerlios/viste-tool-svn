/*
 * PolyDataVisualizationPlugin.cxx
 *
 * 2010-06-25	Tim Peeters
 * - First version
 *
 * 2010-10-19	Evert van Aart
 * - Disabled this plugin for fiber data sets, as those are handled by the
 *   Fiber Visualization plugin.
 * 
 */

#include "PolyDataVisualizationPlugin.h"
#include "ui_polydata.h"

#include <vtkActor.h>
#include <vtkPropAssembly.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

#include <QColorDialog>

#include <QDebug>

namespace bmia {

PolyDataVisualizationPlugin::PolyDataVisualizationPlugin() : Plugin("PolyData")
{
    this->selectedData = -1;
    this->changingSelection = false;

    this->assembly = vtkPropAssembly::New();
    //this->assembly->VisibleOff();
    
    this->widget = new QWidget();
    this->ui = new Ui::PolyDataForm();
    this->ui->setupUi(this->widget);
    // disable the options frame if there is no data
    this->ui->optionsFrame->setEnabled(false);

    // Link events in the GUI to function calls:
    connect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectData(int)));
    connect(this->ui->visibleCheckBox, SIGNAL(toggled(bool)), this, SLOT(setVisible(bool)));
    connect(this->ui->lightingCheckBox, SIGNAL(toggled(bool)), this, SLOT(setLighting(bool)));
    connect(this->ui->colorButton, SIGNAL(clicked()), this, SLOT(changeColor()));
}

PolyDataVisualizationPlugin::~PolyDataVisualizationPlugin()
{
    // TODO: call dataSetRemoved() for all datasets.
    delete this->widget; this->widget = NULL;
    this->assembly->Delete();
}

vtkProp* PolyDataVisualizationPlugin::getVtkProp()
{
    return this->assembly;
}

QWidget* PolyDataVisualizationPlugin::getGUI()
{
    return this->widget;
}

void PolyDataVisualizationPlugin::dataSetAdded(data::DataSet* ds)
{
    Q_ASSERT(ds);
    vtkPolyData* polydata = ds->getVtkPolyData();

	if (polydata == NULL || ds->getKind() == "fibers")
	{
	// The added data set does not have VTK polydata, so it is not
	// interesting for this plug-in.
		// Evert: I've disabled this plugin for fibers, since it was interfering
		// with the fiber visualization plugin
	return;
    } // if

    // Add the new data set to the list of currently available polydata sets:
    this->dataSets.append(ds);

    // Build a pipeline for rendering this data set:
    vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
    mapper->ScalarVisibilityOff();
   mapper->SetInput(polydata);
    vtkActor* actor = vtkActor::New();
    actor->SetMapper(mapper);
    mapper->Delete(); mapper = NULL;
    // Note that the mapper was not actually deleted because it was
    // registered by the actor. And it can still be accessed through
    // actor->GetMapper().

    // Add the actor to the assembly to be rendered:
    this->assembly->AddPart(actor);

    // Add the actor to the list of actors, for easy access to its parameters
    // later on:
    this->actors.append(actor);

    // Add the new data set to the list of data sets in the GUI:
    this->ui->dataList->addItem(ds->getName());
    this->ui->optionsFrame->setEnabled(true);

    // TODO: select the newly added dataset

    this->core()->render();
}

void PolyDataVisualizationPlugin::dataSetChanged(data::DataSet* ds)
{
    // TODO
}

void PolyDataVisualizationPlugin::dataSetRemoved(data::DataSet* ds)
{
    // TODO: implement when unloading of data is implemented.
    // TODO: disable optionsFrame if number of datasets == 0.
}

void PolyDataVisualizationPlugin::selectData(int row)
{
    this->changingSelection = true;
    this->selectedData = row;
    Q_ASSERT(row >= 0); // TODO: if there is no data, do sth else
    // TODO: assert row is in range.
    this->ui->dataSetName->setText(this->dataSets.at(this->selectedData)->getName());
    this->ui->visibleCheckBox->setChecked(this->actors.at(this->selectedData)->GetVisibility());
    this->ui->lightingCheckBox->setChecked(this->actors.at(this->selectedData)->GetProperty()->GetLighting());
    this->changingSelection = false;
}

void PolyDataVisualizationPlugin::setVisible(bool visible)
{
    if (this->changingSelection) return;
    if (this->selectedData == -1) return;
    this->actors.at(this->selectedData)->SetVisibility(visible);
    this->core()->render();
}

void PolyDataVisualizationPlugin::setLighting(bool lighting)
{
    if (this->changingSelection) return;
    if (this->selectedData == -1) return;
    this->actors.at(this->selectedData)->GetProperty()->SetLighting(lighting);
    this->core()->render();
}

void PolyDataVisualizationPlugin::changeColor()
{
    if (this->changingSelection) return;
    if (this->selectedData == -1) return;

    Q_ASSERT(this->actors.at(this->selectedData));
    vtkProperty* property = this->actors.at(this->selectedData)->GetProperty();
    Q_ASSERT(property);

    double oldColorRGB[3];
    property->GetColor(oldColorRGB);

    QColor oldColor;
    oldColor.setRgbF(oldColorRGB[0], oldColorRGB[1], oldColorRGB[2]);    

    QColor newColor = QColorDialog::getColor(oldColor, 0);
    if ( newColor.isValid() )
	{
	property->SetColor(newColor.redF(), newColor.greenF(), newColor.blueF());
	this->core()->render();
	}
}

} // namespace bmia
Q_EXPORT_PLUGIN2(libPolyDataVisualizationPlugin, bmia::PolyDataVisualizationPlugin)
