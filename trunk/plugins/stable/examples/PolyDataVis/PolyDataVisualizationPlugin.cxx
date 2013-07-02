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
 * PolyDataVisualizationPlugin.cxx
 *
 * 2010-06-25	Tim Peeters
 * - First version
 *
 * 2010-10-19	Evert van Aart
 * - Disabled this plugin for fiber data sets, as those are handled by the
 *   Fiber Visualization plugin.
 *
 *  2013-07-02	Mehmet Yusufoglu
 * - Added an opacity slider,corresponding slot and lines to the
 * list box data selection slot. No class variables added.
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
	 connect(this->ui->opacitySlider, SIGNAL(valueChanged(int)), this, SLOT(changeOpacity(int)));
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
	
		vtkObject * obj;
	if (!(ds->getAttributes()->getAttribute("transformation matrix", obj)))
	{
		qDebug() << "No transformation matrix for the polydata" << endl;
		//return;
	}
	else {

	vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);

	actor->SetUserMatrix(m);
	}
    // Add the actor to the assembly to be rendered:
    this->assembly->AddPart(actor);

    // Add the actor to the list of actors, for easy access to its parameters
    // later on:
    this->actors.append(actor);

    // Add the new data set to the list of data sets in the GUI:
    this->ui->dataList->addItem(ds->getName());
    this->ui->optionsFrame->setEnabled(true);

    // TODO: select the newly added dataset
//		this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
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
	//opacity
	this->ui->opacitySlider->setValue(this->actors.at(this->selectedData)->GetProperty()->GetOpacity()*100);
	this->ui->opacityLabel->setText( QString::number( this->actors.at(this->selectedData)->GetProperty()->GetOpacity() ));
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

void PolyDataVisualizationPlugin::changeOpacity(int value)
{
    if (this->changingSelection) return;
    if (this->selectedData == -1) return;
	 Q_ASSERT(this->actors.at(this->selectedData));
	 this->actors.at(this->selectedData)->GetProperty()->SetOpacity(value/100.0);
	 	this->ui->opacityLabel->setText( QString::number(value/100.0));
     this->core()->render();

}

} // namespace bmia
Q_EXPORT_PLUGIN2(libPolyDataVisualizationPlugin, bmia::PolyDataVisualizationPlugin)
