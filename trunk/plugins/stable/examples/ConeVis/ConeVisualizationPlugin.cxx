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
 * ConeVisualizationPlugin.cxx
 *
 * 2010-02-11	Tim Peeters
 * - First version
 *
 * 2010-03-04	Tim Peeters
 * - Add QWidget GUI.
 *
 * 2010-06-23	Tim Peeters
 * - Add comments to use this class as an example plugin.
 *
 * 2010-11-09	Tim Peeters
 * - Now a random cone color! woohoo ;)
 */

#include "ConeVisualizationPlugin.h"

// include the header file that is automatically generated from
// the form that was created using Qt designer
#include "ui_cone.h"

// include the header files for the VTK classes that are needed
// to create the cone actor
#include <vtkConeSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkMath.h>

// QDebug is included so that we can use qDebug() for writing
// debug messages to console output.
#include <QDebug>

namespace bmia {

// Construct a new ConeVisualizationPlugin object.
// There is nothing to do here because the pipeline will be built in init().
ConeVisualizationPlugin::ConeVisualizationPlugin() : plugin::Plugin("Cone")
{
    // Write text as output for debugging.
    // This can be removed when the plugin is tested and stable.
    qDebug()<<"Constructing ConeVisualizationPlugin object.";
}

// The init() function will be called immediately after the plugin object was
// created, and it has been assigned a Core object that can be used.
// Note that in this example we do not make use of the Core object.
void ConeVisualizationPlugin::init()
{
    // Write text output for debugging
    qDebug()<<"ConeVisualizationPlugin::init()";

    // Create the VTK objects and construct the pipeline
    this->cone = vtkConeSource::New();
    this->mapper = vtkPolyDataMapper::New();
    this->mapper->SetInput(this->cone->GetOutput());
    this->actor = vtkActor::New();
    this->actor->SetMapper(this->mapper);

    // Configure standard parameters
    this->cone->SetResolution(25);
    this->cone->SetAngle(30);
    double r = vtkMath::Random(0.0, 1.0);
    double g = vtkMath::Random(0.0, 1.0);
    double b = vtkMath::Random(0.0, 2.0-r-g);
    this->actor->GetProperty()->SetColor(r,g,b);

    // Create the Qt widget to be returned by getGUI()
    this->qWidget = new QWidget();
    this->ui = new Ui::ConeForm();
    this->ui->setupUi(this->qWidget);

    // Link the events of user interactions in the Qt GUI to functions of this class
    connect(this->ui->resolutionSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setResolution(int)));
    connect(this->ui->angleSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setAngle(int)));
}

// Destroy this instance of ConeVisualizationPlugin.
ConeVisualizationPlugin::~ConeVisualizationPlugin()
{
    delete this->qWidget; this->qWidget = NULL;
    this->actor->Delete(); this->actor = NULL;
    this->mapper->Delete(); this->mapper = NULL;
    this->cone->Delete(); this->cone = NULL;
}

vtkProp* ConeVisualizationPlugin::getVtkProp()
{
    return this->actor;
}

QWidget* ConeVisualizationPlugin::getGUI()
{
    return this->qWidget;
}

void ConeVisualizationPlugin::setResolution(int r)
{
    qDebug()<<"Setting resolution to "<<r;
    this->cone->SetResolution(r);
    // After changing the parameter, call the core render() function
    // to update the rendered cone in the view.
    this->core()->render();
}

void ConeVisualizationPlugin::setAngle(int a)
{
    qDebug()<<"Setting angle to "<<a;
    this->cone->SetAngle(double(a));
    this->core()->render();
}

} // namespace bmia

// This macro is obligatory for compiling plugins.
// Make sure the used name corresponds to the name in CMakeLists.txt
// (with lib added in front of it), or the plugin cannot be loaded.
Q_EXPORT_PLUGIN2(libConeVisualizationPlugin, bmia::ConeVisualizationPlugin)
