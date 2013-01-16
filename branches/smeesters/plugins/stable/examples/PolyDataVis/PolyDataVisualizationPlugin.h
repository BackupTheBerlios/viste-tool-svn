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
 * PolyDataVisualizationPlugin.h
 *
 * 2010-06-23	Tim Peeters
 * - First version
 * 
 * 2010-10-19	Evert van Aart
 * - Disabled this plugin for fiber data sets, as those are handled by the
 *   Fiber Visualization plugin.
 * 
 */

// This example plugin shows how to create a plugin that uses
// data and visualizes that in the view.

#ifndef bmia_PolyDataVisualizationPlugin_h
#define bmia_PolyDataVisualizationPlugin_h

class vtkPropAssembly;
class vtkActor;

namespace Ui {
    class PolyDataForm;
}

#include "DTITool.h"

namespace bmia {

/**
 * This class visualizes poly data using the default vtkPolyDataMapper.
 * Plugins must always subclass plugin::Plugin. Because this plugin
 * uses data, visualizes something, and shows a Qt GUI to the user, it
 * also implements the interfaces data::Consumer, plugin::Visualization,
 * and plugin::GUI respectively.
 */
class PolyDataVisualizationPlugin : 	public plugin::Plugin,
					public data::Consumer,
					public plugin::Visualization,
					public plugin::GUI
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::Visualization)
    Q_INTERFACES(bmia::plugin::GUI)

public:
    PolyDataVisualizationPlugin();
    ~PolyDataVisualizationPlugin();

    //void init();

    /**
     * Return the VTK prop that renders all the geometry.
     * This implements the Visualization interface.
     */
    vtkProp* getVtkProp();

    /**
     * Return the Qt widget that gives the user control.
     * This implements the GUI interface.
     */
    QWidget* getGUI();

    /**
     * Implement the Consumer interface
     */
    void dataSetAdded(data::DataSet* ds);
    void dataSetChanged(data::DataSet* ds);
    void dataSetRemoved(data::DataSet* ds);

protected slots:
    void selectData(int row);
    void setVisible(bool visible);
    void setLighting(bool lighting);
    void changeColor();

private:
    /**
     * The collection of all the actors that this plugin can render.
     * This is the object that will be returned by getVtkProp().
     */
    vtkPropAssembly* assembly;

    /**
     * The Qt widget to be returned by getGUI().
     */
    QWidget* widget;

    /**
     * The Qt form created with Qt designer.
     */
    Ui::PolyDataForm* ui;

    /**
     * The added data sets that contain VTK polydata
     */
    QList<data::DataSet*> dataSets;

    /**
     * The actors associated with the data sets in dataSets.
     */
    QList<vtkActor*> actors;

    /**
     * Keep track whether the selection is being changed.
     * If this is set to true, then parameters in the GUI can
     * be updated without updating the rendering.
     */
    bool changingSelection;

    /**
     * The index of the currently selected data set.
     * -1 means no data set is selected.
     */
    int selectedData;

}; // class PolyDataVisualizationPlugin
} // namespace bmia
#endif // bmia_PolyDataVisualizationPlugin_h
