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
