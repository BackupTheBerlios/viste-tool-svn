/*
 * ConeVisualizationPlugin
 *
 * 2010-02-11	Tim Peeters
 * - First version
 *
 * 2010-03-04	Tim Peeters
 * - Implement GUI interface.
 *
 * 2010-06-23	Tim Peeters
 * - Extend documentation to use this as an example plugin.
 */

// This is an example header file that shows how to create a plugin
// that visualizes something using a VTK actor, and how to manipulate
// the parameters of the visualization using a Qt GUI.

// Make sure that the class description below is only loaded once,
// even if this header file would be included multiple times
#ifndef bmia_ConeVisualization_ConeVisualizationPlugin_h
#define bmia_ConeVisualization_ConeVisualizationPlugin_h

// The VTK classes that will be used in this plugin
class vtkActor;
class vtkPolyDataMapper;
class vtkConeSource;

// The Qt class that was created using Qt designer, that will be used
// here as the basis for our Qt widget for defining the visualization
// parameters
namespace Ui {
    class ConeForm;
} // namspace Ui

// Include the header file that includes all the needed header files
// from the tool
#include "DTITool.h"

// Always use namespace bmia to avoid conflicts with other
// software/packages/libraries.
namespace bmia {

/**
 * A simple visualization plugin that does not consume data
 * and renders a cone. Every plugin must be a subclass of the plugin::Plugin
 * class. Because this plugin is a Visualization plugin and a GUI plugin,
 * it also subclasses the plugin::Visualization and plugin::GUI interfaces.
 */
class ConeVisualizationPlugin : public plugin::Plugin, public plugin::Visualization, public plugin::GUI
{
    // Always list Q_OBJECT here for Qt and plugin classes so that the Qt
    // Meta Object Compiler (moc) processes this file.
    Q_OBJECT
    // Call the Q_INTERFACES macro for bmia::plugin::Plugin and for each of the plugin interfaces
    // that this plugin implements.
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::Visualization)
    Q_INTERFACES(bmia::plugin::GUI)

public:
    /**
     * Construct a new ConeVisualization plugin.
     * Please note that at the time the plugin is constructed, no Core object exists
     * yet. The core object can be accessed in the init() function below which is
     * called immediately after the plugin has been loaded. See tool/plugin/Plugin.h
     * for more information.
     */
    ConeVisualizationPlugin();

    /**
     * Destroy the current ConeVisualizationPlugin object.
     */
    ~ConeVisualizationPlugin();

    /**
     * This function is called immediately after the plugin was loaded.
     * See tool/plugin/Plugin.h for more information about when to use init().
     */
    void init();

    /**
     * Return the VTK actor that renders the cone.
     * This function must be implemented for each subclass of the plugin::Visualization
     * interface, and it returns the vtkProp (a vtkActor or vtkVolume) that will be
     * added to the scene when this plugin is used.
     */
    vtkProp* getVtkProp();

    /**
     * Return the widget that is shown in the GUI.
     * This function must be implemented for each subclass of the plugin::GUI
     * interface, and it returns a Qt widget that gives the user the options to change
     * parameters.
     */
    QWidget* getGUI();

protected slots:
    // These two functions will be called when the user changes parameters in the Qt GUI.
    
    /**
     * Set the resolution of the cone.
     */
    void setResolution(int r);
    /**
     * Set the angle of the cone.
     */
    void setAngle(int a);

private:
    // The VTK objects that build the pipeline that renders the cone
    vtkActor* actor;
    vtkPolyDataMapper* mapper;
    vtkConeSource* cone;

    // The Qt widget to return in getGUI() function.
    QWidget* qWidget;
    // The cone form that was created using Qt Designer.
    Ui::ConeForm* ui;

}; // class ConeVisualizationPlugin
} // namespace bmia
#endif // bmia_ConeVisualization_ConeVisualizationPlugin_h
