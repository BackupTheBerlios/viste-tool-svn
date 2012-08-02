/*
 * VTKRayCastPlugin
 *
 * 2010-03-10	Wiljan van Ravensteijn
 * - First version
 */

#ifndef bmia_VTKRayCastPlugin_VTKRayCastPlugin_h
#define bmia_VTKRayCastPlugin_VTKRayCastPlugin_h

class vtkVolume;
class vtkColorTransferFunction;
class vtkPiecewiseFunction;
class QWidget;
class QComboBox;

#include "DTITool.h"

namespace bmia {

/**
 * A simple visualization plugin that does consume volume data
 * and renders it ray casting.
 */
class VTKRayCastPlugin : public plugin::Plugin, public plugin::Visualization, public data::Consumer, public plugin::GUI
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::Visualization)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::GUI)

public:
     /**
     * Construct a new VTKRayCast plugin instance.
     */
    VTKRayCastPlugin();

     /**
     * Destroy the VTKRayCast plugin instance.
     */
    ~VTKRayCastPlugin();

    /**
     * Return the VTK volume that renders the Geometry.
     */
    vtkProp* getVtkProp();

    /**
     * This function is called when a new data set becomes available.
     *
     * @param ds The new data set that was added.
     */
    void dataSetAdded(data::DataSet* ds);

    /**
     * This function is called when an already available data set was changed.
     *
     * @param ds The data set that has been updated.
     */
    void dataSetChanged(data::DataSet* ds);

    /**
     * This function is called when a data set that was available has been removed.
     *
     * @param ds The data set that was removed from the pool of data sets.
     */
    void dataSetRemoved(data::DataSet* ds);

    /**
     * Return the Widget that is shown in the gui
     */
    QWidget* getGUI();

protected:
    QList<data::DataSet*> compatibleDataSets;
    QList<data::DataSet*> compatibleTransferFunctions;
    QList<vtkColorTransferFunction*> transferFunctions;
    QList<vtkPiecewiseFunction*> piecewiseFunctions;
    QWidget* qWidget;
    QComboBox* qComboBoxDataSet;
    QComboBox* qComboBoxTransferFunction;
    vtkVolume* volume;

private slots:

    /**
      * Update the volume for the new dataset that is going to be visualized.
      */
    void fillVolume();

}; // class VTKRayCastPlugin
} // namespace bmia
#endif // bmia_VTKRayCastPlugin_VTKRayCastPlugin_h
