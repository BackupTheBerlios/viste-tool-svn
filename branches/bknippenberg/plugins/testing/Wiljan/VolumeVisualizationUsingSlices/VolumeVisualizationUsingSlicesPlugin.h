/*
 * VolumeVisualizationUsingSlicesPlugin
 *
 * 2010-02-19	Wiljan van Ravensteijn
 * - First version
 */

#ifndef bmia_VolumeVisualizationUsingSlices_VolumeVisualizationUsingSlicesPlugin_h
#define bmia_VolumeVisualizationUsingSlices_VolumeVisualizationUsingSlicesPlugin_h

class vtkVolume;
class vtkColorTransferFunction;
class vtkPiecewiseFunction;
class QWidget;
class QComboBox;

#include "DTITool.h"

namespace bmia {

/**
 * A simple visualization plugin that does consume volume data
 * and renders it using slices.
 */
class VolumeVisualizationUsingSlicesPlugin : public plugin::Plugin, public plugin::Visualization, public data::Consumer, public plugin::GUI
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::Visualization)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::GUI)

public:
    VolumeVisualizationUsingSlicesPlugin();
    ~VolumeVisualizationUsingSlicesPlugin();

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

    void fillVolume();

}; // class VolumeVisualizationUsingSlicesPlugin
} // namespace bmia
#endif // bmia_GeometryVisualization_VolumeVisualizationUsingSlicesPlugin_h
