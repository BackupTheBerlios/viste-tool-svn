#ifndef bmia_DistanceMeasures_h
#define bmia_DistanceMeasures_h

/** Includes - Main Header */

#include "DTITool.h"
//#include "../src/smeesters/plugins/stable/PlanesVisualization/PlanesVisPlugin.h"

/** Includes - Qt */

#include "ui_DistanceMeasures.h"

/** Includes - VTK */

#include <vtkPropAssembly.h>
#include <vtkProp3D.h>
#include <vtkActor.h>
#include <vtkActor2D.h>
#include <vtkPolyData.h>
#include <vtkStringArray.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkPointSetToLabelHierarchy.h>
#include <vtkLabelPlacementMapper.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>

namespace Ui
{
    class DistanceMeasuresForm;
}

namespace bmia
{

/** Generic vector3 struct */
typedef struct
{
    double x, y, z;

} Vec3;

/** Holding pointdata and location of the most anterior point of a fiber */
typedef struct
{
    QList< Vec3* > data;
    QList< double > scalarData;
    int anteriorPointIndex;
    double userPointRefinement;

} FiberData;

/** Holding fiberdata */
typedef struct
{
    data::DataSet* ds;
    QList<FiberData*> selectedLines;
    int userSelectedLine;

} SortedFibers;

class DistanceMeasures :  public plugin::AdvancedPlugin,
                    public plugin::Visualization,
                    public plugin::GUI,
                    public data::Consumer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::AdvancedPlugin)
    Q_INTERFACES(bmia::plugin::GUI)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::Visualization)

public:

    QString getPluginVersion()
    {
        return "1.0.0";
    }

    DistanceMeasures();
    ~DistanceMeasures();

    virtual void init();

    /**
     * Return the VTK actor that renders the cone.
     */
    vtkProp * getVtkProp();

    /**
     * Return the widget that is shown in the GUI
     */
    QWidget * getGUI();

    /**
     * This function is called when a new data set becomes available.
     *
     * @param ds The new data set that was added.
     */
    void dataSetAdded(data::DataSet * d);

    /**
     * This function is called when an already available data set was changed.
     *
     * @param ds The data set that has been updated.
     */
    void dataSetChanged(data::DataSet * d);

    /**
     * This function is called when a data set that was available has been removed.
     *
     * @param ds The data set that was removed from the pool of data sets.
     */
    void dataSetRemoved(data::DataSet * d);

protected slots:

    void buttonSetPointAClicked();
    void buttonSetPointBClicked();

private:

    /** If plugin inherits from plugin::GUI */
    QWidget * widget;

    /** QT form */
    Ui::DistanceMeasuresForm * form;

    /** The collection of all the actors that this plugin can render.
    		This is the object that will be returned by getVtkProp().  */
    vtkPropAssembly * assembly;

    //
    //  Qt communication
    //

    /** Connect GUI controls to their respective "SLOT" functions. */
    void connectAll();

    /** Disconnect all GUI controls */
    void disconnectAll();

    //
    //  Plugin base
    //

	/** Settings dataset */
	data::DataSet * settings;

    /** Fiber information structs */
    QList<SortedFibers*> sortedFibersList;

    /** Find a fiber data struct in the sortedFibersList **/
    int FindInputDataSet(data::DataSet * ds);

    /** Generate text labels **/
    vtkActor2D* GenerateLabels(vtkSmartPointer<vtkPoints> points, vtkSmartPointer<vtkStringArray> labels);

    /** Update posititon coordinates **/
    void UpdateCoordinates();
    int slicePosX, slicePosY, slicePosZ;

    void setMeasuredPoint(int id);
    void calculateDistance();
};

}

#endif  // bmia_DistanceMeasures_h
