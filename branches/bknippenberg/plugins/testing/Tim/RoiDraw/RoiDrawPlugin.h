/*
 * RoiDrawPlugin.h
 *
 * 2010-11-02	Tim Peeters
 * - First version
 *
 * 2010-12-14	Evert van Aart
 * - Hotfix solution for the bug in Windows, which causes errors when
 *   switching windows. This is an improvised solution, a better 
 *   solution should be made in the near future.
 *
 * - Added support for saving ROIs. Handles cannot be saved at the moment,
 *   and a loaded ROI should be added to the drawing window (which is not
 *   the case at the moment), but this will do for now.
 *
 */

#ifndef bmia_RoiDrawPlugin_h
#define bmia_RoiDrawPlugin_h

#include <vtkMatrix4x4.h>

class vtkActor;
class vtkPolyData;
class vtkPropAssembly;

namespace Ui {
    class RoiForm;
}

#include "DTITool.h"

namespace bmia {

class RoiDrawDialog;

namespace data {
  class DataSet;
}

/**
 * This plugin can be used to manipulate (create, rename, draw, delete)
 * regions of interest (ROIs), and to specify seeding parameters from the ROIs.
 */
class RoiDrawPlugin :	public plugin::Plugin,
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
    RoiDrawPlugin();
    ~RoiDrawPlugin();
    void init();

    /**
     * Implement the plugin::GUI interface.
     */
    QWidget* getGUI();

    /**
     * Implement the plugin::Visualization interface.
     */
    vtkProp* getVtkProp();

    /**
     * Implement the data::Consumer interface.
     */
    void dataSetAdded(data::DataSet* ds);
    void dataSetChanged(data::DataSet* ds);
    void dataSetRemoved(data::DataSet* ds);

    /**
     * This function is called by RoiDrawDialog when the user has drawn a new
     * ROI and clicks the apply button. It updates the currently selected ROI
     * with the drawn one.
     */
    void updateROI(vtkPolyData* newdata, vtkMatrix4x4 * m = NULL);


protected slots:

    void selectRoi(int index);
    void draw();
    void newRoi();
    void deleteRoi();
	void saveROI();
    void rename();

    void noSeeding(bool updateDS = true);
    void distanceSeeding();
    void voxelSeeding();
    void updateSeedDistance();
    void updateSeedVoxels();

private:
    QWidget* widget;
    Ui::RoiForm* ui;

    RoiDrawDialog* dialog;

    QList<data::DataSet*> roiData;
    QList<data::DataSet*> volumeData;

    /**
     * Collection of all the actors that
     * show the ROIs in the 3D view.
     */
    vtkPropAssembly* assembly;

    /**
     * Currently selected ROI. This number corresponds
     * to an index in roiData, or is -1 if no ROI
     * is selected.
     */
    int selectedRoi;

    /**
     * Return the currently selected ROI data, or NULL if no ROI data is selected.
     */
    data::DataSet* getSelectedRoi();

    /**
     * Helper function to get the vtkProp3D of a ROI from the assembly.
     */
    vtkActor* getRoiActor(int index);

}; // class RoiDrawPlugin
} // namespace bmia
#endif // bmia_RoiDrawPlugin_h
