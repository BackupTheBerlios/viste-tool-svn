/*
 * RayCastPlugin
 *
 * 2010-03-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-01-14	Evert van Aart
 * - Made the plugin shut up about new, changed or removed data sets
 *
 * 2011-03-02	Evert van Aart
 * - Version 1.0.0.
 * - Fixed rendering for viewports that do not start at (0, 0).
 *
 * 2011-03-28	Evert van Aart
 * - Version 1.0.1.
 * - Made the volume actor non-pickable, so that it does not interfere with
 *   the Fiber Cutting plugin.
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.1.0.
 * - Redesigned the GUI, got rid of the horizontal tabs. 
 * - Removed the "bmia_" prefix.
 *
 * 2011-04-27	Evert van Aart
 * - Version 1.1.1.
 * - Properly implemented "dataSetChanged" for scalar volumes.
 *
 * 2011-07-08	Evert van Aart
 * - Version 1.2.0.
 * - Masking now works correctly for transformed volumes.
 *
 */


#ifndef bmia_RayCastPlugin_RayCastPlugin_h
#define bmia_RayCastPlugin_RayCastPlugin_h


class vtkVolume;
class vtkColorTransferFunction;
class vtkPiecewiseFunction;
class RayCastVolumeMapper;
class vtkVolumeProperty;
class QPixmap;

namespace Ui {
    class RayCastPlugin;
} // namspace Ui

#include "DTITool.h"

namespace bmia {

/**
 * Raycast plugin to visualize scalar volumes.
 */
class RayCastPlugin : public plugin::Plugin, public plugin::Visualization,
                      public data::Consumer, public plugin::GUI
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::Visualization)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::GUI)

public:

	QString getPluginVersion()
	{
		return "1.2.0";
	}

    RayCastPlugin();
    ~RayCastPlugin();

    void init();

    /**
     * Return the VTK actor that renders the cone.
     */
    vtkProp* getVtkProp();

    /**
     * Return the widget that is shown in the GUI
     */
    QWidget* getGUI();

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


protected slots:

    void comboBoxDataChanged();
    void comboBoxRenderMethodChange();
    void comboBoxTransferFunctionChange();
    void comboBoxBitMaskChange();

    void checkBoxX1Triggered(bool checked);
    void checkBoxX2Triggered(bool checked);
    void checkBoxY1Triggered(bool checked);
    void checkBoxY2Triggered(bool checked);
    void checkBoxZ1Triggered(bool checked);
    void checkBoxZ2Triggered(bool checked);
    void isoValueSliderChanged(int value);          // RALPH
    void horizontalSliderX1Changed(int value);
    void horizontalSliderX2Changed(int value);
    void horizontalSliderY1Changed(int value);
    void horizontalSliderY2Changed(int value);
    void horizontalSliderZ1Changed(int value);
    void horizontalSliderZ2Changed(int value);
    void doubleSpinBoxStepsizeChanged(double value);
    void doubleSpinBoxInteractiveStepsizeChanged(double value);
    void doubleSpinBoxIsovalueChanged(double value);
    void doubleSpinBoxIsovalueOpacityChanged(double value);
    void positionSliderXChange(int x);
    void positionSliderYChange(int y);
    void positionSliderZChange(int z);
    void rotationSliderXChange(int x);
    void rotationSliderYChange(int y);
    void rotationSliderZChange(int z);
    void scaleSliderXChange(int x);
    void scaleSliderYChange(int y);
    void scaleSliderZChange(int z);
    void scaleSpinBoxXChange(double x);
    void scaleSpinBoxYChange(double y);
    void scaleSpinBoxZChange(double z);
    void checkBoxGrayScaleToggled(bool checked);
    void doubleSpinBoxGrayScaleValueChanged(double value);
    void widgetIsoValueColorClicked();

    /**
      * Calls this->core()->render();
      */
    void render();




private:
    float volumeRotation[3];
    RayCastVolumeMapper* mapper;
    QList<data::DataSet*> compatibleDataSets;
    QList<data::DataSet*> compatibleTransferFunctions;
    QList<data::DataSet*> compatibleMasks;
    QList<vtkColorTransferFunction*> transferFunctions;
    QList<vtkPiecewiseFunction*> piecewiseFunctions;
    vtkVolume* pVolume;
    QWidget* qWidget;
    Ui::RayCastPlugin* ui;
    QPixmap* iconPixmapIsovalueColor;
    QColor lastIsoValueColor;

    /**
      *  if a dataset has a transformation function, this function retrieves it.
      */
    void calculateTransformationMatrix();

    /**
      * check if all required info is present.
      */
    bool renderCheck();

    /**
      * set all controls to their proper position
      */
    void setControls();


}; // class RayCastPlugin
} // namespace bmia
#endif // bmia_RayCastPlugin_RayCastPlugin_h
