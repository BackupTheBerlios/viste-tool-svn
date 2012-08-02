/*
 * VTKRayCastPlugin.cxx
 *
 * 2010-03-10	Wiljan van Ravensteijn
 * - First version
 */

#include "VTKRayCastPlugin.h"

#include <vtkVolume.h>
#include <vtkGPUVolumeRayCastMapper.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkSLCReader.h>
#include <vtkImageData.h>
#include <QWidget>
#include <QComboBox>
#include <QLabel>
#include <QFormLayout>
#include <QDebug>
#include <vtkPlane.h>

namespace bmia {

    VTKRayCastPlugin::VTKRayCastPlugin() : plugin::Plugin("Volume visualization plugin")
    {
        QFormLayout* layout;


        this->volume = vtkVolume::New();
        this->qWidget = new QWidget();
        layout = new QFormLayout(this->qWidget);

        this->qComboBoxDataSet = new QComboBox(qWidget);
        layout->addRow("Dataset: ", this->qComboBoxDataSet);

        this->qComboBoxTransferFunction = new QComboBox(qWidget);
        layout->addRow("TransferFunction: ", this->qComboBoxTransferFunction);

        this->qWidget->setLayout(layout);

        connect(this->qComboBoxDataSet,SIGNAL(currentIndexChanged(int)),this,SLOT(fillVolume()));
        connect(this->qComboBoxTransferFunction,SIGNAL(currentIndexChanged(int)),this,SLOT(fillVolume()));

    }

    QWidget* VTKRayCastPlugin::getGUI()
    {
        return this->qWidget;
    }

    VTKRayCastPlugin::~VTKRayCastPlugin()
    {
        this->volume->Delete(); this->volume = NULL;
    }

    vtkProp* VTKRayCastPlugin::getVtkProp()
    {
        return this->volume;

    }

    void VTKRayCastPlugin::dataSetAdded(data::DataSet* ds)
    {
        this->core()->out()->logMessage("Received notification of the addition of " + ds->getName() + " of the type " + ds->getKind());
        vtkObject* cpf;
        if (ds->getKind() == "scalar volume")
        {
            this->compatibleDataSets.append(ds);
            this->qComboBoxDataSet->addItem(ds->getName());
        }
        else if ( (ds->getKind() == "transfer function")
            and ( ds->getAttributes()->getAttribute("piecewise function", cpf ) ) )
            {
            this->compatibleTransferFunctions.append(ds);
            this->transferFunctions.append(vtkColorTransferFunction::SafeDownCast(ds->getVtkObject()));
            this->piecewiseFunctions.append(vtkPiecewiseFunction::SafeDownCast(cpf));
            this->qComboBoxTransferFunction->addItem(ds->getName());
        }
    }

    void VTKRayCastPlugin::dataSetChanged(data::DataSet* ds)
    {
        this->core()->out()->logMessage("Received notification of the modification of " + ds->getName());
        vtkObject* cpf;
        if (ds->getKind() == "scalar volume")
        {
        }
        else if (ds->getKind() == "transfer function")
        {
            int index = this->compatibleTransferFunctions.indexOf(ds);
            if (index == -1)
            {
                dataSetAdded(ds);
            }
        }
        this->volume->Update();
        this->core()->render();
    }

    void VTKRayCastPlugin::dataSetRemoved(data::DataSet* ds)
    {
        this->core()->out()->logMessage("Received notification of the removal of " + ds->getName());

        if (ds->getKind() == "scalar volume")
        {
            int index = this->compatibleDataSets.indexOf(ds);
            qDebug() << index;
            if (index == -1)
                return

                        this->compatibleDataSets.removeAt(index);
            this->qComboBoxDataSet->removeItem(index);
        }
        else if (ds->getKind() == "transfer function")
        {
            int index = this->compatibleTransferFunctions.indexOf(ds);
            if (index == -1)
                return
                        this->compatibleDataSets.removeAt(index);
            this->transferFunctions.removeAt(index);
            this->piecewiseFunctions.removeAt(index);
            this->qComboBoxTransferFunction->removeItem(index);
        }
    }

    void VTKRayCastPlugin::fillVolume()
    {
        int volumeIndex;
        int transferFunctionIndex;
        data::DataSet* ds;
        vtkGPUVolumeRayCastMapper* mapper;
        vtkVolumeProperty* property;

        volumeIndex = this->qComboBoxDataSet->currentIndex();
        transferFunctionIndex = this->qComboBoxTransferFunction->currentIndex();

        if (volumeIndex == -1)
        {
            this->volume->SetReferenceCount(0);
            this->volume->Delete();
            this->volume = vtkVolume::New();
            this->core()->render();
            return;
        }
        if (transferFunctionIndex == -1)
            return;

        ds = this->compatibleDataSets.at(volumeIndex);
        this->core()->out()->logMessage("Switching to " + ds->getName());
        mapper = vtkGPUVolumeRayCastMapper::New();

        vtkPlane* plane = vtkPlane::New();
        plane->SetNormal(1,0,0);
        plane->SetOrigin(ds->getVtkImageData()->GetCenter());
        mapper->AddClippingPlane(plane);
        mapper->SetBlendModeToComposite();
        mapper->SetInput(ds->getVtkImageData());
        property = vtkVolumeProperty::New();
        property->SetColor(this->transferFunctions.at(transferFunctionIndex));
        property->SetScalarOpacity(this->piecewiseFunctions.at(transferFunctionIndex));
        property->ShadeOn();
        property->SetAmbient(0.6);
        property->SetInterpolationTypeToLinear();
        this->volume->SetMapper(mapper);
        this->volume->SetProperty(property);

        mapper->Delete();
        property->Delete();
        this->core()->render();
    }




} // namespace bmia
Q_EXPORT_PLUGIN2(libbmia_VTKRayCastPlugin, bmia::VTKRayCastPlugin)
