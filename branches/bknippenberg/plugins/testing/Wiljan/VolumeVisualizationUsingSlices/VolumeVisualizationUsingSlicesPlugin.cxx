/*
 * VolumeVisualizationUsingSlicesPlugin.cxx
 *
 * 2010-02-19	Wiljan van Ravensteijn
 * - First version
 */

#include "VolumeVisualizationUsingSlicesPlugin.h"

#include <vtkVolume.h>
#include <vtkVolumeTextureMapper2D.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkSLCReader.h>
#include <QWidget>
#include <QComboBox>
#include <QLabel>
#include <QFormLayout>
#include <QDebug>

namespace bmia {

    VolumeVisualizationUsingSlicesPlugin::VolumeVisualizationUsingSlicesPlugin() : plugin::Plugin("Volume visualization plugin")
    {
        this->volume = vtkVolume::New();

        this->qWidget = new QWidget();
        QFormLayout* layout = new QFormLayout(this->qWidget);

        this->qComboBoxDataSet = new QComboBox(qWidget);
        layout->addRow("Dataset: ", this->qComboBoxDataSet);

        this->qComboBoxTransferFunction = new QComboBox(qWidget);
        layout->addRow("TransferFunction: ", this->qComboBoxTransferFunction);
        this->qWidget->setLayout(layout);

        connect(this->qComboBoxDataSet,SIGNAL(currentIndexChanged(int)),this,SLOT(fillVolume()));
        connect(this->qComboBoxTransferFunction,SIGNAL(currentIndexChanged(int)),this,SLOT(fillVolume()));




    }

    QWidget* VolumeVisualizationUsingSlicesPlugin::getGUI()
    {
        return this->qWidget;
    }

    VolumeVisualizationUsingSlicesPlugin::~VolumeVisualizationUsingSlicesPlugin()
    {
        this->volume->Delete(); this->volume = NULL;
    }

    vtkProp* VolumeVisualizationUsingSlicesPlugin::getVtkProp()
    {
        return this->volume;

    }

    void VolumeVisualizationUsingSlicesPlugin::dataSetAdded(data::DataSet* ds)
    {
        pxCore->out()->logMessage("Received notification of the addition of " + ds->getName() + " of the type " + ds->getKind());
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

    void VolumeVisualizationUsingSlicesPlugin::dataSetChanged(data::DataSet* ds)
    {
        pxCore->out()->logMessage("Received notification of the modification of " + ds->getName());
        vtkObject* cpf;
        if (ds->getKind() == "scalar volume")
        {
        }
        else if (ds->getKind() == "transfer function")
        {
            int index = this->compatibleTransferFunctions.indexOf(ds);
            if (index != -1)
            {
                this->compatibleTransferFunctions.removeAt(index);
                this->transferFunctions.removeAt(index);
                this->piecewiseFunctions.removeAt(index);
                this->qComboBoxTransferFunction->removeItem(index);
            }
            dataSetAdded(ds);
        }
        this->volume->Update();
    }

    void VolumeVisualizationUsingSlicesPlugin::dataSetRemoved(data::DataSet* ds)
    {
        pxCore->out()->logMessage("Received notification of the removal of " + ds->getName());

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

    void VolumeVisualizationUsingSlicesPlugin::fillVolume()
    {
        int volumeIndex;
        int transferFunctionIndex;
        data::DataSet* ds;
        vtkVolumeTextureMapper2D* mapper;
        vtkVolumeProperty* property;



        volumeIndex = this->qComboBoxDataSet->currentIndex();
        transferFunctionIndex = this->qComboBoxTransferFunction->currentIndex();

        if (volumeIndex == -1)
        {
            this->volume->SetReferenceCount(0);
            this->volume->Delete();
            this->volume = vtkVolume::New();
            pxCore->gui()->vtkRender();
            return;
        }
        if (transferFunctionIndex == -1)
            return;

        ds = this->compatibleDataSets.at(volumeIndex);
        pxCore->out()->logMessage("Switching to " + ds->getName());

        mapper = vtkVolumeTextureMapper2D::New();
        mapper->SetInput(ds->getVtkImageData());
        property = vtkVolumeProperty::New();
        property->SetColor(this->transferFunctions.at(transferFunctionIndex));
        property->SetScalarOpacity(this->piecewiseFunctions.at(transferFunctionIndex));
        this->volume->SetMapper(mapper);
        this->volume->SetProperty(property);

        mapper->Delete();
        property->Delete();
        pxCore->gui()->vtkRender();
    }




} // namespace bmia
Q_EXPORT_PLUGIN2(libbmia_VolumeVisualizationUsingSlicesPlugin, bmia::VolumeVisualizationUsingSlicesPlugin)
