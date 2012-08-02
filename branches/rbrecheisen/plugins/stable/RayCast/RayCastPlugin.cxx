/*
 * RayCastPlugin.cxx
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

#include "RayCastPlugin.h"
#include "ui_RayCastPlugin.h"
#include <QIcon>
#include <QPainter>
#include <QColorDialog>
#include <vtkVolume.h>
#include <vtkObject.h>
#include <vtkPointData.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include "RayCastVolumeMapper.h"
#include <vtkPlane.h>
#include <QDoubleSpinBox>
#include "vtkClippingPlane.h"
#include <vtkMatrix4x4.h>
#include <vtkLinearTransform.h>
#include <vtkTransform.h>
#include <vtkImageMask.h>
#include <QMessageBox>
#include "vtkImageMask2.h"


namespace bmia {

    RayCastPlugin::RayCastPlugin() : plugin::Plugin("Ray Cast Plugin")
    {
        this->qWidget = new QWidget();
        this->ui = new Ui::RayCastPlugin();
        this->ui->setupUi(this->qWidget);
        connect(this->ui->comboBoxDataset,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxDataChanged()));
        connect(this->ui->comboBoxTransferFunction,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxTransferFunctionChange()));
        connect(this->ui->comboBoxBitMask,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxBitMaskChange()));
        connect(this->ui->comboBoxRendermode,SIGNAL(currentIndexChanged(int)),this,SLOT(comboBoxRenderMethodChange()));
        connect(this->ui->checkBoxInvertBitmask,SIGNAL(toggled(bool)),this,SLOT(comboBoxBitMaskChange()));
        connect(this->ui->checkBoxX1,SIGNAL(toggled(bool)),this,SLOT(checkBoxX1Triggered(bool)));
        connect(this->ui->checkBoxX2,SIGNAL(toggled(bool)),this,SLOT(checkBoxX2Triggered(bool)));
        connect(this->ui->checkBoxY1,SIGNAL(toggled(bool)),this,SLOT(checkBoxY1Triggered(bool)));
        connect(this->ui->checkBoxY2,SIGNAL(toggled(bool)),this,SLOT(checkBoxY2Triggered(bool)));
        connect(this->ui->checkBoxZ1,SIGNAL(toggled(bool)),this,SLOT(checkBoxZ1Triggered(bool)));
        connect(this->ui->checkBoxZ2,SIGNAL(toggled(bool)),this,SLOT(checkBoxZ2Triggered(bool)));
        connect(this->ui->isoValueSlider,SIGNAL(valueChanged(int)),this,SLOT(isoValueSliderChanged(int)));
        connect(this->ui->horizontalSliderX1,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderX1Changed(int)));
        connect(this->ui->horizontalSliderX2,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderX2Changed(int)));
        connect(this->ui->horizontalSliderY1,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderY1Changed(int)));
        connect(this->ui->horizontalSliderY2,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderY2Changed(int)));
        connect(this->ui->horizontalSliderZ1,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderZ1Changed(int)));
        connect(this->ui->horizontalSliderZ2,SIGNAL(valueChanged(int)),this,SLOT(horizontalSliderZ2Changed(int)));
        connect(this->ui->doubleSpinBoxStepsize,SIGNAL(valueChanged(double)),this,SLOT(doubleSpinBoxStepsizeChanged(double)));
        connect(this->ui->doubleSpinBoxInteractiveStepsize,SIGNAL(valueChanged(double)),this,SLOT(doubleSpinBoxInteractiveStepsizeChanged(double)));
        connect(this->ui->doubleSpinBoxIsovalue,SIGNAL(valueChanged(double)),this,SLOT(doubleSpinBoxIsovalueChanged(double)));
        connect(this->ui->doubleSpinBoxIsovalueOpacity,SIGNAL(valueChanged(double)),this,SLOT(doubleSpinBoxIsovalueOpacityChanged(double)));

        connect(this->ui->horizontalSliderXPos,SIGNAL(valueChanged(int)),this,SLOT(positionSliderXChange(int)));
        connect(this->ui->horizontalSliderYPos,SIGNAL(valueChanged(int)),this,SLOT(positionSliderYChange(int)));
        connect(this->ui->horizontalSliderZPos,SIGNAL(valueChanged(int)),this,SLOT(positionSliderZChange(int)));

        connect(this->ui->horizontalSliderRotateX,SIGNAL(valueChanged(int)),this,SLOT(rotationSliderXChange(int)));
        connect(this->ui->horizontalSliderRotateY,SIGNAL(valueChanged(int)),this,SLOT(rotationSliderYChange(int)));
        connect(this->ui->horizontalSliderRotateZ,SIGNAL(valueChanged(int)),this,SLOT(rotationSliderZChange(int)));

        connect(this->ui->horizontalSliderScaleX, SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderXChange(int)));
        connect(this->ui->horizontalSliderScaleY, SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderYChange(int)));
        connect(this->ui->horizontalSliderScaleZ, SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderZChange(int)));

        connect(this->ui->checkBoxGrayScale,SIGNAL(toggled(bool)),this,SLOT(checkBoxGrayScaleToggled(bool)));
        connect(this->ui->doubleSpinBoxMinValue,SIGNAL(valueChanged(double)),this,SLOT(doubleSpinBoxGrayScaleValueChanged(double)));
        connect(this->ui->doubleSpinBoxMaxValue,SIGNAL(valueChanged(double)),this,SLOT(doubleSpinBoxGrayScaleValueChanged(double)));

        connect(this->ui->spinBoxScaleX,SIGNAL(valueChanged(double)),this,SLOT(scaleSpinBoxXChange(double)));
        connect(this->ui->spinBoxScaleY,SIGNAL(valueChanged(double)),this,SLOT(scaleSpinBoxYChange(double)));
        connect(this->ui->spinBoxScaleZ,SIGNAL(valueChanged(double)),this,SLOT(scaleSpinBoxZChange(double)));

        connect(this->ui->pushButtonColor,SIGNAL(clicked()),this,SLOT(widgetIsoValueColorClicked()));
        connect(this->ui->checkBoxUseToonShading,SIGNAL(toggled(bool)),this,SLOT(comboBoxRenderMethodChange()));

        volumeRotation[0] = 0;
        volumeRotation[1] = 0;
        volumeRotation[2] = 0;

        lastIsoValueColor = Qt::black;
        iconPixmapIsovalueColor = new QPixmap(15,15);
        QPainter p(iconPixmapIsovalueColor);
		p.fillRect(iconPixmapIsovalueColor->rect(),Qt::gray);
        this->ui->pushButtonColor->setIcon(QIcon(*iconPixmapIsovalueColor));
    }

    void RayCastPlugin::init()
    {
        this->pVolume = vtkVolume::New();
        this->pVolume->VisibilityOff();
		this->pVolume->SetPickable(0);

        this->mapper = RayCastVolumeMapper::New();
        connect(this->mapper,SIGNAL(render()),this,SLOT(render()));
        if (!mapper->extensionsSupported())
            this->core()->out()->logMessage("The required gpu extensions are not supported!");

        this->mapper->setStepsize(this->ui->doubleSpinBoxStepsize->value());
        this->mapper->setInteractiveStepSize(this->ui->doubleSpinBoxInteractiveStepsize->value());
        this->mapper->setRenderMethod(MIP);
        this->pVolume->SetMapper(mapper);
    }

    RayCastPlugin::~RayCastPlugin()
    {
        delete this->qWidget; this->qWidget = NULL;
        this->pVolume->Delete(); this->pVolume = NULL;
        this->mapper->Delete(); this->mapper = NULL;
        delete this->iconPixmapIsovalueColor;
    }

    vtkProp* RayCastPlugin::getVtkProp()
    {
        return this->pVolume;
    }

    QWidget* RayCastPlugin::getGUI()
    {
        return this->qWidget;
    }

    void RayCastPlugin::dataSetAdded(data::DataSet* ds)
    {
//        this->core()->out()->logMessage("Received notification of the addition of " + ds->getName() + " of the type " + ds->getKind());
        vtkObject* cpf;
        if (ds->getKind() == "scalar volume")
        {
            this->compatibleDataSets.append(ds);
            this->ui->comboBoxDataset->addItem(ds->getName());
            if ( QString(ds->getVtkImageData()->GetScalarTypeAsString()) == "unsigned char"
                /* || QString(ds->getVtkImageData()->GetScalarTypeAsString()) == "unsigned short"*/)
            {
                this->ui->comboBoxBitMask->addItem(ds->getName());
                this->compatibleMasks.append(ds);    
            }
        }
        else if ( (ds->getKind() == "transfer function")
            && ( ds->getAttributes()->getAttribute("piecewise function", cpf ) ) )
            {
            this->compatibleTransferFunctions.append(ds);
            this->transferFunctions.append(vtkColorTransferFunction::SafeDownCast(ds->getVtkObject()));
            this->piecewiseFunctions.append(vtkPiecewiseFunction::SafeDownCast(cpf));
            this->ui->comboBoxTransferFunction->addItem(ds->getName());
        }

		this->core()->render();
    }
    void RayCastPlugin::dataSetChanged(data::DataSet* ds)
    {
//        this->core()->out()->logMessage("Received notification of the modification of " + ds->getName());
        if (ds->getKind() == "scalar volume")
        {
			if (this->compatibleDataSets.indexOf(ds) == this->ui->comboBoxDataset->currentIndex() - 1)
			{
				mapper->volumeChanged();
				mapper->reInitClippingPlanes();
				this->renderCheck();
			}
        }
        else if (ds->getKind() == "transfer function")
        {
            int index = this->compatibleTransferFunctions.indexOf(ds);
            if (index != -1)
            {
                this->mapper->transferFunctionChanged();
                this->renderCheck();
            }
            else
            {
                this->dataSetAdded(ds);
            }
        }
    }

    void RayCastPlugin::dataSetRemoved(data::DataSet* ds)
    {
//        this->core()->out()->logMessage("Received notification of the removal of " + ds->getName());

        if (ds->getKind() == "scalar volume")
        {
            int index = this->compatibleDataSets.indexOf(ds);
      
			if (index != -1)
			{
				this->compatibleDataSets.removeAt(index);
				this->ui->comboBoxDataset->removeItem(index + 1);
			}

			int bitMaskIndex = this->compatibleMasks.indexOf(ds);

			if (bitMaskIndex != -1)
			{
				this->compatibleMasks.removeAt(bitMaskIndex);
				this->ui->comboBoxBitMask->removeItem(index + 1);
			}
        }
        else if (ds->getKind() == "transfer function")
        {
            int index = this->compatibleTransferFunctions.indexOf(ds);
            if (index == -1)
                return;

            this->compatibleDataSets.removeAt(index);
            this->transferFunctions.removeAt(index);
            this->piecewiseFunctions.removeAt(index);
            this->ui->comboBoxTransferFunction->removeItem(index + 1);
        }
    }

    void RayCastPlugin::checkBoxX1Triggered(bool checked)
    {
        this->mapper->setEnableClippingPlane(0,checked);
        this->ui->horizontalSliderX1->setEnabled(checked);
        this->core()->render();
    }

    void RayCastPlugin::checkBoxX2Triggered(bool checked)
    {
        this->mapper->setEnableClippingPlane(1,checked);
        this->ui->horizontalSliderX2->setEnabled(checked);
        this->core()->render();
    }

    void RayCastPlugin::checkBoxY1Triggered(bool checked)
    {
        this->mapper->setEnableClippingPlane(2,checked);
        this->ui->horizontalSliderY1->setEnabled(checked);
        this->core()->render();
    }

    void RayCastPlugin::checkBoxY2Triggered(bool checked)
    {
        this->mapper->setEnableClippingPlane(3,checked);
        this->ui->horizontalSliderY2->setEnabled(checked);
        this->core()->render();
    }

    void RayCastPlugin::checkBoxZ1Triggered(bool checked)
    {
        this->mapper->setEnableClippingPlane(4,checked);
        this->ui->horizontalSliderZ1->setEnabled(checked);
        this->core()->render();
    }

    void RayCastPlugin::checkBoxZ2Triggered(bool checked)
    {
        this->mapper->setEnableClippingPlane(5,checked);
        this->ui->horizontalSliderZ2->setEnabled(checked);
        this->core()->render();
    }

    void RayCastPlugin::isoValueSliderChanged(int value)
    {
        this->mapper->setIsoValue((float) value);
        this->ui->doubleSpinBoxIsovalue->blockSignals(true);
        this->ui->doubleSpinBoxIsovalue->setValue((double) value);
        this->ui->doubleSpinBoxIsovalue->blockSignals(false);
        this->core()->render();
    }

    void RayCastPlugin::horizontalSliderX1Changed(int value)
    {
        this->mapper->getPlane(0)->SetOrigin(value,0,0);
        this->core()->render();
    }

    void RayCastPlugin::horizontalSliderX2Changed(int value)
    {
        this->mapper->getPlane(1)->SetOrigin(value,0,0);
        this->core()->render();
    }
    void RayCastPlugin::horizontalSliderY1Changed(int value)
    {
        this->mapper->getPlane(2)->SetOrigin(0,value,0);
        this->core()->render();
    }

    void RayCastPlugin::horizontalSliderY2Changed(int value)
    {
        this->mapper->getPlane(3)->SetOrigin(0,value,0);
        this->core()->render();
    }

    void RayCastPlugin::horizontalSliderZ1Changed(int value)
    {
        this->mapper->getPlane(4)->SetOrigin(0,0,value);
        this->core()->render();
    }

    void RayCastPlugin::horizontalSliderZ2Changed(int value)
    {
        this->mapper->getPlane(5)->SetOrigin(0,0,value);
        this->core()->render();
    }

    void RayCastPlugin::render()
    {
        this->core()->render();
    }

    void RayCastPlugin::doubleSpinBoxStepsizeChanged(double value)
    {
        this->mapper->setStepsize(value);
    }

    void RayCastPlugin::doubleSpinBoxInteractiveStepsizeChanged(double value)
    {
        this->mapper->setInteractiveStepSize(value);
    }

    void RayCastPlugin::doubleSpinBoxIsovalueChanged(double value)
    {
        this->mapper->setIsoValue(value);
        this->ui->isoValueSlider->blockSignals(true);
        this->ui->isoValueSlider->setValue((int) (value < 1 ? 1 : value));
        this->ui->isoValueSlider->blockSignals(false);
        this->core()->render();
    }

    void RayCastPlugin::doubleSpinBoxIsovalueOpacityChanged(double value)
    {
        this->mapper->setIsoValueOpacity(value);
        this->core()->render();
    }

    void RayCastPlugin::positionSliderXChange(int x)
    {
        if (this->pVolume != NULL)
        {
            double position[3];
            this->pVolume->GetPosition(position);
            position[0] = x;
            this->pVolume->SetPosition(position);
            this->core()->render();
        }
    }

    void RayCastPlugin::positionSliderYChange(int y)
    {
        if (this->pVolume != NULL)
        {
            double position[3];
            this->pVolume->GetPosition(position);
            position[1] = y;
            this->pVolume->SetPosition(position);
            this->core()->render();
        }
    }

    void RayCastPlugin::positionSliderZChange(int z)
    {
        if (this->pVolume != NULL)
        {
            double position[3];
            this->pVolume->GetPosition(position);
            position[2] = z;
            this->pVolume->SetPosition(position);
            this->core()->render();
        }
    }

    void RayCastPlugin::rotationSliderXChange(int x)
    {
        float diffX = x - this->volumeRotation[0];
        this->pVolume->RotateX(diffX);
        this->volumeRotation[0] += diffX;
        this->core()->render();
    }

    void RayCastPlugin::rotationSliderYChange(int y)
    {
        float diffY = y - this->volumeRotation[1];
        this->pVolume->RotateY(diffY);
        this->volumeRotation[1] += diffY;
        this->core()->render();
    }

    void RayCastPlugin::rotationSliderZChange(int z)
    {
        float diffZ = z - this->volumeRotation[2];
        this->pVolume->RotateZ(diffZ);
        this->volumeRotation[2] += diffZ;
        this->core()->render();
    }

    void RayCastPlugin::scaleSliderXChange(int x)
    {
        float realScale;
        if ( x > 0)
        {
            realScale = 1 + ((float)x / 10.0f);
        }
        else
        {
            realScale = (((float)100+x) / 100);
        }

        this->ui->horizontalSliderScaleX->disconnect();
        this->ui->spinBoxScaleX->setValue(realScale);
        if (this->ui->checkBoxUniformScaling->isChecked())
        {
            this->pVolume->SetScale(realScale,realScale,realScale);
            this->ui->horizontalSliderScaleY->setValue(x);
            this->ui->horizontalSliderScaleZ->setValue(x);
            this->ui->spinBoxScaleY->setValue(realScale);
            this->ui->spinBoxScaleZ->setValue(realScale);
        }
        else
        {
            double scale[3];
            this->pVolume->GetScale(scale);
            scale[0] = realScale;
            this->pVolume->SetScale(scale);
        }
        connect(this->ui->horizontalSliderScaleX,SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderXChange(int)));
        this->core()->render();
    }

    void RayCastPlugin::scaleSliderYChange(int y)
    {
        float realScale;
        if ( y > 0)
        {
            realScale = 1 + ((float)y / 10.0f);
        }
        else
        {
            realScale = (((float)100+y) / 100);
        }

        this->ui->horizontalSliderScaleY->disconnect();
        this->ui->spinBoxScaleY->setValue(realScale);
        connect(this->ui->horizontalSliderScaleY,SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderYChange(int)));
        if (this->ui->checkBoxUniformScaling->isChecked())
        {
            this->pVolume->SetScale(realScale,realScale,realScale);
            this->ui->horizontalSliderScaleX->setValue(y);
            this->ui->horizontalSliderScaleZ->setValue(y);
            this->ui->spinBoxScaleX->setValue(realScale);
            this->ui->spinBoxScaleZ->setValue(realScale);
        }
        else
        {
            double scale[3];
            this->pVolume->GetScale(scale);
            scale[1] = realScale;
            this->pVolume->SetScale(scale);
        }
        this->core()->render();
    }

    void RayCastPlugin::scaleSliderZChange(int z)
    {
        float realScale;
        if ( z > 0)
        {
            realScale = 1 + ((float)z / 10.0f);
        }
        else
        {
            realScale = (((float)100+z) / 100);
        }

        this->ui->horizontalSliderScaleZ->disconnect();
        this->ui->spinBoxScaleZ->setValue(realScale);
        connect(this->ui->horizontalSliderScaleZ,SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderZChange(int)));
        if (this->ui->checkBoxUniformScaling->isChecked())
        {
            this->pVolume->SetScale(realScale,realScale,realScale);
            this->ui->horizontalSliderScaleX->setValue(z);
            this->ui->horizontalSliderScaleY->setValue(z);
            this->ui->spinBoxScaleX->setValue(realScale);
            this->ui->spinBoxScaleY->setValue(realScale);
        }
        else
        {
            double scale[3];
            this->pVolume->GetScale(scale);
            scale[2] = realScale;
            this->pVolume->SetScale(scale);
        }
        this->core()->render();
    }

    void RayCastPlugin::scaleSpinBoxXChange(double x)
    {
        int sliderPos;
        if ( x < 1 )
        {
            sliderPos = x * 100 - 100;
        }
        else
        {
            sliderPos = (x - 1) * 10;
        }
        this->ui->horizontalSliderScaleX->setValue(sliderPos);
        if (this->ui->checkBoxUniformScaling->isChecked())
        {
            this->ui->horizontalSliderScaleY->setValue(sliderPos);
            this->ui->horizontalSliderScaleZ->setValue(sliderPos);
            this->ui->spinBoxScaleY->setValue(x);
            this->ui->spinBoxScaleZ->setValue(x);
        }
        double scale[3];
        this->pVolume->GetScale(scale);
        scale[0] = x;
        this->pVolume->SetScale(scale);
		this->core()->render();
  }

    void RayCastPlugin::scaleSpinBoxYChange(double y)
    {
        int sliderPos;
        if ( y < 1 )
        {
            sliderPos = y * 100 - 100;
        }
        else
        {
            sliderPos = (y - 1) * 10;
        }
        this->ui->horizontalSliderScaleY->setValue(sliderPos);
        if (this->ui->checkBoxUniformScaling->isChecked())
        {
            this->ui->horizontalSliderScaleX->setValue(sliderPos);
            this->ui->horizontalSliderScaleZ->setValue(sliderPos);
            this->ui->spinBoxScaleX->setValue(y);
            this->ui->spinBoxScaleZ->setValue(y);
        }
        double scale[3];
        this->pVolume->GetScale(scale);
        scale[1] = y;
        this->pVolume->SetScale(scale);
		this->core()->render();
   }

    void RayCastPlugin::scaleSpinBoxZChange(double z)
    {
        int sliderPos;
        if ( z < 1 )
        {
            sliderPos = z * 100 - 100;
        }
        else
        {
            sliderPos = (z - 1) * 10;
        }
        this->ui->horizontalSliderScaleZ->setValue(sliderPos);
        if (this->ui->checkBoxUniformScaling->isChecked())
        {
            this->ui->horizontalSliderScaleX->setValue(sliderPos);
            this->ui->horizontalSliderScaleY->setValue(sliderPos);
            this->ui->spinBoxScaleX->setValue(z);
            this->ui->spinBoxScaleY->setValue(z);
        }
        double scale[3];
        this->pVolume->GetScale(scale);
        scale[2] = z;
        this->pVolume->SetScale(scale);
		this->core()->render();
   }


    void RayCastPlugin::checkBoxGrayScaleToggled(bool checked)
    {
        this->ui->doubleSpinBoxMinValue->setEnabled(checked);
        this->ui->doubleSpinBoxMaxValue->setEnabled(checked);
        this->mapper->setUseGrayScaleValues(checked);
        this->core()->render();
    }

    void RayCastPlugin::doubleSpinBoxGrayScaleValueChanged(double value)
    {
        float minValue = this->ui->doubleSpinBoxMinValue->value();
        float maxValue = this->ui->doubleSpinBoxMaxValue->value();
        this->mapper->setClippingPlanesThreshold(minValue,maxValue);
        this->core()->render();
    }

    void RayCastPlugin::calculateTransformationMatrix()
    {
        this->pVolume->SetOrientation(0,0,0);
        pVolume->SetPosition(0,0,0);
        pVolume->SetScale(1,1,1);
        pVolume->SetOrigin(0,0,0);

		// Use the identity matrix by default
		vtkMatrix4x4 * id = vtkMatrix4x4::New();
		id->Identity();
		this->mapper->setExternalTransformationMatrix(id);
		this->pVolume->PokeMatrix(id);

        // check if matrix present

        data::DataSet* ds;
        int volumeIndex = this->ui->comboBoxDataset->currentIndex();
        volumeIndex -= 1;

        if (volumeIndex < 0)
            return;

        ds = this->compatibleDataSets.at(volumeIndex);
        vtkObject* tfm;
        if (ds->getAttributes()->getAttribute("transformation matrix", tfm ))
        {
            vtkMatrix4x4* transformationMatrix = vtkMatrix4x4::SafeDownCast(tfm);
            if (transformationMatrix == 0)
            {
                this->core()->out()->logMessage("not a valid transformation matrix");
                return;
            }

			// Add the transformation matrix to the prop
			this->pVolume->PokeMatrix(transformationMatrix);
        }
        else
        {
            double* center;
            center = this->pVolume->GetCenter();
            this->pVolume->SetOrigin(center[0],center[1],center[2]);
        }

    }

    void RayCastPlugin::comboBoxDataChanged()
    {
        int volumeIndex;
        data::DataSet* ds;
        volumeIndex = this->ui->comboBoxDataset->currentIndex();
        volumeIndex -= 1;

        if (volumeIndex < 0)
        {
            mapper->SetInput((vtkImageData*)0);
        }
        else
        {
            ds = this->compatibleDataSets.at(volumeIndex);

			if (ds->getVtkImageData()->GetActualMemorySize() == 0)
			{
				ds->getVtkImageData()->Update();
				this->core()->data()->dataSetChanged(ds);
			}

            this->core()->out()->logMessage("Switching to " + ds->getName() + " of type " + QString(ds->getVtkImageData()->GetScalarTypeAsString()));
            this->mapper->SetInput(ds->getVtkImageData());
            this->setControls();
			this->calculateTransformationMatrix();
            mapper->volumeChanged();
            mapper->reInitClippingPlanes();
        }
		renderCheck();
		this->core()->render();
    }

    void RayCastPlugin::comboBoxRenderMethodChange()
    {
        switch(this->ui->comboBoxRendermode->currentIndex())
        {
        case 0:

			// Check if we've got a transfer function available
			if (this->transferFunctions.isEmpty())
			{
				// If not, inform the user...
				QMessageBox::warning(this->getGUI(), "DVR", "Cannot switch to DVR, since there are no suitable transfer functions!", QMessageBox::Ok, QMessageBox::Ok);

				// ...and switch back to MIP rendering
				this->ui->comboBoxRendermode->setCurrentIndex(1);
				this->mapper->setRenderMethod(MIP);
			}
			else 
			{
				// If we do, enable the transfer function combo box...
				this->ui->comboBoxTransferFunction->setEnabled(true);

				// ...and select the first transfer function
				if (this->ui->comboBoxTransferFunction->currentIndex() == 0)
					this->ui->comboBoxTransferFunction->setCurrentIndex(1);

				this->mapper->setRenderMethod(DVR);
			}

            break;

        case 1:
            this->mapper->setRenderMethod(MIP);
            break;
        case 2:
            if (this->ui->checkBoxUseToonShading->isChecked())
                this->mapper->setRenderMethod(TOON);
            else
                this->mapper->setRenderMethod(ISOSURFACE);
            break;
        default:
            Q_ASSERT(false);
        }

		bool useDVR = this->ui->comboBoxRendermode->currentIndex() == 0;
		bool useMIP = this->ui->comboBoxRendermode->currentIndex() == 1;
		bool useISO = this->ui->comboBoxRendermode->currentIndex() == 2;

		this->ui->groupBox->setEnabled(useDVR || useMIP);
		this->ui->transferFunctionLabel->setEnabled(useDVR);
		this->ui->comboBoxTransferFunction->setEnabled(useDVR);

		this->ui->groupBox_2->setEnabled(useISO);
		
        renderCheck();
    }

    void RayCastPlugin::comboBoxTransferFunctionChange()
    {
        int transferFunctionIndex;
        transferFunctionIndex = this->ui->comboBoxTransferFunction->currentIndex();
        transferFunctionIndex -= 1;
        if (transferFunctionIndex  < 0)
        {
            this->pVolume->SetProperty(NULL);
        }
        else
        {
            vtkVolumeProperty* property;
            property = vtkVolumeProperty::New();
            property->SetColor(this->transferFunctions.at(transferFunctionIndex));
            property->SetScalarOpacity(this->piecewiseFunctions.at(transferFunctionIndex));
            this->pVolume->SetProperty(property);
            property->Delete();
            this->mapper->transferFunctionChanged();
        }
        renderCheck();
    }

    bool RayCastPlugin::renderCheck()
    {
        bool ok = true;

        // check if volume is present

        if (this->mapper->GetInput() == 0)
            ok = false;

        // check if transferfunction is present but only if rendermethod is dvr

        if (this->ui->comboBoxRendermode->currentIndex() == 0)
        {
            if (this->ui->comboBoxTransferFunction->currentIndex() <= 0)
                ok = false;
        }

        if (ok)
            this->pVolume->VisibilityOn();
        else
            this->pVolume->VisibilityOff();
        this->core()->render();
        return ok;
    }


void RayCastPlugin::comboBoxBitMaskChange()
{
	// Enable/disable "Invert" checkbox
	this->ui->checkBoxInvertBitmask->setEnabled(this->ui->comboBoxBitMask->currentIndex() > 0);

	data::DataSet * sourceDS;
	data::DataSet * maskDS;

	// Get the index of the volume used for masking
	int volumeIndex = this->ui->comboBoxDataset->currentIndex();

	// Decrement the index (first combo box item is "None")
	volumeIndex -= 1;

	// Do nothing if we've selected "None"
	if (volumeIndex < 0)
		return;

	// Get the input image data set
	sourceDS = this->compatibleDataSets.at(volumeIndex);

	// If the source image has not yet been computed, update it now
	if (sourceDS->getVtkImageData()->GetActualMemorySize() == 0)
	{
		sourceDS->getVtkImageData()->Update();
		this->core()->data()->dataSetChanged(sourceDS);
	}

	// Get the index of the volume used for masking
	int bitMaskIndex = this->ui->comboBoxBitMask->currentIndex();

	// Decrement the index (first combo box item is "No bitmask")
	bitMaskIndex -= 1;

	// If we've selected "No bitmask", use the original source image
	if (bitMaskIndex < 0)
	{
		this->mapper->SetInput(sourceDS->getVtkImageData());
		this->calculateTransformationMatrix();
	}
	// Otherwise, we mask the source image used the mask image
	else
	{
		vtkImageMask2 * maskFilter = new vtkImageMask2;
		
		// Turn inverting on or off, based on the GUI
		maskFilter->setInvert(this->ui->checkBoxInvertBitmask->isChecked());

		// Assign the source image to the first input
		maskFilter->setInput0(sourceDS->getVtkImageData());

		// Get the data set of the masking image
		maskDS = this->compatibleMasks.at(bitMaskIndex);

		// Update the mask image if necessary
		if (maskDS->getVtkImageData()->GetActualMemorySize() == 0)
		{
			sourceDS->getVtkImageData()->Update();
			this->core()->data()->dataSetChanged(sourceDS);
		}

		// Assign the mask image to the second input
		maskFilter->setInput1(maskDS->getVtkImageData());

		// Temporary VTK object
		vtkObject * obj;

		// Matrices for source and mask images
		vtkMatrix4x4 * sourceMatrix;
		vtkMatrix4x4 * maskMatrix;

		// Get transformation matrices from the data sets and add them to the filter
		if (sourceDS->getAttributes()->getAttribute("transformation matrix", obj))
		{
			sourceMatrix = vtkMatrix4x4::SafeDownCast(obj);
			maskFilter->setSourceMatrix(sourceMatrix);
		}
		else
		{
			// Use the identity matrix of no transformation matrix is available
			maskFilter->setSourceMatrix(NULL);
		}

		if (maskDS->getAttributes()->getAttribute("transformation matrix", obj))
		{
			maskMatrix = vtkMatrix4x4::SafeDownCast(obj);
			maskFilter->setMaskMatrix(maskMatrix);
		}
		else
		{
			// Use the identity matrix of no transformation matrix is available
			maskFilter->setMaskMatrix(NULL);
		}

		// Update the mask filter
		maskFilter->Update();
		
		// Use the output of the filter as the input of the mapper
		mapper->SetInput(maskFilter->getOutput());

		// Transformation matrix of the output image. Equal to either the source
		// matrix (if source and mask have the same size), or to the mask matrix
		// (if the two images have different sizes).

		vtkMatrix4x4 * outMatrix = vtkMatrix4x4::New();

		// If the output matrix is non-NULL, copy it
		if (maskFilter->getOutputMatrix())
			outMatrix->DeepCopy(maskFilter->getOutputMatrix());
		// Otherwise, use the identity matrix
		else
			outMatrix->Identity();

		// Apply the matrix to the volume
		this->pVolume->PokeMatrix(outMatrix);

		// Delete the filter
		delete maskFilter;
	}

	// Render the scene
	this->mapper->volumeChanged();
	renderCheck();
}

    void RayCastPlugin::setControls()
    {
        double bounds[6];
        this->pVolume->GetBounds(bounds);
        double range[2];

        int volumeIndex = this->ui->comboBoxDataset->currentIndex();
        volumeIndex -= 1;
        data::DataSet* ds = this->compatibleDataSets.at(volumeIndex);

		if (ds->getVtkImageData()->GetActualMemorySize() == 0)
		{
			ds->getVtkImageData()->Update();
			this->core()->data()->dataSetChanged(ds);
		}

		ds->getVtkImageData()->GetScalarRange(range);

        // clipping planes
        this->ui->horizontalSliderX1->setRange(bounds[0],bounds[1]);
        this->ui->horizontalSliderX1->setValue(bounds[0]);
        this->ui->horizontalSliderX2->setRange(bounds[0],bounds[1]);
        this->ui->horizontalSliderX2->setValue(bounds[1]);
        this->ui->horizontalSliderY1->setRange(bounds[2],bounds[3]);
        this->ui->horizontalSliderY1->setValue(bounds[2]);
        this->ui->horizontalSliderY2->setRange(bounds[2],bounds[3]);
        this->ui->horizontalSliderY2->setValue(bounds[3]);
        this->ui->horizontalSliderZ1->setRange(bounds[4],bounds[5]);
        this->ui->horizontalSliderZ1->setValue(bounds[4]);
        this->ui->horizontalSliderZ2->setRange(bounds[4],bounds[5]);
        this->ui->horizontalSliderZ2->setValue(bounds[5]);

		this->horizontalSliderX1Changed(bounds[0]);
		this->horizontalSliderX2Changed(bounds[1]);
		this->horizontalSliderY1Changed(bounds[2]);
		this->horizontalSliderY2Changed(bounds[3]);
		this->horizontalSliderZ1Changed(bounds[4]);
		this->horizontalSliderZ2Changed(bounds[5]);

        this->ui->checkBoxX1->setChecked(false);
        this->ui->checkBoxX2->setChecked(false);
        this->ui->checkBoxY1->setChecked(false);
        this->ui->checkBoxY2->setChecked(false);
        this->ui->checkBoxZ1->setChecked(false);
        this->ui->checkBoxZ2->setChecked(false);
        this->ui->checkBoxGrayScale->setChecked(false);
        this->ui->doubleSpinBoxMinValue->setRange(range[0],range[1]);
        this->ui->doubleSpinBoxMaxValue->setRange(range[0],range[1]);
        this->ui->doubleSpinBoxMinValue->setValue(range[0]);
        this->ui->doubleSpinBoxMaxValue->setValue(0.1*(range[1]-range[0]));
        this->ui->horizontalSliderX1->setEnabled(false);
        this->ui->horizontalSliderX2->setEnabled(false);
        this->ui->horizontalSliderY1->setEnabled(false);
        this->ui->horizontalSliderY2->setEnabled(false);
        this->ui->horizontalSliderZ1->setEnabled(false);
        this->ui->horizontalSliderZ2->setEnabled(false);
        this->ui->doubleSpinBoxMinValue->setEnabled(false);
        this->ui->doubleSpinBoxMaxValue->setEnabled(false);

        // isovalue
        this->ui->doubleSpinBoxIsovalue->setRange(range[0],range[1]);
        this->ui->doubleSpinBoxIsovalue->setValue(range[0] + 0.1 * (range[1] -range[0]));
        this->ui->doubleSpinBoxIsovalueOpacity->setValue(1.0);
        this->ui->isoValueSlider->setRange(1, (int) range[1]);
        this->ui->isoValueSlider->setValue((int) (1 + 0.1 * (range[1] - range[0])));

        // transformations

        this->ui->horizontalSliderXPos->setValue(0);
        this->ui->horizontalSliderYPos->setValue(0);
        this->ui->horizontalSliderZPos->setValue(0);
        this->ui->horizontalSliderScaleX->disconnect();
        this->ui->horizontalSliderScaleY->disconnect();
        this->ui->horizontalSliderScaleZ->disconnect();
        this->ui->spinBoxScaleX->disconnect();
        this->ui->spinBoxScaleY->disconnect();
        this->ui->spinBoxScaleZ->disconnect();
        this->ui->horizontalSliderScaleX->setValue(0);
        this->ui->horizontalSliderScaleY->setValue(0);
        this->ui->horizontalSliderScaleZ->setValue(0);
        this->ui->spinBoxScaleX->setValue(1);
        this->ui->spinBoxScaleY->setValue(1);
        this->ui->spinBoxScaleZ->setValue(1);
        connect(this->ui->horizontalSliderScaleX, SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderXChange(int)));
        connect(this->ui->horizontalSliderScaleY, SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderYChange(int)));
        connect(this->ui->horizontalSliderScaleZ, SIGNAL(sliderMoved(int)),this,SLOT(scaleSliderZChange(int)));
        connect(this->ui->spinBoxScaleX,SIGNAL(valueChanged(double)),this,SLOT(scaleSpinBoxXChange(double)));
        connect(this->ui->spinBoxScaleY,SIGNAL(valueChanged(double)),this,SLOT(scaleSpinBoxYChange(double)));
        connect(this->ui->spinBoxScaleZ,SIGNAL(valueChanged(double)),this,SLOT(scaleSpinBoxZChange(double)));
        this->ui->spinBoxRotateX->setValue(0);
        this->ui->spinBoxRotateY->setValue(0);
        this->ui->spinBoxRotateZ->setValue(0);
        this->volumeRotation[0] = 0;
        this->volumeRotation[1] = 0;
        this->volumeRotation[2] = 0;

        // clear bitmask

        this->ui->comboBoxBitMask->setCurrentIndex(0);
        this->ui->checkBoxInvertBitmask->setChecked(false);

    }



    void RayCastPlugin::widgetIsoValueColorClicked()
    {
        QColor color = QColorDialog::getColor(this->lastIsoValueColor,0);
        if (color.isValid())
        {
            QPainter p(iconPixmapIsovalueColor);
            p.fillRect(iconPixmapIsovalueColor->rect(),color);
            this->ui->pushButtonColor->setIcon(QIcon(*iconPixmapIsovalueColor));
            this->lastIsoValueColor = color;
            this->mapper->setIsoValueColor(color.redF(),color.greenF(),color.blueF());
            this->core()->render();
        }
    }

} // namespace bmia
Q_EXPORT_PLUGIN2(libbmia_RayCastPlugin, bmia::RayCastPlugin)


