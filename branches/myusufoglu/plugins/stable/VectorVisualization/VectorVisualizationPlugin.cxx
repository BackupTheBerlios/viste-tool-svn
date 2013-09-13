/**
* Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without 
* modification, are permitted provided that the following conditions
* are met:
* 
*   - Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
* 
*   - Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in
*     the documentation and/or other materials provided with the 
*     distribution.
* 
*   - Neither the name of Eindhoven University of Technology nor the
*     names of its contributors may be used to endorse or promote 
*     products derived from this software without specific prior 
*     written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

/*
* VectorVisualizationPlugin.cxx
*
* 2010-06-25	Tim Peeters
* - First version
*
* 2010-10-19	Evert van Aart
* - Disabled this plugin for fiber data sets, as those are handled by the
*   Fiber Visualization plugin.
*
*  2013-07-02	Mehmet Yusufoglu
* - Added an opacity slider,corresponding slot and lines to the
* list box data selection slot. No class variables added.
* 
*/

#include "VectorVisualizationPlugin.h"
#include "ui_vectorvisualization.h"

#include <vtkActor.h>
#include <vtkPropAssembly.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <QColorDialog>

#include <QDebug>

namespace bmia {

	VectorVisualizationPlugin::VectorVisualizationPlugin() : AdvancedPlugin("VectorVisualisation")
	{
		this->selectedData = -1;
		this->pipeFormed =0;
		cout << "TO FALSE"<< endl;  this->changingSelection = false;

		this->assembly = vtkPropAssembly::New();
		//this->assembly->VisibleOff();

		this->widget = new QWidget();
		this->ui = new Ui::VectorVisualizationForm();  
		this->ui->setupUi(this->widget);
		// disable the options frame if there is no data
		this->ui->optionsFrame->setEnabled(false);




		// Link events in the GUI to function calls:
		// Connect the GUI controls
		connect(this->ui->glyphDataCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(inputDataChanged(int))			);
		connect(this->ui->seedPointsCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(seedDataChanged(int))			);
		connect(this->ui->scaleSpin,			SIGNAL(valueChanged(double)),		this, SLOT(setScale(double))				);
		connect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectVectorData(int)));
		connect(this->ui->visibleCheckBox, SIGNAL(toggled(bool)), this, SLOT(setVisible(bool)));
		connect(this->ui->lightingCheckBox, SIGNAL(toggled(bool)), this, SLOT(setLighting(bool)));
		connect(this->ui->colorButton, SIGNAL(clicked()), this, SLOT(changeColor()));
		connect(this->ui->opacitySlider, SIGNAL(valueChanged(int)), this, SLOT(changeOpacity(int)));
	}

	VectorVisualizationPlugin::~VectorVisualizationPlugin()
	{
		// TODO: call dataSetRemoved() for all datasets.
		delete this->widget; this->widget = NULL;
		this->assembly->Delete();
	}

	vtkProp* VectorVisualizationPlugin::getVtkProp()
	{
		return this->assembly;
	}

	QWidget* VectorVisualizationPlugin::getGUI()
	{
		return this->widget;
	}



	void VectorVisualizationPlugin::dataSetAdded(data::DataSet* ds)
	{
		Q_ASSERT(ds);

		cout << ds->getKind().toStdString() << " in datasetadded "<< endl;


		if (ds->getKind() == "seed points" && this->seedDataSets.contains(ds) == false)
		{
			// Check if the data set contains a VTK data object
			if (!(ds->getVtkObject()))
				return;
			//if((vtkImageData::SafeDownCast(ds->getVtkObject() ) ))
			//if(  (vtkImageData::SafeDownCast(ds->getVtkObject() ) )->GetPointData()->GetArray("MaxDirectionUnitVectors0" )   ) 
			//cout << "MaxDirectionUnitVectors0" << endl;
			// If so, add it to the list and the GUI
			this->seedDataSets.append(ds);
			this->ui->seedPointsCombo->addItem(ds->getName());

		}



		if (ds->getKind() == "seed points" && this->seedDataSets.contains(ds))
		{
			// Get the index of the data set
			int dsIndex = this->seedDataSets.indexOf(ds);



			// Change the data set name
			this->ui->seedPointsCombo->setItemText(dsIndex, ds->getName());

			// If we're changing the currently selected data set...
			if (this->ui->seedPointsCombo->currentIndex() == dsIndex && this->glyphFilter)
			{
				// ...update the builder, and render the scene
				//this->glyphFilter->SetInput(0, vtkDataObject::SafeDownCast(this->seedDataSets[dsIndex]->getVtkObject()));
				//this->glyphFilter->Modified();
				this->core()->render();
			}
		}

		if (ds->getKind() == "scalar volume")
		{
			vtkImageData *img_temp   = ds->getVtkImageData();
			if (!img_temp)
				return;

			// Check if the image contains point data with scalars
			vtkPointData * imagePD = img_temp->GetPointData();

			if (!imagePD)
				return;

			if (!(imagePD->GetScalars()))
				return;


			int nArrays;
			nArrays = img_temp->GetPointData()->GetNumberOfArrays() ;  // 1 for the original image N for the arrays added for unit vectors
			if (nArrays==0) 
				return;  
			bool hasVector=false;
			for(unsigned int nr = 0; nr <nArrays  ; nr++)
			{
				//	outUnitVectorList.push_back(vtkDoubleArray::New());
				//outUnitVectorList.at(nr)->SetNumberOfComponents(3);
				//outUnitVectorList.at(nr)->SetNumberOfTuples(maximaVolume->GetNumberOfPoints());

				//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
				QString name(img_temp->GetPointData()->GetArrayName(nr));
				cout << name.toStdString() << endl;
				if(name=="") return;
				if ((img_temp->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( img_temp->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
				{
					hasVector=true;
				}

			}
			if(!hasVector) return;

			img   = ds->getVtkImageData();
			// We can use this data set, so add it to the list and the GUI
			this->glyphDataSets.append(ds);
			this->ui->glyphDataCombo->addItem(ds->getName());

			//insert array names to the list box
			this->insertArrayNamesToTheListBox(this->img);
			//if(this->ui->dataList->count()  > 0)
			this->ui->dataList->setCurrentRow(0);
			//main pipeline
			this->formPipeLine(this->img, this->ui->seedPointsCombo->currentIndex() ); // 2 array number


			// Try to get a transformation matrix from the data set
			vtkObject * obj;
			if ((ds->getAttributes()->getAttribute("transformation matrix", obj)))
			{
				// Try to cast the object to a matrix
				if (vtkMatrix4x4::SafeDownCast(obj))
				{
					//useIdentityMatrix = false;

					// Copy the matrix to a new one, and apply it to the actor
					vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);
					vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
					mCopy->DeepCopy(m);
					this->actor->SetUserMatrix(mCopy);
					mCopy->Delete();
				}
			}

			// Add the actor to the assembly to be rendered:
			this->assembly->AddPart(actor);

			// Add the actor to the list of actors, for easy access to its parameters
			// later on:
			this->actors.append(actor);

			// Add the new data set to the list of data sets in the GUI:

			this->ui->optionsFrame->setEnabled(true);

			// TODO: select the newly added dataset
			//		this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
			//// Depth Peeling
			//this->fullCore()->canvas()->GetRenderer3D()->GetRenderWindow()->SetOffScreenRendering(1);

			this->core()->render();
		}
	}




	void  VectorVisualizationPlugin::formPipeLine(vtkImageData *img, int seedNumber)
	{
		//QString name(img->GetPointData()->GetArrayName(arrayNumber));
		//	cout << name.toStdString() << endl;
		//img->GetPointData()->SetActiveVectors(name.toStdString().c_str());
		cout << "formPipeLine START "  << endl;
		vtkArrowSource  *arrowSource =  vtkArrowSource::New();
		arrowSource->Update();
		glyphFilter =  vtkGlyph3D::New();
		glyphFilter->SetSourceConnection(arrowSource->GetOutputPort());
		glyphFilter->OrientOn();
		glyphFilter->SetVectorModeToUseVector(); // Or to use Normal
		glyphFilter->SetScaling(true);
		glyphFilter->SetScaleFactor(1);
		//	int dsIndex = this->seedDataSets.indexOf(ds);

		// Change the data set name
		///	glyphFilter->SetInput(0, vtkDataObject::SafeDownCast(this->seedDataSets[dsIndex]->getVtkObject()));
		//glyphFilter->SetInput(img);
		glyphFilter->SetScaleModeToDataScalingOff();
		if(seedDataSets.size()>0)
		{
			/*function */

			this->addVectorToSeeds(this->seedDataSets.at(seedNumber), this->ui->dataList->currentItem()->text() );
			vtkPointSet *temo = vtkPointSet::SafeDownCast( this->seedDataSets.at(seedNumber)->getVtkObject());
			//QString name= this->img->GetPointData()->GetArrayName(arrayNumber);
			//cout << name.toStdString() << endl;

			temo->Update();
			//	cout << temo->GetPointData()->GetArray(this->img->GetPointData()->GetArrayName(1))->GetName() << endl;
			//	cout << temo->GetPointData()->GetArray(this->img->GetPointData()->GetArrayName(1))->GetNumberOfTuples() << endl;
			//	cout << temo->GetPointData()->GetArray(this->img->GetPointData()->GetArrayName(1))->GetNumberOfComponents() << endl;
			glyphFilter->SetInput( temo);

			//glyphFilter->SetScaleModeToDataScalingOff();
			//glyphFilter->SetScaleModeToScaleByVector();
			glyphFilter->Modified();
			glyphFilter->Update();
			this->core()->render();
		}
		//glyphFilter->Update();

		// Build a pipeline for rendering this data set:
		vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
		mapper->ScalarVisibilityOff();
		mapper->SetInput(glyphFilter->GetOutput());
		actor = vtkActor::New();


		this->actor->SetVisibility(true);
		this->assembly->SetVisibility(true);
		actor->SetMapper(mapper);
		//mapper->Delete(); mapper = NULL;
		// Note that the mapper was not actually deleted because it was
		// registered by the actor. And it can still be accessed through
		// actor->GetMapper().
		pipeFormed=1;
		cout << "formPipeLine END "  << endl;
	}

	void  VectorVisualizationPlugin::insertArrayNamesToTheListBox(vtkImageData *img)
	{

		int nArrays;
		nArrays = img->GetPointData()->GetNumberOfArrays() ;

		for(unsigned int nr = 0; nr <nArrays  ; nr++)
		{
			//	outUnitVectorList.push_back(vtkDoubleArray::New());
			//outUnitVectorList.at(nr)->SetNumberOfComponents(3);
			//outUnitVectorList.at(nr)->SetNumberOfTuples(maximaVolume->GetNumberOfPoints());

			//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
			QString name(img->GetPointData()->GetArrayName(nr));
			if ((img->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( img->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
			{
				//outUnitVectorListFromFile.push_back( vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str()  )));
				data::DataSet* ds_local = new data::DataSet(name, "vector", vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str())));
				//this->core()->data()->addDataSet(ds);
				this->dataSets.append(ds_local);
				// Add the new data set to the list of currently available polydata sets:
				//this->dataSets.append(ds); // local list

				this->ui->dataList->addItem(ds_local->getName());

			}
		}
	}



	//--------------------------[ inputDataChanged ]--------------------------\\

void VectorVisualizationPlugin::inputDataChanged(int index)
{
	if (index < 0 || index >= this->glyphDataSets.size() ); //|| this->m == NULL)
		return;

	// Delete existing builder
	if (this->glyphFilter)
	{
	//	this->core()->out()->deleteProgressBarForAlgorithm(this->builder);
		this->glyphFilter->Delete();
	}

   
	// Disable rendering while we set the builder options
	this->core()->disableRendering();

	 
	/* Setup the builder
	this->setNormalizationMethod(this->ui->normMethodCombo->currentIndex());
	this->setNormalizationScope(this->ui->normScopeCombo->currentIndex());
	this->setScale(this->ui->scaleSpin->value());
	this->setSharpeningExponent(this->ui->sharpenPowerSpin->value());
	this->enableNormalization(this->ui->normalizeGroup->isChecked());
	this->enableSharpening(this->ui->sharpenGroup->isChecked());
	this->enableSmoothing(this->ui->smoothGroup->isChecked());
	this->updateSmoothOptions();
	this->changeColorMethod(this->ui->colorMethodCombo->currentIndex());
	this->setLUT(this->ui->colorLUTCombo->currentIndex());
	this->setScalarVolume(this->ui->colorScalarsCombo->currentIndex(), false);
	this->setTessellationOrder(this->ui->tessSpin->value());
	*/
	// Re-enable rendering
	this->core()->enableRendering();

	// Update the builder and render the scene
	this->core()->render();
}




	void VectorVisualizationPlugin::dataSetChanged(data::DataSet* ds)
	{
		Q_ASSERT(ds);
		cout << "DATASET CHANGED " << endl;
		if ((ds->getKind() == "scalar volume") && this->glyphDataSets.contains(ds)){
			cout << ds->getKind().toStdString() << "changed"<< endl;

			// Get the index of the data set
			int dsIndex = this->glyphDataSets.indexOf(ds);

			// Change the data set name
			this->ui->glyphDataCombo->setItemText(dsIndex, ds->getName());

			// If we're changing the currently selected data set...
			if (this->ui->glyphDataCombo->currentIndex() == dsIndex && this->glyphFilter)
			{
				// ...update the geometry of the builder, and render the scene
				//this->builder->setInputVolume(this->glyphDataSets[dsIndex]->getVtkImageData());
				//this->builder->computeGeometry(this->ui->tessSpin->value());
				//this->builder->Modified();
				this->core()->render();
			}
		}

		else if (ds->getKind() == "seed points" && this->seedDataSets.contains(ds))
		{
			cout << "seed dataset changed" << endl; 
			// Get the index of the data set
			int dsIndex = this->seedDataSets.indexOf(ds);

			// Change the data set name
			this->ui->seedPointsCombo->setItemText(dsIndex, ds->getName());

			// If we're changing the currently selected data set...
			if (this->ui->seedPointsCombo->currentIndex() == dsIndex && this->glyphFilter)
			{
				if(!this->pipeFormed) return;
			//	cout << "TO TRUE"<< endl; 
			//	this->changingSelection = true;



				if (!this->glyphFilter)
					return;

				if(this->img && this->dataSets.size() >0 && (this->seedDataSets.size() > 0))
					this->addVectorToSeeds(this->seedDataSets.at(dsIndex), this->ui->dataList->currentItem()->text() );
				else
				{
			 
					
					return; }
				vtkPointSet *temo = vtkPointSet::SafeDownCast( this->seedDataSets.at(dsIndex)->getVtkObject());
				//QString name= this->img->GetPointData()->GetArrayName(arrayNumber);
				//cout << name.toStdString() << endl;
				//cout << this->ui->seedPointsCombo->currentIndex() << " " << this->seedDataSets.size() <<  " " << this->ui->seedPointsCombo->currentIndex() << " " << this->dataSets.at(index)->getName().toStdString()  << endl;

				temo->Update();
				//	cout << temo->GetPointData()->GetArray(this->img->GetPointData()->GetArrayName(1))->GetName() << endl;
				//	cout << temo->GetPointData()->GetArray(this->img->GetPointData()->GetArrayName(1))->GetNumberOfComponents() << endl;
				glyphFilter->SetInput( temo);
				this->glyphFilter->Modified();
				this->glyphFilter->Update();
				//cout << "TO FALSE"<< endl;  this->changingSelection = false;
				this->core()->render();
 
			}
		}

	}

	void VectorVisualizationPlugin::dataSetRemoved(data::DataSet* ds)
	{
		// TODO: implement when unloading of data is implemented.
		// TODO: disable optionsFrame if number of datasets == 0.
	}

	// input is seeds set
	void VectorVisualizationPlugin::addVectorToSeeds(data::DataSet* dsSeeds, QString vectorName) 

	{ // Add vector to each seed point
			cout << "addVectorToSeeds Start ===" << endl;
		// Get the seed points
		vtkPointSet * seeds = vtkPointSet::SafeDownCast(dsSeeds->getVtkObject());

		if (!seeds)
		{
			//vtkErrorMacro(<< "Seed points have not been set!");
			return;
		}
		cout << "seed pointset has " << seeds->GetNumberOfPoints() << "points"<< endl;
		// Check if we've got any seed points
		if (seeds->GetNumberOfPoints() <= 0)
			return;

		// Check if the input volume has been set
		if (!(this->img))
		{
			//vtkErrorMacro(<< "Input volume has not been set!");
			return;
		}

		// Get the "Vectors" array from the input volume
		vtkPointData * imgPD = this->img->GetPointData();

		if (!imgPD)
		{
			//vtkErrorMacro(<< "Input volume does not contain point data!");
			return;
		}
		QString name(vectorName);
		cout << name.toStdString() << endl;
		if (!imgPD->GetArray(name.toStdString().c_str()))
		{
			//vtkErrorMacro(<< "Input volume does not contain point data!");

			return;
		}
	 
		vtkDoubleArray * maxUnitVectorImg ;
		maxUnitVectorImg = vtkDoubleArray::SafeDownCast(imgPD->GetArray( name.toStdString().c_str()));
		//cout <<  maxUnitVectorImg->GetNumberOfComponents() << endl;
		//cout << "MaxDirectionUnitVectors0 comp:"<<  maxUnitVectorImg->GetNumberOfComponents() << endl;
		if (!maxUnitVectorImg)
		{
			//vtkErrorMacro(<< "Input volume does not contain a 'Vectors' array!");
			return;
		}

		vtkDoubleArray * maxUnitVectorSeeds = vtkDoubleArray::New();
		maxUnitVectorSeeds->SetNumberOfComponents(3);
		maxUnitVectorSeeds->SetName(name.toStdString().c_str());
		// Loop through all seed points
		for (int pointId = 0; pointId < seeds->GetNumberOfPoints(); ++pointId)
		{
			// Update progress bar
			//if ((pointId % progressStepSize) == 0)
			//{
			//	this->UpdateProgress((double) pointId / (double) seeds->GetNumberOfPoints());
			//}

			// Get the seed point coordinates (glyph center)
			double * p = seeds->GetPoint(pointId);
			//cout << pointId << endl;
			// Find the corresponding voxel
			vtkIdType imagePointId = this->img->FindPoint(p[0], p[1], p[2]);
			 
			maxUnitVectorSeeds->InsertNextTuple3( maxUnitVectorImg->GetTuple3(imagePointId)[0],maxUnitVectorImg->GetTuple3(imagePointId)[1],maxUnitVectorImg->GetTuple3(imagePointId)[2]);
			// maxUnitVectorSeeds->InsertNextTuple3( maxUnitVectorImg->GetTuple3(imagePointId)[0],0.1,0);

			// Check if the seed point lies inside the image

			if (imagePointId == -1)
				continue;

		}
		seeds->GetPointData()->SetScalars(maxUnitVectorSeeds);
		seeds->GetPointData()->SetActiveVectors(name.toStdString().c_str());
		cout << "addVectorToSeeds End=====" << endl;

	}

	void VectorVisualizationPlugin::seedDataChanged(int index)
	{
		cout << "seedDataChanged"  << "Start ==========" << endl;

		if (index < 0 || index >= this->seedDataSets.size())
			return;

		if (!this->glyphFilter)
			return;

		if(this->img && this->dataSets.size() >0 && (this->seedDataSets.size() > 0))
			this->addVectorToSeeds(this->seedDataSets.at(index), this->ui->dataList->currentItem()->text() );
		else return;
		vtkPointSet *temo = vtkPointSet::SafeDownCast( this->seedDataSets.at(index)->getVtkObject());
		//cout << "TO TRUE"<< endl; this->changingSelection = true;
		cout << this->ui->seedPointsCombo->currentIndex() << " " << this->seedDataSets.size() <<  " " << this->ui->seedPointsCombo->currentIndex() << " " << this->dataSets.at(index)->getName().toStdString()  << endl;

		temo->Update();
		//	cout << temo->GetPointData()->GetArray(this->img->GetPointData()->GetArrayName(1))->GetName() << endl;
		glyphFilter->SetInput( temo);
		this->glyphFilter->Modified();
		this->glyphFilter->Update();
		// << "TO FALSE"<< endl; this->changingSelection = false;
		this->core()->render();
			cout << "seedDataChanged END"  << "===========" << endl;
	}

	void VectorVisualizationPlugin::selectVectorData(int row)
	{
		cout << this->dataSets.size() << " selectVectorData  Start========================" <<  row << endl;

		if (row < 0 || this->dataSets.size() <= row)
			return;
		if(! this->seedDataSets.at(this->ui->seedPointsCombo->currentIndex()))
			return;
			if(!this->pipeFormed)
			return;
		//if (this->changingSelection) return;
		//cout << "TO TRUE"<< endl; this->changingSelection = true;
		this->selectedData = row;
		Q_ASSERT(row >= 0); // TODO: if there is no data, do sth else
		if (!this->glyphFilter)
			return;
		  // pipe is formed in add scalar but this is callled  just before the pipeline when arraynasmes are added!!!!
		 
	
		if(this->img && (this->dataSets.size() > 0) && (row < this->dataSets.size()))
			this->ui->dataSetName->setText(this->dataSets.at(this->selectedData)->getName());
		else return;


		if(this->img && (this->dataSets.size() > 0) && (row < this->dataSets.size())&& (this->seedDataSets.size() > 0) && this->ui->seedPointsCombo->currentIndex() < this->seedDataSets.size() )
			this->addVectorToSeeds(  this->seedDataSets.at( this->ui->seedPointsCombo->currentIndex() ), this->dataSets.at(row)->getName()  );
		else return;
		cout << " " << this->seedDataSets.size() <<  " " << this->ui->seedPointsCombo->currentIndex() << " " << this->dataSets.at(row)->getName().toStdString()  << " " << this->dataSets.size() << endl;
		vtkPointSet *temo;
		if(this->seedDataSets.at(this->ui->seedPointsCombo->currentIndex())->getVtkObject())
			temo = vtkPointSet::SafeDownCast( this->seedDataSets.at(this->ui->seedPointsCombo->currentIndex())->getVtkObject());
		else return;
		if(!this->seedDataSets.at(this->ui->seedPointsCombo->currentIndex())->getVtkObject())  return;
		temo->Update();
		glyphFilter->SetInputConnection( temo->GetProducerPort() );
		this->glyphFilter->Modified();
		this->glyphFilter->Update();

		this->core()->render(); 

		//cout << "TO FALSE"<< endl;  this->changingSelection = false;
		cout << this->dataSets.size() << " selectVectorData END ========================" <<  row << endl;
	}

	void VectorVisualizationPlugin::setVisible(bool visible)
	{
		if (this->changingSelection) return;
		if (this->selectedData == -1) return;
		//this->actors.at(this->selectedData)->SetVisibility(visible);
		
		this->actor->SetVisibility(visible);
		this->core()->render();
		 
	}


	 
	 
	void VectorVisualizationPlugin::setLighting(bool lighting)
	{
		if (this->changingSelection) return;
		if (this->selectedData == -1) return;
		this->actors.at(this->selectedData)->GetProperty()->SetLighting(lighting);
		this->core()->render();
	}

	void VectorVisualizationPlugin::changeColor()
	{
		if (this->changingSelection) return;
		if (this->selectedData == -1) return;

		Q_ASSERT(this->actors.at(this->selectedData));
		vtkProperty* property = this->actors.at(this->selectedData)->GetProperty();
		Q_ASSERT(property);

		double oldColorRGB[3];
		property->GetColor(oldColorRGB);

		QColor oldColor;
		oldColor.setRgbF(oldColorRGB[0], oldColorRGB[1], oldColorRGB[2]);    

		QColor newColor = QColorDialog::getColor(oldColor, 0);
		if ( newColor.isValid() )
		{
			property->SetColor(newColor.redF(), newColor.greenF(), newColor.blueF());
			this->core()->render();
		}
	}

	void VectorVisualizationPlugin::changeOpacity(int value)
	{
		if (this->changingSelection) return;
		if (this->selectedData == -1) return;
		Q_ASSERT(this->actors.at(this->selectedData));
		this->actors.at(this->selectedData)->GetProperty()->SetOpacity(1);
		this->ui->opacityLabel->setText( QString::number(value/100.0));
		this->core()->render();

	}
	//-------------------------------[ setScale ]------------------------------\\

	void VectorVisualizationPlugin::setScale(double scale)
	{
		if (this->glyphFilter == NULL)
			return;

		this->glyphFilter->SetScaleFactor(scale);
		this->glyphFilter->Modified();
		this->core()->render();
	}


} // namespace bmia
Q_EXPORT_PLUGIN2(libVectorVisualizationPlugin, bmia::VectorVisualizationPlugin)
