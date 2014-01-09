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
*
*  2013-10-29	Mehmet Yusufoglu
* - Created for displaying vtkImageData having a 3-component (double) vectors for each voxel. 
* Image Data is expected to be read by any other plugin e.g. vtiReaderPlugin, dataset type is "unit vector volume".
* Each vector is shown together with its negative (the vector starting from the same point but in opposite direction) since they are assumed to be maxima vectors of HARDI ODFs. 
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
		this->ui->visibleCheckBox->setChecked(true);



		// Link events in the GUI to function calls:
		// Connect the GUI controls
		connect(this->ui->volumeDataCombo,		SIGNAL(currentIndexChanged(int)),	this, SLOT(inputDataChanged(int))			);
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

		if (ds->getKind() == "unit vector volume")
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
			this->ui->dataList->setCurrentRow(0);
			// We can use this data set, so add it to the list and the GUI
			this->glyphDataSets.append(ds);
			this->ui->volumeDataCombo->addItem(ds->getName());
			///this->ui->volumeDataCombo->setCurrentIndex(this->ui->volumeDataCombo->count()-1);
			//insert array names to the list box
			

			this->ui->optionsFrame->setEnabled(true);

			// TODO: select the newly added dataset
			//		this->fullCore()->canvas()->GetRenderer3D()->ResetCamera();
			//// Depth Peeling
			//this->fullCore()->canvas()->GetRenderer3D()->GetRenderWindow()->SetOffScreenRendering(1);

			this->core()->render();
		}
	}



	 

	void  VectorVisualizationPlugin::formPipeLinesForAllArrays(vtkImageData *img, int seedNumber)
	{
		//QString name(img->GetPointData()->GetArrayName(arrayNumber));
		//	cout << name.toStdString() << endl;
		//img->GetPointData()->SetActiveVectors(name.toStdString().c_str());
		cout << "formPipeLine START "  << endl;
		vtkArrowSource  *arrowSource =  vtkArrowSource::New();
		arrowSource->Update();

		int nArrays;
		nArrays = img->GetPointData()->GetNumberOfArrays() ;  // 1 for the original image N for the arrays added for unit vectors
		if (nArrays==0) 
			return;  
		int size = glyphFilters.size();
		for( int nr = 0; nr <glyphFilters.size()  ; nr++)
		{

			actors.at(nr)->SetVisibility(false);
			glyphFilters.at(nr)->Delete();
			mappers.at(nr)->Delete();
			actors.at(nr)->Delete();
			seedGridsOfASeed.at(nr)->Delete();
		}

		glyphFilters.clear();
		actors.clear();
		mappers.clear();
		seedGridsOfASeed.clear();

		for( int nr = 0; nr < size ; nr++)
		{

			actorsOpposite.at(nr)->SetVisibility(false);
			glyphFiltersOpposite.at(nr)->Delete();
			mappersOpposite.at(nr)->Delete();
			actorsOpposite.at(nr)->Delete();
			seedGridsOfASeedOpposite.at(nr)->Delete();
		}

		glyphFiltersOpposite.clear();
		actorsOpposite.clear();
		mappersOpposite.clear();
		seedGridsOfASeedOpposite.clear();

		for( int nr = 0; nr <nArrays  ; nr++)
		{
			QString name(img->GetPointData()->GetArrayName(nr));
			if ((img->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( img->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
			{ 

				glyphFilters.append( vtkGlyph3D::New() );
				actors.append( vtkActor::New() );
				mappers.append( vtkPolyDataMapper::New() );
				seedGridsOfASeed.append(vtkUnstructuredGrid::New());
				glyphFiltersOpposite.append( vtkGlyph3D::New() );
				actorsOpposite.append( vtkActor::New() );
				mappersOpposite.append( vtkPolyDataMapper::New() );
				seedGridsOfASeedOpposite.append(vtkUnstructuredGrid::New());
			}

		}
		if(seedDataSets.size()<=0)
			return;

		for( int nr = 0; nr <glyphFilters.size() ; nr++)
		{



			glyphFilters[nr] ->SetSourceConnection(arrowSource->GetOutputPort());
			//glyphFilter->OrientOn();
			glyphFilters[nr]->SetVectorModeToUseVector(); // Or to use Normal
			glyphFilters[nr]->SetScaling(true);
			glyphFilters[nr]->SetScaleFactor(1);		 
			glyphFilters[nr]->SetScaleModeToDataScalingOff();
			glyphFilters[nr]->SetScaleModeToScaleByVector();

			/*function */
			glyphFiltersOpposite[nr] ->SetSourceConnection(arrowSource->GetOutputPort());
			glyphFiltersOpposite[nr]->SetVectorModeToUseVector(); // Or to use Normal
			glyphFiltersOpposite[nr]->SetScaling(true);
			glyphFiltersOpposite[nr]->SetScaleFactor(1);
			glyphFiltersOpposite[nr]->SetScaleModeToDataScalingOff();
			glyphFiltersOpposite[nr]->SetScaleModeToScaleByVector();


			vtkPointSet * seeds = vtkPointSet::SafeDownCast(this->seedDataSets.at(seedNumber)->getVtkObject());  
			vtkPoints * newPoints = vtkPoints::New();

			for (int pointId = 0; pointId < seeds->GetNumberOfPoints(); ++pointId)
			{


				// Get the seed point coordinates (glyph center)
				double * p = seeds->GetPoint(pointId);

				// Find the corresponding voxel
				vtkIdType imagePointId = this->img->FindPoint(p[0], p[1], p[2]);
				newPoints->InsertNextPoint(p[0], p[1], p[2]);

			}
			seedGridsOfASeed.at(nr)->SetPoints(newPoints);
			seedGridsOfASeedOpposite.at(nr)->SetPoints(newPoints);
			this->addVectorToUnstructuredGrid( seedGridsOfASeed.at(nr), this->ui->dataList->item(nr)->text() );

			this->addVectorToUnstructuredGrid( seedGridsOfASeedOpposite.at(nr), this->ui->dataList->item(nr)->text(),true );


			glyphFilters[nr]->SetInput( seedGridsOfASeed.at(nr));
			glyphFilters[nr]->Modified();
			glyphFilters[nr]->Update();


			//glyphFilter->Update();

			// Build a pipeline for rendering this data set:

			mappers[nr]->ScalarVisibilityOff();
			mappers[nr]->SetInput(glyphFilters[nr]->GetOutput());



			this->actors[nr]->SetVisibility(true);

			actors[nr]->SetMapper(mappers[nr]);
			this->assembly->AddPart(actors[nr]);

			///////////// OPPOSITE 
			glyphFiltersOpposite[nr]->SetInput( seedGridsOfASeedOpposite.at(nr));

			glyphFiltersOpposite[nr]->Modified();
			glyphFiltersOpposite[nr]->Update();

			// Build a pipeline for rendering this data set:

			mappersOpposite[nr]->ScalarVisibilityOff();
			mappersOpposite[nr]->SetInput(glyphFiltersOpposite[nr]->GetOutput());
			this->actorsOpposite[nr]->SetVisibility(true);

			actorsOpposite[nr]->SetMapper(mappersOpposite[nr]);
			this->assembly->AddPart(actorsOpposite[nr]);

		}//for

		// Try to get a transformation matrix from the data set
		vtkObject * obj;
		if ((this->glyphDataSets.at(this->ui->volumeDataCombo->currentIndex())->getAttributes()->getAttribute("transformation matrix", obj)))
		{
			// Try to cast the object to a matrix
			if (vtkMatrix4x4::SafeDownCast(obj))
			{
				//useIdentityMatrix = false;

				// Copy the matrix to a new one, and apply it to the actor
				vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);
				vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
				mCopy->DeepCopy(m);
			 
				for( int nr = 0; nr <glyphFilters.size()  ; nr++)
				{
					this->actors[nr]->SetUserMatrix(mCopy);
					this->actorsOpposite[nr]->SetUserMatrix(mCopy);
				}
				mCopy->Delete();
			}
		}

		this->core()->render();
		this->assembly->SetVisibility(true);
		//mapper->Delete(); mapper = NULL;

		pipeFormed=1;
		cout << "formPipeLine END "  << endl;
	}


	
	void  VectorVisualizationPlugin::insertArrayNamesToTheListBox(vtkImageData *img)
	{

		int nArrays;
		nArrays = img->GetPointData()->GetNumberOfArrays() ;
		this->ui->dataList->clear();
		for(unsigned int nr = 0; nr <nArrays  ; nr++)
		{
 

			//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
			QString name(img->GetPointData()->GetArrayName(nr));
			if ((img->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( img->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
			{
				//outUnitVectorListFromFile.push_back( vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str()  )));
				data::DataSet* ds_local = new data::DataSet(name, "vector", vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str())));
 
				this->dataSets.append(ds_local);
 

				this->ui->dataList->addItem(ds_local->getName());

			}
		}
		if(nArrays > 0) this->selectedData = 0;
	}



	//--------------------------[ inputDataChanged ]--------------------------\\

	void VectorVisualizationPlugin::inputDataChanged(int index)
	{

		cout << "New data selected " << endl;
		cout << index << " " << this->glyphDataSets.size() << endl;
		if (index < 0 || index >= this->glyphDataSets.size() ) //|| this->m == NULL)
		return;

		// Delete existing builder
		//if (this->glyphFilter)
		//{
			//	this->core()->out()->deleteProgressBarForAlgorithm(this->builder);
			//this->glyphFilter->Delete();
		//}

		// ADD
		// Disable rendering while we set the builder options
		this->core()->disableRendering();


		/* Setup the builder
	 
		*/
		data::DataSet *ds = this->glyphDataSets.at(this->ui->volumeDataCombo->currentIndex());
		 
			this->insertArrayNamesToTheListBox(this->img);//clears others
			//if(this->ui->dataList->count()  > 0)
			
			//main pipeline
			//this->formPipeLine(this->img, this->ui->seedPointsCombo->currentIndex() ); // 2 array number
			this->formPipeLinesForAllArrays(this->img, this->ui->seedPointsCombo->currentIndex() ); // this must add addtional arrays how to do??
		

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
					 
					for( int nr = 0; nr <glyphFilters.size()  ; nr++)



					{
						this->actors[nr]->SetUserMatrix(mCopy);

						this->actorsOpposite[nr]->SetUserMatrix(mCopy);
					}
					mCopy->Delete();
				}
			}

			 //volcount


		// Re-enable rendering
		this->core()->enableRendering();

		// Update the builder and render the scene
		this->core()->render();
	}




	void VectorVisualizationPlugin::dataSetChanged(data::DataSet* ds)
	{
		Q_ASSERT(ds);
		 
		if ((ds->getKind() == "scalar volume") && this->glyphDataSets.contains(ds)){
			//cout << ds->getKind().toStdString() << "changed"<< endl;

			// Get the index of the data set
			int dsIndex = this->glyphDataSets.indexOf(ds);

			// Change the data set name
			this->ui->volumeDataCombo->setItemText(dsIndex, ds->getName());

			// If we're changing the currently selected data set...
			if (this->ui->volumeDataCombo->currentIndex() == dsIndex  )
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
			//cout << "seed dataset changed" << endl; 
			// Get the index of the data set
			int dsIndex = this->seedDataSets.indexOf(ds);

			// Change the data set name
			//this->ui->seedPointsCombo->setItemText(dsIndex, ds->getName());

			// If we're changing the currently selected data set...
			if (this->ui->seedPointsCombo->currentIndex() == dsIndex  )
			{
				if(!this->pipeFormed) return;
				//	cout << "TO TRUE"<< endl; 
				//	this->changingSelection = true;

				 
				if(this->img && this->dataSets.size() >0 && (this->seedDataSets.size() > 0) && (this->ui->seedPointsCombo->itemText(dsIndex)== ds->getName()))
			this->formPipeLinesForAllArrays(this->img, this->ui->seedPointsCombo->currentIndex() );
		else return;  


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
	void VectorVisualizationPlugin::addVectorToUnstructuredGrid(vtkUnstructuredGrid *gridForArrayForSeed, QString vectorName, bool Opposite) 
	{ // Add vector to each seed point
		cout << "addVector " <<  vectorName.toStdString() << "  ToSeeds Start ===" << endl;
		// Get the seed points
		vtkPointSet * seeds = vtkPointSet::SafeDownCast(gridForArrayForSeed);

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
		if (!maxUnitVectorImg)
		{
			//vtkErrorMacro(<< "Input volume does not contain a 'Vectors' array!");
			return;
		}

		vtkDoubleArray * maxUnitVectorSeeds = vtkDoubleArray::New();
		maxUnitVectorSeeds->SetNumberOfComponents(3);
		maxUnitVectorSeeds->SetName(name.toStdString().c_str());
		// Loop through all seed points
		double *unitv;
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
			unitv= (double *) maxUnitVectorImg->GetTuple3(imagePointId);
			if(Opposite) 
			{ unitv[0]=-1*unitv[0]; 
			unitv[1]=-1*unitv[1]; unitv[2]=-1*unitv[2]; }
			maxUnitVectorSeeds->InsertNextTuple3( unitv[0],unitv[1],unitv[2]);
			//if(unitv[0]==0 &&  unitv[1] ==0 && unitv[0]==0)
			// cout << "ZERO UNIUT VECTOR"<< endl; 

			// maxUnitVectorSeeds->InsertNextTuple3( maxUnitVectorImg->GetTuple3(imagePointId)[0],0.1,0);

			// Check if the seed point lies inside the image

			if (imagePointId == -1)
				continue;

		}
		seeds->GetPointData()->AddArray(maxUnitVectorSeeds);
		//seeds->GetPointData()->SetActiveAttribute(name.toStdString().c_str(),1);
		seeds->GetPointData()->SetActiveVectors(name.toStdString().c_str());
		seeds->Modified();
		seeds->Update();
		cout << "addVectorToSeeds " <<  name.toStdString() << " End=====" << endl;

	}

	void VectorVisualizationPlugin::seedDataChanged(int index)
	{
		cout << "seedDataChanged"  << "Start ==========" << endl;

		if (index < 0 || index >= this->seedDataSets.size())
			return;

		//if (!this->glyphFilter)
		//return;

		if(this->img && this->dataSets.size() >0 && (this->seedDataSets.size() > 0))
			this->formPipeLinesForAllArrays(this->img, this->ui->seedPointsCombo->currentIndex() );
		else return;  
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
		if(this->actors.size()> row)
			this->ui->visibleCheckBox->setChecked(this->actors[row]->GetVisibility());

		if(this->img && (this->dataSets.size() > 0) && (row < this->dataSets.size()))
			this->ui->dataSetName->setText(this->dataSets.at(this->selectedData)->getName());
		else return;
		//this->seedDataSets.at( this->ui->seedPointsCombo->currentIndex() )->getVtkObject()->SetAc
		vtkPointSet * seeds = vtkPointSet::SafeDownCast(  this->seedDataSets.at( this->ui->seedPointsCombo->currentIndex() )->getVtkObject());
		seeds->GetPointData()->SetActiveVectors(this->dataSets.at( this->selectedData )->getName().toStdString().c_str());
	 
		//cout << "TO FALSE"<< endl;  this->changingSelection = false;
		this->core()->render();
		cout << this->dataSets.size() << " selectVectorData END ========================" <<  row << endl; 
	}

	void VectorVisualizationPlugin::setVisible(bool visible)
	{
		//if (this->changingSelection) return;
		//if (this->selectedData == -1) return;
		//this->actors.at(this->selectedData)->SetVisibility(visible);
		QString name(this->ui->dataList->currentItem()->text());
		cout << name.toStdString() << endl; 
		//vtkPointSet::SafeDownCast( this->seedDataSets.at(this->ui->seedPointsCombo->currentIndex())->getVtkObject())->GetPointData()->SetActiveVectors(name.toStdString().c_str());

		cout << this->ui->dataList->currentRow() << endl;
		cout << visible << endl;
		this->actors[this->ui->dataList->currentRow()]->SetVisibility(visible);
		this->actorsOpposite[this->ui->dataList->currentRow()]->SetVisibility(visible);
		this->core()->render();

	}




	void VectorVisualizationPlugin::setLighting(bool lighting)
	{
		if (this->changingSelection) return;

		//	if (this->selectedData == -1) return;

		Q_ASSERT(this->actors[this->ui->dataList->currentRow()]);
		vtkProperty* property = this->actors[this->ui->dataList->currentRow()]->GetProperty();
		Q_ASSERT(property);
		property->SetLighting(lighting);

		Q_ASSERT(this->actorsOpposite[this->ui->dataList->currentRow()]);
		vtkProperty* propertyOpposite = this->actorsOpposite[this->ui->dataList->currentRow()]->GetProperty();
		Q_ASSERT(propertyOpposite);
		propertyOpposite->SetLighting(lighting);

		this->core()->render();
	}

	void VectorVisualizationPlugin::changeColor()
	{
		if (this->changingSelection) return;
		//	if (this->selectedData == -1) return;

		Q_ASSERT(this->actors[this->ui->dataList->currentRow()]);
		vtkProperty* property = this->actors[this->ui->dataList->currentRow()]->GetProperty();
		Q_ASSERT(property);

		Q_ASSERT(this->actorsOpposite[this->ui->dataList->currentRow()]);
		vtkProperty* propertyOpposite = this->actorsOpposite[this->ui->dataList->currentRow()]->GetProperty();
		Q_ASSERT(propertyOpposite);

		double oldColorRGB[3];
		property->GetColor(oldColorRGB);

		QColor oldColor;
		oldColor.setRgbF(oldColorRGB[0], oldColorRGB[1], oldColorRGB[2]);    

		QColor newColor = QColorDialog::getColor(oldColor, 0);
		if ( newColor.isValid() )
		{
			property->SetColor(newColor.redF(), newColor.greenF(), newColor.blueF());
			propertyOpposite->SetColor(newColor.redF(), newColor.greenF(), newColor.blueF());
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
		if (this->glyphFilters[this->ui->dataList->currentRow()] == NULL)
			return;

		this->glyphFilters[this->ui->dataList->currentRow()]->SetScaleFactor(scale);
		this->glyphFilters[this->ui->dataList->currentRow()]->Modified();

		this->glyphFiltersOpposite[this->ui->dataList->currentRow()]->SetScaleFactor(scale);
		this->glyphFiltersOpposite[this->ui->dataList->currentRow()]->Modified();
		this->core()->render();
	}


} // namespace bmia
Q_EXPORT_PLUGIN2(libVectorVisualizationPlugin, bmia::VectorVisualizationPlugin)
