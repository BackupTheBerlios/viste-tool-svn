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
		this->changingSelection = false;

		this->assembly = vtkPropAssembly::New();
		//this->assembly->VisibleOff();

		this->widget = new QWidget();
		this->ui = new Ui::VectorVisualizationForm();  
		this->ui->setupUi(this->widget);
		// disable the options frame if there is no data
		this->ui->optionsFrame->setEnabled(false);




		// Link events in the GUI to function calls:
		connect(this->ui->dataList, SIGNAL(currentRowChanged(int)), this, SLOT(selectData(int)));
		connect(this->ui->visibleCheckBox, SIGNAL(toggled(bool)), this, SLOT(setVisible(bool)));
		connect(this->ui->depthPeelingCheckBox, SIGNAL(toggled(bool)), this, SLOT(setDepthPeeling(bool)));
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

	
	
	void VectorVisualizationPlugin::addVectorToSeeds(data::DataSet* ds) 

			 { // Add vector to each seed point

			 // Get the seed points
	vtkPointSet * seeds = vtkPointSet::SafeDownCast(ds->getVtkObject());

	if (!seeds)
	{
		//vtkErrorMacro(<< "Seed points have not been set!");
		return;
	}

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

	if (!imgPD->GetArray("MaxDirectionUnitVectors0"))
	{
		//vtkErrorMacro(<< "Input volume does not contain point data!");
		return;
	}

	
	vtkDoubleArray * maxUnitVectorImg = vtkDoubleArray::SafeDownCast(imgPD->GetArray("MaxDirectionUnitVectors0"));

	if (!maxUnitVectorImg)
	{
		//vtkErrorMacro(<< "Input volume does not contain a 'Vectors' array!");
		return;
	}

	vtkDoubleArray * maxUnitVectorSeeds = vtkDoubleArray::New();

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

		// Find the corresponding voxel
		vtkIdType imagePointId = this->img->FindPoint(p[0], p[1], p[2]);

		maxUnitVectorSeeds->InsertNextTuple3( maxUnitVectorImg->GetTuple3(imagePointId)[0],maxUnitVectorImg->GetTuple3(imagePointId)[1],maxUnitVectorImg->GetTuple3(imagePointId)[2]);
		
		// Check if the seed point lies inside the image
		
		if (imagePointId == -1)
			continue;

	}

	}


	
	
	
	void VectorVisualizationPlugin::dataSetAdded(data::DataSet* ds)
	{
		Q_ASSERT(ds);
		 img ;
		 cout << ds->getKind().toStdString() << endl;

	
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
		addVectorToSeeds(ds);
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
			img   = ds->getVtkImageData();
				if (!img)
			return;

		// Check if the image contains point data with scalars
		vtkPointData * imagePD = img->GetPointData();

		if (!imagePD)
			return;

		if (!(imagePD->GetScalars()))
			return;
			int nArrays;
			nArrays = img->GetPointData()->GetNumberOfArrays() ;  // 1 for the original image N for the arrays added for unit vectors
			std::vector<vtkDoubleArray *> outUnitVectorListFromFile;



			for(unsigned int nr = 0; nr <nArrays  ; nr++)
			{
				//	outUnitVectorList.push_back(vtkDoubleArray::New());
				//outUnitVectorList.at(nr)->SetNumberOfComponents(3);
				//outUnitVectorList.at(nr)->SetNumberOfTuples(maximaVolume->GetNumberOfPoints());

				//outUnitVectorList.at(nr)->SetName( arrName.toStdString().c_str() );  //fist vector array for each point (keeps only the first vector)
				QString name(img->GetPointData()->GetArrayName(nr));
				if ((img->GetPointData()->GetArray(name.toStdString().c_str()  )->GetDataType() == VTK_DOUBLE) && ( img->GetPointData()->GetArray( name.toStdString().c_str() )->GetNumberOfComponents() ==3))
				{
					outUnitVectorListFromFile.push_back( vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str()  )));
					data::DataSet* ds_local = new data::DataSet(name, "vector", vtkDoubleArray::SafeDownCast( img->GetPointData()->GetArray(name.toStdString().c_str())));
					//this->core()->data()->addDataSet(ds);
					this->dataSets.append(ds_local);
					// Add the new data set to the list of currently available polydata sets:
					//this->dataSets.append(ds); // local list

					this->ui->dataList->addItem(ds_local->getName());



				}
				
			}
				QString name(img->GetPointData()->GetArrayName(1));
				cout << name.toStdString() << endl;
				img->GetPointData()->SetActiveVectors(name.toStdString().c_str());
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
					glyphFilter->SetInput(img);
					glyphFilter->SetScaleModeToDataScalingOff();
					
					glyphFilter->Update();
					
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

	void VectorVisualizationPlugin::dataSetChanged(data::DataSet* ds)
	{
		// TODO
	}

	void VectorVisualizationPlugin::dataSetRemoved(data::DataSet* ds)
	{
		// TODO: implement when unloading of data is implemented.
		// TODO: disable optionsFrame if number of datasets == 0.
	}

	void VectorVisualizationPlugin::selectData(int row)
	{
		this->changingSelection = true;
		this->selectedData = row;
		Q_ASSERT(row >= 0); // TODO: if there is no data, do sth else
		// TODO: assert row is in range.
		this->ui->dataSetName->setText(this->dataSets.at(this->selectedData)->getName());
		cout << this->dataSets.at(this->selectedData)->getName().toStdString() << endl;
		img->GetPointData()->SetActiveVectors(this->dataSets.at(this->selectedData)->getName().toStdString().c_str());
		//img->Update();
		//img->Modified();
		//img->GetPointData()->Update();
		//img->GetPointData()->Modified();
		//img->GetPointData()->SetActiveAttribute(this->dataSets.at(this->selectedData)->getName().toStdString().c_str(),vtkDataSetAttributes::VECTORS);
		//this->glyphFilter->SetInputArrayToProcess(row+1,0,0,vtkDataObject::FIELD_ASSOCIATION_POINTS ,this->dataSets.at(this->selectedData)->getName().toStdString().c_str());
		this->actor->SetVisibility(true);
		this->glyphFilter->Modified();
		
		this->glyphFilter->Update();
		
		//this->ui->visibleCheckBox->setChecked(this->actors.at(this->selectedData)->GetVisibility());
		//this->ui->lightingCheckBox->setChecked(this->actors.at(this->selectedData)->GetProperty()->GetLighting());
		//this->ui->depthPeelingCheckBox->setChecked(this->fullCore()->canvas()->GetRenderer3D()->GetUseDepthPeeling());
		//opacity
		//this->ui->opacitySlider->setValue(this->actors.at(this->selectedData)->GetProperty()->GetOpacity()*100);
		//this->ui->opacityLabel->setText( QString::number( this->actors.at(this->selectedData)->GetProperty()->GetOpacity() ));
		this->changingSelection = false;
		this->core()->render();
	}

	void VectorVisualizationPlugin::setVisible(bool visible)
	{
		if (this->changingSelection) return;
		if (this->selectedData == -1) return;
		//this->actors.at(this->selectedData)->SetVisibility(visible);
		this->core()->render();
		//if(this->fullCore()->canvas()->GetRenderer3D()->GetLastRenderingUsedDepthPeeling())
		//cout << " depth peeling used" << endl; 
		cout << "IsDepthPeelingSupported (offscreen true):" << IsDepthPeelingSupported(this->fullCore()->canvas()->GetRenderWindow(), this->fullCore()->canvas()->GetRenderer3D(), true) << endl;
		// cout << "IsDepthPeelingSupported (offscreen false):" << IsDepthPeelingSupported(this->fullCore()->canvas()->GetRenderWindow(), this->fullCore()->canvas()->GetRenderer3D(), false);

	}


	/**
	* Setup the rendering environment for depth peeling (general depth peeling
	* support is requested).
	* @see IsDepthPeelingSupported()
	* @param renderWindow a valid openGL-supporting render window
	* @param renderer a valid renderer instance
	* @param maxNoOfPeels maximum number of depth peels (multi-pass rendering)
	* @param occulusionRation the occlusion ration (0.0 means a perfect image,
	* >0.0 means a non-perfect image which in general results in faster rendering)
	* @return TRUE if depth peeling could be set up
	*/
	bool VectorVisualizationPlugin::SetupEnvironmentForDepthPeeling(
		vtkRenderWindow *renderWindow,
		vtkRenderer *renderer, int maxNoOfPeels,
		double occlusionRatio)
	{
		if (!renderWindow || !renderer)
			return false;

		// 1. Use a render window with alpha bits (as initial value is 0 (false)):
		renderWindow->SetAlphaBitPlanes(true);

		// 2. Force to not pick a framebuffer with a multisample buffer
		// (as initial value is 8):
		renderWindow->SetMultiSamples(0);

		// 3. Choose to use depth peeling (if supported) (initial value is 0 (false)):
		renderer->SetUseDepthPeeling(true);

		// 4. Set depth peeling parameters
		// - Set the maximum number of rendering passes (initial value is 4):
		renderer->SetMaximumNumberOfPeels(maxNoOfPeels);
		// - Set the occlusion ratio (initial value is 0.0, exact image):
		renderer->SetOcclusionRatio(occlusionRatio);

		return true;
	}

	/**
	* Find out whether this box supports depth peeling. Depth peeling requires
	* a variety of openGL extensions and appropriate drivers.
	* @param renderWindow a valid openGL-supporting render window
	* @param renderer a valid renderer instance
	* @param doItOffscreen do the test off screen which means that nothing is
	* rendered to screen (this requires the box to support off screen rendering)
	* @return TRUE if depth peeling is supported, FALSE otherwise (which means
	* that another strategy must be used for correct rendering of translucent
	* geometry, e.g. CPU-based depth sorting)
	*/

	bool VectorVisualizationPlugin::IsDepthPeelingSupported( vtkRenderWindow *renderWindow,
		vtkRenderer *renderer,
		bool doItOffScreen)
	{
		if (!renderWindow || !renderer)
		{
			return false;
		}

		bool success = true;

		// Save original renderer / render window state
		bool origOffScreenRendering = renderWindow->GetOffScreenRendering() == 1;
		bool origAlphaBitPlanes = renderWindow->GetAlphaBitPlanes() == 1;
		int origMultiSamples = renderWindow->GetMultiSamples();
		bool origUseDepthPeeling = renderer->GetUseDepthPeeling() == 1;
		int origMaxPeels = renderer->GetMaximumNumberOfPeels();
		double origOcclusionRatio = renderer->GetOcclusionRatio();

		cout << "origOffScreenRendering" << origOffScreenRendering << endl;
		cout << "origAlphaBitPlanes" << origAlphaBitPlanes << endl;
		cout << "origMultiSamples" << origMultiSamples << endl;
		cout << "origUseDepthPeeling" << origUseDepthPeeling << endl;
		cout << "origMaxPeels" << origMaxPeels << endl;
		cout << "origOcclusionRatio" << origOcclusionRatio << endl;






		// Activate off screen rendering on demand
		renderWindow->SetOffScreenRendering(doItOffScreen);

		// Setup environment for depth peeling (with some default parametrization)
		success = success && SetupEnvironmentForDepthPeeling(renderWindow, renderer,
			50, 0.1);

		// Do a test render
		renderWindow->Render();

		// Check whether depth peeling was used
		success = success && renderer->GetLastRenderingUsedDepthPeeling();

		// recover original state
		renderWindow->SetOffScreenRendering(origOffScreenRendering);
		renderWindow->SetAlphaBitPlanes(origAlphaBitPlanes);
		renderWindow->SetMultiSamples(origMultiSamples);
		renderer->SetUseDepthPeeling(origUseDepthPeeling);
		renderer->SetMaximumNumberOfPeels(origMaxPeels);
		renderer->SetOcclusionRatio(origOcclusionRatio);

		return success;
	}




	void VectorVisualizationPlugin::setDepthPeeling(bool value)
	{
		if (this->changingSelection) return;
		if (this->selectedData == -1) return;
		return;
		//vtkRenderer *renderer= this->fullCore()->canvas()->GetRenderer3D();
		//vtkRenderWindow *renderWindow = this->fullCore()->canvas()->GetRenderer3D()->GetRenderWindow();
		vtkRenderer *renderer=  this->fullCore()->canvas()->GetRenderer3D();  
		vtkRenderWindow *renderWindow =  this->fullCore()->canvas()->GetRenderWindow();

		renderer->SetUseDepthPeeling(value);
		if(value)
		{
			renderWindow->SetAlphaBitPlanes(1); // default 0 can be put to main window!!
			renderWindow->SetMultiSamples(0); //default 8 Set the number of multisamples to use for hardware antialiasing.

			renderer->SetMaximumNumberOfPeels(50); // default 4
			////renderer->GetRenderWindow()->For
			renderer->SetOcclusionRatio(0.1); 
			//renderWindow->SetOffScreenRendering(1);
		}
		else
		{
			renderWindow->SetAlphaBitPlanes(0); // default 0 can be put to main window!!
			renderWindow->SetMultiSamples(8); //default 8 Set the number of multisamples to use for hardware antialiasing.

			renderer->SetMaximumNumberOfPeels(4); // default 4
			////renderer->GetRenderWindow()->For
			renderer->SetOcclusionRatio(0.0); // default value is 0.0
		}


		renderer->Render();
		bool origOffScreenRendering = renderWindow->GetOffScreenRendering() == 1;
		bool origAlphaBitPlanes = renderWindow->GetAlphaBitPlanes() == 1;
		int origMultiSamples = renderWindow->GetMultiSamples();
		bool origUseDepthPeeling = renderer->GetUseDepthPeeling() == 1;
		int origMaxPeels = renderer->GetMaximumNumberOfPeels();
		double origOcclusionRatio = renderer->GetOcclusionRatio();

		cout << "origOffScreenRendering" << origOffScreenRendering << endl;
		cout << "origAlphaBitPlanes" << origAlphaBitPlanes << endl;
		cout << "origMultiSamples" << origMultiSamples << endl;
		cout << "origUseDepthPeeling" << origUseDepthPeeling << endl;
		cout << "origMaxPeels" << origMaxPeels << endl;
		cout << "origOcclusionRatio" << origOcclusionRatio << endl;


		this->core()->render();

		if(renderer->GetLastRenderingUsedDepthPeeling())
			cout << " depth peeling used" << endl; 
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

} // namespace bmia
Q_EXPORT_PLUGIN2(libVectorVisualizationPlugin, bmia::VectorVisualizationPlugin)
