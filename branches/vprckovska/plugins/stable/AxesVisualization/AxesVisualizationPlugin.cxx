/*
 * AxesVisualizationPlugin.cxx
 *
 * 2010-05-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-04-04	Evert van Aart
 * - Version 1.0.0.
 * - Completely new approach to drawing the axes: The axes widget is now added to
 *   the medical canvas by means of a "vtkOrientationMarkerWidget".
 *
 * 2011-06-08	Evert van Aart
 * - Version 1.0.1.
 * - Fixed a bug that caused crashes when this plugin was unloaded.
 *
 */


#include "AxesVisualizationPlugin.h"


namespace bmia {


AxesVisualizationPlugin::AxesVisualizationPlugin() : plugin::AdvancedPlugin("Axes")
{
	// Setup the GUI
	this->qWidget = new QWidget();
	this->ui = new Ui::AxesVisualizationForm();
	this->ui->setupUi(this->qWidget);

	// Set pointer to NULL
	this->callBack = NULL;
}


void AxesVisualizationPlugin::init()
{
	// Get the collection of subcanvasses
	vtkMedicalCanvas * canvas = this->fullCore()->canvas();
	vtkAbstractSubCanvasCollection * subcanvasses = canvas->GetSubCanvasses();

	// Loop through all subcanvasses
	for (int i = 0; i < subcanvasses->GetNumberOfItems(); ++i)
	{
		vtkSubCanvas * currentSubCanvas = (vtkSubCanvas *) subcanvasses->GetItem(i);

		// Currently, we only look at the 3D view
		if (currentSubCanvas->subCanvasName != "3D View")
			continue;

		// Add the subcanvas to the GUI
		this->ui->subCanvasCombo->addItem(QString(currentSubCanvas->subCanvasName.c_str()));

		// Create an axes actor
		vtkAxesActor * newActor = vtkAxesActor::New();
		newActor->SetTotalLength(800.0,800.0,800.0);
		newActor->SetConeRadius(0.2);
		newActor->SetConeResolution(16);
		newActor->SetPickable(0);

		// Configure the font of the axes actor
		newActor->GetXAxisCaptionActor2D()->GetTextActor()->SetTextScaleMode(vtkTextActor::TEXT_SCALE_MODE_NONE);
		vtkTextProperty * xText = newActor->GetXAxisCaptionActor2D()->GetTextActor()->GetTextProperty();
		xText->SetFontSize(16);

		newActor->GetYAxisCaptionActor2D()->GetTextActor()->SetTextScaleMode(vtkTextActor::TEXT_SCALE_MODE_NONE);
		vtkTextProperty * yText = newActor->GetYAxisCaptionActor2D()->GetTextActor()->GetTextProperty();
		yText->SetFontSize(16);

		newActor->GetZAxisCaptionActor2D()->GetTextActor()->SetTextScaleMode(vtkTextActor::TEXT_SCALE_MODE_NONE);
		vtkTextProperty * zText = newActor->GetZAxisCaptionActor2D()->GetTextActor()->GetTextProperty();
		zText->SetFontSize(16);

		// Create a marker widget for the current subcanvas
		vtkOrientationMarkerWidget * newMarker = vtkOrientationMarkerWidget::New();
		newMarker->SetInteractor(currentSubCanvas->GetInteractor());
		newMarker->SetCurrentRenderer(currentSubCanvas->GetRenderer());

		// Create an initial viewport (size = 0.2, location = bottom left)
		double * viewPort;
		viewPort = currentSubCanvas->GetViewport();
		double vpWidth  = ((viewPort[2] - viewPort[0]) < 0.2) ? (viewPort[2] - viewPort[0]) : (0.2);
		double vpHeight = ((viewPort[3] - viewPort[1]) < 0.2) ? (viewPort[3] - viewPort[1]) : (0.2);

		newMarker->SetViewport(	viewPort[0], 
								viewPort[1],
								viewPort[0] + vpWidth,
								viewPort[1] + vpHeight);

		// Turn the widget on, but disable interaction (resizing and moving)
		newMarker->SetOrientationMarker(newActor);
		newMarker->SetEnabled(1);
		newMarker->InteractiveOff();

		// Create a struct for the information about these axes
		AxesInfo newAxesInfo;

		// Set default values
		newAxesInfo.actor				= newActor;
		newAxesInfo.marker				= newMarker;
		newAxesInfo.subcanvas			= currentSubCanvas;
		newAxesInfo.isVisible			= true;
		newAxesInfo.size				= 0.2;
		newAxesInfo.pos					= MPOS_BL;
		newAxesInfo.applyTransformation = false;
		newAxesInfo.matrixIndex			= 0;

		// Add the information struct to the list
		this->infoList.append(newAxesInfo);
	}

	// Copy the default settings to the GUI
	this->settingsToGUI();

	// Create the callback class
	this->callBack = AxesCallback::New();
	this->callBack->plugin = this;

	// Add the callback to the canvas as an observer
	this->fullCore()->canvas()->AddObserver(vtkCommand::UserEvent + 
		BMIA_USER_EVENT_SUBCANVASSES_RESIZED, this->callBack);

	// Connect the GUI elements
	connect(this->ui->positionTLButton,			SIGNAL(clicked()),					this, SLOT(setPosToTL()));
	connect(this->ui->positionTRButton,			SIGNAL(clicked()),					this, SLOT(setPosToTR()));
	connect(this->ui->positionBRButton,			SIGNAL(clicked()),					this, SLOT(setPosToBR()));
	connect(this->ui->positionBLButton,			SIGNAL(clicked()),					this, SLOT(setPosToBL()));
	connect(this->ui->positionCButton,			SIGNAL(clicked()),					this, SLOT(setPosToC() ));
	connect(this->ui->sizeSlide,				SIGNAL(valueChanged(int)),			this, SLOT(changeSize(int)));
	connect(this->ui->showCheck,				SIGNAL(toggled(bool)),				this, SLOT(changeVisibility(bool)));
	connect(this->ui->subCanvasCombo,			SIGNAL(currentIndexChanged(int)),	this, SLOT(settingsToGUI()));
	connect(this->ui->showAllButton,			SIGNAL(clicked()),					this, SLOT(showAll()));
	connect(this->ui->hideAllButton,			SIGNAL(clicked()),					this, SLOT(hideAll()));
	connect(this->ui->transformationMatrixCombo,SIGNAL(currentIndexChanged(int)),	this, SLOT(setTransformationMatrix()));
	connect(this->ui->transformEnableCheck,		SIGNAL(toggled(bool)),				this, SLOT(setApplyTransformation(bool)));
}


AxesVisualizationPlugin::~AxesVisualizationPlugin()
{
	// Delete the callback
	if (this->callBack)
	{
		this->callBack->Delete();
	}

	// Remove the GUI
	if (this->qWidget)
	{
		delete this->qWidget; 
	}

	// Delete all matrices
	for (int i = 0; i < this->uniqueMatrices.size(); ++i)
	{
		this->uniqueMatrices.at(i)->Delete();
	}

	// Clear the list of matrices
	this->uniqueMatrices.clear();

	// Clear the list of data sets
	this->matrixDataSets.clear();

	// Delete all widgets
	for (int i = 0; i < this->infoList.size(); ++i)
	{
		AxesInfo currentAxesInfo = this->infoList.at(i);
		currentAxesInfo.marker->EnabledOff();
		currentAxesInfo.marker->Delete();
	}

	// Clear the axes information list
	this->infoList.clear();
}


QWidget * AxesVisualizationPlugin::getGUI()
{
	return this->qWidget;
}


void AxesVisualizationPlugin::subcanvassesResized()
{
	// Do nothing if no axes widgets exist
	if (this->infoList.size() == 0)
		return;

	// Get the collection of subcanvasses
	vtkMedicalCanvas * canvas = this->fullCore()->canvas();
	vtkAbstractSubCanvasCollection * subcanvasses = canvas->GetSubCanvasses();

	// Loop through all subcanvasses
	for (int i = 0; i < subcanvasses->GetNumberOfItems(); ++i)
	{
		vtkAbstractSubCanvas * currentSubCanvas = subcanvasses->GetItem(i);

		std::string subCanvasName = currentSubCanvas->subCanvasName;

		// We're currently only interested in the 3D view
		if (subCanvasName != "3D View")
			continue;

		// Since we're currently only working with one set of axes, we can just
		// get the first item of the list, which will be the axes for the 3D view

		AxesInfo currentAxesInfo = this->infoList[0];

		// Update the 3D view axes
		this->updateMarker(currentAxesInfo);
	}
}


void AxesVisualizationPlugin::updateMarker(AxesInfo info)
{
	// Get the current viewport
	double * viewPort;
	viewPort = info.subcanvas->GetViewport();

	// If the viewport has non-zero area, configure the axes
	if (((viewPort[2] - viewPort[0]) > 0.0) && ((viewPort[3] - viewPort[1]) > 0.0))
	{
		// Show or hide the axes
		info.actor->SetVisibility(info.isVisible);

		// Compute a new viewport based on the size and position of the axes
		double newViewPort[4];
		this->computeViewPort(viewPort, newViewPort, info);
		info.marker->SetViewport(newViewPort[0], newViewPort[1], newViewPort[2], newViewPort[3]);

		// If the "Apply transformation" option has been checked, AND the current
		// matrix index is within range, AND there's at least one available transformation
		// widget, we apply a transformation to the axes

		if (	info.applyTransformation && info.matrixIndex >= 0 && 
				info.matrixIndex < this->ui->transformationMatrixCombo->count() &&
				this->ui->transformationMatrixCombo->count() > 0)
		{
			// Get the current transformation matrix
			vtkMatrix4x4 * currentMatrix = this->uniqueMatrices.at(info.matrixIndex);

			// Create an identity matrix
			vtkMatrix4x4 * simplifiedMatrix = vtkMatrix4x4::New();
			simplifiedMatrix->Identity();

			// Transform and normalize the three orthogonal unit vectors
			double ix[4] = {1.0, 0.0, 0.0, 0.0};
			currentMatrix->MultiplyPoint(ix, ix);
			vtkMath::Normalize(ix);

			double iy[4] = {0.0, 1.0, 0.0, 0.0};
			currentMatrix->MultiplyPoint(iy, iy);
			vtkMath::Normalize(iy);

			double iz[4] = {0.0, 0.0, 1.0, 0.0};
			currentMatrix->MultiplyPoint(iz, iz);
			vtkMath::Normalize(iz);

			// Use the transformed and normalized vectors to define a new matrix.
			// This prevents highly anisotropic scaling from messing up the axes.
			
			simplifiedMatrix->Element[0][0] = ix[0];	simplifiedMatrix->Element[0][1] = iy[0];	simplifiedMatrix->Element[0][2] = iz[0];
			simplifiedMatrix->Element[1][0] = ix[1];	simplifiedMatrix->Element[1][1] = iy[1];	simplifiedMatrix->Element[1][2] = iz[1];
			simplifiedMatrix->Element[2][0] = ix[2];	simplifiedMatrix->Element[2][1] = iy[2];	simplifiedMatrix->Element[2][2] = iz[2];

			// Set the new matrix
			info.actor->SetUserMatrix(simplifiedMatrix);
			simplifiedMatrix->Delete();

		} // if [apply transformation matrix]

		else
		{
			// Otherwise, simply create and assign an identity matrix
			vtkMatrix4x4 * id = vtkMatrix4x4::New();
			id->Identity();
			info.actor->SetUserMatrix(id);
			id->Delete();
		}

	} // if [viewport is visible]

	// If the viewport has zero area, simply hide the axes
	else
	{
		info.actor->VisibilityOff();
	}

	// Render the screen
	this->core()->render();
}


void AxesVisualizationPlugin::computeViewPort(double * mainViewPort, double * outViewPort, AxesInfo info)
{
	// Use the desired size, unless it is larger than viewport size
	double vpWidth  = ((mainViewPort[2] - mainViewPort[0]) < info.size) ? (mainViewPort[2] - mainViewPort[0]) : (info.size);
	double vpHeight = ((mainViewPort[3] - mainViewPort[1]) < info.size) ? (mainViewPort[3] - mainViewPort[1]) : (info.size);

	// Axes position
	switch(info.pos)
	{
		// Bottom Left
		case MPOS_BL:
			outViewPort[0] = mainViewPort[0];
			outViewPort[1] = mainViewPort[1];
			outViewPort[2] = mainViewPort[0] + vpWidth;
			outViewPort[3] = mainViewPort[1] + vpHeight;	
			break;

		// Top Left
		case MPOS_TL:
			outViewPort[0] = mainViewPort[0];
			outViewPort[1] = mainViewPort[3] - vpHeight;
			outViewPort[2] = mainViewPort[0] + vpWidth;
			outViewPort[3] = mainViewPort[3];
			break;

		// Top Right
		case MPOS_TR:
			outViewPort[0] = mainViewPort[2] - vpWidth;
			outViewPort[1] = mainViewPort[3] - vpHeight;
			outViewPort[2] = mainViewPort[2];
			outViewPort[3] = mainViewPort[3];
			break;

		// Bottom Right
		case MPOS_BR:
			outViewPort[0] = mainViewPort[2] - vpWidth;
			outViewPort[1] = mainViewPort[1];
			outViewPort[2] = mainViewPort[2];
			outViewPort[3] = mainViewPort[1] + vpHeight;	
			break;

		// Center
		case MPOS_C:
			outViewPort[0] = ((mainViewPort[2] - mainViewPort[0]) / 2.0) - (vpWidth / 2.0)  + mainViewPort[0];
			outViewPort[1] = ((mainViewPort[3] - mainViewPort[1]) / 2.0) - (vpHeight / 2.0) + mainViewPort[1];
			outViewPort[2] = ((mainViewPort[2] - mainViewPort[0]) / 2.0) + (vpWidth / 2.0)  + mainViewPort[0];
			outViewPort[3] = ((mainViewPort[3] - mainViewPort[1]) / 2.0) + (vpHeight / 2.0) + mainViewPort[1];
			break;

		// This should never happen
		default:
			outViewPort[0] = 0.0;
			outViewPort[1] = 0.0;
			outViewPort[2] = 0.0;
			outViewPort[3] = 0.0;
			break;
	}
}


void AxesVisualizationPlugin::setPosToTL()
{
	// Get the current marker information, and change its position
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].pos = MPOS_TL;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::setPosToTR()
{
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].pos = MPOS_TR;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::setPosToBR()
{
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].pos = MPOS_BR;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::setPosToBL()
{
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].pos = MPOS_BL;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::setPosToC()
{
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].pos = MPOS_C;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::changeSize(int newSize)
{
	// Get the current marker information, and change its size
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].size = (double) newSize / 100.0;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::changeVisibility(bool show)
{
	// Get the current marker information, and change its visibility
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].isVisible = show;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::setTransformationMatrix()
{
	// Get the current marker information, and change its transformation matrix
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].matrixIndex = this->ui->transformationMatrixCombo->currentIndex();
	this->updateMarker(this->infoList[markerId]);

	// Update the screen a second time. This is nasty, but updating only once does
	// not seem to work correctly, the axes change size suddenly when the user
	// moves the camera.

	this->core()->render();
}


void AxesVisualizationPlugin::setApplyTransformation(bool apply)
{
	// Get the current marker information, and set whether or not transform it
	int markerId = this->ui->subCanvasCombo->currentIndex();
	this->infoList[markerId].applyTransformation = apply;
	this->updateMarker(this->infoList[markerId]);
}


void AxesVisualizationPlugin::settingsToGUI()
{
	// Get the information of the current marker
	int markerId = this->ui->subCanvasCombo->currentIndex();
	AxesInfo info = this->infoList[markerId];

	// Set the GUI widget values based on the information struct values
	this->ui->showCheck->setChecked(info.isVisible);
	this->ui->sizeSlide->setValue((int) (info.size * 100.0));
	this->ui->sizeSpin->setValue((int) (info.size * 100.0));
	this->ui->transformEnableCheck->setChecked(info.applyTransformation);
	this->ui->transformationMatrixCombo->setCurrentIndex(info.matrixIndex);
	this->ui->transformationMatrixCombo->setEnabled(info.applyTransformation);
	this->ui->transformationMatrixLabel->setEnabled(info.applyTransformation);
}


void AxesVisualizationPlugin::showAll()
{
	// Loop through the list
	for (int i = 0; i < this->infoList.size(); ++i)
	{
		// Turn visibility on
		this->infoList[i].isVisible = true;
		this->updateMarker(this->infoList[i]);
	}

	this->settingsToGUI();
}


void AxesVisualizationPlugin::hideAll()
{
	for (int i = 0; i < this->infoList.size(); ++i)
	{
		this->infoList[i].isVisible = false;
		this->updateMarker(this->infoList[i]);
	}

	this->settingsToGUI();
}


void AxesVisualizationPlugin::dataSetAdded(bmia::data::DataSet * ds)
{
	// Check if the data set contains a transformation matrix
	vtkObject * obj;
	if (!(ds->getAttributes()->getAttribute("transformation matrix", obj)))
		return;

	vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);

	if (!m)
		return;

	// Check if the matrix is unique
	if (!(this->isMatrixUnique(m)))
		return;

	// Make a copy of the matrix
	vtkMatrix4x4 * mCopy = vtkMatrix4x4::New();
	mCopy->DeepCopy(m);

	// Store the data set pointer and the matrix
	this->uniqueMatrices.append(mCopy);
	this->matrixDataSets.append(ds);

	// Add the matrix to the GUI
	this->ui->transformationMatrixCombo->addItem(ds->getName());

	// If this is the first matrix, initialize the transformations
	if (this->ui->transformationMatrixCombo->count() == 1)
	{
		this->initializeMatrices();
	}
}


void AxesVisualizationPlugin::dataSetChanged(bmia::data::DataSet * ds)
{
	if (!(this->matrixDataSets.contains(ds)))
	{
		// Data set previously did not contain a unique matrix; check if the
		// matrix (which may have changed) is now unique, and if so, add it to
		// the GUI.

		this->dataSetAdded(ds);
		return;
	}

	// If the data set no longer contains a valid matrix, act as though it's been removed
	vtkObject * obj;
	if (!(ds->getAttributes()->getAttribute("transformation matrix", obj)))
	{
		this->dataSetRemoved(ds);
		return;
	}

	vtkMatrix4x4 * m = vtkMatrix4x4::SafeDownCast(obj);

	if (!m)
	{
		this->dataSetRemoved(ds);
		return;
	}

	int dsIndex = this->matrixDataSets.indexOf(ds);

	// Update the name
	this->ui->transformationMatrixCombo->setItemText(dsIndex, ds->getName());

	vtkMatrix4x4 * oldMatrix = this->uniqueMatrices.at(dsIndex);

	// Matrix hasn't changed, so we're done here
	if (this->areMatricesEqual(oldMatrix, m))
		return;

	// Update the matrix
	oldMatrix->DeepCopy(m);

	// Update all markers that use this matrix
	for (int i = 0; i < this->infoList.size(); ++i)
	{
		AxesInfo info = this->infoList.at(i);

		if (info.matrixIndex == dsIndex)
		{
			this->updateMarker(info);
		}
	}
}


void AxesVisualizationPlugin::dataSetRemoved(bmia::data::DataSet * ds)
{
	// Check if the data set has in the past produced a unique transformation matrix
	int dsIndex = this->matrixDataSets.indexOf(ds);

	disconnect(this->ui->transformationMatrixCombo,	SIGNAL(currentIndexChanged(int)), this, SLOT(setTransformationMatrix()));

	// Remove the matrix from the GUI
	this->ui->transformationMatrixCombo->removeItem(dsIndex);

	connect(this->ui->transformationMatrixCombo,	SIGNAL(currentIndexChanged(int)), this, SLOT(setTransformationMatrix()));

	// Remove the matrix and the data set pointer from their respective lists
	this->uniqueMatrices.removeAt(dsIndex);
	this->matrixDataSets.removeAt(dsIndex);

	// Loop through all markers
	for (int i = 0; i < this->infoList.size(); ++i)
	{
		AxesInfo & info = this->infoList[i];

		// If the marker used the deleted matrix, turn off transformations
		if (info.matrixIndex == dsIndex)
		{
			info.applyTransformation = false;
			info.matrixIndex = 0;
			this->updateMarker(info);

			if (i == this->ui->subCanvasCombo->currentIndex())
				this->settingsToGUI();
		}
		// Otherwise, decrement the matrix index if necessary
		else if (info.matrixIndex > dsIndex)
		{
			info.matrixIndex--;

			if (i == this->ui->subCanvasCombo->currentIndex())
				this->settingsToGUI();
		}
	}
}


bool AxesVisualizationPlugin::isMatrixUnique(vtkMatrix4x4 * m)
{
	// Check if input matrix exists
	for (int i = 0; i < this->uniqueMatrices.size(); ++i)
	{
		vtkMatrix4x4 * currentMatrix = this->uniqueMatrices.at(i);

		if (this->areMatricesEqual(currentMatrix, m))
			return false;
	}

	return true;
}


bool AxesVisualizationPlugin::areMatricesEqual(vtkMatrix4x4 * a, vtkMatrix4x4 * b)
{
	// Compare all relevant elements
	if (	a->Element[0][0] == b->Element[0][0]	&& 
			a->Element[0][1] == b->Element[0][1]	&& 
			a->Element[0][2] == b->Element[0][2]	&& 
			a->Element[0][3] == b->Element[0][3]	&& 
			a->Element[1][0] == b->Element[1][0]	&& 
			a->Element[1][1] == b->Element[1][1]	&& 
			a->Element[1][2] == b->Element[1][2]	&& 
			a->Element[1][3] == b->Element[1][3]	&& 
			a->Element[2][0] == b->Element[2][0]	&& 
			a->Element[2][1] == b->Element[2][1]	&& 
			a->Element[2][2] == b->Element[2][2]	&& 
			a->Element[2][3] == b->Element[2][3]	)
		return true;

	return false;
}


void AxesVisualizationPlugin::initializeMatrices()
{
	// Apply first matrix to all active markers
	for (int i = 0; i < this->infoList.size(); ++i)
	{
		AxesInfo & info = this->infoList[i];

		info.applyTransformation = true;
		info.matrixIndex = 0;
		this->updateMarker(info);

		if (i == this->ui->subCanvasCombo->currentIndex())
			this->settingsToGUI();
	}
}


void AxesVisualizationPlugin::AxesCallback::Execute(vtkObject * caller, unsigned long event, void * callData)
{
	// If the subcanvasses have been resized, update the axes markers
	if (event == vtkCommand::UserEvent + BMIA_USER_EVENT_SUBCANVASSES_RESIZED)
	{
		this->plugin->subcanvassesResized();
	}
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libbmia_AxesVisualizationPlugin, bmia::AxesVisualizationPlugin)
