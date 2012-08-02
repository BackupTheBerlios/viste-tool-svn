/*
 * IllustrativeClusterDisplacement.h
 *
 * 2009-11-25	Ron Otten
 * - First version.
 * 
 * 2011-03-25	Evert van Aart
 * - Ported to DTITool3.
 *
 */


#include "IllustrativeClusterDisplacement.h"


namespace bmia
{

IllustrativeClusterDisplacement::IllustrativeClusterDisplacement(vtkMedicalCanvas * rCanvas) : 
							mWidgetPlaced(false), 
							mFocusDefined(false),
							mExplosionScale(1), 
							mSlideScale(1)
{
	this->canvas = rCanvas;

	this->mFocusWidget = vtkSmartPointer<vtkBoxWidget>::New();
	this->mFocusWidget->SetInteractor(this->canvas->GetInteractor());
	this->mFocusWidget->SetDefaultRenderer(this->canvas->GetSubCanvas3D()->GetRenderer());
	this->mFocusWidget->Off();

	vtkCamera * camera = this->canvas->GetSubCanvas3D()->GetRenderer()->GetActiveCamera();
	
	this->mCameraCallback = CameraCallback::New();
	this->mCameraCallback->setOwner(this);		
	
	camera->AddObserver(vtkCommand::ModifiedEvent, mCameraCallback);

	this->actorsHaveBeenReset = true;
	this->isActive = true;
}


IllustrativeClusterDisplacement::~IllustrativeClusterDisplacement()
{
	vtkCamera * camera = this->canvas->GetSubCanvas3D()->GetRenderer()->GetActiveCamera();

	camera->RemoveObserver(this->mCameraCallback);		
	this->mCameraCallback->Delete(); 
	this->mCameraCallback = NULL;

	this->removeAllActors();
}


void IllustrativeClusterDisplacement::enableFocusSelectionWidget(bool enable)
{
	if (enable)
	{
		// Provide a sane initial placement and size
		if (!(this->mWidgetPlaced))
		{
			this->mWidgetPlaced = true;

			vtkCamera * camera = this->canvas->GetRenderer3D()->GetActiveCamera();
			double * focalPoint = camera->GetFocalPoint();
			double radius = 0.25 * camera->GetDistance();

			this->mFocusWidget->PlaceWidget(
				focalPoint[0] - radius, focalPoint[0] + radius,
				focalPoint[1] - radius, focalPoint[1] + radius,
				focalPoint[2] - radius, focalPoint[2] + radius
			);
		}

		this->mFocusWidget->On();
	}
	else
		this->mFocusWidget->Off();
}


void IllustrativeClusterDisplacement::setFocusToSelection()
{
	// Can't select anything when the widget isn't placed yet
	if (!(this->mWidgetPlaced)) 
		return;

	vtkSmartPointer<vtkPolyData> widgetData = vtkSmartPointer<vtkPolyData>::New();
	this->mFocusWidget->GetPolyData(widgetData);

	/* From the vtkBoxWidget::getPolyData API documentation:
	 *
	 * The polydata consists of 6 quadrilateral faces and 15 points. The first eight points define
	 * the eight corner vertices; the next six define the -x,+x, -y,+y, -z,+z face points; and the
	 * final point (the 15th out of 15 points) defines the center of the hexahedron.
	 */
	  
	vtkPoints * widgetPoints = widgetData->GetPoints();

	Vector3 obbAxes[3];
	Vector3 obbCenter(widgetPoints->GetPoint(14));

	obbAxes[0] = (Vector3(widgetPoints->GetPoint(9)) - obbCenter);
	obbAxes[1] = (Vector3(widgetPoints->GetPoint(11)) - obbCenter);
	obbAxes[2] = (Vector3(widgetPoints->GetPoint(13)) - obbCenter);

	Vector3 obbMaxExtents(
		obbAxes[0].normalize(), obbAxes[1].normalize(), obbAxes[2].normalize()
		);

	Vector3 obbMinExtents(-obbMaxExtents);
	
	this->mFocusBox = OrientedBoundingBox(obbCenter, obbAxes, obbMinExtents, obbMaxExtents);
	this->mFocusDefined = true;		
}

void IllustrativeClusterDisplacement::setFocusToCurrentCluster(vtkPolyData * pd)
{
	if (pd != NULL)		
	{
		this->mFocusBox = OrientedBoundingBox(pd->GetPoints());
		this->mFocusDefined = true;
	}
}

void IllustrativeClusterDisplacement::updateInput()
{
	if (!(this->isActive))
	{
		if (!(this->actorsHaveBeenReset))
		{
			DisplacedActorMap::iterator itActors = this->mDisplacedActors.begin();
			for(; itActors != this->mDisplacedActors.end(); ++itActors)
			{
				itActors->second->targetPosition = itActors->second->originalPosition;				
			}			

			this->actorsHaveBeenReset = true;
		}

		return;
	}

	// Makes no sense when we haven't pointed out a focus region
	if (!(this->mFocusDefined)) 
		return;

	vtkCamera * camera = this->canvas->GetSubCanvas3D()->GetRenderer()->GetActiveCamera();

	Vector3 cameraAxes[3];
	cameraAxes[2] = Vector3(camera->GetDirectionOfProjection()).normalizedCopy();
	cameraAxes[1] = Vector3(camera->GetViewUp()).normalizedCopy();
	cameraAxes[0] = cameraAxes[1].crossProduct(cameraAxes[2]).normalizedCopy();

	ConvexHull focusHull(mFocusBox, cameraAxes[2], cameraAxes[1]);

	DisplacedActorMap::iterator itActors = this->mDisplacedActors.begin();

	for(; itActors != this->mDisplacedActors.end(); ++itActors)
	{
		this->updateDisplacement(itActors->second, this->mFocusBox, focusHull, cameraAxes);	

		// Provide a small bit of actual animation here, as the QTimer that drives
		// the animation is 'locked' while the user rotates VTK's camera -- or performs
		// any action that the VTK interactor responds to using its timers, for that
		// matter. Bah!		
		// Explicitly do /not/ call updateAnimation, as that would cause an additional
		// frame to be rendered, leading to stutter.

		this->animateDisplacement(itActors->second, 10, 0.1);
	}		

	this->actorsHaveBeenReset = false;
}


void IllustrativeClusterDisplacement::updateAnimation(int msec)
{
	DisplacedActorMap::iterator itActors = this->mDisplacedActors.begin();

	for(; itActors != this->mDisplacedActors.end(); ++itActors)
	{	
		this->animateDisplacement(itActors->second, msec, 0.1);
	}

	// Don't bother re-rendering if nothing could be animated
	if (this->mDisplacedActors.size() > 0)
	{
		this->canvas->GetSubCanvas3D()->GetRenderer()->GetRenderWindow()->Render();
	}
}


void IllustrativeClusterDisplacement::addActor(vtkActor * actor)
{
	DisplacedActorMap::const_iterator itActors = this->mDisplacedActors.find(actor);		
	if (itActors != this->mDisplacedActors.end()) 
		return;

	DisplacedActor * dispActor = new DisplacedActor();
	dispActor->actor = actor;
	dispActor->boundingBox = OrientedBoundingBox(actor);

	dispActor->originalPosition = Vector3(actor->GetPosition());
	dispActor->currentPosition = dispActor->originalPosition;
	dispActor->targetPosition = dispActor->originalPosition;

	this->mDisplacedActors.insert(DisplacedActorMap::value_type(actor, dispActor));
}

void IllustrativeClusterDisplacement::removeActor(vtkActor* actor)
{
	DisplacedActorMap::iterator itActors = this->mDisplacedActors.find(actor);
	if (itActors == this->mDisplacedActors.end()) return;

	delete itActors->second;
	this->mDisplacedActors.erase(itActors);
}

void IllustrativeClusterDisplacement::removeAllActors()
{
	DisplacedActorMap::iterator itActors = this->mDisplacedActors.begin();
	for(; itActors != this->mDisplacedActors.end(); ++itActors)
	{
		delete itActors->second;
	}

	this->mDisplacedActors.clear();
}

void IllustrativeClusterDisplacement::updateDisplacement(DisplacedActor * displacedActor, OrientedBoundingBox focusBox, ConvexHull focusHull, Vector3 cameraAxes[3])
{
	Vector3 centerFocusBox = focusBox.getCenter();
	Vector3 centerFocusHull = focusHull.getCenter();
		
	OrientedBoundingBox contextBox = displacedActor->boundingBox;
	Vector3 centerContextBox = contextBox.getCenter();

	// Reset the target position
	displacedActor->targetPosition = displacedActor->originalPosition;

	double worldScale = this->mExplosionScale;
	double viewScale = this->mSlideScale;
	Vector3 dirPenetration = (centerContextBox - centerFocusBox).normalizedCopy();

//	if (dirPenetration.isZeroLength()) dirPenetration = Vector3::UNIT_X;
	if (dirPenetration.isZeroLength())
	{
		displacedActor->targetPosition = displacedActor->originalPosition;
		return;
	}

	double dp = cameraAxes[2].dotProduct(dirPenetration);

	if (dp > 0)
	{
		worldScale = (1.0 - dp) * this->mExplosionScale;
		viewScale = 0;

		dirPenetration = cameraAxes[2];
	}

	// WORLD SPACE EXPLOSION
	if (worldScale > 0)
	{
		double amtPentration =
			contextBox.findPenetrationAlong(focusBox, dirPenetration);

		Vector3 offset = worldScale * amtPentration * dirPenetration;

		displacedActor->targetPosition = displacedActor->originalPosition + offset;			
		contextBox.setPosition(displacedActor->targetPosition);
	}				
	
	// VIEW SPACE SLIDING
	if (viewScale > 0)
	{
		// Project the oriented bounding box into a screen parallel 2D convex hull
		ConvexHull contextHull(contextBox, cameraAxes[2], cameraAxes[1]);
		Vector3 centerContextHull = contextHull.getCenter();

		Vector3 projDirPenetration = (centerContextHull - centerFocusHull).normalizedCopy();
		if (projDirPenetration.isZeroLength()) projDirPenetration = cameraAxes[0];

		double amtPenetration =
			contextHull.findPenetrationAlong(focusHull, projDirPenetration);	
					
		Vector3 offset = viewScale * amtPenetration * projDirPenetration;

		// Go back from the view aligned plane to world space
		offset =
			offset.x * cameraAxes[0] +
			offset.y * cameraAxes[1] +
			offset.z * cameraAxes[2];

		displacedActor->targetPosition = displacedActor->targetPosition + offset;
	}			
}

void IllustrativeClusterDisplacement::animateDisplacement(DisplacedActor * displacedActor, int msec, float distancePerMsec)
{
	Vector3 dir = displacedActor->targetPosition - displacedActor->currentPosition;

	double requiredLength = dir.normalize();
	double givenLength    = distancePerMsec * msec;

	dir *= std::min<double>(requiredLength, givenLength);

	displacedActor->currentPosition += dir;			
	displacedActor->actor->SetPosition(displacedActor->currentPosition.ptr());		
}
}
