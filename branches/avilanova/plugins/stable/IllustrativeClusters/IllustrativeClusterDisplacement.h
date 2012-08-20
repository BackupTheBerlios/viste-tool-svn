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


#ifndef bmia_IllustrativeClusterDisplacement_h
#define bmia_IllustrativeClusterDisplacement_h


/** Includes - C++ */

#include <map>

/** Includes - VTK */

#include <vtkSmartPointer.h>
#include <vtkBoxWidget.h>
#include <vtkCommand.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkCamera.h>

/** Includes - Custom Files */

#include "Math/Vector3.h"
#include "Math/ConvexHull.h"
#include "Math/OrientedBoundingBox.h"
#include "gui/MetaCanvas/vtkMedicalCanvas.h"
#include "gui/MetaCanvas/vtkSubCanvas.h"


using ICMath::Vector3;
using ICMath::ConvexHull;
using ICMath::OrientedBoundingBox;


namespace bmia
{

class IllustrativeClusterDisplacement
{
	public:
	
		IllustrativeClusterDisplacement(vtkMedicalCanvas * rCanvas);
		virtual ~IllustrativeClusterDisplacement();

		void addActor(vtkActor* actor);
		void removeActor(vtkActor* actor);
		void removeAllActors();

		void updateInput();
		void updateAnimation(int msec); // Must be called from a QTimer event in the UI, as VTK's timers are a complete mess

		void enableFocusSelectionWidget(bool enable);
		void setFocusToSelection();
		void setFocusToCurrentCluster(vtkPolyData * pd);

		inline void setScales(float explosion, float slide)
		{
			mExplosionScale = explosion;
			mSlideScale = slide;
		}

		inline void setIsActive(bool rActive)
		{
			isActive = rActive;
		}

	protected:
	
		class CameraCallback : public vtkCommand
		{
			public:
			
				static CameraCallback * New() 
				{ 
					return new CameraCallback; 
				};

				CameraCallback() 
				{

				}

				virtual void Execute(vtkObject * caller, unsigned long eventId, void * arguments)
				{
					if (mOwner != NULL) 
						mOwner->updateInput();
				}

				inline void setOwner(IllustrativeClusterDisplacement * owner) 
				{ 
					mOwner = owner; 
				}

			private:
		
				IllustrativeClusterDisplacement * mOwner;
	
		}; // class CameraCallback

	
		class DisplacedActor
		{
			public:
		
				vtkActor *           actor;
		
				OrientedBoundingBox boundingBox;

				Vector3 originalPosition;
				Vector3 currentPosition;
				Vector3 targetPosition;	

		}; // class DisplacedActor

		typedef std::map<vtkActor *, DisplacedActor *> DisplacedActorMap;

	private:
	
		void updateDisplacement(DisplacedActor * displacedActor, OrientedBoundingBox focusBox, ConvexHull focusHull, Vector3 cameraAxes[3]);
		void animateDisplacement(DisplacedActor * displacedActor, int msec, float distancePerMsec);

		CameraCallback * mCameraCallback;

		double mExplosionScale;
		double mSlideScale;

		bool mWidgetPlaced;
		vtkSmartPointer<vtkBoxWidget> mFocusWidget;	

		bool mFocusDefined;
		OrientedBoundingBox mFocusBox;

		DisplacedActorMap mDisplacedActors;

		vtkMedicalCanvas * canvas;

		bool actorsHaveBeenReset;

		bool isActive;

}; // class IllustrativeClusterDisplacement


} // namespace bmia


#endif // bmia_IllustrativeClusterDisplacement_h