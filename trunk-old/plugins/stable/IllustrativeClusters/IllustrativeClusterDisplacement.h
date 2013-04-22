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