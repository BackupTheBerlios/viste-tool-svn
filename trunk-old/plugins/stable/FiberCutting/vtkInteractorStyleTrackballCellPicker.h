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

#ifndef vtkInteractorStyleTrackballCellPicker_h
#define vtkInteractorStyleTrackballCellPicker_h

/** Includes - Standard C++ */
#include <iostream>

/** Includes - VTK */
#include <vtkSmartPointer.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRendererCollection.h>

#include <vtkPointPicker.h>
#include <vtkCellPicker.h>
#include <vtkWorldPointPicker.h>
#include <vtkPropPicker.h>
#include <vtkPropCollection.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkProperty.h>

/** Includes - Qt */
#include <QList>

#include "data/DataSet.h"

namespace bmia{

/** 
This class allows the user to select one single fiber from all bundles of "fiber" type datasets.
The selected fiber would be highlighted.Continue picking would modify the last time's selection.
Note: Set up renderer and comparedDataSets brfore performing. And once a bundle of fibers is selected
users are constrained on this bundle of fibers.
*/
class vtkInteractorStyleTrackballCellPicker : public vtkInteractorStyleTrackballCamera
{

	public:
		/** 
		Constructor 
		*/
		vtkInteractorStyleTrackballCellPicker();
		
		/** 
		Destructor 
		*/
		~vtkInteractorStyleTrackballCellPicker();
		
		/**
		Constructor Call
		*/
		static vtkInteractorStyleTrackballCellPicker* New();
		
		/** 
		VTK "Type" macros. 
		*/
		vtkTypeMacro(vtkInteractorStyleTrackballCellPicker, vtkInteractorStyleTrackballCamera);

		/**
		Set up renderer
		*/
		void SetRenderProcess(vtkRenderer * renderer);

		/**
		Set up the ComparedDataSets which contains all the "fiber" type datasets.
		*/
		void SetComparedDataSets(QList<data::DataSet*> comaredDataSets);

		/**
		Set up the index of the already picked dataset
		*/
		void SetHasPickedDataSet_Index(int index);

		/**
		Rewrite the OnLeftButtonDown function inherited 
		*/
		virtual void OnLeftButtonDown(); 
		
		/**
		The vtkRenderer
		*/
		vtkRenderer			* Renderer;

		/**
		vtkPolyData for the picked fiber 
		*/
		vtkPolyData			* PickedFiber_PolyData;

		/**
		vtkPolyDataMapper for the picked fiber
		*/
		vtkPolyDataMapper	* PickedFiber_Mapper;

		/**
		vtkActor for the picked fiber
		*/
		vtkActor			* PickedFiber_Actor;

		/**
		vtkPolyData for the picked dataset
		*/
		vtkPolyData			* PickedDataSet_PolyData;
		
		/**
		QList contains for the loaded "fiber" type datasets
		*/
		QList<data::DataSet*> ComparedDataSets;
		
		/**
		Flag of the fiber selection action
		*/
		int HasGotFiber;
		
		/**
		A record of the index for the current picked dataset
		*/
		int PickedDataSet_Index;

		/**
		A record of the index for the picked fiber
		*/
		int PickedFiber_Index;

		/**
		The index of the dataset which selection are working on
		*/
		int HasPickedDataSet_Index;
};

}//namespace bmia
#endif 