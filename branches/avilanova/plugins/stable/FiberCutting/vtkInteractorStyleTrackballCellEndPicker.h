#ifndef vtkInteractorStyleTrackballPointPicker_h
#define vtkInteractorStyleTrackballPointPicker_h


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

//#include <vtkCubeSource.h>
#include <vtkSphereSource.h>
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
#include <QMessageBox>

#include "data/DataSet.h"


namespace bmia{

/** 
This class allows the user to select one single fiber from all bundles of "fiber" type datasets.
The selected fiber would be highlighted.Continue picking would modify the last time's selection.
Note: Set up renderer and comparedDataSets brfore performing. And once a bundle of fibers is selected
users are constrained on this bundle of fibers.
*/

class vtkInteractorStyleTrackballCellEndPicker : public vtkInteractorStyleTrackballCamera
{


	public:

		/** 
		Constructor 
		*/
		vtkInteractorStyleTrackballCellEndPicker();

		/** 
		Destructor 
		*/
		~vtkInteractorStyleTrackballCellEndPicker();
		
		/**
		Constructor Call
		*/
		static vtkInteractorStyleTrackballCellEndPicker* New();
		
		/** 
		VTK "Type" macros. 
		*/
		vtkTypeMacro(vtkInteractorStyleTrackballCellEndPicker, vtkInteractorStyleTrackballCamera);

		/**
		Set up renderer
		*/
		void SetRender(vtkRenderer * renderer);

		/**
		Set up GUI
		*/
		void SetGUI(QWidget * gui);

		/**
		Set up the ComparedDataSets which contains all the "fiber" type datasets.
		*/
		void SetComparedDataSets(QList<data::DataSet*> comaredDataSets);

		/**
		Set up the index of the picked dataset
		*/
		void SetPickedDataSet_Index(int index);

		/**
		Set up the index of the picked fiber in the picked dataset
		*/
		void SetPickedFiber_Index(int index);

		/**
		Rewrite the OnLeftButtonDown function inherited 
		*/
		virtual void OnLeftButtonDown(); 
		
		/**
		Automatic mark the first endpoint
		*/
		void AutoEnd1();
		
		/**
		Automatic mark the second endpoint
		*/
		void AutoEnd2();

		/**
		GUI
		*/
		QWidget * GUI;
		
		/**
		vtkRenderer
		*/
		vtkRenderer			* Renderer;
		
		/**
		vtkPolyData for picked dataset
		*/
		vtkPolyData			* PickedDataSet_PolyData;

		/**
		vtkSphereSource for endpoint mark
		*/
		vtkSphereSource		* End1,			*End2;

		/**
		vtkPolyDataMapper for endpoints
		*/
		vtkPolyDataMapper	* EndMapper1,	*EndMapper2;
		
		/**
		vtkActor for endpoints
		*/
		vtkActor			* EndActor1,	*EndActor2;	
		
		/**
		number of endpoints
		*/
		int NumberOfEnds;

		/**
		index of the endpoints
		*/
		int PointIndex[2];

		/**
		index of the picked fiber
		*/
		int PickedFiber_Index;

		/**
		index of the picked dataset
		*/
		int PickedDataSet_Index;

		/**
		QList contains for the loaded "fiber" type datasets
		*/
		QList<data::DataSet*> ComparedDataSets;

		/**
		Flag for uncomfirmed endpoint
		*/
		int HasTempEnd1;
		int HasTempEnd2;
};

}//namespace bmia
#endif 