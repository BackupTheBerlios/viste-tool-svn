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
* VectorVisualizationPlugin.h
*
*  2013-10-29	Mehmet Yusufoglu
* - Created for displaying vtkImageData having 3 component (double) vectors. 
* Image Data is expected to be read by any other plugin e.g. vtiReaderPlugin, dataset type is "unit vector volume".
* Each vector is shown together with its negative since they are assumed to be maxima vectors of HARDI ODFs.
* 
*/

// This example plugin shows how to create a plugin that uses
// data and visualizes that in the view.

#ifndef bmia_VectorVisualizationPlugin_h
#define bmia_VectorVisualizationPlugin_h

class vtkPropAssembly;
class vtkActor;
class vtkRenderWindow;
namespace Ui {
	class VectorVisualizationForm;
}

#include "DTITool.h"
#include <vtkDoubleArray.h>

// For Maxima Unit Vector Visualisation
#include <vtkGlyph3D.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkArrowSource.h>
#include <vtkUnstructuredGrid.h>
namespace bmia {

	/**
	* This class visualizes poly data using the default vtkPolyDataMapper.
	* Plugins must always subclass plugin::Plugin. Because this plugin
	* uses data, visualizes something, and shows a Qt GUI to the user, it
	* also implements the interfaces data::Consumer, plugin::Visualization,
	* and plugin::GUI respectively.
	*/
	class VectorVisualizationPlugin : 	public plugin::AdvancedPlugin,
		public data::Consumer,
		public plugin::Visualization,
		public plugin::GUI
	{
		Q_OBJECT
			Q_INTERFACES(bmia::plugin::Plugin)
			Q_INTERFACES(bmia::plugin::AdvancedPlugin)
			Q_INTERFACES(bmia::data::Consumer)
			Q_INTERFACES(bmia::plugin::Visualization)
			Q_INTERFACES(bmia::plugin::GUI)

	public:
		VectorVisualizationPlugin();
		~VectorVisualizationPlugin();

		//void init();

		/**
		* Return the VTK prop that renders all the geometry.
		* This implements the Visualization interface.
		*/
		vtkProp* getVtkProp();

		/**
		* Return the Qt widget that gives the user control.
		* This implements the GUI interface.
		*/
		QWidget* getGUI();

		/**
		* Implement the Consumer interface
		*/
		void dataSetAdded(data::DataSet* ds);
		void dataSetChanged(data::DataSet* ds);
		void dataSetRemoved(data::DataSet* ds);


		void insertArrayNamesToTheListBox(vtkImageData *img);
		void addPointsAndVectorToUnstructuredGrids(int seedNumber);
		void formPipeLine(vtkImageData *img, int arrayNumber);
		void formPipeLinesForAllArrays(vtkImageData *img, int arrayNumber);

		vtkActor* actor;
		vtkImageData *img ;
		void addVectorToSeeds(data::DataSet* ds,
			QString vectorName) ;
		void addVectorToUnstructuredGrid(vtkUnstructuredGrid *gridForArrayForSeed, QString vectorName, bool Opposite=0) ;
		protected slots:
			void seedDataChanged(int index);
			void inputDataChanged(int index); //volume of vectors
			void selectVectorData(int row);
			void setVisible(bool visible);
			void setLighting(bool lighting);
			void changeColor();

			void setScale(double scale);

	private:
		/**
		* The collection of all the actors that this plugin can render.
		* This is the object that will be returned by getVtkProp().
		*/
		vtkPropAssembly* assembly;

		/**
		* The Qt widget to be returned by getGUI().
		*/
		QWidget* widget;

		/**
		* The Qt form created with Qt designer.
		*/
		Ui::VectorVisualizationForm* ui;

		/**
		* The added data sets that contain array of the vtkimagedata
		*/
		QList<data::DataSet*> dataSets;

		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkActor*> actors;

		QList<vtkPolyDataMapper*> mappers;

		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkGlyph3D*> glyphFilters;

		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkUnstructuredGrid *> seedGridsOfASeed;


		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkActor*> actorsOpposite;

		QList<vtkPolyDataMapper*> mappersOpposite;

		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkGlyph3D*> glyphFiltersOpposite;

		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkUnstructuredGrid *> seedGridsOfASeedOpposite;



		/**
		* The actors associated with the data sets in dataSets.
		*/
		QList<vtkPolyData*> polydatas;

		/**
		* Keep track whether the selection is being changed.
		* If this is set to true, then parameters in the GUI can
		* be updated without updating the rendering.
		*/
		bool changingSelection;

		/**
		* The index of the currently selected data set.
		* -1 means no data set is selected.
		*/
		int selectedData;


		/** Volumes including vector arrays */
		QList<data::DataSet *> glyphDataSets;

		/** List of all available seed point data sets. */

		QList<data::DataSet *> seedDataSets;
		vtkGlyph3D *glyphFilter;
		bool pipeFormed;

	}; // class VectorVisualizationPlugin
} // namespace bmia
#endif // bmia_VectorVisualizationPlugin_h
