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
 * FiberCuttingPlugin.h
 *
 * 2010-11-15	Yang YU
 * - First Version.
 *
 * 2010-11-16	Yang YU
 * - Realize basic function of toolbox, (next/previous page, plain text, setting parameters).
 *
 * 2010-11-16	Yang YU
 * - Add fiber data load in and visualization (Data Page).
 *
 * 2010-11-25	Yang YU
 * - Realize the point picker.
 *
 * 2011-03-14	Evert van Aart
 * - Version 1.0.0.
 * - Changes to the tutorial text in the GUI and the text of the error/warning dialogs. 
 *
 * 2011-03-28	Evert van Aart
 * - Version 1.1.0.
 * - Increased stability of plugin.
 * - Rewording of GUI elements and message boxes.
 * - Redesign of GUI, enable/disable buttons to make workflow more clear.
 * - Unconfirmed points now show up in red, confirmed points are green. 
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.1.1.
 * - When saving output fibers, the plugin now automatically selects the 
 *   data directory defined in the default profile. 
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.1.2.
 * - Removed the progress bar for the output writer, since it is pretty much
 *   instantaneous anyway.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.1.3.
 * - Improved attribute handling.
 *
 * 2011-06-06	Evert van Aart
 * - Version 1.1.4.
 * - Fixed crash when deleting fiber data sets from the Fiber Visualization plugin.
 *
 */


#ifndef bmia_FiberCutting_FiberCuttingPlugin_h
#define bmia_FiberCutting_FiberCuttingPlugin_h


// Forward Declarations

class vtkPropAssembly;
class vtkActor;


namespace Ui
{
	class FiberCuttingForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_FiberCutting.h"

/** Includes - VTK */

#include <vtkPropAssembly.h>
#include <vtkActor.h>
#include <vtkProperty.h>

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataWriter.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

#include <vtkCamera.h>
#include <vtkRenderer.h>

#include "vtkInteractorStyleTrackballCellPicker.h"
#include "vtkInteractorStyleTrackballCellEndPicker.h"

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCellPicker.h>
#include <vtkCell.h>

#include "gui/MetaCanvas/vtkMedicalCanvas.h"
#include "gui/MetaCanvas/vtkSubCanvas.h"
#include "gui/MainWindowInterface.h"

/** Includes - Qt */

#include <QColorDialog>
#include <QDebug>
#include <QMessageBox>
#include <QFileDialog>


namespace bmia
{
/** This class realized a workflow which allows the user to select their 
	interested fiber, and also cut the selected fiber. The whole workflow
	has be performed step by step. And in each step it uses different 
	interactorstyle to realize different functionalities.

	Plugins must always subclass plugin::Plugin. Because this plugin
	uses data, visualizes something, and shows a Qt GUI to the user, it
	also implements the interfaces data::Consumer, plugin::Visualization,
	and plugin::GUI, respectively.
*/
class FiberCuttingPlugin : public plugin::AdvancedPlugin,
								   public data  ::Consumer, 
								   public plugin::Visualization,
								   public plugin::GUI
								   
	{
		/** 
		Qt Macros 
		*/

		Q_OBJECT
		Q_INTERFACES(bmia::plugin::Plugin)
		Q_INTERFACES(bmia::data::Consumer)
		Q_INTERFACES(bmia::plugin::Visualization)
		Q_INTERFACES(bmia::plugin::GUI)
		Q_INTERFACES(bmia::plugin::AdvancedPlugin)


		public:

			/** Get the current plugin version. */

			QString getPluginVersion()
			{
				return "1.1.4";
			}

			/** 
			Constructor 
			*/
			FiberCuttingPlugin();

			/** 
			Destructor */
			~FiberCuttingPlugin();

			/**
			Return the VTK actor that renders the fibers.
			This function must be implemented for each subclass of the plugin::Visualization interface
			*/
			vtkProp * getVtkProp();

			/**
			Return the widget that is shown in the GUI
			*/
			QWidget * getGUI();

			/** 
			The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. 
			*/
			void dataSetAdded(data::DataSet * ds);
	    
			/** 
			The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. 
			*/
			void dataSetChanged(data::DataSet * ds);

			/** 
			The data manager calls this function whenever an existing
			data set is removed. 
			*/
			void dataSetRemoved(data::DataSet * ds);


		
		protected slots:


			/**
			Confirm the picked endpoint
			*/
			void ConfirmEndPoint();

			/**
			Automatic mark the original endpoint of a fiber
			*/
			void AutoEnd();

			/**
			Clear all the endpoint marks
			*/
			void ClearAll();
			
			/**
			StartButton's responce function of the whole fiber cutting processing which enables the user to select fibers.
			*/
			void StartButton();

			/**
			FiberSelectPage_NextButton's responce function which enables the user to determine the endpoint.
			*/
			void FiberSelectPage_NextButton();
		
			/**
			FiberCuttingPage_PreviousButton's responce function which enables the user to return to fiber select page.
			*/
			void FiberCuttingPage_PreviousButton();

			/**
			FiberCuttingPage_NextButton's responce function which enables the user to go to fiber cutting page.
			*/
			void FiberCuttingPage_NextButton();

			/**
			SaveRepeatPage_RepeatButton's responce function which enables the user to repeat the fiber selection and cutting on different fibers.
			*/
			void SaveRepeatPage_RepeatButton();
			
			/**
			SaveRepeatPage_QuitButton's responce function which allow the user to quit the processing.
			*/
			void SaveRepeatPage_QuitButton();
			
			/**
			SaveRepeatPage_SaveButton's responce function to perform data saving.
			*/
			void SaveRepeatPage_SaveButton();


		private:
			/**
			The collection of all the actors that this plugin can render.
			This is the object that will be returned by getVtkProp().
			*/
			vtkPropAssembly* assembly;

			/**
			The Qt widget to be returned by getGUI().
			*/
			QWidget* widget;

			/**
			The Qt form created with Qt designer.
			*/
			Ui::FiberCuttingForm * ui;

			/**
			The added data sets that contain VTK polydata
			*/
			QList<data::DataSet*> dataSets;


			//vtkPolyData * selectBunch_polydata;
			//QList<vtkActor *> selectBunch_actors;
			

			/**
			Jump between toolbox pages. Turn to previous page. 
			*/
			void PagePrevious();

			/**
			Jump between toolbox pages. Turn to next page.
			*/
			void PageNext();
			
			/**
			vtkInteractorStyle for view
			*/
			vtkInteractorStyleTrackballCamera * styleTrackball;
			
			/**
			vtkInteractorStyle which allow user to select fibers
			*/
			vtkInteractorStyleTrackballCellPicker* styleTrackballCP;

			/**
			vtkInteractorStyle which allow user to pick endpoint
			*/
			vtkInteractorStyleTrackballCellEndPicker * styleTrackballCEP;

			/**
			A flag for AutoEnd action which recorded the status of current AutoEnd location
			*/
			int AutoEndFlag;
			
			/**
			An array that saves the picked fiber index 
			*/
			QList<int> ModifiedFiberList;

			/**
			Current modified fiber index
			*/
			int ModifiedFiberIndex;
	};
}// namespace bmia

#endif //bmia_FiberCuttingPlugin_h