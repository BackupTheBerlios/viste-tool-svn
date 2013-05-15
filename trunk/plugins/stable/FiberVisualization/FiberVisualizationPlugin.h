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
 * FiberVisualizationPlugin.h
 *
 * 2010-07-15	Tim Peeters
 * - First version
 * 
 * 2010-09-15	Evert van Aart
 * - Implemented "dataSetRemoved".
 *
 * 2010-10-05	Evert van Aart
 * - Added support for different fiber shapes and coloring methods.
 * - An object of type "FiberVisualizationPipeline" is created for each
 *   fiber set. This object contains all visualization settings and filters
 *   for that fiber set.
 * - Implemented "dataSetChanged".
 * 
 * 2010-10-22	Evert van Aart
 * - Added support for coloring using the "CellData" array.
 * 
 * 2010-10-25	Evert van Aart
 * - Merged coloring option "None" into option "Single Color". When "Single
 *   Color" is selected, user can use buttons to color the fibers white, use
 *   an automatically generated color (using "vtkQtColorChart"), or select
 *   a custom color. For now, automatic coloring is the default for new fibers.
 *
 * 2011-01-12	Evert van Aart
 * - Added support for Look-Up Tables (transfer functions).
 * - Added a combo box for the eigensystem images.
 * - Before switching to a new coloring method or fiber shape, the class now checks
 *   whether the required data is available (i.e., eigensystem image when using 
 *   MEV coloring), and it will display a message box if this check fails.
 * - Implemented "dataSetChanged" and "dataSetRemoved" for all input data set types.
 * 
 * 2011-01-20	Evert van Aart
 * - Added support for transformation matrices.
 * - Write ".tfm" file when saving fibers.
 * 
 * 2011-02-01	Evert van Aart
 * - Added support for bypassing the simplification filter.
 * - Added a "Delete Fibers" button.
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.0.
 * - When saving fibers, the plugin now automatically selects the 
 *   data directory defined in the default profile. 
 *
 * 2011-04-18	Evert van Aart
 * - Moved the simplifcation filter to the "Helpers" library.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.2.
 * - Improved attribute handling.
 *
 * 2011-07-12	Evert van Aart
 * - Version 1.0.3.
 * - "isVisible" attribute is now also checked for new data sets, not just changed ones.
 *
 */


#ifndef bmia_FiberVisualizationPlugin_h
#define bmia_FiberVisualizationPlugin_h


/** Forward Declarations */

namespace Ui 
{
    class FiberVisualizationForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "vtkStreamlineToStreamTube.h"
#include "vtkStreamlineToHyperStreamline.h"
#include "Helpers/vtkStreamlineToSimplifiedStreamline.h"
#include "HWShading/vtkFiberMapper.h"
#include "Helpers/TransformationMatrixIO.h"

/** Includes - VTK */

#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkPropAssembly.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkQtChartColors.h>
#include <vtkPolyDataWriter.h>
#include <vtkScalarsToColors.h>
#include <vtkMatrix4x4.h>

/** Includes - GUI */

#include "ui_FiberVisualization.h"

/** Includes - Qt */

#include <QColorDialog>
#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>


namespace bmia {


class vtkFiberMapper;
class FiberVisualizationPipeline;


/** This class visualizes fibers created using one of the available fiber
	tracking methods. Visualization is done using a filter pipeline, 
	consisting of one or more processing filters and a mapper. The filters
	and their settings are stored in the "FiberVisualizationPipeline" class;
	each fiber set has one object of this class attached to it.

	Plugins must always subclass plugin::Plugin. Because this plugin
	uses data, visualizes something, and shows a Qt GUI to the user, it
	also implements the interfaces data::Consumer, plugin::Visualization,
	and plugin::GUI, respectively.
*/


class FiberVisualizationPlugin : 	public plugin::Plugin,
									public data::Consumer,
									public plugin::Visualization,
									public plugin::GUI
{
	/** Qt Macros */

    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::Visualization)
    Q_INTERFACES(bmia::plugin::GUI)

	public:
		
		/** Get current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.3";
		}

		/** Constructor */

		FiberVisualizationPlugin();

		/** Destructor */
    
		~FiberVisualizationPlugin();

		/** Return the VTK prop that renders all the geometry.
			This implements the Visualization interface. */

		vtkProp * getVtkProp();

		/** Returns the Qt widget that gives the user control.
			This implements the GUI interface. */
    
		QWidget * getGUI();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		void dataSetAdded(data::DataSet * ds);
    
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */

		void dataSetChanged(data::DataSet * ds);

		/** The data manager calls this function whenever an existing
			data set is removed. */
    
		void dataSetRemoved(data::DataSet * ds);

		/** FiberShape Enumeration. Determines the shape of the fibers, e.g.
			Streamlines or Streamtubes. Note that the order of the enumeration
			elements should be the same as in the "shapeCombo" combo box in the GUI. */
	
		enum FiberShape
		{
			FS_Streamlines = 0,
			FS_Streamtubes,
			FS_Hyperstreamtubes,
			FS_Hyperstreamprisms,
			FS_Streamribbons
		};

		/** FiberColor Enumeration. Determines the coloring method of the fibers.
			Note that the order of the enumeration elements should be the same as
			in the "coloringTypeComboBox" combo box in the GUI. */

		enum FiberColor
		{
			FC_SingleColor = 0,
			FC_FiberData,
			FC_CellData,
			FC_MEV,
			FC_Direction,
			FC_AI
		};

	protected slots:
    
		/** Change selected fiber set.
			@param row	Newly selected row. */

		void selectData(int row);

		/** Change the color of a fiber set when the "Single Color" coloring
			method is selected. */

		void changeSingleColor();

		/** Change the color of a fiber set to white. */

		void changeSingleColorToWhite();

		/** Change the color of a fiber set to an automatically selected color. */

		void changeSingleColorToAuto();

		/** Copy the settings from the GUI to the "FiberVisualizationPipeline"
			associated with the currently selected fiber set. If necessary,
			this will create new filters and/or a new mapper, and rebuild
			the pipeline. */
    
		void settingsFromGUIToPipeline();

		/** Enabled or disable certain GUI controls, depending on the values
			of other controls. For example, if the coloring method is set to 
			"None", all other color-related controls are disabled. */
	
		void setGUIEnable();

		/** Copy settings of current fiber set to all fiber sets. */
	
		void applySettingsToAll();

		/** Save the selected fiber data set to a VTK file. */

		void writeFibersToFile();

		/** Called when the user changes the coloring method. This function checks
			whether the information needed for this new coloring method is available.
			For example, when the "Cell Data" method is selected, it checks if the 
			fiber set contains cell data. If the check fails, it resets the coloring
			method to "Single Color" and displays a message box. */

		void changeColoringMethod();

		/** Called when the user changes the fiber shape. This function checks
			whether the information needed for this new fiber shape is available.
			specifically, the three "hyper" shapes (tubes, prisms and ribbons) need
			eigensystem data, so this function check whether such an image is available.
			If not, it displays a message box, and resets the shape to "Streamlines". */

		void changeShape();

		/** Deletes a fiber data set. */

		void deleteFibers();

	private:

		/** Structure containing a pointer to a "vtkActor" object, as well
			as the "FiberVisualizationPipeline" object that is used to 
			setup this actor. */

		struct actorInfo
		{
			vtkActor * actor;
			FiberVisualizationPipeline * actorPipeline;
		};

		/** Currently selected eigensystem image data. */

		vtkImageData * currentEigenData;

		/** Lists of available data sets. */
    
		QList<data::DataSet *> aiSets;
		QList<data::DataSet *> eigenSets;
		QList<data::DataSet *> fiberSets;
		QList<data::DataSet *> lutSets;

		/** The collection of all the actors that this plugin can render.
			This is the object that will be returned by getVtkProp().  */
    
		vtkPropAssembly * assembly;

		/** The Qt widget to be returned by getGUI(). */
    
		QWidget * widget;

		/** The Qt form created with Qt designer. */
    
		Ui::FiberVisualizationForm * ui;

		/** The actors associated with the data sets in fiberSets. */
    
		QList<actorInfo> actors;

		/** The index of the currently selected data set. A value of "-1" 
			means that no data set is selected. */

		int selectedData;

		/** Add a data set of one of the supported types (fibers, anisotropy index images, 
			eigensystem images, and LUTs) to the corresponding list and to the GUI.
			@param ds		New data set. */

		bool addAIDataSet(data::DataSet * ds);
		bool addLUTDataSet(data::DataSet * ds);
		bool addEigenDataSet(data::DataSet * ds);
		bool addFiberDataSet(data::DataSet * ds);

		/** Copy the settings stored in the pipeline associated with the selected
			fiber set to the controls in the GUI. */
    
		void settingsFromPipelineToGUI();

		/** Connect GUI controls to their respective "SLOT" functions. */
    
		void connectAll();

		/** Disconnect all GUI controls */
    
		void disconnectAll();

		/** Returns a pointer to the current set of fibers. */

		vtkPolyData * getSelectedFibers();

}; // class FiberVisualizationPlugin


} // namespace bmia


#endif // bmia_FiberVisualizationPlugin_h
