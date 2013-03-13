/*
 * Crop3DPlugin.h
 *
 * 2013-02-10	Mehmet Yusufoglu
 * - Version 1.0.0.
 * - First version
 */


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
 * Crop3DPlugin.h
 *
 *  2013-02-10 Mehmet Yusufoglu
 *
 *  Plugin for cropping 3D Data.
 */


#ifndef bmia_Crop3DPlugin_h
#define bmia_Crop3DPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "Helpers/vtkImageOrthogonalSlicesActor.h"
#include "Helpers/vtkImageSliceActor.h"
#include "vtkMEVColoringFilter.h"
#include "core/Core.h"
#include "gui/MetaCanvas/vtkMedicalCanvas.h"
#include "gui/MetaCanvas/vtkMetaCanvasUserEvents.h"
#include "gui/MetaCanvas/vtkSubCanvas.h"

/** Includes - GUI */

#include "ui_Crop3D.h"

/** Includes - VTK */

#include <vtkColorTransferFunction.h>
#include <vtkImageData.h>
#include <vtkLookupTable.h>
#include <vtkMatrix4x4.h>
#include <vtkPoints.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCamera.h>
#include <vtkMath.h>
#include <vtkCommand.h>
#include <vtkExtractVOI.h>

#include <vtkBoxWidget2.h>
#include <vtkBoxRepresentation.h>
#include <vtkTransform.h>
#include <vtkBoundingBox.h>
#include <vtkPropCollection.h>

/** Includes - Qt */

#include <QMessageBox>
#include <QtDebug>


/** Forward Declarations */

namespace Ui 
{
	class Crop3DForm;
}


namespace bmia {


/** Forward Declarations */

class Crop3DPluginCallback;


/** This plugin visualizes cross-sections of a 3D scalar volume using texture 
	mapping and three orthogonal planes. Currently, two different methods are
	supported: Scalar volumes mapped through a Look-Up Table (also called Transfer
	Function), and eigenvectors of DTI tensors mapped to RGB values, with optional
	weighting from a scalar volume. As such, it accepts three different input types:
	scalar volumes, eigensystem volumes, and transfer function.

	In addition to the visualization of the planes (in the 3D view as well as in the
	three 2D views), this plugin outputs the actors representing the planes (so that
	they can be used by, for example, the ROI drawing plugin), and one set of seed
	points for each of the planes, which is automatically updated when the plane
	is moved by the user.
 */

class Crop3DPlugin : public plugin::AdvancedPlugin,
						public plugin::Visualization,
						public plugin::GUI,
						public data::Consumer
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::plugin::AdvancedPlugin)
	Q_INTERFACES(bmia::plugin::Visualization)
	Q_INTERFACES(bmia::plugin::GUI)
	Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.1.1";
		}

		/** Constructor. */

		Crop3DPlugin();

		/** Destructor. */

		~Crop3DPlugin();

		/** Initialize the plugin. Called after the canvas has been set up. */
    
		void init();

		/** Return the VTK actor that displays the planes. */
    
		vtkProp * getVtkProp();

		/** Return the Qt widget that represents the GUI. */
     
		QWidget * getGUI();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		void dataSetAdded(data::DataSet * ds);
		
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds		Modified data set. */

		void dataSetChanged(data::DataSet* ds);

		/** The data manager calls this function whenever an existing
			data set is removed.
			@param ds		Modified data set. */

		void dataSetRemoved(data::DataSet* ds);

		/** Returns the 2D subcanvas of the main window with the indicated index.
			Used by the callback class to find the selected subcanvas.
			@param i		Index of the target subcanvas. */

		vtkSubCanvas * getSubcanvas(int i);

		/** Reset the subcanvas with index "i". Wrapper function for "reset2DCamera"
			for outside access. 
			@param i		Index of the target subcanvas. */

		void resetSubCanvasCamera(int i);

		 /** Crops the 3D data an dproduces 3D data according to values set by the sliders. */
		void crop3DDataSet(data::DataSet * ds);

		/** Ranges of sliders which are used to select 3D ROI boundaries are set depending on the dimensions of the data. */
		void set3DROISliderLimits();

		/** Get the slider values as boundaries of 3D ROI set by the user */ 
		void Crop3DPlugin::get3DROIBoundaries(int *bnd);

	protected slots:

		/** Change the scalar volume used for LUT-based visualization. Does nothing
			if the visualization method is not set to "LUT".
			@param index	Index of the new scalar volume. */

		void changeScalarVolume(int index);

		/** Change the Look-Up Table (Transfer Function) used to color the planes.
			Does nothing if the visualization method is not set to "LUT".
			@param index	Index of the new transfer function. */

		void changeLUT(int index);

		/** Change the weighting volume, which is used to change the brightness
			of the voxels when using RGB-based visualization of DTI volumes. Does
			nothing is the method is not set to "RGB". 
			@param index	Index of the new weighting volume. */

		void changeWeightVolume(int index);

		/** Change the DTI eigensystem, which is used for RGB-based visualization. 
			Does nothing is the method is not set to "RGB". 
			@param index	Index of the new DTI volume. */

		void changeDTIVolume(int index);

		/** Change the visualization method to LUT-based coloring. */

		void applyLUTColoring();

		/** Change the visualization method to RGB-based coloring. */

		void applyRGBColoring();

		/** Change the position of the X slice. 
			@param x			New slice position. 
			@param updateData	If false, no data sets will be updated, and scene will
								not be rendered. */

		void setXSlice(int x, bool updateData = true);

		/** Change the position of the Y slice. 
			@param y			New slice position. 
			@param updateData	If false, no data sets will be updated, and scene will
								not be rendered. */

		void setYSlice(int y, bool updateData = true);

		/** Change the position of the Z slice. 
			@param Z			New slice position. 
			@param updateData	If false, no data sets will be updated, and scene will
								not be rendered. */

		void setZSlice(int z, bool updateData = true);

		/** Show or hide the X slice.
			@param v		Show slice if true, hide it otherwise. */

		void setXVisible(bool v);

		/** Show or hide the Y slice.
			@param v		Show slice if true, hide it otherwise. */

		void setYVisible(bool v);

		/** Show or hide the Z slice.
			@param v		Show slice if true, hide it otherwise. */

		void setZVisible(bool v);

		/** Turn linear interpolation of voxel colors on or off.
			@param i		Turn interpolation on if true, off otherwise. */

		void setInterpolation(bool i);

			/** Call the cropping function  */
		void cropData();

		  /** Set the visibility of 3D ROI box */
		void setRoiBoxVisible(bool v);

		/** changeRoi Boundary */
		void changeRoiBoundary(int value);

		

	private:

		/** Seed points of the X slice. */

		vtkPoints * seedsX;
 
		/** Seed points of the Y slice. */

		vtkPoints * seedsY;

		/** Seed points of the Z slice. */

		vtkPoints* seedsZ;

		/** Actor for the three orthogonal planes. */

		vtkImageOrthogonalSlicesActor * actor;

		/** Qt widget representing the GUI of the plugin. */

		QWidget * qWidget;

		/** Main GUI object. */

		Ui::Crop3DForm * ui;

		/** Default black-to-white LUT. */

		vtkLookupTable * defaultLUT;

		/** Data sets for the seed points on the planes. */

		data::DataSet * seedDataSets[3];

		/** Data sets containing the individual slice actors. */

		data::DataSet* sliceActorDataSets[3];

		/** List containing all input scalar volume data sets. */

		QList<data::DataSet *> scalarVolumeDataSets;

		/** List containing all available Transfer Functions (LUTs). */

		QList<data::DataSet*> lutDataSets;

		/** List containing all input DTI eigensystem data sets. */

		QList<data::DataSet*> dtiDataSets;

		/** Filter used to compute RGB color based on the direction of the main
			eigenvector computed from the DTI tensors. */

		vtkMEVColoringFilter * MEVColoringFilter;

		/** Callback class used to handle events. */

		Crop3DPluginCallback * callBack;

		/** Update the seed points for one of the planes. 
			@param points		Output seed point set. 
			@param bounds		Bounds of the current plane.
			@param steps		Step size (voxel size) for all three dimensions. */

	    static void updateSeeds(vtkPoints * points, double bounds[6], double steps[3]);

		/** Reset the camera for one of the 2D views. We need a slightly different
			method for resetting the camera than usual, in order to be able to
			deal with transformed images.
			@param renderer		Renderer of the 2D view.
			@param sliceActor	Actor for the 2D plane.
			@param axis			Axis of the 2D view (0, 1, or 2). */

		void reset2DCamera(vtkRenderer * renderer, vtkImageSliceActor * sliceActor, int axis);

		/** Check if the currently selected weighting volume matches the selected 
			DTI volume in terms of dimensions. Return false is this is not the case. */

		bool checkWeightVolumeMatch();

		/** Setup plane positions and transformation matrices for the new image.
			Called when the user changes the scalar volume (for LUT-based visualization),
			or the DTI volume (for RGB-based visualization).
			@param ds			Data set of the newly selected image. */

		void configureNewImage(data::DataSet * ds);

		/** Connect or disconnect the GUI widgets to/from their respective slot functions.
			@param doConnect	Connect controls if true, disconnect if false. */

		void connectControls(bool doConnect);

		/** Box widget drawn for the selected bounding box which defines the region to be cropped out. */
		vtkBoxWidget2	*roiBox;
		 vtkWidgetRepresentation *boxRep;
		

}; // class Crop3DPlugin


/** This class listens to events of type "BMIA_USER_EVENT_SUBCANVAS_CAMERA_RESET",
	which are omitted by the metacanvas when the user presses the "R" key. When
	resetting the camera of the 2D views, we cannot use the default "ResetCamera"
	function of the renderer, since this is wrong for transformed planes. This
	callback allows us to overwrite the function of the "R" key. On successfully
	resetting a 2D camera, we set the abort flag, indicating that the event has 
	been handled. If "R" was pressed over the 3D view, we do not set this flag,
	and the default camera resetting function will be used. 
*/
	
class Crop3DPluginCallback : public vtkCommand
{
	public:

		/** Constructor Call */

		static Crop3DPluginCallback * New() { return new Crop3DPluginCallback; }

		/** Execute the callback. 
			@param caller		Not used.
			@param event		Event ID.
			@param callData		Pointer to the selected subcanvas. */

		void Execute(vtkObject * caller, unsigned long event, void * callData);

		/** Pointer to the plane visualization plugin. */

		Crop3DPlugin * plugin;
};


} // namespace bmia


#endif // bmia_PlanesVisPlugin_h
