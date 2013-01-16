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
 * FiberTrackingPlugin.h
 *
 * 2010-09-13	Evert van Aart
 * - First Version.
 *
 * 2010-09-15	Evert van Aart
 * - Added error message boxes.
 * - Added additional checks for input data.
 * - Improved adding output to data manager.
 *
 * 2010-09-20	Evert van Aart
 * - Added support for Whole Volume Seeding.
 *
 * 2010-09-30	Evert van Aart
 * - When recomputing a fiber set, the plugin will now change the existing 
 *   data set, instead of deleting and re-adding it. This way, the visualization
 *   options of the data set (such as shape, color, etcetera) will not be reset.
 * 
 * 2010-11-23	Evert van Aart
 * - Added "Overwrite existing fibers" checkbox to the GUI.
 * - If this box is checked, the plugin checks if a fiber set with the same DTI image,
 *   ROI, and tracking method (ROI or WVS) has been added to the data manager at some 
 *   point. If so, and if this data set still exists, the fibers in that data set are
 *   overwritten. If not, a new data set is generated. 
 * - This should work better with the new data management approach, which no longer
 *   requires unique data set names.
 *
 * 2010-12-10	Evert van Aart
 * - Implemented "dataSetChanged" and "dataSetRemoved".
 *
 * 2011-01-24	Evert van Aart
 * - Added support for transformation matrices.
 *
 * 2011-02-09	Evert van Aart
 * - Version 1.0.0.
 * - Added support for maximum scalar threshold values.
 * - In the GUI, changed references to "AI" values to "scalar" values. The idea
 *   is that any type of scalar data can be used as stopping criterium, not just
 *   AI values. This essentially enables the use of masking volumes.
 *
 * 2011-03-14	Evert van Aart
 * - Version 1.0.1.
 * - Fixed a bug in which it was not always detected when a fiber moved to a new
 *   voxel. Because of this, the fiber tracking process kept using the data of the
 *   old cell, resulting in fibers that kept going in areas of low anisotropy.
 *
 * 2011-03-16	Evert van Aart
 * - Version 1.0.2.
 * - Fixed a bug that could cause crashes if a fiber left the volume. 
 *
 * 2011-04-06	Evert van Aart
 * - Version 1.0.3.
 * - Maximum AI value no longer automatically snaps to the scalar range maximum
 *   when changing the AI image used.
 *
 * 2011-04-18	Evert van Aart
 * - Version 1.0.4.
 * - Properly update the fibers when their seed points change.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.5.
 * - Improved progress reporting.
 * - Slightly improved speed for Whole Volume Seeding.
 * - GUI now enables/disables the WVS controls when WVS is checked/unchecked.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.6.
 * - Improved attribute handling.
 *
 * 2011-06-01	Evert van Aart
 * - Version 1.1.0.
 * - Added geodesic fiber-tracking.
 * - GUI now only show controls for the selected fiber tracking method.
 * - Moved some functions to separate files to avoid one huge file.
 *
 * 2011-06-06	Evert van Aart
 * - Version 1.1.1.
 * - Fixed a bug in WVS that allowed fibers shorter than the minimum fiber length
 *   to still be added to the output.
 *
 * 2011-07-07	Evert van Aart
 * - Version 1.2.0.
 * - Added CUDA support.
 *
 * 2011-08-16	Evert van Aart
 * - Version 1.2.1.
 * - Running out of memory when tracking fibers should no longer crash the program.
 *
 */



#ifndef bmia_FiberTrackingPlugin_h
#define bmia_FiberTrackingPlugin_h


/** Define the UI class */

namespace Ui 
{
    class FiberTrackingForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_FiberTracking.h"

/** Includes - Custom Files */

#include "vtkFiberTrackingFilter.h"
#include "vtkFiberTrackingWVSFilter.h"
#include "geodesicPreProcessor.h"

/** Includes - VTK */

#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <QDebug>
#include <QMessageBox>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QSlider>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSpacerItem>
#include <QGroupBox>
#include <QGridLayout>
#include <QRadioButton>


namespace bmia {


class FiberTrackingGeodesicFilter;


/** Fiber Tracking plugin. This class mainly takes care of managing the
	input data set and the created fiber data sets, in combination with
	the GUI. The actual fiber tracking process is done in the "vtkFiber-
	TrackingFilter" class.
*/

class FiberTrackingPlugin : 	public plugin::Plugin,
								public data::Consumer,
								public plugin::GUI
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Consumer)
    Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Get current plugin version. */

		QString getPluginVersion()
		{
			return "1.2.1";
		}

		/** Constructor */

		 FiberTrackingPlugin();

		 /** Destructor */

		~FiberTrackingPlugin();

		/** Returns a pointer to the GUI object */

		QWidget * getGUI();

		/** Define behavior when new data sets are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */

		void dataSetAdded(data::DataSet * ds);
		void dataSetChanged(data::DataSet * ds);
		void dataSetRemoved(data::DataSet * ds);

	protected slots:

		/** Called when the user presses the "Update" button. */

		void updateFibers(void);

		/** Update the minimum and maximum of the two scalar threshold spin boxes
			to the range of the currently selected scalar volume image. */

		void updateScalarRange();

		/** Change the fiber tracking method. This configures the GUI for the new method
			(e.g., for Whole Volume Seeding, the WVS Options page is added to the toolbox.
			@param index	Index of the new method (within the "TrackingMethod" enum). */

		void changeMethod(int index);

		/** Update the controls for geodesic fiber-tracking. For example, this 
			function is called when the additional angles pattern is changes, 
			after which it hides all angle-related controls that do not apply
			to the selected pattern. */

		void updateGeodesicGUI();

		/** Checks if CUDA is supported, and returns false otherwise. Implemented
			in the "_Config_" source files. For the regular plugin (without CUDA),
			this function always returns true. */

		bool isCUDASupported();

		/** Change the name of the plugin. Used to add "(CUDA)" to the CUDA-version. 
			Implemented in the "_Config_" files. */
			
		void changePluginName();

	private:

		/** The Qt widget returned by "getGUI" */

		QWidget * widget;

		/** Structure containing pointers to the active controls for Whole Volume Seeding,
			as well as a pointer to the widget representing the WVS Options page. While
			the WVS Options page consists of more widgets that the ones contained in this
			structure, the rest of the controls are passive controls that are never directly
			accessed, and thus do not need to be stored here. */

		struct wvsGUIElements
		{
			QWidget *			wvsWidget;				/**< Main widget of the WVS Options page. */
			QDoubleSpinBox *	wvsSeedDistanceSpin;	/**< Spinner for the seed distance. */
			QSpinBox *			wvsMinDistanceSpin;		/**< Spinner for the distance percentage. */
			QSlider *			wvsMinDistanceSlide;	/**< Slider for the distance percentage. */
		};

		/** Pointer to a structure containing pointers to the GUI controls related to
			Whole Volume Seeding. Created in "setupGUIForWVS" when switching to 
			WVS, and deleted when switching to another method. */

		wvsGUIElements * wvsGUI;
		
		/** Structure containing pointers to the active controls for geodesic fiber-tracking,
			as well as a pointers to the widget representing the toolbox pages. While
			these toolbox pages consists of more widgets that the ones contained in this
			structure, the rest of the controls are passive controls that are never directly
			accessed, and thus do not need to be stored here. */

		struct geodesicGUIElements
		{
			QWidget * geodesicTrackingWidget;			/**< Main widget for the 'Additional Tracking Options' page. */
			QWidget * geodesicPPWidget;					/**< Main widget for the 'Preprocessing' page. */
			QWidget * geodesicPerformanceWidget;		/**< Main widget for the 'Performance' page. */

			QGroupBox *			aaGroup;				/**< Group box for the additional angles options. */
			QComboBox *			aaPatternCombo;			/**< Combo box for the additional angles pattern. */
			QSpinBox *			aaConeNumberSpin;		/**< Spinner for the number of angles (for cones). */
			QDoubleSpinBox *	aaConeWidthSpin;		/**< Spinner for the cone width. */
			QSpinBox *			aaSpherePSpin;			/**< Spinner for the number of angles for phi (for spheres). */
			QSpinBox *			aaSphereTSpin;			/**< Spinner for the number of angles for theta (for spheres). */
			QComboBox *			aaIcoTessOrderCombo;	/**< Spinner for the tessellation order (for icosahedron). */
			QGridLayout *		aaGLayout;				/**< Grid layout containing the additional angle options. */

			QCheckBox *			ppEnableCheck;			/**< Check box for enabling preprocessing. */
			QComboBox *			ppSharpenCombo;			/**< Combo box for the sharpening method. */
			QSpinBox *			ppGainSpin;				/**< Spinner for the tensor gain. */
			QDoubleSpinBox *	ppThresholdSpin;		/**< Spinner for the sharpening threshold. */
			QSpinBox *			ppExponentSpin;			/**< spinner for the tensor exponent for sharpening. */
			QGridLayout *		ppSharpenGLayout;		/**< Grid layout containing the sharpening options. */

			QCheckBox *			stopScalarCheck;		/**< Check box for enabling the scalar stopping criterion. */
			QCheckBox *			stopLengthCheck;		/**< Check box for enabling the length stopping criterion. */
			QCheckBox *			stopAngleCheck;			/**< Check box for enabling the angle stopping criterion. */

			QComboBox *			odeCombo;				/**< Combo box for the ODE solver. */

			QRadioButton *		perfARadio;				/**< Radio button for the slowest performance profile. */
			QRadioButton *		perfBRadio;				/**< Radio button for the medium performance profile. */
			QRadioButton *		perfCRadio;				/**< Radio button for the fastest performance profile. */
		};

		/** Pointer to a structure containing pointers to the GUI controls related to
			geodesic fiber-tracking. Created in "setupGUIForGeodesics" when switching to 
			geodesic fiber-tracking, and deleted when switching to another method. */

		geodesicGUIElements * geodesicGUI;


		/** The form created by Qt Designer */

		Ui::FiberTrackingForm * ui;

		/** Lists containing pointers to added data set, divided into three
			categories: seed points, DTI tensors, and Anisotropy Index images. */

		QList<data::DataSet *> seedList;
		QList<data::DataSet *> dtiDataList;
		QList<data::DataSet *> aiDataList;

		/** List available tracking methods. */

		enum TrackingMethod
		{
			TM_Streamlines = 0,		/**< Streamlines(default). */
			TM_WVS,					/**< Whole Volume Seeding. */
			TM_Geodesic				/**< Geodesic Fiber Tracking. */
		};

		/** For each output data set pointer, we store the pointers of the DTI 
			data set and the ROI data set used to create the output fibers, as
			well as the method used (e.g., ROIs or WVS). We also store the name
			of the data set; if the user manually changes the name of the output
			fibers, this plugin will no longer change its name automatically. */

		struct outputInfo
		{
			data::DataSet * dti;
			data::DataSet * seed;
			data::DataSet * output;
			TrackingMethod  method;
			QString         oldName;
		};

		/** List of added output data sets and their related information. */

		QList<outputInfo> outputInfoList;

		/** Returns true if the "outputInfoList" list contains an "outputInfo" item
			whose DTI data, seed point data, and method all match those of the input
			"newFiberObject". It will also copy the output data set pointer of this
			matching item to "newFiberInfo", and return the index of the matching in
			"outputInfoList". Returning true means that the new output fibers should
			be copied to an existing data set; returning false means that we will create
			a new data instead. 
			@param newFiberInfo		Output information of new fibers. 
			@param outputIndex		List index of matching information object. */

		bool overwriteDataSet(outputInfo * newFiberInfo, int * outputIndex);

		/** Add a new data set to one of the three lists */

		bool addDTIDataSet(data::DataSet * ds);
		bool addAIDataSet(data::DataSet * ds);
		bool addSeedPoints(data::DataSet * ds);

		/** Change existing data sets. */

		void changeDTIDataSet(data::DataSet * ds);
		void changeAIDataSet(data::DataSet * ds);
		void changeSeedPoints(data::DataSet * ds);

		/** Remove existing data sets. */

		void removeDTIDataSet(data::DataSet * ds);
		void removeAIDataSet(data::DataSet * ds);
		void removeSeedPoints(data::DataSet * ds);

		/** Main fiber tracking filter for ROI seeding. */

		vtkFiberTrackingFilter * dtiFiberTrackingFilter;

		/** Performs fiber tracking for all Regions of Interest in the "seedList"
			@param dtiImageData		DTI Tensors
			@param aiImageData		AI Scalars */

		void doStreamlineFiberTracking(vtkImageData * dtiImageData, vtkImageData * aiImageData);

		/** Performs fiber tracking using the Whole Volume Seeding technique.
			@param dtiImageData		DTI Tensors
			@param aiImageData		AI Scalars */

		void doWVSFiberTracking(vtkImageData * dtiImageData, vtkImageData * aiImageData);

		/** Performs geodesic fiber tracking for all Regions of Interest in the "seedList"
			@param dtiImageData		DTI Tensors
			@param aiImageData		AI Scalars */

		void doGeodesicFiberTracking(vtkImageData * dtiImageData, vtkImageData * aiImageData);

		/** Adds output fibers to the data manager. If a set of fibers with the same
			DTI image, seeding region, and tracking method already exists, we overwrite
			the fibers in this existing data set.
			@param fibers			Output fibers.
			@param fiberName		Name of the fiber set. 
			@param method			Tracking method used to generate fibers.
			@param seed				Data set of seeding region. */

		void addFibersToDataManager(vtkPolyData * fibers, QString fiberName, TrackingMethod method, data::DataSet * seed);

		/** Remove all pages except for the first one from the toolbox, and delete
			optional GUI components (like "wvsGUI"). The first page is spared, as
			it contains tracking parameters which apply to all methods. */

		void clearToolbox();

		/** Enable all controls that may have been disabled for one of the fiber
			tracking methods. For example, the seed list is disabled for Whole
			Volume Seeding, so when we switch to another method, we need to
			re-enable it. */

		void enableAllControls();

		/** Setup the GUI for WVS. Adds a "Whole Volume Seeding Options" page. */

		void setupGUIForWVS();

		/** Setup the GUI for Streamlines. Calls "enableAllControls" and "clearToolbox". */

		void setupGUIForStreamlines();

		/** Setup the GUI for geodesic fiber-tracking. Calls "enableAllControls" and "clearToolbox". */

		void setupGUIForGeodesics();

}; // class FiberTrackingPlugin


} // namespace bmia


#endif // bmia_FiberTrackingPlugin_h
