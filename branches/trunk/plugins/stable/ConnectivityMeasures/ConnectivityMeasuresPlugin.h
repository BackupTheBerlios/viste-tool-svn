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
 * ConnectivityMeasuresPlugin.h
 *
 * 2011-05-11	Evert van Aart
 * - Version 1.0.0.
 * - First version.
 *
 * 2011-06-06	Evert van Aart
 * - Version 1.1.0.
 * - Increased stability.
 * - Added an option for applying the ranking measure value to each fiber point
 *   (thus getting a single color for each fiber).
 *
 * 2011-08-22	Evert van Aart
 * - Version 1.1.1.
 * - Fixed a crash in the ranking filter.
 * - Added progress bars for all filters.
 *
 */


#ifndef bmia_ConnectivityMeasuresPlugin_h
#define bmia_ConnectivityMeasuresPlugin_h


/** Define the UI class */

namespace Ui 
{
	class ConnectivityMeasuresForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_ConnectivityMeasures.h"

/** Includes - Qt */

#include <QMessageBox>

/** Includes - VTK */

#include <vtkObject.h>
#include <vtkMatrix4x4.h>

/** Includes - Custom Files */

#include "vtkGenericConnectivityMeasureFilter.h"
#include "vtkGeodesicConnectionStrengthFilter.h"


namespace bmia {


/** Class Declarations */

class vtkFiberRankingFilter;


/** This plugin can be used to compute a Connectivity Measure for a set of fibers,
	which quantifies the connection strength for each individual fiber. A CM value
	is computed for each fiber point, which can be visualized using a LUT in the
	Fiber Visualization plugin. Additionally, the user can choose to output only
	the strongest fibers, thus showing only those computed fibers most likely
	to correspond to actual fibers. 
*/

class ConnectivityMeasuresPlugin :	public plugin::Plugin,
									public plugin::GUI,
									public data::Consumer
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::plugin::GUI)
	Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return current plugin version. */

		QString getPluginVersion()
		{
			return "1.1.0";
		}

		/** Constructor. */

		ConnectivityMeasuresPlugin();

		/** Destructor. */

		~ConnectivityMeasuresPlugin();

		/** Initialize the plugin. */

		void init();

		/** Return the widget that is shown in the GUI. */

		QWidget * getGUI();

		/** The data manager calls this function whenever a new 
			data set is added to the manager. 
			@param ds	New data set. */

		virtual void dataSetAdded(data::DataSet * ds);
    
		/** The data manager calls this function whenever an existing
			data set is modified in some way. 
			@param ds	Modified data set. */

		virtual void dataSetChanged(data::DataSet * ds);

		/** The data manager calls this function whenever an existing
			data set is removed. */

		virtual void dataSetRemoved(data::DataSet * ds);

		/** Enumeration for the different connectivity measures. */

		enum ConnectivityMeasure
		{
			CM_GeodesicConnectionStrength = 0	/**< Ratio between euclidean and geodesic length. */
		};

		/** The output method for the ranking filter. By default, we send all
			input fibers to the output, but using the ranking filter, we can also
			choose to only show the 'strongest' fibers, either as a fixed number
			of fibers, or as a percentage of the number of input fibers. */

		enum RankingOutput
		{
			RO_AllFibers = 0,		/**< Show all fibers. */
			RO_BestNumber,			/**< Show the best fibers (fixed number). */
			RO_BestPercentage		/**< Show the best fibers (percentage of input). */
		};

		/** The measure used for ranking. In the ranking filter, each fiber is
			assigned a single scalar value, and only the fibers with the highest
			value are sent to the output. The ranking measure determines how this
			scalar value is computed from the Connectivity Measure values (one 
			value per fiber point). */

		enum RankingMeasure
		{
			RM_FiberEnd = 0,		/**< Use the CM value of the last fiber point. */
			RM_Average				/**< Use the average CM value. */
		};

	protected slots:

		/** Change the selected fibers. Loads the settings for the newly selected
			set of fibers to the GUI, and enables/disables controls.
			@param index	Index of the selected fibers. */

		void changeInputFibers(int index);
		
		/** Enables or disable GUI controls based on the settings of the currently
			selected fibers. */

		void enableControls();

		/** Runs the Connectivity Measure filter, and, if required, the Fiber
			Ranking filter, and update the output data set for the selected fibers. */

		void update();

	private:

		/** Structure containing settings and information for one set of input fibers. */

		struct FiberInformation
		{
			data::DataSet * inDS;							/**< Input data set. */
			data::DataSet * outDS;							/**< Output data set. */
			data::DataSet * auxImage;						/**< Auxiliary image. */
			vtkFiberRankingFilter * rankingFilter;			/**< Fiber Ranking filter. */
			vtkGenericConnectivityMeasureFilter * cmFilter;	/**< Connectivity Measure filter. */
			ConnectivityMeasure measure;					/**< Connectivity Measure used. */
			bool doNormalize;								/**< Normalization on/off. */
			RankingOutput ranking;							/**< Output method of the ranking filter. */
			RankingMeasure rankBy;							/**< Measure for ranking the fibers. */
			int numberOfFibers;								/**< Number of output fibers. */
			int percentage;									/**< Percentage of input fibers. */
			bool useSingleValue;							/**< Whether or not to use a single scalar value per fiber. */
		};

		/** List containing the setting structures for each of the input fiber sets. */

		QList<FiberInformation> infoList;

		/** List of DTI image data sets. DTI images are needed to compute certain
			Connectivity Measures, such as the Geodesic Connection Strength. */

		QList<data::DataSet *> dtiImages;

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt Designer. */

		Ui::ConnectivityMeasuresForm * ui;

		/** When updating output fibers, we also change the input data set. However,
			since the only change we make to the input data set is adding one 
			attribute (the "isVisible" attribute, used to hide it in the Fiber
			Visualization plugin), we do not want the "dataSetChanged" function
			to actually do anything in response to this change. To this end, we
			store the input data set pointer before calling "dataSetChanged",
			and check for this pointer in the "dataSetChanged" function. */

		data::DataSet * ignoreDataSet;

		/** Connect or disconnect GUI controls to the slot functions.
			@param doConnect	Connect if true, disconnect otherwise. */

		void connectControls(bool doConnect);

		/** Copy the settings for one of the input fiber sets to the GUI.
			@param index		Index of the fiber data set. */

		void settingsToGUI(int index);

		/** Locate an input fiber data set in the "infoList" list, and return its
			index. Returns -1 if the data set does not exist in "infoList".
			@param ds			Target fiber data set. */

		int findInputDataSet(data::DataSet * ds);

}; // class ConnectivityMeasuresPlugin


} // namespace bmia


#endif // bmia_ConnectivityMeasuresPlugin_h
