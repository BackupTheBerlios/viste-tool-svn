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
 * ClusteringPlugin.h
 *
 * 2010-10-21	Evert van Aart
 * - First Version.
 *
 * 2011-02-02	Evert van Aart
 * - Implemented "dataSetChanged" and "dataSetRemoved".
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.0.
 * - Improved attribute handling.
 *
 */


/** ToDo List for "ClusteringPlugin"
	Last updated 09-11-2010 by Evert van Aart

	- Implement "dataSetChanged" and "dataSetRemoved".
	- "getNumberOfInputClusters" currently only works if all cluster IDs are in
	  the range 0-(N-1), where N is the number of input clusters. If, for example,
	  an input file contains the IDs 0, 1 and 3, the function will return "3", and 
	  the behaviour for the last cluster (which has clusterId 3, but is in row 2)
	  is undefined.
	- Related to the previous point: I think Rieneke starts counting input clusters
	  at one, while I made it assuming that they start at zero. Either change the 
	  input ".clu" files to make the IDs start at zero, or decrement the IDs in the 
	  code here.
    - The GPU mapper for Streamlines doesn't really work when the RGB colors are 
	  stored in the "CellData" scalars array, the colors become all messed up as 
	  soon as lighting is turned on. Tim should take a look at his mapper.
	- When re-computing the output clusters, we currently first throw away all 
	  existing output clusters - stored in the "addedDataSets" list - and then 
	  add the new ones. This resets the visualization options; perhaps there's a
	  way to overwrite the existing data sets, without first removing them?
*/


#ifndef bmia_ClusteringPlugin_h
#define bmia_ClusteringPlugin_h


/** Define the UI class */

namespace Ui 
{
    class ClusteringForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "ClusteringSettingsIO.h"

/** Includes - GUI */

#include "ui_Clustering.h"

/** Includes - VTK */

#include <vtkStructuredPoints.h>
#include <vtkDataArray.h>
#include <vtkQtChartColors.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkCellData.h>
#include <vtkObject.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <qinputdialog.h>
#include <QMessageBox>
#include <QColorDialog>


namespace bmia {


/** This class allows the suer to perform manual reclustering over an over-
	segmented set of fibers. The inputs are a set of fibers, grouped into a 
	single "vtkPolyData" object, and a clustering information object, which is 
	of type "vtkStructuredPoints". The clustering information data set contains
	as many points as there are lines in the fiber set, and for each point, it
	contains a scalar value corresponding to the input cluster that line belongs
	to. Users can define a set of output clusters, and assign all fibers in one
	input cluster to a single output cluster. Each output cluster is stored in 
	one "vtkPolyData", allowing users to set visualization options per cluster,
	and to export individual clusters. 
*/

class ClusteringPlugin :	public plugin::Plugin,
							public plugin::GUI,
							public data::Consumer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::plugin::GUI)
    Q_INTERFACES(bmia::data::Consumer)


	public:

		/** Return the current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.0";
		}

		/** Constructor */
    
		ClusteringPlugin();

		/** Destructor */

		~ClusteringPlugin();

		/** Return the widget that is shown in the GUI. */

		QWidget * getGUI();

		/** Qt widget returned by "getGUI". */

		QWidget * widget;

		/** Qt Form created by Qt Designer. */

		Ui::ClusteringForm * ui;

		/** Define behaviour when new data set are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */
		
		void dataSetAdded(data::DataSet* ds);
		void dataSetChanged(data::DataSet* ds);
		void dataSetRemoved(data::DataSet* ds);

		/** Add a new cluster to the list of output clusters. The name of the cluster
			has been read from a ".bun" file, using "ClusteringSettingsIO". 
			@param name		New output cluster name. */

		void addOutputClusterFromFile(QString name);

	protected slots:
	
		/** Apply colors of the "colorChart" array to the input fiber set. */
		
		void colorInputFibers();

		/** Add a new cluster to the list of output clusters. */
	
		void addOutputCluster();

		/** Rename an existing output cluster. */

		void renameOutputCluster();

		/** Delete an existing output cluster. */
	
		void removeOutputCluster();

		/** Group the input fibers into multiple output fiber sets, one for
			each output cluster. Colors output fibers using either an 
			automatic color from the "colorChart" array, or a manual color. */
	
		void updateOutputClusters();

		/** Initialize the table widget with the input clusters. Called 
			whenever the input clustering information is changed. */
	
		void createClusterList();

		/** Called when the user changes the value of the output cluster
			combo box. Checks either the Automatic or Manual color radio
			button, based on the value of "useAutoColor" for the newly 
			selected output cluster. Disable/enable the color picker 
			button as needed. */
	
		void outputClusterChanged();

		/** Store desired coloring method - Automatic or Manual - in the 
			"outputClusterInformation" object of the selected cluster. */
		
		void setColorAutoOrManual();

		/** Create a color picker dialog, allowing the user to choose a 
			new output color (when using Manual coloring). */
	
		void setManualColor();

		/** Show the input fibers and hide the output fibers. */

		void showInputHideOutput();

		/** Show the output fibers and hide the input fibers. */

		void showOutputHideInput();

		/** Write settings (output cluster name per input cluster ID) to
			an output file with extension ".bun". The suer can select the
			file using a file dialog. */

		void writeSettings();

		/** Read settings from a ".bun" file. */

		void readSettings();

		/** Called whenever the input fiber data set changes. */

		void changeFibers();

	private:

		/** List containing the available input fiber sets. */

		QList<data::DataSet *> fiberList;

		/** List containing the available clustering information object. */

		QList<data::DataSet *> clusterList;

		/** List containing all data sets that this plugin has added to the
			data manager. When re-computing output clusters, existing output
			clusters are first deleted. */
	
		QList<data::DataSet *> addedDataSets;

		/** Struct containing information about each output cluster: Its 
			name, its (manual) color, and whether it should use automatic
			or manual coloring. */

		struct outputClusterInformation
		{
			QString name;
			QColor color;
			bool useAutoColor;
		};

		/** List containing information for each output cluster. */
	
		QList<outputClusterInformation> outputInfoList;

		/** Number of unique cluster IDs in the input clustering information. */

		int numberOfInputClusters;

		/** Chart containing a number of pre-defined, unqie colors. */
	
		vtkQtChartColors colorChart;

		/** When true, changes in the input fiber data set will be ignored by the 
			"dataSetChanged" function. This prevents infinite loops: The function
			"colorInputFibers" calls "dataSetChanged", which in turns calls 
			"colorInputFibers". */

		bool ignoreFiberDSChanged;

		/** Compute the number of unique cluster IDs in the input. 
			@param in	Input clustering information. */
	
		int getNumberOfInputClusters(vtkStructuredPoints * in);

		/** Return whether or not the input name already exists in the 
			list of output clusters. */

		bool outputClusterExists(QString in);

		/** Return a default output cluster name, which is formatted as 
			"Output Cluster X", with X the lowest integer that does not
			create a duplicate name. */
	
		QString getNextDefaultName();

		/** Delete all elements in the table widget of the GUI. */

		void clearTable();

		/** Set whether or not to show the input fibers. 
			@param show		Show input fibers? */

		void showInput(bool show);

		/** Set whether or not to show the output fibers. 
			@param show		Show output fibers? */

		void showOutput(bool show);

		/** Initialize the colors of the color chart. */
	
		void initColorChart()
		{
			colorChart.clearColors();
			colorChart.addColor(QColor(	 0,   0, 255));
			colorChart.addColor(QColor(  0, 255,   0));
			colorChart.addColor(QColor(255,   0,   0));
			colorChart.addColor(QColor(255, 255,   0));
			colorChart.addColor(QColor(  0, 255, 255));
			colorChart.addColor(QColor(255,   0, 255));
			colorChart.addColor(QColor(255, 255, 255));
			colorChart.addColor(QColor(  0,   0, 127));
			colorChart.addColor(QColor(  0, 127,   0));
			colorChart.addColor(QColor(127,   0,   0)); 
			colorChart.addColor(QColor(127, 255,   0));
			colorChart.addColor(QColor(127, 127,   0));
			colorChart.addColor(QColor(255, 127,   0));
			colorChart.addColor(QColor(127,   0, 127));
			colorChart.addColor(QColor(127,   0, 255));
			colorChart.addColor(QColor(255,   0, 127));
			colorChart.addColor(QColor(  0, 127, 127));
			colorChart.addColor(QColor(  0, 255, 127));
			colorChart.addColor(QColor(  0, 127, 255)); 
			colorChart.addColor(QColor(127, 255, 255)); 
			colorChart.addColor(QColor(127, 127, 255));
			colorChart.addColor(QColor(255, 127, 255));
			colorChart.addColor(QColor(127, 255, 127));
			colorChart.addColor(QColor(255, 255, 127));
			colorChart.addColor(QColor(255, 127, 127));
			colorChart.addColor(QColor(127, 127, 127));
			colorChart.addColor(QColor( 63, 127, 127));
			colorChart.addColor(QColor(191, 127, 127));
			colorChart.addColor(QColor(127,  63, 127));
			colorChart.addColor(QColor(127, 191, 127));
			colorChart.addColor(QColor(127, 127,  63));
			colorChart.addColor(QColor(127, 127, 191));
			colorChart.addColor(QColor( 63,  63, 127));
			colorChart.addColor(QColor( 63, 191, 127));
			colorChart.addColor(QColor(191,  63, 127));
			colorChart.addColor(QColor(191, 191, 127));
			colorChart.addColor(QColor( 63, 127,  63));
			colorChart.addColor(QColor( 63, 127, 191));
			colorChart.addColor(QColor(191, 127,  63));
			colorChart.addColor(QColor(191, 127, 191));
			colorChart.addColor(QColor(127,  63,  63));
			colorChart.addColor(QColor(127,  63, 191));
			colorChart.addColor(QColor(127, 191,  63));
			colorChart.addColor(QColor(127, 191, 191));
			colorChart.addColor(QColor( 63,  63,  63));
			colorChart.addColor(QColor( 63,  63, 191));
			colorChart.addColor(QColor( 63, 191,  63));
			colorChart.addColor(QColor(191,  63,  63));
			colorChart.addColor(QColor(191, 191,  63));
			colorChart.addColor(QColor(191,  63, 191));
			colorChart.addColor(QColor( 63, 191, 191));
			colorChart.addColor(QColor(191, 191, 191));
			colorChart.addColor(QColor(  0,  63,   0));
			colorChart.addColor(QColor(  0, 191,   0));
			colorChart.addColor(QColor( 63,   0,   0));
			colorChart.addColor(QColor( 63,  63,   0));
			colorChart.addColor(QColor( 63, 127,   0));
			colorChart.addColor(QColor( 63, 191,   0));
			colorChart.addColor(QColor( 63, 255,   0));
			colorChart.addColor(QColor(127,  63,   0));
			colorChart.addColor(QColor(127, 191,   0));
			colorChart.addColor(QColor(191,   0,   0));
			colorChart.addColor(QColor(191,  63,   0));
			colorChart.addColor(QColor(191, 127,   0));
			colorChart.addColor(QColor(191, 191,   0));
			colorChart.addColor(QColor(191, 255,   0));
			colorChart.addColor(QColor(255,  63,   0));
			colorChart.addColor(QColor(255, 191,   0));
			colorChart.addColor(QColor(  0,  63, 255));
			colorChart.addColor(QColor(  0, 191, 255));
			colorChart.addColor(QColor( 63,   0, 255));
			colorChart.addColor(QColor( 63,  63, 255));
			colorChart.addColor(QColor( 63, 127, 255));
			colorChart.addColor(QColor( 63, 191, 255));
			colorChart.addColor(QColor( 63, 255, 255));
			colorChart.addColor(QColor(127,  63, 255));
			colorChart.addColor(QColor(127, 191, 255));
			colorChart.addColor(QColor(191,   0, 255));
			colorChart.addColor(QColor(191,  63, 255));
			colorChart.addColor(QColor(191, 127, 255));
			colorChart.addColor(QColor(191, 191, 255));
			colorChart.addColor(QColor(191, 255, 255));
			colorChart.addColor(QColor(255,  63, 255));
			colorChart.addColor(QColor(255, 191, 255));
			colorChart.addColor(QColor(  0,   0,  63));
			colorChart.addColor(QColor(  0,   0, 191));
			colorChart.addColor(QColor(  0,  63,  63));
			colorChart.addColor(QColor(  0,  63, 127));
			colorChart.addColor(QColor(  0,  63, 191));
			colorChart.addColor(QColor(  0, 127,  63));
			colorChart.addColor(QColor(  0, 127, 191));
			colorChart.addColor(QColor(  0, 191,  63));
			colorChart.addColor(QColor(  0, 191, 127));
			colorChart.addColor(QColor(  0, 191, 191));
			colorChart.addColor(QColor(  0, 255,  63));
			colorChart.addColor(QColor(  0, 255, 191));
			colorChart.addColor(QColor(255,   0,  63));
			colorChart.addColor(QColor(255,   0, 191));
			colorChart.addColor(QColor(255,  63,  63));
			colorChart.addColor(QColor(255,  63, 127));
			colorChart.addColor(QColor(255,  63, 191));
			colorChart.addColor(QColor(255, 127,  63));
			colorChart.addColor(QColor(255, 127, 191));
			colorChart.addColor(QColor(255, 191,  63));
			colorChart.addColor(QColor(255, 191, 127));
			colorChart.addColor(QColor(255, 191, 191));
			colorChart.addColor(QColor(255, 255,  63));
			colorChart.addColor(QColor(255, 255, 191));
			colorChart.addColor(QColor( 63,   0,  63));
			colorChart.addColor(QColor( 63,   0, 127));
			colorChart.addColor(QColor( 63,   0, 191));
			colorChart.addColor(QColor(127,   0,  63));
			colorChart.addColor(QColor(127,   0, 191));
			colorChart.addColor(QColor(191,   0,  63));
			colorChart.addColor(QColor(191,   0, 127));
			colorChart.addColor(QColor(191,   0, 191));
			colorChart.addColor(QColor( 63, 255,  63));
			colorChart.addColor(QColor( 63, 255, 127));
			colorChart.addColor(QColor( 63, 255, 191));
			colorChart.addColor(QColor(127, 255,  63));
			colorChart.addColor(QColor(127, 255, 191));
			colorChart.addColor(QColor(191, 255,  63));
			colorChart.addColor(QColor(191, 255, 127));
			colorChart.addColor(QColor(191, 255, 191));
		};

}; // class FiberFilterPlugin


} // namespace bmia


#endif // bmia_FiberFilterPlugin_h
