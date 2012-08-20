/*
 * IllustrativeClustersPlugin.h
 *
 * 2011-03-24	Evert van Aart
 * - Version 1.0.0.
 * - First version. Based on the work by Ron Otten from the old tool, but adapted
 *   to the new plugin system and data management system.
 *
 * 2011-07-12	Evert van Aart
 * - Version 1.0.1.
 * - User can now show/hide fibers and clusters from the GUI.
 *
 */


#ifndef bmia_IllustrativeClustersPlugin_h
#define bmia_IllustrativeClustersPlugin_h


/** Includes - Main Headers */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_IllustrativeClusters.h"

/** Includes - Custom Files */

#include "IllustrativeCluster.h"
#include "IllustrativeClusterDisplacement.h"
#include "vtkIllustrativeFiberBundleMapper.h"
#include "data/DataSet.h"

/** Includes - Qt */

#include <QList>
#include <QColorDialog>
#include <QStringList>
#include <QTimer>

/** Includes - VTK */

#include <vtkPropAssembly.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>


/** Forward Declarations */

namespace Ui 
{
	class IllustrativeClustersForm;
}


namespace bmia {


/** This class can be used to visualize groups of fibers using the illustrative
	fiber visualization method by Ron Otten. Each fiber data set is converted into
	an "IllustrativeCluster" object, which contains a pointer to the actor, another
	pointer to the (custom) mapper, and the settings (configuration) for that
	bundle. The mapper ("vtkIllustrativeFiberBundleMapper") takes care of the 
	actual drawing; this plugin class is mainly used to transfer the settings
	between the GUI and the cluster objects. The class also contains a displacement
	manager, which can displace the cluster actors to focus on one of the bundles.
*/

class IllustrativeClustersPlugin :	public plugin::AdvancedPlugin,
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

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */

		IllustrativeClustersPlugin();

		/** Destructor */

		~IllustrativeClustersPlugin();

		/** Initializes the plugin. */

		void init();

		/** Returns the Qt widget that gives the user control.
			This implements the GUI interface. */
    
		QWidget * getGUI();

		/** Return the VTK prop that renders all the geometry.
			This implements the Visualization interface. */
    
		vtkProp * getVtkProp();

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

	protected slots:

		/** Turn the displacement of clusters on or off. 
			@param enable	Turn displacement on/off. */

		void toggleDisplacement(bool enable);

		/** Turn the displacement widget on or off. This widget allows the user
			to specify a region of the image on which he wants to focus.
			@param enable	Turn widget on/off. */

		void toggleDisplacementWidget(bool enable);

		/** Update the options of the displacement manager. In this case, the
			options are the scale of the spherical explosion, and that of the
			screen-aligned slide. This function is called whenever one of the
			related GUI widgets changes its value. */

		void updateDisplacementOptions();

		/** Tell the displacement manager to focus on the displacement widget,
			which can be set by the user. */

		void focusOnWidget();

		/** Tell the displacement manager to focus on the current fiber cluster. */

		void focusOnCluster();

		/** Change the line color, using a "QColorDialog" color picking window. */

		void changeLineColor();

		/** Change the fill color, using a "QColorDialog" color picking window. */

		void changeFillColor();

		/** Copy the current fill color to the line color. */

		void copyFillColorToLine();

		/** Copy the current line color to the fill color. */

		void copyLineColorToFill();

		/** Copy the values of the configuration of the currently selected cluster
			to the GUI widgets. This is generally called when the user selects a
			new cluster: the settings for this cluster are then loaded to the GUI. */

		void copyConfigToGUI();

		/** Copy the values of the GUI widgets to the configuration of the specified
			cluster. This function is called when the user clicks one of the "Apply"
			buttons. When clicking "Apply to All", we generally do not want to use 
			the same color for all clusters; the "copyColors" parameter allows us
			to specify whether the line- and fill colors stored in the GUI should
			also be copied to the configuration. 
			@param cluster		Target cluster.
			@param copyColors	Should we copy the colors to the configuration? */

		void copyGUItoConfig(IllustrativeCluster * cluster, bool copyColors = true);

		/** Apply the current GUI settings to the selected cluster. */

		void applyToCurrent();

		/** Apply the current GUI settings (except for the colors) to all clusters.
			The current GUI colors are only applied to the selected cluster. */

		void applyToAll();

		/** Updates the position of the clusters through the displacement manager,
			and re-draws the screen. Called by the "QTimer" object. */

		void animateDisplacement();

		/** Show the current cluster. */

		void showCluster();

		/** Hide the current cluster. */

		void hideCluster();

		/** Show the input fibers. This is achieved by setting the "isVisible"
			attribute of the input fiber data set to 1.0, and calling "dataSetChanged".
			The Fiber Visualization plugin then turns on the corresponding fibers. */

		void showFibers();

		/** Show the input fibers. This is achieved by setting the "isVisible"
			attribute of the input fiber data set to -1.0, and calling "dataSetChanged".
			The Fiber Visualization plugin then turns off the corresponding fibers. */

		void hideFibers();

	private:

		/** The Qt widget to be returned by "getGUI". */

		QWidget * widget;

		/** The Qt form created with Qt designer. */

		Ui::IllustrativeClustersForm * ui;

		/** Assembly containing all VTK actors (one per cluster). */

		vtkPropAssembly * assembly;

		/** List of input data sets. */

		QList <data::DataSet *> inDataSets;

		/** List of clusters. Size and order should at all times match those of
			the "inDataSets" list. */

		QList <IllustrativeCluster *> clusterList;

		/** Update the settings of the specified mapper, using the configuration.
			Both the mapper pointer and the input configuration come from the same
			"Illustrative Cluster" object. 
			@param mapper			Mapper of a cluster. 
			@param configuration	Configuration of the same cluster. */

		void updateMapperSettings(vtkSmartPointer<vtkIllustrativeFiberBundleMapper> mapper,
									const IllustrativeCluster::Configuration & configuration);

		/** Update the mapper of the specified cluster, using the cluster's configuration.
			@param clusterId		Cluster index in the "clusterList" list. */

		void updateMapperSettings(int clusterID);

		/** Set a new color to a "QFrame" object. In the GUI, these frames are
			used to show previews of the line- and fill colors.
			@param colorFrame		Target GUI frame. 
			@param newColor			New color for the frame. */

		void setFrameColor(QFrame * colorFrame, QColor newColor);

		/** Return the current color of a "QFrame" object. 
			@param colorFrame		Target GUI frame. */

		QColor getFrameColor(QFrame * colorFrame);

		/** Delete a single cluster, and remove its actor from the assembly. 
			@param clusterID		Index of the target cluster. */

		void deleteCluster(int clusterID);

		/** Displacement manager, takes care of moving the cluster actors around
			in such a way that the focus area or cluster is clearly visible. */

		IllustrativeClusterDisplacement * displacement;

		/** List of predefined line colors. Used to create nice initial colors. */

		QStringList lineColorList;

		/** List of predefined fill colors. Used to create nice initial colors. */

		QStringList fillColorList;

		/** Index for the "lineColorList" and "fillColorList" lists. Incremented	
			whenever a new data set is added. */

		int colorListIndex;

		/** Timer, used to animate the displacement of clusters. */

		QTimer * mTimerAnimation;


}; // class IllustrativeClustersPlugin


} // namespace bmia


#endif // bmia_IllustrativeClustersPlugin_h