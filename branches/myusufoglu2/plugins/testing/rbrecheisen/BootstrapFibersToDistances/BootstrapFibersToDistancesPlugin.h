#ifndef bmia_BootstrapFibersToDistances_BootstrapFibersToDistancesPlugin_h
#define bmia_BootstrapFibersToDistances_BootstrapFibersToDistancesPlugin_h

// Includes DTITool
#include "DTITool.h"
#include <vtkDistanceTable.h>

// Includes plugin UIC
#include "ui_BootstrapFibersToDistancesPlugin.h"

// Includes QT
#include <QList>
#include <QMap>

// Includes VTK
#include <vtkPolyData.h>
#include <vtkIntArray.h>

/** @class BootstrapFibersToDistancesPlugin
	@brief Computes distances between fiber originating at same seed point

	This class works together closely with \see BootstrapFiberTrackingPlugin.
	It computes pairwise distances between streamlines originating from the
	same seed point. Based on these distances we can determine a 'mean'
	streamline. The output of this class is a table containing the distance
	of each streamline to the 'mean' streamline associated with its the
	same seed point */
namespace bmia
{
	class BootstrapFibersToDistancesPlugin :	public plugin::Plugin,
												public plugin::GUI,
												public data::Consumer
	{
		Q_OBJECT
		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::GUI )
		Q_INTERFACES( bmia::data::Consumer )

	public:

		/** Constructor */
		BootstrapFibersToDistancesPlugin();

		/** Destructor */
		virtual ~BootstrapFibersToDistancesPlugin();

		/** Returns plugin's QT widget
			@return The widget containing this plugin's UI */
		QWidget * getGUI();

		/** Handles datasets just added to the reposity
			@param dataset The dataset */
		void dataSetAdded( data::DataSet * dataset );

		/** Handles datasets just removed from the reposity
			@param dataset The dataset */
		void dataSetRemoved( data::DataSet * dataset );

		/** Handles datasets just changed in the reposity
			@param dataset The dataset */
		void dataSetChanged( data::DataSet * dataset );

	private:

		/** Sets up signal/slot connections for all GUI components
			of this plugin */
		void setupConnections();

		/** Sets up the UI's combo box containing the names of
			different distance measures */
		void setupDistanceMeasures();

	private slots:

		/** Computes a distance table for the selected fibers
			and fiber ID's */
		void compute();

		/** Loads fibers and fiber ID's from disk and adds them
			to the data repository */
		void load();

		/** Saves currently selected fibers and fiber ID's */
		void save();

	private:

		QWidget									* _widget;				// The QT widget holding this plugin's GUI
		QList< vtkPolyData * >					_fiberSets;				// The list of loaded fiber sets
		QList< vtkIntArray * >					_fiberIdSets;			// The list of loaded fiber ID sets
		QList< int >							_numberSeedPointsList;	// The list with number of seed points
		QMap< QString, vtkDistanceTable * >		_tableSets;				// The list of distance tables
		Ui::BootstrapFibersToDistancesForm		* _form;				// UI form describing plugin's GUI
	};
}

#endif
