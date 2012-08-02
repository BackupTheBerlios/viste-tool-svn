#ifndef bmia_ClusteringSettingsIO_h
#define bmia_ClusteringSettingsIO_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Qt */

#include <QFileDialog>
#include <QTableWidget>
#include <QTextStream>
#include <QComboBox>
#include <QFile>


namespace bmia {


/** Forward Class Declarations */

class ClusteringPlugin;

/** Class in charge of reading and writing ".bun" files, which contain, for each
	input cluster ID in the "ClusteringPlugin" class, the name of the selected
	output cluster (e.g., "0 Output Cluster A", "1 Output Cluster B", etcetera).
	When saving settings or loading a settings file, a file dialog is created using
	"QFileDialog". */

class ClusteringSettingsIO
{
	public:

		/** Constructor */

		ClusteringSettingsIO();

		/** Destructor */

		~ClusteringSettingsIO();

		/** Write the settings to an output file. 
			@param table	Table widget of the clustering plugin. */

		static void writeOutputFile(QTableWidget * table);

		/** Opens a setting file, prepares it for reading. */

		bool openFileForReading();

		/** Check if all cluster IDs in the settings file are within the
			range defined by the clustering information file selected in the
			clustering plugin (".clu" file). 
			@param maxInputID	Maximum cluster ID. */

		bool checkClusteringIDs(int maxInputID);

		/** Add output cluster names in the settings files to the combo
			boxes in the clustering plugin GUI. 
			@param plugin		Pointer to the clustering plugin. */

		void populateOutputClusters(ClusteringPlugin * plugin);

		/** Set output cluster of each input cluster to the one defined
			in the settings file. 
			@param table	Table widget of the clustering plugin. */

		void setOutputClusters(QTableWidget * table);

	private:

		/** Input file pointer. */

		QFile in;

		/** Text stream used to read input file. */

		QTextStream instream;

		/** Initial position of input file. */

		qint64 start;

};

}

#endif // bmia_ClusteringSettingsIO_h