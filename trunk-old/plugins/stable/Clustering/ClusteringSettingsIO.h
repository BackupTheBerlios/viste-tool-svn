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