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

/**
 * FiberOutputPlugin.h
 *
 * 2010-12-21	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_FiberOutputPlugin_h
#define bmia_FiberOutputPlugin_h


/** Define the UI class */

namespace Ui 
{
	class FiberOutputForm;
}


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - GUI */

#include "ui_FiberOutput.h"

/** Includes - Custom Files */

#include "FiberOutput.h"
#include "FiberOutputTXT.h"
#include "FiberOutputXML.h"

/** Includes - Qt */

#include <QDebug>
#include <QFileDialog>
#include <QMessageBox>


namespace bmia {


/** Outputs DTI data. Users can select scalar images as output (e.g., FA, Cl, etcera);
	in addition, they can output the tensor values and the eigenvectors of the tensors.
	The data source (i.e., the points for which the data is sent to the output) can be
	either seed points, or fibers. When using seed points, the seeding option should be
	set to "Seed Voxels", although this plugin does not verify this. When using fibers,
	data will be written for each point along the fiber. Additionally, this plugin can
	compute and output the length of the fibers, and their volume. Mean values and
	variances can be computed for all measures. Output can be written as either a group
	of ".txt" files, or a single ".xml" file (which can be loaded in Excel).
*/

class FiberOutputPlugin : 	public plugin::Plugin,
							public data::Consumer,
							public plugin::GUI
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)
	Q_INTERFACES(bmia::plugin::GUI)

	public:

		/** Constructor */

		 FiberOutputPlugin();

		 /** Destructor */

		~FiberOutputPlugin();

		/** Returns a pointer to the GUI object */

		QWidget * getGUI();

		/** The form created by Qt Designer */

		Ui::FiberOutputForm * ui;

		/** Define behaviour when new data sets are added to the data manager, and
			when existing sets are removed or modified. This defines the consumer 
			interface of the plugin. */

		void dataSetAdded(data::DataSet * ds);
		void dataSetChanged(data::DataSet * ds);
		void dataSetRemoved(data::DataSet * ds);

	protected slots:

		/** Enable or disable controls based on the current control states. Also hides
			and shows GUI controls (e.g., when the data source has been set to "fibers", 
			the list containing all ROIs is hidden, since we do not need it). */

		void enableControls();

		/** Write output as a group of ".txt" files. Creates a "FiberOutputTXT" object,
			and passes it to "writeOutput". Also asks for the filename. */

		void writeTXT();

		/** Write output as a single ".xml" file. Creates a "FiberOutputXML" object,
			and passes it to "writeOutput". Also asks for the filename. */

		void writeXML();

	private:

		/** Write the output file(s) using the specified "FiberOutput" object. 
			@param out		Output writer, either for ".txt" or ".xml" files.
			@param fileName	Target filename, obtained through a file dialog. */

		void writeOutput(FiberOutput * out, QString fileName);

		/** In the DTITool, scalar volumes (like the FA image) are generally
			named "<DTI Image Name> <Short Measure Name>" (e.g., "MyLongImageName FA").
			When writing the output, we only want the short measure name, so in this
			function, we remove everything up to and including the last space. Of course,
			this does not work if the image name does not fit the above format, so a
			more elegant solution should be implemented in the future. For example,
			we could pass the short measure name as an attribute.
			@param longName	Long measure name. */

		std::string getShortMeasureName(QString longName);

		/** The Qt widget returned by "getGUI" */

		QWidget * widget;

		/** Lists containing data set pointers for each of the five supported data set
			types: DTI Tensors ("DTI"), Eigensystems ("eigen"), scalar measures ("scalar 
			volume"), fiber sets ("fibers"), and seed points ("seed points"). */

		QList<data::DataSet *> dtiImageDataSets;
		QList<data::DataSet *> eigenImageDataSets;
		QList<data::DataSet *> measureImageDataSets;
		QList<data::DataSet *> fiberDataSets;
		QList<data::DataSet *> seedDataSets;


}; // FiberOutputPlugin


} // namespace bmia


#endif // bmia_FiberOutputPlugin_h