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
 * HARDIMeasuresPlugin.h
 *
 * 2011-04-29	Evert van Aart
 * - Version 1.0.0.
 * - First version.
 *
 * 2011-05-04	Evert van Aart
 * - Version 1.0.1.
 * - Added the volume measure.
 *
 * 2011-05-13	Evert van Aart
 * - Version 1.0.2.
 * - Improved attribute handling.
 *
 * 2011-08-05	Evert van Aart
 * - Version 1.0.3.
 * - Fixed an error in the computation of the unit vectors.
 *
 */


#ifndef bmia_HARDIMeasures_HARDIMeasuresPlugin_h
#define bmia_HARDIMeasures_HARDIMeasuresPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Qt */

#include <QDebug>
#include <QList>

/** Includes - VTK */

#include <vtkExecutive.h>
#include <vtkMatrix4x4.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkAlgorithm.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkObject.h>

/** Includes - Custom Files */

#include "HARDI/SphereTriangulator.h"
#include "vtkDiscreteSphereToScalarVolumeFilter.h"


namespace bmia {


/** This plugin converts a volume containing HARDI data to one or more scalar
	measure volumes, which can then be visualized using for example the Planes
	Visualization plugin. All output volumes are computed by request, meaning
	that the output volume will be empty until data is requested. Thus, plugins
	using data sets generated by this plugin should use the "Update" function
	on the "vtkImageData" object before using it.
*/

class HARDIMeasuresPlugin :	public plugin::Plugin, 
							public data::Consumer
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Consumer)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "1.0.3";
		}

		/** Constructor */

		HARDIMeasuresPlugin();

		/** Destructor */

		~HARDIMeasuresPlugin();

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

	protected:

	
	private:

		/** Structure to keep track of all outputs for a specific input image. */

		struct OutputInformation
		{
			data::DataSet * input;
			QList<data::DataSet *> outputs;
		};

		/** List of all input images that have been handled by this plugin. */

		QList<OutputInformation> images;

		/** List of all filters used to compute scalar measures for discrete sphere function volumes. */

		QList<vtkAlgorithm *> discreteSphereFilters;

		/** Find an input image in the list of images, and return its index. If
			the specified data set is not in the list of input images, -1 is returned.
			@param ds		Input data set. */

		int findInputImage(data::DataSet * ds);

		/** Find the indices of an output data set. If the specified data set
			has been generated by this plugin, true is returned, and the indices
			of the image (within the "images" list, and within the "outputs" list
			of the corresponding "OutputInformation" object) are set. If the
			target pointer is not stored in any list, false is returned, and
			the indices remain unchanged.
			@param ds		Output data set.
			@param imageID	Index of the "OutputInformation" object containing "ds".
			@param outID	Index of "ds" within the "outputs" list. */

		bool findOutputImage(data::DataSet * ds, int & imageID, int & outID);

}; // class HARDIMeasuresPlugin


} // namespace bmia


#endif // bmia_HARDIMeasures_HARDIMeasuresPlugin_h