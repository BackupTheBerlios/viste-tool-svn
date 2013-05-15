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
 * PolyDataReaderPlugin.h
 *
 * 2010-01-20	Tim Peeters
 * - First version
 *
 * 2010-06-23	Tim Peeters
 * - Add some comments to use this reader as an example plugin.
 * - Rename from GeometryReaderPlugin to PolyDataReaderPlugin
 *
 * 2011-01-24	Evert van Aart
 * - Added support for reading transformation matrix files.
 *
 * 2011-02-07	Evert van Aart
 * - Automatically fix ROIs that were stored using the wrong format.
 *
 * 2011-03-09	Evert van Aart
 * - Version 1.0.0.
 * - Enabled reading of ".sr" files, which were the seeding region files of the
 *   old DTITool. These files are handled in the same way as ".pol" files.
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


#ifndef bmia_PolyDataReader_PolyDataReaderPlugin_h
#define bmia_PolyDataReader_PolyDataReaderPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "Helpers/TransformationMatrixIO.h"

/** Includes - VTK */

#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkMatrix4x4.h>
#include <vtkIdList.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>

/** Includes - Qt */

#include <QtCore/QFileInfo>
#include <QtCore/QDir>


namespace bmia {


/** This reader plugin enables us to read in VTK polydata files. The data type 
	of the output data set depends on the extension of the input file.
*/

class PolyDataReaderPlugin : public plugin::Plugin, public data::Reader
{
	Q_OBJECT
	Q_INTERFACES(bmia::plugin::Plugin)
	Q_INTERFACES(bmia::data::Reader)

	public:

		/** Return the current plugin version. */

		QString getPluginVersion()
		{
			return "1.0.1";
		}

		/** Constructor */

		PolyDataReaderPlugin();

		/** Destructor */

		~PolyDataReaderPlugin();

		/** Returns the list of file extensions supported by this reader plugin. */
    
		QStringList getSupportedFileExtensions();

 		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

	
		/** Load polydata data from the given file.
			@param filename	Input file name. */
    
		void loadDataFromFile(QString filename);

	protected:



	private:

		/** In some Region-of-Interest files, the ROIs are stored in a different way
			than we expect in the rest of the software. Throughout the tool, we use 
			one line per polygon, but in some input files, one line per line segment
			is used. This function first checks if the ROI is stored in such a way
			(in which case the number of lines will be one less than the number of
			points), and if so, it will change the ROI so that it only uses one line. 
			@param roi		Input ROI, read from a file. */

		void fixROI(vtkPolyData * roi);

}; // class PolyDataReaderPlugin


} // namespace bmia


#endif // bmia_PolyDataReader_PolyDataReaderPlugin_h
