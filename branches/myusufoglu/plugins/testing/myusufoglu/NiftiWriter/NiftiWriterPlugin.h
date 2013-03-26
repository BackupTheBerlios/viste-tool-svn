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
 * bmiaNiftiWriter.cxx
 *
 * * 2013-03-16   Mehmet Yusufoglu
 * - Create the class. Writes the scalar data in Nifti format.
 * - 
 */


#ifndef bmia_NiftiWriterPlugin_h
#define bmia_NiftiWriterPlugin_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "bmiaNiftiWriter.h"

/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <QString>
#include <QStringList>
#include <QMessageBox>


namespace bmia {


/** This plugin reads NIfTI (.nii) files, and interprets them. The output is 
	always a data set containing a "vtkImageData" object, but the kind of the data
	set depends on the parameters of the NIfTI file. For example, if the file 
	has six components and its intent code is set to "SYMMATRIX", we interpret
	it as a set of DTI tensors. 
*/

class NiftiWriterPlugin : public plugin::Plugin, public data::Writer
{
    Q_OBJECT
    Q_INTERFACES(bmia::plugin::Plugin)
    Q_INTERFACES(bmia::data::Writer)

	public:

		/** Return the current version of the plugin. */

		QString getPluginVersion()
		{
			return "2.1.1";
		}

		/** Constructor */
    
		NiftiWriterPlugin();

		/** Destructor */

		~NiftiWriterPlugin();

		/** Returns the list of file extensions supported by this reader plugin. */
    
		QStringList getSupportedFileExtensions();

		/** Returns a list containing short descriptions of the supported file
			types. The number of descriptions and their order should match those
			of the list returned by "getSupportedFileExtensions". */

		QStringList getSupportedFileDescriptions();

 

		void niftiStructure(data::DataSet *ds);

		 void writeDataToFile(QString filename, data::DataSet *ds);


	private:

 

		
}; // class NiftiWriterPlugin


} // namespace bmia


#endif // bmia_NiftiWriterPlugin_h
