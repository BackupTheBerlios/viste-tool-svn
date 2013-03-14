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
 * NiftiReaderPlugin.cxx
 *
 * 2010-08-11	Tim Peeters
 * - First version.
 *
 * 2011-03-21	Evert van Aart
 * - Version 1.0.0.
 * - Turned off debug mode of the reader, reducing its verbosity. 
 * - Reformatted code, added more comments. 
 *
 * 2011-04-04	Evert van Aart
 * - Version 2.0.0.
 * - Completely rebuilt the reader. The new reader is capable of dealing with 
 *   MiND extensions, which are necessary for reading in HARDI data. Furthermore,
 *   the new reader is more easy to extend with support of other data types.
 *
 * 2011-05-10	Evert van Aart
 * - Version 2.1.0.
 * - Added support for spherical harmonics using MiND. 
 *
 * 2011-08-22	Evert van Aart
 * - Version 2.1.1.
 * - Which transformation matrix to use is now determined correctly based on the
 *   "qform_code" and "sform_code" of the NIfTI image.
 *
 *  2013-02-06 Mehmet Yusufoglu
 *  core()->out() is passed to bmiaNiftiReader constructor so that the reader can ask to the user
 * which transformation will be used if qform_code and sform_code both are positive.
 */


/** Includes */


#include "NiftiReaderPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

NiftiReaderPlugin::NiftiReaderPlugin() : plugin::Plugin("NIfTI Reader")
{
	
}


//------------------------------[ Destructor ]-----------------------------\\

NiftiReaderPlugin::~NiftiReaderPlugin()
{
	
}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList NiftiReaderPlugin::getSupportedFileExtensions()
{
	QStringList list;
	list.push_back("nii");
	//list.push_back("nii.gz");
	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList NiftiReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("NIfTI Files");
	//list.push_back("GZipped NIfTI Files");
	return list;
}


 

//--------------------------[ loadDataFromFile ]---------------------------\\

void NiftiReaderPlugin::loadDataFromFile(QString filename)
{
    // Write a message to the log
    this->core()->out()->logMessage("Trying to load data from file " + filename + ".");

    // Create a new reader object and set the filename
	bmiaNiftiReader * reader = new bmiaNiftiReader(this->core()->out());
    
    // Read the data from file by updating the VTK reader object
    QString err = reader->readNIfTIFile(filename.toLatin1().data());

	// If something went wrong and/or no output data was created, display an error
	if (err.isEmpty() == false || reader->outData.size() == 0)
	{
		QMessageBox::warning(NULL, "NIfTI Reader", "Reading NIfTI file failed with the following error:\n\n" + err);
		delete reader;
		return;
	}

	QString dsName;

	// Get the filename without its path by removing everything up to and including
	// the last slash or backslash.

	if (filename.contains("/"))
	{
		int lastSlash = filename.lastIndexOf("/");
		dsName = filename.right(filename.length() - lastSlash - 1);
	} 
	else if (filename.contains("\\"))
	{
		int lastBSlash = filename.lastIndexOf("\\");
		dsName = filename.right(filename.length() - lastBSlash - 1);
	}

	// Also remove the ".nii" extension
	if (dsName.contains("."))
	{
		dsName = dsName.left(dsName.lastIndexOf("."));
	}

	data::DataSet * ds;

	// Get the transformation matrix from the NIfTI file
	vtkMatrix4x4 * m = reader->getTransformMatrix();

	// Switch based on the data type of the NIfTI file
	switch(reader->imageDataType)
	{
		// Single scalar volume
		case bmiaNiftiReader::NDT_ScalarVolume:
			ds = new data::DataSet(dsName, "scalar volume", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// DTI tensors (second-order)
		case bmiaNiftiReader::NDT_DTITensors:
			ds = new data::DataSet(dsName, "DTI", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// Discrete sphere function
		case bmiaNiftiReader::NDT_DiscreteSphere:
			ds = new data::DataSet(dsName, "discrete sphere", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// Spherical harmonics
		case bmiaNiftiReader::NDT_SphericalHarm:
			ds = new data::DataSet(dsName, "spherical harmonics", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// Triangle files always describe the geometry for another data set, e.g.,
		// for a set of vertices on a sphere, the triangle set describes the triangles
		// connecting these vertices. Therefore, it makes no sense to have a set 
		// of triangle definitions without the accompanying geometry. Here, we
		// warn the user that triangle NIfTI files should not be opened directly;
		// instead, they should automatically be loaded by the NIfTI reader.

		case bmiaNiftiReader::NDT_Triangles:
			QMessageBox::warning(NULL, "NIfTI Reader", QString("NIfTI files of intent type NIFTI_INTENT_TRIANGLE should not be read directly.\n") + 
									QString("If the target folder contains a file with the same name as the selected NIfTI file, but without the\n") + 
									QString("'_geom' extension, you should read this file instead."));
			break;

		// Generic vectors
		case bmiaNiftiReader::NDT_GenericVector:

			// Loop through all output volumes
			for (int i = 0; i < reader->outData.size(); ++i)
			{
				ds = new data::DataSet(dsName + " [" + QString::number(i) + "/" + 
											QString::number(reader->outData.size())
											+ "]", "scalar volume", reader->outData.at(i));
				this->addDataSet(ds, m);
			}

			break;

		default:

			// This shouldn't happen
			this->core()->out()->showMessage("Failed to load NIfTI file!");
			return;
	}

    // Delete the reader that was used to read the data
	delete reader;
}


//------------------------------[ addDataSet ]-----------------------------\\

void NiftiReaderPlugin::addDataSet(bmia::data::DataSet * ds, vtkMatrix4x4 * m)
{
	// If we've got a transformation matrix, add it to the data set
	if (m)
	{
		ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(m));
	}

	// Add the new data set to the data manager
	this->core()->data()->addDataSet(ds);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libNiftiReaderPlugin, bmia::NiftiReaderPlugin)
