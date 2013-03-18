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


/** Includes */


#include "NiftiWriterPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

NiftiWriterPlugin::NiftiWriterPlugin() : plugin::Plugin("NIfTI Writer")
{
	
}


//------------------------------[ Destructor ]-----------------------------\\

NiftiWriterPlugin::~NiftiWriterPlugin()
{
	
}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList NiftiWriterPlugin::getSupportedFileExtensions()
{
	QStringList list;
	list.push_back("nii");
	//list.push_back("nii.gz");
	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList NiftiWriterPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("NIfTI Files");
	//list.push_back("GZipped NIfTI Files");
	return list;
}


 

//--------------------------[ loadDataFromFile ]---------------------------\\

void NiftiWriterPlugin::loadDataFromFile(QString filename)
{
    // Write a message to the log
    this->core()->out()->logMessage("Trying to load data from file " + filename + ".");

    // Create a new reader object and set the filename
	bmiaNiftiWriter * reader = new bmiaNiftiWriter(this->core()->out());
    
    // Read the data from file by updating the VTK reader object
    QString err = reader->readNIfTIFile(filename.toLatin1().data());

	// If something went wrong and/or no output data was created, display an error
	if (err.isEmpty() == false || reader->outData.size() == 0)
	{
		QMessageBox::warning(NULL, "NIfTI Writer", "Writing NIfTI file failed with the following error:\n\n" + err);
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
		case bmiaNiftiWriter::NDT_ScalarVolume:
			ds = new data::DataSet(dsName, "scalar volume", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// DTI tensors (second-order)
		case bmiaNiftiWriter::NDT_DTITensors:
			ds = new data::DataSet(dsName, "DTI", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// Discrete sphere function
		case bmiaNiftiWriter::NDT_DiscreteSphere:
			ds = new data::DataSet(dsName, "discrete sphere", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// Spherical harmonics
		case bmiaNiftiWriter::NDT_SphericalHarm:
			ds = new data::DataSet(dsName, "spherical harmonics", reader->outData.at(0));
			this->addDataSet(ds, m);
			break;

		// Triangle files always describe the geometry for another data set, e.g.,
		// for a set of vertices on a sphere, the triangle set describes the triangles
		// connecting these vertices. Therefore, it makes no sense to have a set 
		// of triangle definitions without the accompanying geometry. Here, we
		// warn the user that triangle NIfTI files should not be opened directly;
		// instead, they should automatically be loaded by the NIfTI reader.

		case bmiaNiftiWriter::NDT_Triangles:
			QMessageBox::warning(NULL, "NIfTI Writer", QString("NIfTI files of intent type NIFTI_INTENT_TRIANGLE should not be read directly.\n") + 
									QString("If the target folder contains a file with the same name as the selected NIfTI file, but without the\n") + 
									QString("'_geom' extension, you should read this file instead."));
			break;

		// Generic vectors
		case bmiaNiftiWriter::NDT_GenericVector:

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

void NiftiWriterPlugin::addDataSet(bmia::data::DataSet * ds, vtkMatrix4x4 * m)
{
	// If we've got a transformation matrix, add it to the data set
	if (m)
	{
		ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(m));
	}

	// Add the new data set to the data manager
	this->core()->data()->addDataSet(ds);
}
void NiftiWriterPlugin::writeDataToFile(QString filename, vtkImageData *image)
{

}


} // namespace bmia


Q_EXPORT_PLUGIN2(libNiftiWriterPlugin, bmia::NiftiWriterPlugin)
