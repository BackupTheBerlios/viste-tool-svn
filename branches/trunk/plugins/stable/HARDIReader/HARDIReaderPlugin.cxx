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
 * HARDIReaderPlugin.h
 *
 * 2010-11-29	Evert van Aart
 * - First version.
 *
 * 2011-01-24	Evert van Aart
 * - Added support for transformation matrices
 *
 * 2011-04-19	Evert van Aart
 * - Version 1.0.0.
 * - Raw HARDI data is now outputted in the format excepted by the Geometry Glyphs
 *   plugin, so with an array of angles and an array defining the triangles.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


/** Includes */

#include "HARDIReaderPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

HARDIReaderPlugin::HARDIReaderPlugin() : plugin::Plugin("HARDI Reader")
{

}


//------------------------------[ Destructor ]-----------------------------\\

HARDIReaderPlugin::~HARDIReaderPlugin()
{

}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList HARDIReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;

    list.push_back("hardi");	// Raw HARDI data
	list.push_back("sharm");	// Spherical Harmonics

	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList HARDIReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;

	list.push_back("Raw HARDI Data");
	list.push_back("Spherical Harmonics");

	return list;
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void HARDIReaderPlugin::loadDataFromFile(QString filename)
{
    // Write initial message to the log
    this->core()->out()->logMessage("Trying to load data from file "+filename);

	// Set the output data set name
    QDir dir(filename);
    QFileInfo fi(filename);
    QString name = fi.dir().dirName() +"/"+ fi.baseName();

    // Set the kind of the output data set
    QString kind = "";
    if (filename.endsWith("hardi")) 
	{
		kind = "discrete sphere";
	}
	else if (filename.endsWith("sharm")) 
	{
		kind = "spherical harmonics";
	}

	Q_ASSERT(!kind.isEmpty());
	
	// Reader output
	vtkImageData * out = NULL;
	
	// Output data set
	data::DataSet * ds = NULL;
	
	// Raw HARDI data files (one ".hardi" file and N ".dat" files)
	if (kind == "discrete sphere")
	{
		// Create a new reader and set the filename
		vtkHARDIReader * reader = vtkHARDIReader::New();
		reader->SetFileName(filename.toAscii().data());
		
		// Start a progress dialog that shows the progress of reading the data
		this->core()->out()->createProgressBarForAlgorithm(reader, "HARDI Reader");

		// Read the data from file by updating the VTK reader object.
		reader->Update();

		// Stop reporting progress
		this->core()->out()->deleteProgressBarForAlgorithm(reader);
		
		// Get output
		ds = new data::DataSet(name, kind, reader->GetOutput());
		
		std::string err;

		// Try to read the transformation matrix from a ".tfm" file
		vtkMatrix4x4 * m = TransformationMatrixIO::readMatrix(filename.toStdString(), err);

		// If we succeeded, add the matrix to the data set
		if (m)
		{
			ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(m));
		}
		// If an error occurred while reading the matrix file, print it
		else if (!(err.empty()))
		{
			this->core()->out()->showMessage(QString(err.c_str()));
		}

		// Get the attributes of the data set
		data::Attributes * dsAtt = ds->getAttributes();

		// The attributes should always exist
		Q_ASSERT(dsAtt);

		// Set the B-value and the two gradient arrays as attributes
		dsAtt->addAttribute("B-value", reader->getB());
		dsAtt->addAttribute("gradientArray", (vtkObject *) reader->getGradientArray());

		// Add the new data set to the data manager
		this->core()->data()->addDataSet(ds);

		// Delete the reader that was used to read the data
		reader->Delete(); 
		
	} // if [kind == "hardi-raw"]

	// Spherical harmonics (one ".sharm" file and N ".dat" files)
	else if (kind == "spherical harmonics")
	{
		// Create a new reader and set the filename
		vtkSHARMReader * reader = vtkSHARMReader::New();
		reader->SetFileName(filename.toAscii().data());

		// Start a progress dialog that shows the progress of reading the data
		this->core()->out()->createProgressBarForAlgorithm(reader, "HARDI Reader");

		// Read the data from file by updating the VTK reader object.
		reader->Update();

		// Stop reporting progress
		this->core()->out()->deleteProgressBarForAlgorithm(reader);

		// Get output
		ds = new data::DataSet(name, kind, reader->GetOutput());

		std::string err;

		// Try to read the transformation matrix from a ".tfm" file
		vtkMatrix4x4 * m = TransformationMatrixIO::readMatrix(filename.toStdString(), err);

		// If we succeeded, add the matrix to the data set
		if (m)
		{
			ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(m));
		}
		// If an error occurred while reading the matrix file, print it
		else if (!(err.empty()))
		{
			this->core()->out()->showMessage(QString(err.c_str()));
		}

		// Get the attributes of the data set
		data::Attributes * dsAtt = ds->getAttributes();

		// The attributes should always exist
		Q_ASSERT(dsAtt);

		// Get the list of parameters loaded by the reader
		std::list<std::string> attributeNames  = reader->getParamNames();
		std::list<double>      attributeValues = reader->getParamValues();

		// Create iterators for both lists
		std::list<std::string>::iterator nameIter;
		std::list<double>::iterator      valueIter;

		// Loop through all parameters, and save them as attributes
		for ( nameIter  = attributeNames.begin(), valueIter  = attributeValues.begin();
			  nameIter != attributeNames.end(),   valueIter != attributeValues.end();
			  nameIter++, valueIter++ )
		{
			dsAtt->addAttribute(QString((*nameIter).c_str()), (*valueIter));
		}

		// Add the new data set to the data manager
		this->core()->data()->addDataSet(ds);

		// Delete the reader that was used to read the data
		reader->Delete(); 

	} // if [kind == "spherical harmonics"]

}


} // namespace bmia


Q_EXPORT_PLUGIN2(libHARDIReaderPlugin, bmia::HARDIReaderPlugin)
