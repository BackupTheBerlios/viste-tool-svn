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
 * StructuredPointsReaderPlugin.cxx
 *
 * 2010-10-20	Evert van Aart
 * - First version. Added this to allow reading of ".clu" files.
 *
 * 2010-12-10	Evert van Aart
 * - Added automatic shortening of file names.
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.0.
 * - Improved progress reporting.
 *
 */


/** Includes */

#include "StructuredPointsReaderPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

StructuredPointsReaderPlugin::StructuredPointsReaderPlugin() : plugin::Plugin("PolyData Reader")
{

}


//------------------------------[ Destructor ]-----------------------------\\

StructuredPointsReaderPlugin::~StructuredPointsReaderPlugin()
{

}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList StructuredPointsReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;

    list.push_back("clu");	// Clustering information

	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList StructuredPointsReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("Clustering Information");
	return list;
}

//-----------------------------[ Constructor ]-----------------------------\\

void StructuredPointsReaderPlugin::loadDataFromFile(QString filename)
{
    // Write initial message to the log
    this->core()->out()->logMessage("Trying to load data from file "+filename);

    // Create a new reader and set the filename
    vtkStructuredPointsReader * reader = vtkStructuredPointsReader::New();
    reader->SetFileName(filename.toAscii().data());
    
    // Start a progress dialog that shows the progress of reading the data
    this->core()->out()->createProgressBarForAlgorithm(reader, "Structured Points Reader", "Reading VTK Structured Points file...");

    // Read the data from file by updating the VTK reader object.
    reader->Update();

    // Stop reporting progress
    this->core()->out()->deleteProgressBarForAlgorithm(reader);

	// Set the output data set name
	QDir dir(filename);
	QFileInfo fi(filename);
	QString name = fi.dir().dirName() +"/"+ fi.baseName();

    // Set the kind of the output data set
    QString kind = "";
    if (filename.endsWith("clu")) 
	{
		kind = "clusters";
	}

	Q_ASSERT(!kind.isEmpty());

	// Create a new data set
    data::DataSet * ds = new data::DataSet(name, kind, reader->GetOutput());

    // Add the new data set to the data manager
    this->core()->data()->addDataSet(ds);

    // Delete the reader that was used to read the data
    reader->Delete(); 
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libStructuredPointsReaderPlugin, bmia::StructuredPointsReaderPlugin)
