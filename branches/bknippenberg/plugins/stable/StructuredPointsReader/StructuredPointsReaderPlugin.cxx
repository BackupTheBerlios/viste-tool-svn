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
