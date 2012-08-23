/*
 * DTIReaderPlugin.cxx
 *
 * 2009-11-27	Tim Peeters
 * - First version
 *
 * 2011-01-14	Evert van Aart
 * - Structural information images are now added as separate
 *   scalar volume data sets, which allows them to be visualized
 *   using planes, volume mapping, etceta.
 *
 * 2011-01-24	Evert van Aart
 * - Added support for reading ".tfm" transformation matrix files.
 *
 * 2011-03-31	Evert van Aart
 * - Version 1.0.0.
 * - Allowed the reader to read doubles.
 *
 * 2011-04-21	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


/** Includes */

#include "DTIReaderPlugin.h"
#include "vtkDTIReader2.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

DTIReaderPlugin::DTIReaderPlugin() : plugin::Plugin("DTI Reader")
{

}


//------------------------------[ Destructor ]-----------------------------\\

DTIReaderPlugin::~DTIReaderPlugin()
{

}


//---------------------[ getSupportedFileExtensions ]----------------------\\

QStringList DTIReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;
    list.push_back("dti");
    return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList DTIReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("DTI Data");
	return list;
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void DTIReaderPlugin::loadDataFromFile(QString filename)
{
    // Create the DTI reader
    vtkDTIReader2 * reader = vtkDTIReader2::New();

	// Check if the reader is able to open the file
    if (!reader->CanReadFile(filename.toAscii().data()))
	{
		this->core()->out()->showMessage("Could not read data from '" + filename + "' using DTIReaderPlugin.");
		return;
	}

	// Report the progress of the reader
    this->core()->out()->createProgressBarForAlgorithm(reader, "DTI Reader");

	// Set the target file name
    reader->SetFileName(filename.toAscii().data());

	// Update the reader to read the DTI data
    reader->Update();

	// Stop reporting the progress of the reader
	this->core()->out()->deleteProgressBarForAlgorithm(reader);

	// Get the name of the data set
    QDir dir(filename);
    QFileInfo fi(filename);
    QString DTIName = fi.dir().dirName() +"/"+ fi.baseName();

	// Get the output
	vtkImageData * dti = reader->GetOutput();

	// Check if the output exists
	if (!dti)
	{
		this->core()->out()->showMessage("No DTI tensor data read from '" + filename + "'!");
		return;
	}

	// Add the DTI data set to the data manager
	data::DataSet * ds = new data::DataSet(DTIName, "DTI", reader->GetOutput());

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

	this->core()->data()->addDataSet(ds);

	// Set the name of the structural data
	QString StructName = DTIName + " [I]";

	// Get the structural data object
	vtkImageData * i = reader->getStructuralInformation();

	// Check if the image exists
	if (!i)
	{
		this->core()->out()->showMessage("Structural data ('I' file) could not be read from '" + filename + "'!");
	}
	else
	{
		// Add the structural data set to the data manager
		data::DataSet * ds = new data::DataSet(StructName, "scalar volume", i);

		// Add the matrix to the data set
		if (m)
		{
			ds->getAttributes()->addAttribute("transformation matrix", vtkObject::SafeDownCast(m));
		}

		ds->getAttributes()->addAttribute("isAI", 0.0);
		this->core()->data()->addDataSet(ds);
	}

	// Delete the reader
    reader->Delete(); 
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libDTIReaderPlugin, bmia::DTIReaderPlugin)
