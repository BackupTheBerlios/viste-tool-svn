/*
 * PolyDataReaderPlugin.cxx
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


/** Includes */

#include "PolyDataReaderPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

PolyDataReaderPlugin::PolyDataReaderPlugin() : plugin::Plugin("PolyData reader")
{

}


//-----------------------------[ Destructor ]------------------------------\\

PolyDataReaderPlugin::~PolyDataReaderPlugin()
{

}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList PolyDataReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;
    list.push_back("fbs");	// Fibers
    list.push_back("vtk");	// VTK PolyData
    list.push_back("pol");	// Seeding ROI
	list.push_back("sr");	// Seeding ROI (DTITool2 format)
	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList PolyDataReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("Fibers");
	list.push_back("VTK PolyData");
	list.push_back("Region of Interest");
	list.push_back("Region of Interest (DTITool2)");
	return list;
}


//--------------------------------[ fixROI ]-------------------------------\\

void PolyDataReaderPlugin::fixROI(vtkPolyData * roi)
{
	// The ROI only needs to be fixed if the number of lines is one
	// less than the number of points; this means that each line segment
	// is stored as its own line, instead of using one line per polygon.

	if (roi->GetNumberOfPoints() != (roi->GetNumberOfLines() + 1))
		return;

	// Get the points of the ROI
	vtkPoints * points = roi->GetPoints();

	if (!points)
		return;

	// Copy all point IDs to a new list
	vtkIdList * roiList = vtkIdList::New();
	for (vtkIdType ptId = 0; ptId < points->GetNumberOfPoints(); ++ptId)
	{
		roiList->InsertNextId(ptId);
	}

	// Create a new lines array, and insert the new point list
	vtkCellArray * lines = vtkCellArray::New();
	lines->InsertNextCell(roiList);
	roi->SetLines(lines);
	lines->Delete();
	roiList->Delete();
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void PolyDataReaderPlugin::loadDataFromFile(QString filename)
{
	// Write a message to the log
	this->core()->out()->logMessage("Trying to load data from file " + filename);

	// Create a new reader object and set the filename
	vtkPolyDataReader * reader = vtkPolyDataReader::New();
	reader->SetFileName(filename.toAscii().data());
    
	// Start a progress dialog that shows the progress of reading the data
	this->core()->out()->createProgressBarForAlgorithm(reader, "PolyData Reader", "Reading VTK PolyData file...");

	// Read the data from file by updating the VTK reader object.
	reader->Update();

	// Stop reporting progress
	this->core()->out()->deleteProgressBarForAlgorithm(reader);

	// Define the name of the dataset as the filename.
	QDir dir(filename);
	QFileInfo fi(filename);
	QString name = fi.dir().dirName() +"/"+ fi.baseName();

	// Construct a new DataSet object with that has the output
	// of the VTK reader as VTK data set.
	QString kind = "";

	if (filename.endsWith(".fbs"))		
	{
		kind = "fibers";
	}
	else if (filename.endsWith(".vtk"))	
	{
		kind = "polydata";
	}
	else if (filename.endsWith(".pol"))	
	{
		kind = "regionOfInterest";
	}
	else if (filename.endsWith(".sr"))	
	{
		kind = "regionOfInterest";
	}

	// This is theoretically impossible
	Q_ASSERT(!kind.isEmpty());

	data::DataSet * ds = NULL;

	// For fibers and generic polydata, we do not need to perform any 
	// additional steps.
	if (kind == "fibers" || kind == "polydata")
	{
		ds = new data::DataSet(name, kind, reader->GetOutput());
	}
	// For ROIs, we need to check if the ROI polydata has been stored
	// in the correct format (i.e., one line per polygon, instead of one
	// line per line segment). If this is not the case, we "fix" it here.
	else
	{
		vtkPolyData * roi = reader->GetOutput();
		this->fixROI(roi);
		ds = new data::DataSet(name, kind, roi);
	}

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

	// Add the new DataSet to the data manager:
	this->core()->data()->addDataSet(ds);

	// Delete the reader that was used to read the data.
	reader->Delete(); reader = NULL;
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libPolyDataReaderPlugin, bmia::PolyDataReaderPlugin)
