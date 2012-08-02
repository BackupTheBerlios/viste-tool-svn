/*
 * VolumeReaderPlugin.cxx
 *
 * 2010-02-10	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-01-04	Evert van Aart
 * - Added additional comments, added more descriptive error messages.
 * - Remove "vtkDataHeaderReader" from this plugin, since it was not used.
 *
 * 2011-01-17	Evert van Aart
 * - Fixed reading for files with unsigned characters.
 *
 * 2011-02-11	Evert van Aart
 * - Version 1.0.0.
 * - Enabled reading of decimal spacing values (instead of integers).
 *
 * 2011-04-26	Evert van Aart
 * - Version 1.0.1.
 * - Improved progress reporting.
 *
 */


/** Includes */

#include "VolumeReaderPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

VolumeReaderPlugin::VolumeReaderPlugin() : plugin::Plugin("Volume Reader")
{

}


//------------------------------[ Destructor ]-----------------------------\\

VolumeReaderPlugin::~VolumeReaderPlugin()
{

}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList VolumeReaderPlugin::getSupportedFileExtensions()
{
	QStringList list;
	list.push_back("vol");
	return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList VolumeReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("Volume Data");
	return list;
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void VolumeReaderPlugin::loadDataFromFile(QString filename)
{
	// Create the Qt file handler
	QFile file(filename);

	// Try to open the file
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		this->core()->out()->logMessage("The file " + filename + " could not be opened!");
		return;
	}

	// Read the first line, which contains the name of the ".raw" file
	QString rawFileName = file.readLine();

	// Only use the part of the line after the equal sign
	int lastIndexOf = rawFileName.lastIndexOf('=');
	rawFileName = rawFileName.right(rawFileName.length() - lastIndexOf - 1);
	rawFileName = rawFileName.trimmed();

	// Create a file information object for the given ".raw" file
	QFileInfo rawFileInfo(rawFileName);

	// If the ".raw" file cannot be found directly...
	if (!rawFileInfo.exists())
	{
		// ...try to create a new file information object, which contains
		// an absolute path to the same directory containing the ".vol" file.

		QFileInfo tempFileInfo(filename);
		QDir dir = tempFileInfo.dir();
		rawFileName = dir.absolutePath() + "/" + rawFileName;
		tempFileInfo.setFile(rawFileName);

		// Break if this new file information object still does not exist
		if (!tempFileInfo.exists())
		{
			this->core()->out()->logMessage("File " + rawFileName + " in " + filename + " not found");
			file.close();
			return;
		}
	}

	// Read the second line, which contains the file type
	QString fileType = file.readLine();

	// Check if the file type exists
	if (fileType.isEmpty())
	{
		this->core()->out()->logMessage("Could not read file type in volume file " + filename + "!");
		file.close();
		return;
	}

	// Find the last occurrence of the equal sign in the second line
	lastIndexOf = fileType.lastIndexOf('=');

	// Check if the line contains an equal sign
	if (lastIndexOf == -1)
	{
		this->core()->out()->logMessage("Could not read file type in volume file " + filename + "!");
		file.close();
		return;
	}

	// Keep only the part after the equal sign
	fileType = fileType.right(fileType.length() - lastIndexOf - 1);
	fileType = fileType.trimmed();

	// Read the third line, which contains the dimensions
	QString strDimensions = file.readLine();

	// Check if the volume dimensions exist
	if (strDimensions.isEmpty())
	{
		this->core()->out()->logMessage("Could not read dimensions in volume file " + filename + "!");
		file.close();
		return;
	}

	// Find the last occurrence of the equal sign in the third line
	lastIndexOf = strDimensions.lastIndexOf('=');

	// Check if the line contains an equal sign
	if (lastIndexOf == -1)
	{
		this->core()->out()->logMessage("Could not read dimensions in volume file " + filename + "!");
		file.close();
		return;
	}

	// Keep only the part after the equal sign
	strDimensions = strDimensions.right(strDimensions.length() - lastIndexOf - 1);
	strDimensions = strDimensions.trimmed();

	// Vector containing the volume dimensions
	QVector<int> dimensions;

	// Current dimension
	int number;

	// True if reading the numbers was successful
	bool ok = true;

	// Split the string containing the dimensions into substrings
	QStringList strListDimensions = strDimensions.split(QRegExp("\\s+"), QString::SkipEmptyParts);

	// Convert the substrings to integers and add them to the vector
	foreach(QString string, strListDimensions)
	{
		number = string.toInt(&ok);
		
		if (ok)
		{
			dimensions.append(number);
		}
	}

	// Vector should now contain three dimensions
	if (dimensions.count() != 3)
	{
		this->core()->out()->logMessage("Could not parse dimensions in file " + filename  + "!");
		file.close();
		return;
	}

	// Read the fourth line, which contains the spacing
	QString strPixelSpacing = file.readLine();

	if (strPixelSpacing.isEmpty())
	{
		this->core()->out()->logMessage("Could not read spacing in volume file " + filename + "!");
		file.close();
		return;
	}

	lastIndexOf = strPixelSpacing.lastIndexOf('=');

	if (lastIndexOf == -1)
	{
		this->core()->out()->logMessage("Could not read spacing in volume file " + filename + "!");
		file.close();
		return;
	}

	strPixelSpacing = strPixelSpacing.right(strPixelSpacing.length() - lastIndexOf - 1);
	strPixelSpacing = strPixelSpacing.trimmed();

	// Parse the spacing in the same way as the dimensions
	QVector<double> spacing;
	QStringList strListPixelSpacing = strPixelSpacing.split(QRegExp("\\s+"), QString::SkipEmptyParts);

	foreach(QString string, strListPixelSpacing)
	{
		double tempSpacing = string.toDouble(&ok);

		if (ok)
		{
			spacing.append(tempSpacing);
		}
	}

	if (spacing.count() != 3)
	{
		this->core()->out()->logMessage("Could not parse spacing in file " + filename  + "!");
		file.close();
		return;
	}

	// Read the fifth line, which contains the number of bits
	QString strBits = file.readLine();

	if (strBits.isEmpty())
	{
		this->core()->out()->logMessage("Could not read number of bits in volume file " + filename + "!");
		file.close();
		return;
	}

	lastIndexOf = strBits.lastIndexOf('=');

	if (lastIndexOf == -1)
	{
		this->core()->out()->logMessage("Could not read number of bits in volume file " + filename + "!");
		return;
	}

	strBits = strBits.right(strBits.length() - lastIndexOf - 1);
	strBits = strBits.trimmed();

	// Parse the number of bits
	int bits = strBits.toInt(&ok);

	// Check if the number of bits was successfully parsed
	if (!ok)
	{
		this->core()->out()->logMessage("Could not parse number of bits in file " + filename  + "!");
		file.close();
		return;
	}

	// Read the sixth line, which contains the number of components
	QString strComponents = file.readLine();

	if (strComponents.isEmpty())
	{
		this->core()->out()->logMessage("Could not read number of components in volume file " + filename + "!");
		file.close();
		return;
	}

	lastIndexOf = strComponents.lastIndexOf('=');

	if (lastIndexOf == -1)
	{
		this->core()->out()->logMessage("Could not read number of components in volume file " + filename + "!");
		file.close();
		return;
	}

	strComponents = strComponents.right(strComponents.length() - lastIndexOf - 1);
	strComponents = strComponents.trimmed();

	// Parse the number of components
	int components = strComponents.toInt(&ok);

	// Check if the number of components was successfully parsed
	if (!ok)
	{
		this->core()->out()->logMessage("Could not parse number of components in file " + filename  + "!");
		file.close();
		return;
	}

	// Optional transformation matrix
	vtkMatrix4x4 * vtkMatrix = NULL;

	// If we're not yet at the end of the file, this means that a transformation
	// matrix is included, so we read it now.

	if (!file.atEnd())
	{
		// Output matrix
		double matrix[16];

		QString strMatrix = file.readLine();

		if (strMatrix.isEmpty())
		{
			this->core()->out()->logMessage("Could not read full transformation matrix in volume file " + filename + "!");
			file.close();
			return;
		}

		lastIndexOf = strMatrix.lastIndexOf('=');

		if (lastIndexOf == -1)
		{
			this->core()->out()->logMessage("Could not read full transformation matrix in volume file " + filename + "!");
			file.close();
			return;
		}

		strMatrix = strMatrix.right(strMatrix.length() - lastIndexOf - 1);
		strMatrix = strMatrix.trimmed();

		// Split the current row into substrings divided by spaces
		QStringList strListMatrixRow = strMatrix.split(QRegExp("\\s+"), QString::SkipEmptyParts);

		// Transformation matrix value
		double val;

		// Matrix index
		int counter = 0;

		// Row should contain four elements
		if (strListMatrixRow.length() > 4)
		{
			this->core()->out()->logMessage("Incorrect number of elements in transformation matrix in file " + filename + "!");
			return;
		}

		// For each of the substrings...
		foreach(QString string, strListMatrixRow)
		{
			// Try to convert the string to a double
			val = string.toDouble(&ok);

			// Check if the value was successfully parsed
			if (!ok)
			{
				this->core()->out()->logMessage("Failed to parse transformation matrix in file " + filename + "!");
				file.close();
				return;
			}

			// Add the value to the transformation matrix
			matrix[counter++] = val;
		}

		// Loop through the remaining rows of the matrix
		for(int i = 1; i < 4; ++i)
		{
			counter = 0;

			// Read the next line
			strMatrix = file.readLine();

			if (strMatrix.isEmpty())
			{
				this->core()->out()->logMessage("Could not read full transformation matrix in volume file " + filename + "!");
				file.close();
				return;
			}

			strListMatrixRow = strMatrix.split(QRegExp("\\s+"), QString::SkipEmptyParts);
		
			if (strListMatrixRow.length() > 4)
			{
				this->core()->out()->logMessage("Incorrect number of elements in transformation matrix in file " + filename + "!");
				file.close();
				return;
			}

			foreach(QString string, strListMatrixRow)
			{
				val = string.toDouble(&ok);
			
				if (!ok)
				{
					this->core()->out()->logMessage("Failed to parse transformation matrix in file " + filename + "!");
					file.close();
					return;
				}

				matrix[i * 4 + counter++] = val;
			}

		} // for [matrix rows]

		// Create a new "vtkMatrix4x4" object, and store the transformation matrix in it
		vtkMatrix = vtkMatrix4x4::New();
		vtkMatrix->DeepCopy(matrix);

	} // if [read transformation matrix]

	// Close the ".vol" file
	file.close();

	// Output data set pointer
	data::DataSet * ds;

	// Image reader used to load the ".raw" file
	vtkImageReader2 * imageReader;

	// Set the data type of the scalars
	int scalarType = VTK_UNSIGNED_CHAR;

	if (bits == 8)
	{
		scalarType = VTK_UNSIGNED_CHAR;
	}
	else if (bits == 16)
	{
		scalarType = VTK_UNSIGNED_SHORT;
	}
	else if (bits == 32)
	{
		scalarType = VTK_FLOAT;
	}
	else if (bits == 64)
	{
		scalarType = VTK_DOUBLE;
	}
	else
	{
		this->core()->out()->logMessage("Unsupported number of bits in file " + filename + "!");
		return;
	}

	// Create the image reader
	imageReader = vtkImageReader2::New();

	// Set the options of the image reader
	imageReader->SetDataExtent(0, (dimensions[0] - 1), 0, (dimensions[1] - 1), 0, (dimensions[2] - 1));
	imageReader->SetDataSpacing(spacing[0], spacing[1], spacing[2]);
	imageReader->SetNumberOfScalarComponents(components);
	imageReader->SetFileDimensionality(3);
	imageReader->SetDataScalarType(scalarType);
	imageReader->FileLowerLeftOn();

	// Start the progress for the image reader
	this->core()->out()->createProgressBarForAlgorithm(imageReader, "Volume Reader", "Reading volume data...");

	// Set the filename stored in the ".vol" file, and perform the reading process
	imageReader->SetFileName(rawFileName.toStdString().c_str());
	imageReader->Update();

	// Stop the progress for the image reader
	this->core()->out()->deleteProgressBarForAlgorithm(imageReader);

	ds = new data::DataSet(rawFileName, "scalar volume", imageReader->GetOutput());

	// If applicable, add the transformation matrix as an attribute
	if (vtkMatrix != NULL)
	{
		ds->getAttributes()->addAttribute("transformation matrix", vtkMatrix);
        vtkMatrix->Print( std::cout );
	}

	// Add the data set to the data manager
	this->core()->data()->addDataSet(ds);

	// Delete the reader
	imageReader->Delete();
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libVolumeReaderPlugin, bmia::VolumeReaderPlugin)
