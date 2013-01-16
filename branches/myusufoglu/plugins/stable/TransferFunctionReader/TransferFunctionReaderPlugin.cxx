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
 * TransferFunctionReaderPlugin.cxx
 *
 * 2010-03-03	Wiljan van Ravensteijn
 * - First version
 *
 * 2011-01-04	Evert van Aart
 * - Improved error handling, added some comments.
 *
 * 2011-01-17	Evert van Aart
 * - Fixed a bug that prevented transfer functions with a piecewise function
 *   from being loaded correctly.
 * - Transfer functions are now given a shorter name by default.
 *
 */


/** Includes */

#include "TransferFunctionReaderPlugin.h"



namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

TransferFunctionReaderPlugin::TransferFunctionReaderPlugin() : plugin::Plugin("Transfer Function Reader")
{

}


//------------------------------[ Destructor ]-----------------------------\\

TransferFunctionReaderPlugin::~TransferFunctionReaderPlugin()
{

}


//----------------------[ getSupportedFileExtensions ]---------------------\\

QStringList TransferFunctionReaderPlugin::getSupportedFileExtensions()
{
    QStringList list;
    list.push_back("tf");
    return list;
}


//---------------------[ getSupportedFileDescriptions ]--------------------\\

QStringList TransferFunctionReaderPlugin::getSupportedFileDescriptions()
{
	QStringList list;
	list.push_back("Transfer Functions");
	return list;
}


//---------------------------[ loadDataFromFile ]--------------------------\\

void TransferFunctionReaderPlugin::loadDataFromFile(QString filename)
{
	// Array containing the current line
	QByteArray line;

	// Minimum and maximum of the range
	double minRange;
	double maxRange;

	// True if the transfer function has a piecewise function
	bool hasPiecewiseFunction;

	// Used to determine if strings are parsed successfully
	bool ok = true;

	// Contains substrings describing one anchor point
	QStringList pointStrings;

	// Print status message to the log
	this->core()->out()->logMessage("Trying to load data from file " + filename);

	// Create the Qt file handler
	QFile file(filename);

	// Try to open the input file
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		this->core()->out()->logMessage("Could not open file " + filename + "!");
		return;
	}

	// Read the line, which contains the minimum of the range
	line = file.readLine();

	// Convert the line to a string, and remove the final line break
	QString lineStr(line);
	lineStr.chop(1);

	// Convert the string to a double
	minRange = lineStr.toDouble(&ok);

	// Check if the number was parsed correctly
	if (!ok) 
	{
		this->core()->out()->logMessage("Failed to read range minimum in file " + filename + "!");
		file.close();
		return;
	}

	// Repeat the above for the range maximum
	line = file.readLine();
	lineStr = QString(line);
	lineStr.chop(1);
	maxRange = lineStr.toDouble(&ok);

	if (!ok) 
	{
		this->core()->out()->logMessage("Failed to read range maximum in file " + filename + "!");
		file.close();
		return;
	}

	// Read the next line, which contains whether or not this transfer function has a piecewise function
	line = file.readLine();

	// Convert the line to a string, and remove the final line break
	lineStr = QString(line);
	lineStr.chop(1);

	// Set the boolean variable "hasPiecewiseFunction"
	if (lineStr == "True")
	{
		hasPiecewiseFunction = true;
	}
	else
	{
		hasPiecewiseFunction = false;
	}

	// Create a color transfer function object
	vtkColorTransferFunction * ctf = vtkColorTransferFunction::New();

	// Create a piecewise function object if necessary
	vtkPiecewiseFunction * cpf;

	if (hasPiecewiseFunction)
	{
		cpf = vtkPiecewiseFunction::New();
	}

	// Loop through the rest of the lines, which contain the anchor points of the transfer function
	while (!file.atEnd()) 
	{
		// Read the next line, convert it to a string, and remove the final line break
		line = file.readLine();
		lineStr = QString(line);
		lineStr.chop(1);

		// Split the current line at the spaces
		pointStrings = lineStr.split(' ', QString::SkipEmptyParts);

		// Line should contain four or five numbers
		if (pointStrings.size() != 4 + ((hasPiecewiseFunction) ? (1) : (0)))
		{
			this->core()->out()->logMessage("Wrong number of elements for RGB point in file " + filename + "!");
			file.close();
			return;
		}

		// Parse point elements one by one
		double x = pointStrings.at(0).toDouble(&ok);

		if(!ok)
		{
			this->core()->out()->logMessage("Failed to parse RGB point element in file " + filename + "!");
			continue;
		}

		double r = pointStrings.at(1).toDouble(&ok);

		if(!ok)
		{
			this->core()->out()->logMessage("Failed to parse RGB point element in file " + filename + "!");
			continue;
		}

		double g = pointStrings.at(2).toDouble(&ok);

		if(!ok)
		{
			this->core()->out()->logMessage("Failed to parse RGB point element in file " + filename + "!");
			continue;
		}

		double b = pointStrings.at(3).toDouble(&ok);

		if(!ok)
		{
			this->core()->out()->logMessage("Failed to parse RGB point element in file " + filename + "!");
			continue;
		}

		// Add the RGB point to the transfer function
		ctf->AddRGBPoint(x, r, g, b);

		// If we've got a piecewise function...
		if (hasPiecewiseFunction)
		{
			// ...convert the fifth substring to a double...
			double y = pointStrings.at(4).toDouble(&ok);

			if(!ok)
			{
				this->core()->out()->logMessage("Failed to parse RGB point element in file " + filename + "!");
				continue;
			}

			// ...and add it to the piecewise function object
			cpf->AddPoint(x, y);
		}
	}

	// Close the input file
	file.close();

	// Short name of the data set
	QString shortName = filename;

	// Find the last slash
	int lastSlash = filename.lastIndexOf("/");

	// If the filename does not contain a slash, try to find a backslash
	if (lastSlash == -1)
	{
		lastSlash = filename.lastIndexOf("\\");
	}

	// Throw away everything up to and including the last slash
	if (lastSlash != -1)
	{
		shortName = shortName.right(shortName.length() - lastSlash - 1);
	}

	// Find the last dot in the remainder of the filename
	int lastPoint = shortName.lastIndexOf(".");

	// Throw away everything after and including the last dot
	if (lastPoint != -1)
	{
		shortName = shortName.left(lastPoint);
	}

	// Create a new data set for the transfer function
	data::DataSet * ds = new data::DataSet(shortName, "transfer function", ctf);

	// Add the range and the piecewise function as attributes
	ds->getAttributes()->addAttribute("minRange", minRange);
	ds->getAttributes()->addAttribute("maxRange", maxRange);

	if (hasPiecewiseFunction)
	{
		ds->getAttributes()->addAttribute("piecewise function", cpf);
	}

	// Add the data set to the manager
	this->core()->data()->addDataSet(ds);
}


} // namespace bmia


Q_EXPORT_PLUGIN2(libTransferFunctionReaderPlugin, bmia::TransferFunctionReaderPlugin)
