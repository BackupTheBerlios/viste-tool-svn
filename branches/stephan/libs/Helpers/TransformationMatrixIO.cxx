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
 * TransformationMatrixIO.h
 *
 * 2011-01-24	Evert van Aart
 * - First version. 
 *
 */


/** Includes */

#include "TransformationMatrixIO.h"


namespace bmia {


//------------------------------[ readMatrix ]-----------------------------\\

vtkMatrix4x4 * TransformationMatrixIO::readMatrix(std::string filename, std::string & errorMessage)
{
	errorMessage = "";

	// find the last dot of the filename
	int lastDot = filename.rfind('.');

	// The filename should include at least one dot
	if (lastDot == std::string::npos)
	{
		errorMessage = "Incorrect filename!";
		return NULL;
	}

	// Replace the extension of the filename with "mat"
	std::string matFilename = filename.substr(0, lastDot + 1) + "tfm";

	QString matFilenameQ(matFilename.c_str());

	// Create and open the matrix file
	QFile matFile(matFilenameQ);

	if (!matFile.open(QFile::ReadOnly))
	{
		return NULL;
	}

	// Array for the new matrix
	double matrixArray[16];

	// Row counter
	int row = 0;

	// Create a text stream for the input file
	QTextStream in(&matFile);

	// Read the first line
	QString line = in.readLine();

	// Process all lines
	while (!line.isNull() && row < 4)
	{
		// Simplify the line (remove trailing spaces, double spaces, etc)
		line = line.simplified();

		// Split the line into four parts
		QStringList rowStrings = line.split(" ", QString::SkipEmptyParts);

		// We should have four values per line
		if (rowStrings.size() != 4)
		{
			errorMessage = "Wrong formatting in matrix file '" + matFilename + "'!";
			matFile.close();
			return NULL;
		}

		bool ok = true;

		for (int col = 0; col < 4; ++col)
		{
			// Parse the four sub-strings to doubles
			double matrixValue = rowStrings.at(col).toDouble(&ok);

			if (!ok)
			{
				errorMessage = "Could not parse value(s) in matrix file '" + matFilename + "'!";
				matFile.close();
				return NULL;
			}

			// Store the double in the matrix array
			matrixArray[row * 4 + col] = matrixValue;
		}

		row++;

		// Read the next line
		line = in.readLine();
	};

	// Input file should contain four lines
	if (row != 4)
	{
		errorMessage = "Wrong number of rows in matrix file '" + matFilename + "'!";
		matFile.close();
		return NULL;
	}

	matFile.close();

	// Create a new matrix, and copy the array to this matrix
	vtkMatrix4x4 * m = vtkMatrix4x4::New();
	m->DeepCopy(matrixArray);
	return m;
}


//-----------------------------[ writeMatrix ]-----------------------------\\

bool TransformationMatrixIO::writeMatrix(std::string filename, vtkMatrix4x4 * m, std::string & errorMessage)
{
	errorMessage = "";

	// find the last dot of the filename
	int lastDot = filename.rfind('.');

	// The filename should include at least one dot
	if (lastDot == std::string::npos)
	{
		errorMessage = "Incorrect filename!";
		return NULL;
	}

	// Replace the extension of the filename with "mat"
	std::string matFilename = filename.substr(0, lastDot + 1) + "tfm";

	QString matFilenameQ(matFilename.c_str());

	// Create and open the matrix file
	QFile matFile(matFilenameQ);

	if (!matFile.open(QFile::WriteOnly))
	{
		errorMessage = "Failed to open matrix file '" + matFilename + "'!";
		return false;
	}

	// Create a text stream
	QTextStream out(&matFile);

	// Write the sixteen matrix elements to the output file
	out << m->Element[0][0] << " " << m->Element[0][1] << " " << m->Element[0][2] << " " << m->Element[0][3] << endl;
	out << m->Element[1][0] << " " << m->Element[1][1] << " " << m->Element[1][2] << " " << m->Element[1][3] << endl;
	out << m->Element[2][0] << " " << m->Element[2][1] << " " << m->Element[2][2] << " " << m->Element[2][3] << endl;
	out << m->Element[3][0] << " " << m->Element[3][1] << " " << m->Element[3][2] << " " << m->Element[3][3];

	matFile.flush();
	matFile.close();

	// Done!
	return true;
}


} // namespace bmia