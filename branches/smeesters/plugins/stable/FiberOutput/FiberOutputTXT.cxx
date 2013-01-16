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

/** 
 * FiberOutputTXT.cxx
 *
 * 2008-01-28	Jasper Levink
 * - First Version.
 *
 * 2010-12-20	Evert van Aart
 * - First version for the DTITool3.
 *
 */


/** Includes */

#include "FiberOutputTXT.h"


using namespace std;


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

FiberOutputTXT::FiberOutputTXT() : FiberOutput()
{
	// Initialize the strings
	this->fileLocation = "";
	this->filePrefix   = "";
}


//------------------------------[ Destructor ]-----------------------------\\

FiberOutputTXT::~FiberOutputTXT()
{

}


//------------------------------[ outputInit ]-----------------------------\\

void FiberOutputTXT::outputInit() 
{
	string stringFileName = this->fileName;

	// Index specifies where the location ends (i.e., where the filename begins)
	int endOfLocation;

	// Find the index of the last slash (for Unix systems) or backslash (for Windows)
	int lastIndexOfSlash	 = stringFileName.find_last_of("/");
	int lastIndexOfBackSlash = stringFileName.find_last_of("\\");

	// Find the index of the last dot
	int lastIndexOfDot = stringFileName.find_last_of(".");

	if(lastIndexOfSlash > lastIndexOfBackSlash)
		endOfLocation = lastIndexOfSlash;
	else
		endOfLocation = lastIndexOfBackSlash;

	// Get the file location from the input string (directory containing the file)
	this->fileLocation = stringFileName.substr(0, endOfLocation + 1);

	// Get the file prefix (i.e., filename without extension)
	this->filePrefix = stringFileName.substr(endOfLocation + 1, lastIndexOfDot - endOfLocation - 1);
}


//----------------------------[ outputHeader ]-----------------------------\\

void FiberOutputTXT::outputHeader() 
{
	// Open the file
	this->outfile.open(this->fileName);

	// Check if the file is actually open
	if (!this->outfile.is_open())
		return;

	// Print whether we use fibers or ROIs as the data source
	string strDataSource = "Datasource: ";

	if (this->dataSource == DS_ROI)
	{
		strDataSource += "ROI";
	} 
	else if(this->dataSource == DS_Fibers)
	{
		strDataSource += "Fiber"; 
	}
	
	// Print the selected parameters
	string strParameters = "Parameters: ";

	// Loop through the selected measures
	for(int currentMeasure = 0; currentMeasure < this->numberOfSelectedMeasures; ++currentMeasure) 
	{
		// Append measure name
		InputInfo currentInputInfo = this->scalarImageList.at(currentMeasure);
		strParameters += currentInputInfo.name;

		if(currentMeasure < this->numberOfSelectedMeasures - 1) 
		{
			strParameters += ", ";
		}
	}

	// Print whether or not we use the "Per Voxel" settings
	string strPerVoxel = "Per Voxel: ";

	if(this->perVoxel) 
	{
		strPerVoxel += "yes";
	}
	else 
	{
		strPerVoxel += "no";
	}
	
	// Print whether or not we the "Mean and Variance" setting
	string strMeanAndVar = "Mean and Variance: ";

	if(this->meanAndVar) 
	{
		strMeanAndVar+="yes"; 
	}
	else 
	{
		strMeanAndVar += "no";
	}

	// Write information to header file
	this->outfile
		<< "HEADER FILE" << endl
		<< "" << endl
		<< "This is a DTI data file exported by DTITool." << endl
		<< "The output consists of several tab-spaced txt-files." << endl
		<< "" << endl
		<< "More info about DTITool:" <<endl
		<< "http://www.bmia.bmt.tue.nl/software/dtitool" << endl
		<< ""<< endl
		<< "DTIfile: " + this->tensorImageInfo.name << endl
		<< "" << endl
		<< strDataSource << endl
		<< "" << endl
		<< strParameters << endl
		<< "" << endl
		<< strPerVoxel << endl
		<< "" << endl
		<< strMeanAndVar << endl
		<< "" << endl
		<< "File list: " << endl;

		// Writing information for each voxel
		if(this->perVoxel) 
		{
			// Using ROIs as data source
			if(this->dataSource == DS_ROI)
			{
				// Loop through all selected ROIs
				for(int currentROI = 0; currentROI < this->numberOfSelectedROIs; ++currentROI) 
				{
					// Copy name to output
					InputInfo currentInputInfo = this->seedList.at(currentROI);
					this->outfile << this->filePrefix << "_" << prepareROINameForFileName((string) "ROI " + currentInputInfo.name) << ".txt" << endl;
				} 
			}
			// Using fibers as data source
			else if(this->dataSource == DS_Fibers) 
			{ 
				// Add "_Fibers.txt" to the prefix
				this->outfile << this->filePrefix << "_Fibers.txt" << endl;
			} 
		}

		if(this->meanAndVar)
		{
			// File containing mean and variance
			this->outfile << this->filePrefix << "_MeanAndVariance.txt" << endl;
		}

		// Close the header file
		this->outfile.close();
}


//-------------------------[ outputInitWorksheet ]-------------------------\\

void FiberOutputTXT::outputInitWorksheet(string titel) 
{
	// Set the full file name
	string fullFileName = this->fileLocation + this->filePrefix + "_" + prepareROINameForFileName(titel) + ".txt";

	// Create a new text file
	this->outfile.open(fullFileName.c_str());	
}


//---------------------------[ outputWriteRow ]----------------------------\\

void FiberOutputTXT::outputWriteRow(string * content, int contentSize, int styleID) 
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Loop through the content array
	for(int currentContentIndex = 0; currentContentIndex < contentSize; ++currentContentIndex) 
	{
		// Write the current string to the output
		this->outfile << content[currentContentIndex];
		
		if(currentContentIndex != contentSize - 1) 
		{
			// Write a tab as a column separator
			this->outfile << "\t";
		}
	}

	this->outfile << endl;
}


//----------------------------[ outputWriteRow ]---------------------------\\

void FiberOutputTXT::outputWriteRow(double * content, int contentSize, string label, int styleID) 
{
	// Check if the output file is open
	if (!this->outfile.is_open())
		return;

	// Add the label, if it has been defined
	if(label.length() > 0)
	{
		this->outfile << label << "\t";
	}

	// Loop through the content array
	for(int currentContentIndex = 0; currentContentIndex < contentSize; ++currentContentIndex) 
	{
		// Write the current string to the output
		this->outfile << content[currentContentIndex];

		if(currentContentIndex != contentSize - 1) 
		{
			// Write a tab as a column separator
			this->outfile << "\t";
		}
	}

	this->outfile << endl;
}


//--------------------------[ outputEndWorksheet ]-------------------------\\

void FiberOutputTXT::outputEndWorksheet()
{
	// Close the current output file
	if (this->outfile.is_open())
	{
		this->outfile.close();
	}
}


//------------------------------[ outputEnd ]------------------------------\\

void FiberOutputTXT::outputEnd()
{
	// Nothing to do here
}


//----------------------[ prepareROINameForFileName ]----------------------\\

string FiberOutputTXT::prepareROINameForFileName(string roiName) 
{
	// Create a string with reserved characters
	string reservedCharacters = " |\\?*<\":>/";
	unsigned int currentReservedCharacter;

	// Check for every character in "roiName" whether it is a reserved character
	for(unsigned int currentCharacterIndex = 0; currentCharacterIndex < roiName.length(); ++currentCharacterIndex) 
	{
		// Get the current character
		char currentCharacter = roiName.at(currentCharacterIndex);

		// True if we have found a reserved character
		bool found = false;

		// Index of reserved characters
		currentReservedCharacter = 0;

		while(!found && currentReservedCharacter < reservedCharacters.length()) 
		{
			// Check if the current character is reserved
			if(currentCharacter == (char)reservedCharacters.at(currentReservedCharacter)) 
			{
				// If so, replace it with an underscore
				roiName.replace(currentCharacterIndex,1,"_");
				found = 1;
			}

			currentReservedCharacter++;
		}
	}

	// Return the new filename
	return roiName;
}


} // namespace bmia