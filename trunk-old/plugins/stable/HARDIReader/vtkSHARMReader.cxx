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
 * vtkSHARMReader.h
 *
 * 2010-12-01	Evert van Aart
 * - First version. Based on "vtkSHARMReader" of the old DTITool.
 *
 */


/** Includes */

#include "vtkSHARMReader.h"


namespace bmia {


vtkStandardNewMacro(vtkSHARMReader);


//-----------------------------[ Constructor ]-----------------------------\\

vtkSHARMReader::vtkSHARMReader()
{
	// Set default values of the class variables
	this->InformationExecuted 	= false;
	this->CurrentLine 			= "";
	this->Dimensions[0] 		= 0;
	this->Dimensions[1] 		= 0;
	this->Dimensions[2] 		= 0;
	this->DataSpacing[0] 		= 1.0;
	this->DataSpacing[1] 		= 1.0;
	this->DataSpacing[2] 		= 1.0;
	this->numberOfComponents    = 0;
	this->fType					= FileType_SHARM1;
	this->dataType				= VTK_DOUBLE;

	// Set pointers to NULL
	this->outputArray			= NULL;
	this->minMaxArray			= NULL;
	this->componentReader		= NULL;;
	this->componentImageData	= NULL;;
	this->componentPointData	= NULL;;
	this->componentScalars		= NULL;;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkSHARMReader::~vtkSHARMReader()
{
	// Clean up everything
	this->CleanUp();
}


//-------------------------------[ CleanUp ]-------------------------------\\

void vtkSHARMReader::CleanUp()
{
	// Set default values of the class variables
	this->InformationExecuted 	= false;
	this->CurrentLine 			= "";
	this->Dimensions[0] 		= 0;
	this->Dimensions[1] 		= 0;
	this->Dimensions[2] 		= 0;
	this->DataSpacing[0] 		= 1.0;
	this->DataSpacing[1] 		= 1.0;
	this->DataSpacing[2] 		= 1.0;
	this->numberOfComponents    = 0;
	this->fType					= FileType_SHARM1;
	this->dataType				= VTK_DOUBLE;

	// Close and delete the input stream
	if (this->IStream)
	{
		if (this->IStream->is_open())
		{
			this->IStream->close();
		}

		delete this->IStream;
		this->IStream = NULL;
	}

	// Delete all entries in the filename list
	this->ComponentFileNames.clear();

	// Clear parameter lists
	this->paramNames.clear();
	this->paramValues.clear();

	// Delete individual VTK objects and set them to NULL

	if (this->outputArray)
	{
		this->outputArray->Delete();
		this->outputArray = NULL;
	}

	if (this->minMaxArray)
	{
		this->minMaxArray->Delete();
		this->minMaxArray = NULL;
	}

	if (this->componentReader)
	{
		this->componentReader->Delete();
		this->componentReader = NULL;
	}

	if (this->componentImageData)
	{
		this->componentImageData->Delete();
		this->componentImageData = NULL;
	}

	if (this->componentPointData)
	{
		this->componentPointData->Delete();
		this->componentPointData = NULL;
	}

	if (this->componentScalars)
	{
		this->componentScalars->Delete();
		this->componentScalars = NULL;
	}
}


//-----------------------------[ CanReadFile ]-----------------------------\\

int vtkSHARMReader::CanReadFile(const char * fname)
{
	// Open the file
	QFile fileTest(fname);

	if (!(fileTest.open(QFile::ReadOnly)))
	{
		// Return zero on failure
		return 0;
	}

	// Close the file again
	fileTest.close();

	// Return one on success
	return 1;
}


//-------------------------------[ NextLine ]------------------------------\\

bool vtkSHARMReader::NextLine()
{
	// Next line in the file
	std::string nextline;

	// Check if the stream exists
	if (this->IStream == NULL)
	{
		vtkErrorMacro(<<"Input stream has not been set");
		return false;
	}

	// Get the next line from the input stream
	std::getline(*(this->IStream), nextline);

	// Check if something went wrong...
	if (this->IStream->fail() || nextline.size() == 0)
	{
		return false;
	}

	// If the line ends in a carriage return, remove it
	if (nextline[nextline.size() - 1] == '\r')
	{
		nextline.erase(nextline.size() - 1, 1);
	}

	// Copy string to class variable
	this->CurrentLine = nextline;

	return true;
}


//--------------------------[ ExecuteInformation ]-------------------------\\

void vtkSHARMReader::ExecuteInformation()
{
	// Information execution has not finished yet
	this->InformationExecuted = false;
 
	// Get the filename of the ".sharm" file
	char * fname = this->GetFileName();

	// Check if the filename has been set
	if (!fname)
	{
		vtkDebugMacro(<<"No input file defined.");
		return;
	}

	// Open an input stream for the ".sharm" file
	this->IStream = new ifstream(this->FileName, ios::in);

	// Check if opening the file succeeded
	if (this->IStream->fail())
	{
		vtkErrorMacro(<< "Unable to open file: "<< this->FileName);
		this->CleanUp();
		return;
	}

	// Try to read the header
	if (!this->ReadHeader())
	{
		vtkWarningMacro(<<"Incorrect file format, aborting reading process...");
		this->CleanUp();
		return;
	}

	// Try to read the data type
	if (!this->ReadDataType())
	{
		vtkWarningMacro(<<"Error reading data type, aborting reading process...");
		this->CleanUp();
		return;	
	}

	// Try to read the B-value	
	if (!this->ReadBValue())
	{
		vtkWarningMacro(<<"Error reading B-value type, aborting reading process...");
		this->CleanUp();
		return;		
	}

	// Try to read the voxel size (spacing)	
	if (!this->ReadVoxelSize())
	{
		vtkWarningMacro(<<"Error reading voxels size, aborting reading process...");
		this->CleanUp();
		return;			
	}

	// Try to read the dimensions
	if (!this->ReadDimensions())
	{
		vtkWarningMacro(<<"Error reading dimensions, aborting reading process...");
		this->CleanUp();
		return;	
	}

	// Try to read the dimensions
	if (!this->ReadParameters())
	{
		vtkWarningMacro(<<"Error reading parameters, aborting reading process...");
		this->CleanUp();
		return;	
	}

	// Try to read the filenames of the ".dat" component files	
	if(!this->ReadComponentFileNames())
	{
		vtkWarningMacro(<<"Error reading component file names, aborting reading process...");
		this->CleanUp();
		return;			
	}

	// Reset the input stream
	this->IStream->close();
	delete this->IStream;
	this->IStream = NULL;

	// Information has successfully been executed
	this->InformationExecuted = true;
}


//------------------------------[ ReadHeader ]-----------------------------\\

bool vtkSHARMReader::ReadHeader()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading header for file: " << this->FileName);
		return false;
	}

	// The first line contains the file type of the header
	if (this->CurrentLine == "SHARM1")
	{
		this->fType = FileType_SHARM1;
	}
	else if (this->CurrentLine == "SHARM2")
	{
		this->fType = FileType_SHARM2;
	}
	else if (this->CurrentLine == "SHARM3")
	{
		this->fType = FileType_SHARM3;
	}
	else if (this->CurrentLine == "SHCOEFF")
	{
		this->fType = FileType_SHCOEFF;
	}
	// Return an error if no supported file type was found
	else
	{
		vtkErrorMacro(<<"Unrecognized file type '"<< this->CurrentLine.c_str() << "' for file " << this->FileName);
		return false;
	}

	return true;
}


//-----------------------------[ ReadDataType ]----------------------------\\

bool vtkSHARMReader::ReadDataType()
{
	// No data type specified for "SHCOEFF" files, so we just use doubles
	if (this->fType == FileType_SHCOEFF)
	{
		this->dataType = VTK_DOUBLE;
		return true;
	}

	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading data type for file: " << this->FileName);
		return false;
	}

	// Unsigned short
	if (this->CurrentLine == "type double")
	{
		this->dataType = VTK_DOUBLE;
	}
	// Floating point
	else if (this->CurrentLine == "type float")
	{
		this->dataType = VTK_FLOAT;
	}
	// Unknown data type
	else
	{
		vtkErrorMacro(<<"Unknown data type: " << this->CurrentLine.c_str() << ", on line 3! " << " for file: " << this->FileName);
		return false;
	}

	return true;
}


//------------------------------[ ReadBValue ]-----------------------------\\

bool vtkSHARMReader::ReadBValue()
{
	// Do nothing unless the file type is "SHARM3"
	if (this->fType != FileType_SHARM3)
		return true;

	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading B-value for file: " << this->FileName);
		return false;
	}

	// Current line position
	unsigned int linepos = 0;

	// Read B-value
	double b = vtkBetterDataReader::ReadInt(this->CurrentLine, 2, linepos);

	// Add the B-value to the list of parameters
	this->paramNames.push_back("B-value");
	this->paramValues.push_back(b);

	return true;
}


//----------------------------[ ReadVoxelSize ]----------------------------\\

bool vtkSHARMReader::ReadVoxelSize()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading voxel size.");
		return false;
	}

	// Current position in the line
	unsigned int linepos = 0;

	// Read the three spacing components
	for (int i = 0; i < 3; i++)
	{
		this->DataSpacing[i] = vtkBetterDataReader::ReadFloat(this->CurrentLine, linepos, linepos);
	}

	return true;
}


//----------------------------[ ReadDimensions ]---------------------------\\

bool vtkSHARMReader::ReadDimensions()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading dimensions for file: " << this->FileName);
		return false;
	}

	// Current line position
	unsigned int linepos = 0;

	//Read the dimensions
	for (int i = 0; i < 3; i++)
	{
		this->Dimensions[i] = vtkBetterDataReader::ReadFloat(this->CurrentLine,	linepos, linepos);
	}

	return true;
}


//----------------------------[ ReadParameters ]---------------------------\\

bool vtkSHARMReader::ReadParameters()
{
	// Do nothing if the file type is "SHCOEFF"
	if (this->fType == FileType_SHCOEFF)
	{
		return true;
	}

	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading parameters for file: " << this->FileName);
		return false;
	}

	// Current line position
	unsigned int linepos = 0;

	// Get the model type
	std::string typeString = vtkBetterDataReader::ReadWord(this->CurrentLine, 0, linepos);

	// Add the glyph type to the list of parameters
	if (typeString == "DOTParametric") 
	{
		this->paramNames.push_back("Glyph Type");
		this->paramValues.push_back((double) GlyphType_DOT_PARAMETRIC);
	} 
	else if (typeString == "adc") 
	{
		this->paramNames.push_back("Glyph Type");
		this->paramValues.push_back((double) GlyphType_ADC);
	} 
	else if (typeString == "qball") 
	{
		this->paramNames.push_back("Glyph Type");
		this->paramValues.push_back((double) GlyphType_QBALL);
	}
	// Unknown glyph type; this is considered a non-fatal error.
	else
	{
		vtkWarningMacro(<< "Unknown glyph type: " << typeString);
	}

	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading parameters for file: " << this->FileName);
		return false;
	}

	// Get the current stream position (for rewinding)
	int streamPos = this->IStream->tellg();

	// Continue until we reach the end of the file
	while (this->CurrentLine[0] !='\n')
	{
		// Reset the current line position
		linepos = 0;

		// If we find the first line containing a ".dat" file name...
		if (this->CurrentLine.rfind(".dat") != string::npos)
		{
			// Rewind the stream position to the start of this line...
			this->IStream->seekg(streamPos);

			// ...and break from the loop. In a properly formatted header file,
			// this should always be the exist point of the while-loop.
			break;
		}

		// Get parameter name and value (as a double)
		string paramName  = vtkBetterDataReader::ReadWord(  this->CurrentLine, linepos, linepos);
		double paramValue = vtkBetterDataReader::ReadDouble(this->CurrentLine, linepos, linepos);

		// Add parameters to the list
		this->paramNames.push_back(paramName);
		this->paramValues.push_back(paramValue);

		// Get the current stream position
		streamPos = this->IStream->tellg();

		// Try to read the next line, and break the while loop if it fails
		if (!this->NextLine())
		{
			break;
		}
	}

	// Check if the end of the file has been reached
	if (this->IStream->eof())
	{
		vtkErrorMacro(<< "End of file reached while reading parameters for file: " << this->FileName);
		return false;
	}

	return true;
}


//------------------------[ ReadComponentFileNames ]-----------------------\\

bool vtkSHARMReader::ReadComponentFileNames()
{
	// Get the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading component filename.");
		return false;
	}

	// Clear any existing file names
	this->ComponentFileNames.clear();

	// Current line position
	unsigned int linepos = 0;

	// Reset the number of components
	this->numberOfComponents = 0;

	// Read all lines in the file
	while (this->CurrentLine[0] !='\n')
	{
		// Reset the line position
		linepos = 0;

		// Get the last word of the line, which is the filename of the component data set
		std::string componentFileName = vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);

		// Get the filename of the ".sharm" file...
		std::string filePath(this->FileName);

		// ...and find the last occurrence of "/"
		int pathEnd = filePath.rfind("/");

		// If "/" was found, add the component name to the end of the path. 

		if (pathEnd != std::string::npos)
		{
			componentFileName.insert(0, filePath.substr(0, pathEnd + 1));
		}

		// If we could not find "/", try to find "\" instead.
		else
		{
			pathEnd = filePath.rfind("\\");

			if (pathEnd != std::string::npos)
			{
				componentFileName.insert(0, filePath.substr(0, pathEnd + 1));
			}
		}

		// Add component filename to the list
		ComponentFileNames.push_back(componentFileName);

		// Increment the component counter
		this->numberOfComponents++;

		// Try to read the next line, and break the while loop if it fails
		if (!this->NextLine())
		{
			break;
		}
	} // while [CurrentLine]

	// Right now, we only support fourth-order spherical harmonics (15 components)
	// and eight-order spherical harmonics (30 components).

	if (this->numberOfComponents != 15 && this->numberOfComponents != 30 && this->numberOfComponents != 45)
	{
		vtkErrorMacro(<< "Number of spherical harmonic components is " << this->numberOfComponents
					  << " , which is not supported.");

		return false;
	}
	
	return true;
}


//-----------------------------[ ExecuteData ]-----------------------------\\

void vtkSHARMReader::ExecuteData(vtkDataObject * out)
{
	// Start progress meter
	this->UpdateProgress(0.0);

	// Check if the execution of the information was successful
	if (!this->InformationExecuted)
	{
		vtkErrorMacro(<<"InformationExecuted() did not end successfully. " << " Aborting execution of data.");
		this->CleanUp();
		return;
	}

	// Get the output of the reader
	vtkImageData * output = this->GetOutput();

	// Check if the output has been set.
	if (!output)
	{
		vtkErrorMacro(<<"No output defined!");
		this->CleanUp();
		return;
	}

	// Check if the output contains point data
	if (!(output->GetPointData()))
	{
		vtkErrorMacro(<<"No output point data defined!");
		this->CleanUp();
		return;	
	}

	// Create an iterator for the list of gradient filenames
	std::list<std::string>::iterator iter;

	// Gradient filename from list
	std::string fname;

	// Index of the current component
	int componentIndex = 0;

	// Create the output array
	this->outputArray = vtkDataArray::CreateDataArray(this->dataType);
	this->outputArray->SetNumberOfComponents(this->numberOfComponents);

	// Loop through all stored filenames
	for (iter  = this->ComponentFileNames.begin(), componentIndex = 0;
		 iter != this->ComponentFileNames.end(); 
		 ++iter, ++componentIndex)
	{
		// Get the current filename
		fname = (*iter);

		// Create the reader
		this->componentReader = vtkImageReader2::New();
		this->componentReader->SetFileName(fname.c_str());
		this->componentReader->SetFileDimensionality(3);
		this->componentReader->SetNumberOfScalarComponents(1);
		this->componentReader->SetDataScalarTypeToDouble();
		this->componentReader->FileLowerLeftOff();
		this->componentReader->SetDataExtent(	0, this->Dimensions[0] - 1, 
												0, this->Dimensions[1] - 1, 
												0, this->Dimensions[2] - 1);

		// Update text of progress bar
		std::string progressText("Reading component file '");
		progressText += fname;
		progressText += "'...";
		this->SetProgressText(progressText.c_str());

		// Update the progress bar
		this->UpdateProgress(((float) componentIndex) / (float) (this->numberOfComponents + 1));

		// Update the component reader to read the file
		this->componentReader->Update();

		// Get the output of the reader
		this->componentImageData = this->componentReader->GetOutput();

		// Check if the output exists
		if (!this->componentImageData)
		{
			vtkErrorMacro(<<"Component reader of component '" << fname << "' does not have any output!");
			this->CleanUp();
			return;
		}

		// Fetch the point data of the output
		this->componentPointData = this->componentImageData->GetPointData();

		// Check if the point data exists
		if (!this->componentPointData)
		{
			vtkErrorMacro(<<"No point data defined for component '" << fname << "'!");
			this->CleanUp();
			return;
		}

		// Fetch the scalar array from the point data
		this->componentScalars =  this->componentPointData->GetScalars();

		// Check if the scalar data exists
		if (!this->componentScalars)
		{
			vtkErrorMacro(<<"No scalars defined for component '" << fname << "'!");
			this->CleanUp();
			return;
		}

		//If this is the first file, finish initialization of the output
		if (iter == this->ComponentFileNames.begin())
		{
			this->outputArray->SetNumberOfTuples(this->componentScalars->GetNumberOfTuples());

			output->CopyStructure(this->componentImageData);
			output->SetDimensions(this->componentImageData->GetDimensions());
			output->SetOrigin(0.0, 0.0, 0.0);
			output->SetSpacing(this->DataSpacing[0], this->DataSpacing[1], this->DataSpacing[2]);
		}

		// Copy all voxel values to the correct component in the output
		this->outputArray->CopyComponent(componentIndex, this->componentScalars, 0);

		// Delete the component reader
		this->componentReader->Delete();
		this->componentReader = NULL;

	} // for [ComponentFileNames]

	// Add the output arrays
	output->GetPointData()->SetScalars(this->outputArray);

	// Try to read the "min.dat" and "max.dat" files
	this->readMinMaxFiles(this->outputArray->GetNumberOfTuples());

	// Finalize the output
	output->SetUpdateExtentToWholeExtent();
	output->Squeeze();

	// Done!
	this->UpdateProgress(1.0);
}


//---------------------------[ readMinMaxFiles ]---------------------------\\

void vtkSHARMReader::readMinMaxFiles(vtkIdType numberOfTuples)
{
	// Default names for minima- and maxima-files
	std::string fileNames[2];
	fileNames[0] = "min.dat";
	fileNames[1] = "max.dat";

	// Get the filename of the ".sharm" file...
	std::string filePath(this->FileName);

	// ...and find the last occurrence of "/"
	int pathEnd = filePath.rfind("/");

	// If "/" was found, add the file name to the end of the path. 
	if (pathEnd != std::string::npos)
	{
		fileNames[0].insert(0, filePath.substr(0, pathEnd + 1));
		fileNames[1].insert(0, filePath.substr(0, pathEnd + 1));
	}

	// Otherwise, try to find "\"
	else
	{
		pathEnd = filePath.rfind("\\");

		if (pathEnd != std::string::npos)
		{
			fileNames[0].insert(0, filePath.substr(0, pathEnd + 1));
			fileNames[1].insert(0, filePath.substr(0, pathEnd + 1));
		}
	}
	
	// Check if both files exist. If not, we don't read the minima and maxima
	if (this->CanReadFile(fileNames[0].c_str()) == 0 || this->CanReadFile(fileNames[0].c_str()) == 0)
	{
		vtkWarningMacro(<< "Could not find 'min.dat' and/or 'max.dat'.");
		return;
	}

	// Create array for minima and maxima
	this->minMaxArray = vtkDataArray::CreateDataArray(this->dataType);
	this->minMaxArray->SetNumberOfComponents(2);
	this->minMaxArray->SetNumberOfTuples(numberOfTuples);
	this->minMaxArray->SetName("minmax");

	// Process minima and maxima
	for (int i = 0; i < 2; ++i)
	{
		// Create the reader
		this->componentReader = vtkImageReader2::New();
		this->componentReader->SetFileName(fileNames[i].c_str());
		this->componentReader->SetFileDimensionality(3);
		this->componentReader->SetNumberOfScalarComponents(1);
		this->componentReader->SetDataScalarTypeToDouble();
		this->componentReader->FileLowerLeftOff();
		this->componentReader->SetDataExtent(	0, this->Dimensions[0] - 1, 
												0, this->Dimensions[1] - 1, 
												0, this->Dimensions[2] - 1);

		// Update text of progress bar
		std::string progressText("Reading component file '");
		progressText += fileNames[i].c_str();
		progressText += "'...";
		this->SetProgressText(progressText.c_str());

		// Update the component reader to read the file
		this->componentReader->Update();

		// Get the output of the reader
		this->componentImageData = this->componentReader->GetOutput();

		// Check if the output exists
		if (!this->componentImageData)
		{
			vtkErrorMacro(<<"Component reader of component '" << fileNames[i] << "' does not have any output!");
			return;
		}

		// Fetch the point data of the output
		this->componentPointData = this->componentImageData->GetPointData();

		// Check if the point data exists
		if (!this->componentPointData)
		{
			vtkErrorMacro(<<"No point data defined for component '" << fileNames[i] << "'!");
			return;
		}

		// Fetch the scalar array from the point data
		this->componentScalars =  this->componentPointData->GetScalars();

		// Check if the scalar data exists
		if (!this->componentScalars)
		{
			vtkErrorMacro(<<"No scalars defined for component '" << fileNames[i] << "'!");
			return;
		}

		// Copy all voxel values to the correct component in the output
		this->minMaxArray->CopyComponent(i, this->componentScalars, 0);

		// Delete the component reader
		this->componentReader->Delete();
		this->componentReader = NULL;

	} // for [i]

	// Done, add the array to the output
	this->GetOutput()->GetPointData()->AddArray(this->minMaxArray);
}


} // namespace bmia
