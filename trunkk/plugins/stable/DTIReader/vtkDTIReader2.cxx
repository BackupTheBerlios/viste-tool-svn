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
 * vtkDTIReader2.cxx
 *
 * 2006-01-05	Tim Peeters
 * - First version
 *
 * 2006-01-10	Tim Peeters
 * - Finished implementation. A lot of code is copied from the old
 *   "vtkTensorDataReader.cxx", but here the new "vtkDTIComponentReader"
 *   classes and "vtkImageReader2" functionality are used.
 *
 * 2006-01-16	Tim Peeters
 * - Hmm.. I did not finish the implementation on 2006-01-10 ;).
 * - Added code for merging of input datasets and creating an output
 *   tensor dataset.
 *
 * 2006-03-08	Tim Peeters
 * - Fixed resetting of spacing of the output data somewhere when the
 *   pipeline is executed by properly setting "DataSpacing" in "ExecuteInformation".
 * - Use "DataSpacing" from superclass instead of "VoxelSize".
 *
 * 2006-05-12	Tim Peeters
 * - Add support for a matrix to transform the data that is read with.
 *
 * 2010-09-03	Tim Peeters
 * - From now on, store the (symmetrical!) tensors in a 6-valued scalar
 *   array in the output instead of 9-valued tensor array.
 * 
 * 2010-11-18	Evert van Aart
 * - Moved "vtkBetterDataReader" and "vtkDTIComponentReader" to the libraries
 *   direction, since I need them for the new HARDI reader.
 *
 * 2011-01-14	Evert van Aart
 * - Structural set is now created as a separate "vtkImageData" object.
 * 
 * 2011-03-14	Evert van Aart
 * - Changed data type of structural image to double. 
 *
 * 2011-03-31	Evert van Aart
 * - Allowed the reader to read doubles.
 *
 */


/** Includes */

#include "vtkDTIReader2.h"


namespace bmia {


vtkStandardNewMacro(vtkDTIReader2);


//-----------------------------[ Constructor ]-----------------------------\\

vtkDTIReader2::vtkDTIReader2()
{
	// Set default values of the class variables
	this->InformationExecuted 	= false;
	this->CurrentLine 			= "";
	this->DataSpacing[0] 		= 1.0;
	this->DataSpacing[1] 		= 1.0;
	this->DataSpacing[2] 		= 1.0;
	this->ScanDirection			= BMIA_SCAN_DIRECTION_UNDEFINED;
	this->DataType				= VTK_UNSIGNED_SHORT;

	// Set pointers to NULL
	this->IStream				= NULL;
	this->TensorTransformMatrix = NULL;
	this->componentReader		= NULL;
	this->componentImageData	= NULL;
	this->componentPointData	= NULL;
	this->componentScalars		= NULL;
	this->outputTensorArray		= NULL;
	this->structImage			= NULL;
	this->structScalars			= NULL;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkDTIReader2::~vtkDTIReader2()
{
	// Clean up everything
	this->CleanUp();	
}


//-------------------------------[ CleanUp ]-------------------------------\\

void vtkDTIReader2::CleanUp()
{
	// Set default values of the class variables
	this->InformationExecuted 	= false;
	this->CurrentLine 			= "";
	this->DataSpacing[0] 		= 1.0;
	this->DataSpacing[1] 		= 1.0;
	this->DataSpacing[2] 		= 1.0;
	this->ScanDirection			= BMIA_SCAN_DIRECTION_UNDEFINED;
	this->DataType				= VTK_UNSIGNED_SHORT;

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

	// Clear the component information list
	this->ComponentInfoList.clear();

	// Delete individual VTK objects and set them to NULL

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

	if (this->outputTensorArray)
	{
		this->outputTensorArray->Delete();
		this->outputTensorArray = NULL;
	}

	if (this->structImage)
	{
		this->structImage->Delete();
		this->structImage = NULL;
	}

	if (this->structScalars)
	{
		this->structScalars->Delete();
		this->structScalars = NULL;
	}

	// Delete the transformation matrix
	this->CleanTensorTransformMatrix();
}


//-----------------------------[ CanReadFile ]-----------------------------\\

int vtkDTIReader2::CanReadFile(const char* fname)
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

bool vtkDTIReader2::NextLine()
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

void vtkDTIReader2::ExecuteInformation()
{
	// Information execution has not finished yet
	this->InformationExecuted = false;

	// Get the filename of the .dti file
	char * fname = this->GetFileName();

	// Check if the filename has been set
	if (!fname)
	{
		vtkDebugMacro(<<"No input file defined.");
		return;
	}

	// Open an input stream for the ".dti" file
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

	// Try to read the scan direction
	if (!this->ReadScanDirection())
	{
		vtkWarningMacro(<<"Error reading scan direction, aborting reading process...");
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

	// Try to read the filenames of the ".dat" component files	
	if(!this->ReadComponentFileNames())
	{
		vtkWarningMacro(<<"Error reading component file names, aborting reading process...");
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

	// Try to read the transformation matrix
	this->ReadTensorTransformMatrix();

	// Reset the input stream
	this->IStream->close();
	delete this->IStream;
	this->IStream = NULL;

	// Information has successfully been executed
	this->InformationExecuted = true;
}


//------------------------------[ ReadHeader ]-----------------------------\\

bool vtkDTIReader2::ReadHeader()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading header for file: " << this->FileName);
		return false;
	}

	// Check if the header starts with the following line
	if (this->CurrentLine != "/* DTI BMT format */" )
	{
		vtkErrorMacro(<< "Unrecognized file type: " << this->CurrentLine.c_str() << " for file: " << this->FileName);
		return false;
    }

	return true;
}


//--------------------------[ ReadScanDirection ]--------------------------\\

bool vtkDTIReader2::ReadScanDirection()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading scan direction for file: " << this->FileName);
		return false;
	}

	// The scan direction line should contain only one character
	if (this->CurrentLine.size() != 1)
	{
		vtkErrorMacro(<<"Unsupported scan direction! Scan direction should be 'T', 'S' or 'C'.");
		return false;
	}

	// Get the only character of the line
	char c = this->CurrentLine[0];

	// Parse the scan direction
	switch (c)
	{
		case 'T':
			this->ScanDirection = BMIA_SCAN_DIRECTION_TRANSVERSAL;
			break;
		case 'S':
			this->ScanDirection = BMIA_SCAN_DIRECTION_SAGITTAL;
			break;
		case 'C':
			this->ScanDirection = BMIA_SCAN_DIRECTION_CORONAL;
			break;
		default:
			vtkErrorMacro(<<"Unsupported scan direction! Scan direction should be 'T', 'S' or 'C'.");
			return false;
	}
  
	return true;
}


//----------------------------[ ReadDataType ]-----------------------------\\

bool vtkDTIReader2::ReadDataType()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading data type for file: " << this->FileName);
		return false;
	}

	// Unsigned short
	if (this->CurrentLine == "ushort")
	{
		this->DataType = VTK_UNSIGNED_SHORT;
	}
	// Floating point
	else if (this->CurrentLine == "float")
	{
		this->DataType = VTK_FLOAT;
	}
	// Doubles
	else if (this->CurrentLine == "double")
	{
		this->DataType = VTK_DOUBLE;
	}
	// Unknown data type
	else
	{
		vtkErrorMacro(<<"Unknown data type: " << this->CurrentLine.c_str() << ", on line 3! " << " for file: " << this->FileName);
		return false;
	}

	return true;
}


//------------------------[ ReadComponentFileNames ]-----------------------\\

bool vtkDTIReader2::ReadComponentFileNames()
{
	// Clear the component information list
	this->ComponentInfoList.clear();

	// Read all component file names
	for (int i = 0; i < BMIA_NUMBER_OF_COMPONENTS; ++i)
    {
		// Get the next line
		if (!this->NextLine())
		{
			vtkErrorMacro(<<"Premature EOF reading component filename.");
			return false;
		}

		// Current line position
		unsigned int linepos = 0;

		// Index of the current components
		DTIComponent componentIndex;

		// Fetch the first word of the current line
		std::string componentString = vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);

		// Parse the component name
		if (componentString == "XX")
		{
			componentIndex = BMIA_XX_COMPONENT_INDEX;
		}
		else if (componentString == "XY")
		{
			componentIndex = BMIA_XY_COMPONENT_INDEX;
		}
		else if (componentString == "XZ")
		{
			componentIndex = BMIA_XZ_COMPONENT_INDEX;
		}
		else if (componentString == "YY")
		{
			componentIndex = BMIA_YY_COMPONENT_INDEX;
		}
		else if (componentString == "YZ")
		{
			componentIndex = BMIA_YZ_COMPONENT_INDEX;
		}
		else if (componentString == "ZZ")
		{
			componentIndex = BMIA_ZZ_COMPONENT_INDEX;
		}
		else if (componentString == "I")
		{
			componentIndex = BMIA_I_COMPONENT_INDEX;
		}
		else
		{
			vtkErrorMacro(<< "Invalid component index " << componentString.c_str() << "!");
			return false;
		}

		// Determine the file name of the component
		string componentFileName = vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);

		// Filename should not contain slashes or backslashes
		if (componentFileName.find("/", 0) == std::string::npos && componentFileName.find("\\", 0) == std::string::npos)
		{
			// Get the filename, check if it contains a slash
			std::string filePath(this->FileName);
			int pathEnd = filePath.rfind("/", filePath.size());

			// If so, prepend the directory to the component file name
			if (pathEnd != vtkstd::string::npos)
			{
				componentFileName = filePath.substr(0, pathEnd + 1) + componentFileName;
			}

			// If the file path does not contain a slash, check if it contains a backslash
			else
			{
				pathEnd = filePath.rfind("\\", filePath.size());

				// If so, prepend the directory to the component file name
				if (pathEnd != vtkstd::string::npos)
				{
					componentFileName = filePath.substr(0, pathEnd + 1) + componentFileName;
				}
			}

		} // if [component name does not contain slashes]

		else
		{
			vtkErrorMacro(<< "Component file name should not contain forward slashes or backslashes!");
			return false;
		}

		// Add the information to the list
		componentInfo newComponentInfo;
		newComponentInfo.name = componentFileName;
		newComponentInfo.ID   = componentIndex;
		this->ComponentInfoList.push_back(newComponentInfo);

	} // for [every DTI component]

	return true;
}


//----------------------------[ ReadVoxelSize ]----------------------------\\

bool vtkDTIReader2::ReadVoxelSize()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading voxel size for file: " << this->FileName);
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


//----------------------[ ReadTensorTransformMatrix ]----------------------\\

void vtkDTIReader2::ReadTensorTransformMatrix()
{
	// Initialize the transformation matrix
	this->TensorTransformMatrix = new float * [6];

	// Create the full 6x6 matrix
	for (int i = 0; i < 6; ++i) 
	{
		this->TensorTransformMatrix[i] = new float[6];
	}

	// Loop through all six lines describing the transformation matrix
	for (int line = 0; line < 6; ++line)
    {
		// Try to read the next line
		if (!this->NextLine())
		{
			this->CleanTensorTransformMatrix();
			vtkDebugMacro(<<"Tensor transform matrix could not be read. " << "No tensor transform matrix will be used.");
			return;
		}

		// Current line position
		unsigned int linepos = 0;

		// Read the six floats on the current line
		for (int i = 0; i < 6; ++i)
		{
			this->TensorTransformMatrix[line][i] = vtkBetterDataReader::ReadFloat(this->CurrentLine, linepos, linepos);
		} 

	} // for [all six lines]

	return;
}


//----------------------[ CleanTensorTransformMatrix ]---------------------\\

void vtkDTIReader2::CleanTensorTransformMatrix()
{
	if (this->TensorTransformMatrix)
	{
		// Delete all six lines of the matrix
		for (int i = 0; i < 6; ++i)
		{
			if (this->TensorTransformMatrix[i])
			{
				delete[] this->TensorTransformMatrix[i];
				this->TensorTransformMatrix[i] = NULL;
			}
		}
    
		// Delete the matrix row array
		delete [] this->TensorTransformMatrix;
		this->TensorTransformMatrix = NULL;
	}
}


//-----------------------------[ ExecuteData ]-----------------------------\\

void vtkDTIReader2::ExecuteData(vtkDataObject* out)
{
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

	// Create the output array
	this->outputTensorArray = vtkDataArray::CreateDataArray(VTK_FLOAT);
	this->outputTensorArray->SetNumberOfComponents(6);

	// Create the output image for the structural data
	if (this->structImage)
		this->structImage->Delete();

	this->structImage = vtkImageData::New();

	// Create the point data and scalar array
	vtkPointData * structPointData = this->structImage->GetPointData();
	this->structScalars = (vtkDataArray *) vtkDataArray::CreateArray(VTK_DOUBLE);

	// Create an iterator for the list of component information objects
	std::list<componentInfo>::iterator iter;

	//  Numerical iterator value
	int i;

	// Component filename from list
	std::string fname;

	// Index of the component
	DTIComponent componentID;

	// Image extents for the first image ("0") and subsequent images ("X")
	int extent0[6];
	int extentX[6];

	// Number of tuples for the first image ("0") and subsequent images ("X")
	int numberOfTuples0;
	int numberOfTuplesX;

	// Loop through all stored filenames
	for (	iter  = this->ComponentInfoList.begin(), i = 0;
			iter != this->ComponentInfoList.end(); 
			++iter, ++i)
	{
		// Get the current filename and component ID
		fname		= (*iter).name;
		componentID = (*iter).ID;

		// Create the reader
		this->componentReader = vtkDTIComponentReader::New();
		this->componentReader->SetFileName(fname.c_str());
		this->componentReader->SetDataScalarType(this->DataType);

		// Update the progress bar
		std::string progressString = "Reading component file '" + fname + "'...";
		this->SetProgressText(progressString.c_str());
		this->UpdateProgress(((float) i) / (float) (BMIA_NUMBER_OF_COMPONENTS));

		// Update the reader to read the file
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

		// If this is the first image, get the extent and number of tuples
		if (i == 0)
		{
			componentImageData->GetExtent(extent0);
			numberOfTuples0 = componentScalars->GetNumberOfTuples();

			// Finish initialization of the output images
			output->CopyStructure(this->componentImageData);
			output->SetDimensions(this->componentImageData->GetDimensions());
			output->SetOrigin(0.0, 0.0, 0.0);
			output->SetSpacing(this->DataSpacing[0], this->DataSpacing[1], this->DataSpacing[2]);

			this->structImage->CopyStructure(this->componentImageData);
			this->structImage->SetDimensions(this->componentImageData->GetDimensions());
			this->structImage->SetOrigin(0.0, 0.0, 0.0);
			this->structImage->SetSpacing(this->DataSpacing[0], this->DataSpacing[1], this->DataSpacing[2]);

			// Finish initialization of the output arrays
			this->outputTensorArray->SetNumberOfTuples(numberOfTuples0);
			
			this->structScalars->SetNumberOfComponents(1);
			this->structScalars->SetNumberOfTuples(numberOfTuples0);
		}
		// Otherwise, compare the extent and number of tuples of the current image
		// with those of the first image, and return an error if a mismatch is found
		else
		{
			componentImageData->GetExtent(extentX);

			if (	extent0[0] != extentX[0] || extent0[1] != extentX[1] || extent0[2] != extentX[2] || 
					extent0[3] != extentX[3] || extent0[4] != extentX[4] || extent0[5] != extentX[5] )
			{
				vtkErrorMacro(<<"Extent mismatch for component '" << fname << "'!");
				this->CleanUp();
				return;
			}

			numberOfTuplesX = componentScalars->GetNumberOfTuples();

			if (numberOfTuples0 != numberOfTuplesX)
			{
				vtkErrorMacro(<<"Number of tuples mismatch for component '" << fname << "'!");
				this->CleanUp();
				return;
			}
		} 

		// Copy all voxel values to the correct component in the output

		switch (componentID)
		{
			case BMIA_XX_COMPONENT_INDEX:
				this->outputTensorArray->CopyComponent(0, this->componentScalars, 0);
				break;
			case BMIA_XY_COMPONENT_INDEX:
				this->outputTensorArray->CopyComponent(1, this->componentScalars, 0);
				break;
			case BMIA_XZ_COMPONENT_INDEX:
				this->outputTensorArray->CopyComponent(2, this->componentScalars, 0);
				break;
			case BMIA_YY_COMPONENT_INDEX:
				this->outputTensorArray->CopyComponent(3, this->componentScalars, 0);
				break;
			case BMIA_YZ_COMPONENT_INDEX:
				this->outputTensorArray->CopyComponent(4, this->componentScalars, 0);
				break;
			case BMIA_ZZ_COMPONENT_INDEX:
				this->outputTensorArray->CopyComponent(5, this->componentScalars, 0);
				break;
			case BMIA_I_COMPONENT_INDEX:
				structPointData->CopyAllocate(this->componentPointData, numberOfTuples0);
				this->structScalars->CopyComponent(0, this->componentScalars, 0);
				break;
			default:
				vtkErrorMacro(<< "Unknown component identifier!");
				this->CleanUp();
				return;
		}

		// Delete the component reader
		this->componentReader->Delete();
		this->componentReader = NULL;

	} // for [componentInfoList]

	// Transform the tensor array
	this->transformTensors(outputTensorArray);

	// Store the scalar array of the structural image
	structPointData->SetScalars(this->structScalars);

	// Store the tensor array
	this->outputTensorArray->SetName("Tensors");
	output->GetPointData()->AddArray(this->outputTensorArray);
	output->GetPointData()->SetActiveScalars("Tensors");
  
	// Finalize the output
	output->SetUpdateExtentToWholeExtent();
	output->Squeeze();

	// Done!
	this->UpdateProgress(1.0);
}

  
//---------------------------[ transformTensors ]--------------------------\\

void vtkDTIReader2::transformTensors(vtkDataArray * inTensors)
{
	// Do nothing if no matrix has been set
	if (!this->TensorTransformMatrix)
		return;

	// Reset the progress
	this->SetProgressText("Transforming DTI image...");
	this->SetProgress(0.0);

	// Set the step size for the progress bar
	int progressStepSize = (int) ((float) inTensors->GetNumberOfTuples() / 25.0f);
	progressStepSize += (progressStepSize == 0) ? (1) : (0);

	// Current input and output tensors
	double inTensor[6];
	double outTensor[6];

	// Transformation matrix
	float ** matrix = this->TensorTransformMatrix;

	for (vtkIdType ptId = 0; ptId < inTensors->GetNumberOfTuples(); ++ptId)
	{
		// Get the input tensor
		inTensors->GetTuple(ptId, inTensor);

		// Transform the tensor
		outTensor[0] =	matrix[0][0] * inTensor[0] + matrix[0][1] * inTensor[3] + matrix[0][2] * inTensor[5] + 
						matrix[0][3] * inTensor[1] + matrix[0][4] * inTensor[2] + matrix[0][5] * inTensor[4];
		outTensor[1] =	matrix[3][0] * inTensor[0] + matrix[3][1] * inTensor[3] + matrix[3][2] * inTensor[5] +
						matrix[3][3] * inTensor[1] + matrix[3][4] * inTensor[2] + matrix[3][5] * inTensor[4];
		outTensor[2] =	matrix[4][0] * inTensor[0] + matrix[4][1] * inTensor[3] + matrix[4][2] * inTensor[5] +
						matrix[4][3] * inTensor[1] + matrix[4][4] * inTensor[2] + matrix[4][5] * inTensor[4];
		outTensor[3] =	matrix[1][0] * inTensor[0] + matrix[1][1] * inTensor[3] + matrix[1][2] * inTensor[5] +
						matrix[1][3] * inTensor[1] + matrix[1][4] * inTensor[2] + matrix[1][5] * inTensor[4];
		outTensor[4] =	matrix[5][0] * inTensor[0] + matrix[5][1] * inTensor[3] + matrix[5][2] * inTensor[5] +
						matrix[5][3] * inTensor[1] + matrix[5][4] * inTensor[2] + matrix[5][5] * inTensor[4];
		outTensor[5] =	matrix[2][0] * inTensor[0] + matrix[2][1] * inTensor[3] + matrix[2][2] * inTensor[5] +
						matrix[2][3] * inTensor[1] + matrix[2][4] * inTensor[2] + matrix[2][5] * inTensor[4];

		// Write the output tensor to the array
		inTensors->InsertTuple(ptId, outTensor);

		// Update the progress bar
		if ((ptId % progressStepSize) == 0)
		{
			this->SetProgress(ptId / (float) inTensors->GetNumberOfTuples());
		}
	}
}


} // namespace bmia
