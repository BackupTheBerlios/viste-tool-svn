/*
 * vtkHARDIReader.cxx
 *
 * 2010-11-18	Evert van Aart
 * - First version. 
 *
 * 2011-04-19	Evert van Aart
 * - Changed the reader so that the output has the same format as the discrete
 *   sphere functions read in by the NIfTI reader. This way, the Geometry Glyphs
 *   plugin can be used to visualize the HARDI data read by this plugin.
 * - Added triangulation for the input HARDI data.
 *
 * 2011-04-26	Evert van Aart
 * - Improved progress reporting.
 *
 */



/** Includes */

#include "vtkHARDIReader.h"


namespace bmia {


vtkStandardNewMacro(vtkHARDIReader);

	
//-----------------------------[ Constructor ]-----------------------------\\

vtkHARDIReader::vtkHARDIReader()
{
	// Set default values of the class variables
	this->InformationExecuted 	= false;
	this->readDimensions 		= false;
	this->CurrentLine 			= "";
	this->zeroGradientCounter 	= 0;
	this->b 					= 0.0;
	this->DataType 				= VTK_UNSIGNED_SHORT;
	this->numberOfGradients 	= 0;
	this->Dimensions[0] 		= 0;
	this->Dimensions[1] 		= 0;
	this->Dimensions[2] 		= 0;
	this->DataSpacing[0] 		= 1.0;
	this->DataSpacing[1] 		= 1.0;
	this->DataSpacing[2] 		= 1.0;

	// Set pointers to NULL
	this->grayValuesArray		= NULL;
	this->zeroGradientsArray	= NULL;;
	this->componentReader		= NULL;;
	this->componentImageData	= NULL;;
	this->componentPointData	= NULL;;
	this->componentScalars		= NULL;;
	this->gradientArray			= NULL;;
	this->anglesArray			= NULL;;
}


//------------------------------[ Destructor ]-----------------------------\\

vtkHARDIReader::~vtkHARDIReader()
{
	// Clean up everything
	this->CleanUp();
}


//-------------------------------[ CleanUp ]-------------------------------\\

void vtkHARDIReader::CleanUp()
{
	// Set default values of the class variables
	this->InformationExecuted 	= false;
	this->readDimensions 		= false;
	this->CurrentLine 			= "";
	this->zeroGradientCounter 	= 0;
	this->b 					= 0.0;
	this->DataType 				= VTK_UNSIGNED_SHORT;
	this->numberOfGradients 	= 0;
	this->Dimensions[0] 		= 0;
	this->Dimensions[1] 		= 0;
	this->Dimensions[2] 		= 0;
	this->DataSpacing[0] 		= 1.0;
	this->DataSpacing[1] 		= 1.0;
	this->DataSpacing[2] 		= 1.0;

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
	
	// Delete individual VTK objects and set them to NULL
	
	if (this->grayValuesArray)
	{
		this->grayValuesArray->Delete();
		this->grayValuesArray = NULL;
	}

	if (this->zeroGradientsArray)
	{
		this->zeroGradientsArray->Delete();
		this->zeroGradientsArray = NULL;
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

	if (this->gradientArray)
	{
		this->gradientArray->Delete();
		this->gradientArray = NULL;
	}

	if (this->anglesArray)
	{
		this->anglesArray->Delete();
		this->anglesArray = NULL;
	}
}


//-------------------------------[ NextLine ]------------------------------\\

bool vtkHARDIReader::NextLine()
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


//-----------------------------[ CanReadFile ]-----------------------------\\
	
int vtkHARDIReader::CanReadFile(const char * fname)
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


//--------------------------[ ExecuteInformation ]-------------------------\\

void vtkHARDIReader::ExecuteInformation()
{
	// Information execution has not finished yet
	this->InformationExecuted = false;

	// Get the filename of the .hardi file
	char * fname = this->GetFileName();

	// Check if the filename has been set
	if (!fname)
	{
		vtkDebugMacro(<<"No input file defined.");
		return;
	}

	// Initialize the progress bar
	this->SetProgressText("Parsing .hardi header file...");
	this->UpdateProgress(0.0);

	// Open an input stream for the ".hardi" file
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

	// Try to read the dimensions, if necessary.
	if (this->readDimensions)
	{
		if (!this->ReadDimensions())
		{
			vtkWarningMacro(<<"Error reading dimensions, aborting reading process...");
			this->CleanUp();
			return;	
		}
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

	// Try to read the number of gradients	
	if (!this->ReadNumberOfGradients())
	{
		vtkWarningMacro(<<"Error reading number of gradients, aborting reading process...");
		this->CleanUp();
		return;			
	}

	// Try to read the filenames of the .dat component files	
	if(!this->ReadGradientFileNamesDat())
	{
		vtkWarningMacro(<<"Number of components does not match the number of gradients\n" <<
							"as defined in the .hardi file. Aborting reading process...");
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

bool vtkHARDIReader::ReadHeader()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading header for file: " << this->FileName);
		return false;
	}
	
	// The file should start with one of these keywords
	if ( (this->CurrentLine != "HARDI00" ) && 
	     (this->CurrentLine != "HARDI01" ) && 
		 (this->CurrentLine != "HARDI02" ) && 
		 (this->CurrentLine != "HARDI03" ) )
	{
		vtkErrorMacro(<<"Unrecognized file type '"<< this->CurrentLine.c_str() << "' for file " << this->FileName);
		return false;
	}

	// If this is a HARDI03 file, we should read the dimenions from the .hardi file itself
	if (this->CurrentLine == "HARDI03")
		this->readDimensions = true;
		
	return true;
}
	
	
//----------------------------[ ReadDimensions ]---------------------------\\

bool vtkHARDIReader::ReadDimensions()
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


//------------------------------[ ReadBValue ]-----------------------------\\

bool vtkHARDIReader::ReadBValue()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading B-value for file: " << this->FileName);
		return false;
	}

	// Current line position
	unsigned int linepos = 0;

	// Read B-value
	this->b = vtkBetterDataReader::ReadInt(this->CurrentLine, 2, linepos);
	
	return true;
}
	
	
//------------------------[ ReadNumberOfGradients ]------------------------\\

bool vtkHARDIReader::ReadNumberOfGradients()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading number of gradients for file: " << this->FileName);
		return false;
	}
	
	// Current line position
	unsigned int linepos = 0;

	// Read the number of gradients
	this->numberOfGradients = vtkBetterDataReader::ReadInt(this->CurrentLine, 0, linepos);

	return true;
}
	
	
//-----------------------------[ ReadDataType ]----------------------------\\

bool vtkHARDIReader::ReadDataType()
{
	// Try to read the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading data type for file: " << this->FileName);
		return false;
	}

	// Unsigned short
	if (this->CurrentLine == "type ushort")
	{
		this->DataType = VTK_UNSIGNED_SHORT;
	}
	// Floating point
	else if (this->CurrentLine == "type float")
	{
		this->DataType = VTK_FLOAT;
	}
	// Unknown data type
	else
	{
		vtkErrorMacro(<<"Unknown data type: " << this->CurrentLine.c_str() << ", on line 3! " << " for file: " << this->FileName);
		return false;
	}

	return true;
}

	
//----------------------------[ ReadVoxelSize ]----------------------------\\

bool vtkHARDIReader::ReadVoxelSize()
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

	
//-----------------------[ ReadGradientFileNamesDat ]----------------------\\

bool vtkHARDIReader::ReadGradientFileNamesDat()
{
	vtkDebugMacro(<<"Reading component file names");

	// Get the next line
	if (!this->NextLine())
	{
		vtkErrorMacro(<<"Premature EOF reading component filename.");
		return false;
	}

	// Initialize the two gradient arrays
	this->gradientArray = vtkDoubleArray::New();
	this->gradientArray->SetNumberOfComponents(3);

	this->anglesArray = vtkDoubleArray::New();
	this->anglesArray->SetNumberOfComponents(2);
	this->anglesArray->SetName("Spherical Directions");

	// Current line position
	unsigned int linepos = 0;
	
	// Gradient direction
	double gradient[3];

	// Reset the number of zero gradients
	this->zeroGradientCounter = 0;
	
	// Number of gradients read
	int gradientCounter = 0;
	
	// Index for the matrices of non-zero gradients
	int nonZeroGradientIndex = 0;

	// Read all lines in the file
	while (this->CurrentLine[0] !='\n')
	{
		// Reset the line position
		linepos = 0;

		// Read the first word, which is the component name. We do not use it at this point.
		std::string componentString = vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);

		// Read the gradient directions
		for(int i = 0; i < 3;i++) 
		{
			gradient[i] = (double) vtkBetterDataReader::ReadFloat(this->CurrentLine, linepos, linepos);
		}

		// Normalize the gradient
		vtkMath::Normalize(gradient);
		
		// Convert the unit gradient vector to a zenith angle (0 to pi, angle with
		// the positive Z-axis) and an azimuth angle (-pi to pi, angle with the 
		// positive X-axis). This is the format used by discrete sphere functions. 

		double zenith = asin(gradient[2]);

		double azimuth = 0.0;

		// Compute the azimuth
		if (cos(zenith) != 0.0)
		{
			azimuth = acos(gradient[0] / cos(zenith));

			if (gradient[1] < 0.0)
				azimuth *= -1.0;
		}

		// Make the zenith the angle with the positive Z-axis
		zenith = (vtkMath::Pi() / 2.0f) - zenith;
			
		// Add the new gradient to the list of all gradients
		this->gradientArray->InsertNextTuple3(gradient[0], gradient[1], gradient[2]);

		// Check if the gradient is non-zero
		if (gradient[0] != 0 || gradient[1] != 0 || gradient[2] != 0)
		{ 
			// If so, add the angles to the list of non-zero gradients...
			this->anglesArray->InsertNextTuple2(azimuth, zenith);

			// Increase the index for these two matrices
			nonZeroGradientIndex++;
		}
		else
		{
			// If the gradient is zero, increment the counter
			this->zeroGradientCounter++;
		}
		
		// Get the last word of the line, which is the filename of the component data set
		std::string componentFileName = vtkBetterDataReader::ReadWord(this->CurrentLine, linepos, linepos);

		// Get the filename of the ".hardi" file...
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
		
		// Successfully read one line, so increment the counter
		gradientCounter++;

		// Try to read the next line, and break the while loop if it fails
		if (!this->NextLine())
		{
			break;
		}
	} // while [CurrentLine]

	// Reading was successful if the number of lines read matches the number of gradients
	return (gradientCounter == this->numberOfGradients);
}


//-----------------------------[ ExecuteData ]-----------------------------\\

void vtkHARDIReader::ExecuteData(vtkDataObject * out)
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
	
	// The gray values array contains scalar values from all components with non-zero gradient
	this->grayValuesArray = vtkDoubleArray::New();
	
	// The zero-gradients array contains the sum of all components with zero gradients
	this->zeroGradientsArray = vtkFloatArray::New();

	// Set the names of the output arrays
	this->grayValuesArray->SetName("Vectors");
	this->zeroGradientsArray->SetName("Zero Gradient Average");

	// Gray values array has one component for each non-zero gradient
	this->grayValuesArray->SetNumberOfComponents(this->numberOfGradients - this->zeroGradientCounter);
	
	// Zero-gradients array only has one component
	this->zeroGradientsArray->SetNumberOfComponents(1);
	
	// Dimensions and extent of the first image.
	int dimRef[3];
	int extentRef[6];

	// Dimensions and extent of the other images. These should be the same for all components.
	int dim[3];
	int extent[6];
	
	// Index of current component file
	int componentIndex;

	// Index of current component, excluding those with non-zero gradients
	int componentIndexNonZero;
	
	// Number of voxels in the input images
	int numberOfVoxels = 0;

	// Loop through all stored filenames
	for (iter = this->ComponentFileNames.begin(), componentIndex = 0, componentIndexNonZero = 0; 
		iter != this->ComponentFileNames.end(); 
		++iter, ++componentIndex)
	{
		// Get the current filename
		fname = (*iter);

		// Create a new component reader
		this->componentReader = vtkDTIComponentReader::New();
		
		// Set filename and data type of the reader
		this->componentReader->SetFileName(fname.c_str());
		this->componentReader->SetDataScalarType(this->DataType);

		// If we already read the dimensions from the header, we shouldn't do it in the component files.
		if (this->readDimensions)
		{
			// Don't read dimensions from the component files
			this->componentReader->ReadDimensionsOff();
			
			// Manually set the extent of the reader using the read dimensions
			this->componentReader->SetDataExtent(	0, this->Dimensions[0] - 1, 
													0, this->Dimensions[1] - 1, 
													0, this->Dimensions[2] - 1	);
		
		}
		
		// Update text of progress bar
		std::string progressText("Reading component file '");
		progressText += fname;
		progressText += "'...";
		this->SetProgressText(progressText.c_str());

		// Update the progress bar
		this->UpdateProgress(((float) componentIndex) / (float) (this->numberOfGradients + 1));
		
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
		
		//If this is the first file, get the reference dimensions and extent, and initialize the output arrays.
		if (iter == this->ComponentFileNames.begin())
		{
			this->componentImageData->GetDimensions(dimRef);
			this->componentImageData->GetExtent(extentRef);
			
			// Get the number of voxels
			numberOfVoxels = this->componentScalars->GetNumberOfTuples();

			// Set the number of voxels for both arrays
			   this->grayValuesArray->SetNumberOfTuples(numberOfVoxels);
			this->zeroGradientsArray->SetNumberOfTuples(numberOfVoxels);
			
			// Fill the zero-gradients array with zeros.
			this->zeroGradientsArray->FillComponent(0, 0);
			
			// Configure the output
			output->CopyStructure(this->componentImageData);
			output->SetDimensions(dimRef);
			output->SetOrigin(0.0, 0.0, 0.0);
			output->SetSpacing(this->DataSpacing[0], this->DataSpacing[1], this->DataSpacing[2]);
		}
		// Otherwise, check if the dimensions and extent match the reference values
		else
		{
			this->componentImageData->GetDimensions(dim);
			this->componentImageData->GetExtent(extent);

			if (	   dim[0] !=    dimRef[0] ||    dim[1] !=    dimRef[1] ||       dim[2] != dimRef[2] ||
					extent[0] != extentRef[0] || extent[1] != extentRef[1] || extent[2] != extentRef[2] || 
					extent[3] != extentRef[3] || extent[4] != extentRef[4] || extent[5] != extentRef[5] )
			{
				vtkErrorMacro(<<"Wrong dimensions and/or extents for component '" << fname << "'!");
				this->CleanUp();
				return;
			}
		}
		
		// Note: At this point in the old code, it would loop through all voxels, and for each voxel,
		// 	    recompute the 1D index if a boolean called "simulation" has been set. Remapping would
		//      be done by calling "y = dim[1] - y - 1" and "z = dim[2] - z - 1". This is wildly in-
		//      efficient, and far too specific (i.e., I'm guessing it was only used for a handful of
		//      data sets, and mirroring the Y and Z axes isn't default for HARDI). For these reasons,
		//      I am leaving it out for now. If it is still needed, we should probably use the 
		//      "vtkImageReslice" filter, and maybe include a "Modify image..." button when loading
		//	    the HARDI data set.
		
		// Get the current gradient from the gradient array
		double gradient[3];
		this->gradientArray->GetTuple(componentIndex, gradient);

		// If the gradient is non-zero, we copy the entire scalars array to the current component
		// in the gray values array, and increase in the index of the current component index.
		if ( gradient[0] != 0.0 ||
			 gradient[1] != 0.0 ||
			 gradient[2] != 0.0 )
		{
			this->grayValuesArray->CopyComponent(componentIndexNonZero, this->componentScalars, 0);
			componentIndexNonZero++;
		}
		// Otherwise, we need to add the scalar array to the zero-gradients array
		else
		{
			// Loop through all voxels in the image
			for (vtkIdType voxelId = 0; voxelId < numberOfVoxels; ++voxelId)
			{
				// Add the scalar value in the input image, divided by the number of zero gradients,
				// to the current value in the zero-gradients array. This way, we compute the average
				// scalar value of all components with zero gradients.
				
				this->zeroGradientsArray->SetComponent(voxelId, 0, this->componentScalars->GetTuple1(voxelId) + 
																(this->zeroGradientsArray->GetTuple1(voxelId) / (double) this->zeroGradientCounter));
			}
		}

		// Delete the component reader
		this->componentReader->Delete();
		this->componentReader = NULL;
		
	} // for [ComponentFileNames]

	// Update progress bar text
	this->SetProgressText("Triangulating...");

	// Triangulate the  sphere (for geometry glyphs)
	vtkIntArray * triangles = vtkIntArray::New();
	SphereTriangulator * triangulator = new SphereTriangulator;
	bool success = triangulator->triangulateFromAnglesArray(anglesArray, triangles);
	triangles->SetName("Triangles");

	// Add the output arrays
	output->GetPointData()->AddArray(this->zeroGradientsArray);
	output->GetPointData()->AddArray(this->grayValuesArray);

	if (this->anglesArray)
		output->GetPointData()->AddArray(this->anglesArray);

	if (success && triangles)
		output->GetPointData()->AddArray(triangles);

	this->zeroGradientsArray->Delete();
	this->grayValuesArray->Delete();

	if (this->anglesArray)
		this->anglesArray->Delete();

	if (triangles)
		triangles->Delete();

	// Finalize the output
	output->SetUpdateExtentToWholeExtent();
	output->Squeeze();

	// Done!
	this->UpdateProgress(1.0);
}


} // namespace bmia
