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
 * bmiaNiftiReader.cxx
 *
 * 2011-04-04	Evert van Aart
 * - First version. This reader is based on the old "vtkNiftiReader" class by Tim
 *   Peeters, but with some changes. The reader now supports multiple types of 
 *   input data, including triangles ("NIFTI_INTENT_TRIANGLE"), which means that
 *   its output will not always be "vtkImageData". For this reason, the reader
 *   is no longer a child of "vtkImageDataReader2". Furthermore, the reader not
 *   includes support for MiNT extensions.
 *
 * 2011-05-10	Evert van Aart
 * - Added support for spherical harmonics using MiND. 
 *
 * 2011-08-22	Evert van Aart
 * - Which transformation matrix to use is now determined correctly based on the
 *   "qform_code" and "sform_code" of the NIfTI image.
 *
 * 2013-01-28   Mehmet Yusufoglu
 * - Add a pointer parameter pointing the instance of UserOut class as an argument to the constructor.  
 * - Userout class pointer is used to ask which transfomation is used if both qform_code ans sform_code are larger than zero. 
 */


/** Includes */

#include "bmiaNiftiReader.h"


// Copy the "COMP"-th component of the input array, of data type "C_TYPE", to
// an array of doubles of size "arraySize". Used to create scalar volumes.

#define createDoubleScalarArrayMacro(C_TYPE, COMP)								\
	{																			\
		C_TYPE * inArrayCasted = (C_TYPE *) this->NiftiImage->data;				\
		for (int i = 0; i < arraySize; ++i) 									\
			outDoubleArray[i] = (double) inArrayCasted[i + COMP * arraySize];	\
	}

// Copy "ELEMENTS" elements from an input array of type "C_TYPE" to an output
// doubles array. The indices of the elements are mapped through an "indexMap"
// array, which should have size "ELEMENTS". Used to create tensors; for second-
// order tensors, "ELEMENTS" should be six.

#define createDoubleMappedArrayMacro(C_TYPE, ELEMENTS)							\
	{																			\
		C_TYPE * inArrayCasted = (C_TYPE *) this->NiftiImage->data;				\
		for (int i = 0; i < arraySize; ++i) {					 				\
			for (int j = 0; j < ELEMENTS; ++j) {								\
				outDoubleArray[j + ELEMENTS * i] = (double) inArrayCasted[i + indexMap[j] * arraySize];	\
	} } }

// Copy "ELEMENTS" elements from an input array of type "C_TYPE" to an output
// doubles array. Like "createDoubleMappedArrayMacro", but without the index
// mapping. Used for generic vectors.

#define createDoubleVectorArrayMacro(C_TYPE, ELEMENTS)							\
	{																			\
		C_TYPE * inArrayCasted = (C_TYPE *) this->NiftiImage->data;				\
		for (int i = 0; i < arraySize; ++i) {									\
			for (int j = 0; j < ELEMENTS; ++j) {								\
				outDoubleArray[j + ELEMENTS * i] = (double) inArrayCasted[i + j * arraySize];	\
	} } }

// Like "createDoubleVectorArrayMacro", but with integers

#define createIntVectorArrayMacro(C_TYPE, ELEMENTS)								\
	{																			\
		C_TYPE * inArrayCasted = (C_TYPE *) this->NiftiImage->data;				\
		for (int i = 0; i < arraySize; ++i) {									\
			for (int j = 0; j < ELEMENTS; ++j) {								\
			outIntArray[j + ELEMENTS * i] = (int) inArrayCasted[i + j * arraySize];	\
	} } }



namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

bmiaNiftiReader::bmiaNiftiReader(UserOutput * rUserOut)
{
	// Initialize pointers to NULL
	this->NiftiImage		= NULL;
	this->transformMatrix	= NULL;
	this->progress			= NULL;
	this->userOut			= rUserOut;
}


//------------------------------[ Destructor ]-----------------------------\\

bmiaNiftiReader::~bmiaNiftiReader()
{
	// Use the "cleanUp" function. This will 'delete' the VTK data objects generated
	// by this reader, but if the reader was successful, all these data objects 
	// will have been registered to new data sets, which means that they will not 
	// really be deleted in the "cleanUp" function.

	this->cleanUp();
}


//-------------------------------[ cleanUp ]-------------------------------\\

void bmiaNiftiReader::cleanUp()
{
	// Delete the NIfTI image object
	if (this->NiftiImage)
	{
		nifti_image_free(this->NiftiImage);
		this->NiftiImage = NULL;
	}

	// Delete the transformation matrix
	if (this->transformMatrix)
	{
		this->transformMatrix->Delete();
		this->transformMatrix = NULL;
	}

	// Delete all existing output objects
	for (int i = 0; i < this->outData.size(); ++i)
	{
		vtkObject * obj = this->outData.at(i);
		obj->Delete();
	}

	// Clear the output list
	this->outData.clear();

	// Delete the progress bar
	if (this->progress)
	{
		delete this->progress;
		this->progress = NULL;
	}
}


//-----------------------------[ CanReadFile ]-----------------------------\\

int bmiaNiftiReader::CanReadFile(const char * filename)
{
	// Open the file
	QFile fileTest(filename);

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


//----------------------------[ readNIfTIFile ]----------------------------\\

QString bmiaNiftiReader::readNIfTIFile(const char * filename, bool showProgress)
{
	// Check if the filename is set
	if (!filename)
	{
		return "Input filename has not been set.";
	}

	// Store filename
	this->filenameQ = QString(filename);

	// Delete existing NIfTI image
	if (this->NiftiImage)
	{
		nifti_image_free(this->NiftiImage);
		this->NiftiImage = NULL;
	}

	this->progress = NULL;

	if (showProgress)
	{
		// Create a progress dialog
		this->progress = new QProgressDialog;
		this->progress->setLabelText("Reading NIfTI file...");
		this->progress->setWindowTitle("NIfTI Reader");
		this->progress->setMaximum(100);
		this->progress->setMinimum(0);
		this->progress->setMinimumDuration(500);
		this->progress->setValue(25);
	}

	// Read the specified NIfTI file
	this->NiftiImage = nifti_image_read(filename, 1);

	if (!(this->NiftiImage))
	{
		this->cleanUp();
		return "Failed to read the NIfTI file.";
	}

	if (this->progress)
	{
		// Update progress bar
		this->progress->setValue(50);
		this->progress->setLabelText("Parsing NIfTI data...");
	}

	// Warn the user if we could not determine the data type
	if (!(this->determineDataType()))
	{
		this->cleanUp();
		return "Failed to determine the NIfTI data format!";
	}

	vtkObject * obj = NULL;

	// Switch based on the data type of the NIfTI file
	switch(this->imageDataType)
	{
		// Scalar Volumes: Create one "vtkImageData" object with one-component scalar array

		case NDT_ScalarVolume:
			obj = vtkObject::SafeDownCast(this->parseScalarVolume());

			if (obj)
			{
				this->outData.append(obj);
			}
			else
			{
				this->cleanUp();
				return "Failed to parse scalar volume!";
			}

			break;

		// DTI Tensors: Create one "vtkImageData" object with six-component tensor array

		case NDT_DTITensors:
			obj = vtkObject::SafeDownCast(this->parseDTIVolume());

			if (obj)
			{
				this->outData.append(obj);
			}
			else
			{
				this->cleanUp();
				return "Failed to parse DTI tensor volume!";
			}

			break;

		// Discrete Sphere: Create one "vtkImageData" object with n-component vector
		// array (with n the number of vertices on the sphere), and a double array
		// with two components and n tuples describing the sphere directions.

		case NDT_DiscreteSphere:
			obj = vtkObject::SafeDownCast(this->parseDiscreteSphereVolume());

			if (obj)
			{
				this->outData.append(obj);
			}
			else
			{
				this->cleanUp();
				return "Failed to parse discrete sphere volume!";
			}

			break;

		// Spherical harmonics
		case NDT_SphericalHarm:
			obj = vtkObject::SafeDownCast(this->parseSphericalHarmonicsVolume());

			if (obj)
			{
				this->outData.append(obj);
			}
			else
			{
				this->cleanUp();
				return "Failed to parse spherical harmonics volume!";
			}

			break;

		// Triangles: Create one "vtkIntArray" object with three components

		case NDT_Triangles:
			obj = vtkObject::SafeDownCast(this->parseTriangles());

			if (obj)
			{
				this->outData.append(obj);
			}
			else
			{
				this->cleanUp();
				return "Failed to parse triangles!";
			}

			break;

		// Generic Vector: Create n "vtkImageData" arrays, each with a one-component
		// scalar array, where n is the vector length of the NIfTI file.

		case NDT_GenericVector:
			
			for (int i = 0; i < this->NiftiImage->dim[5]; ++i)
			{
				obj = vtkObject::SafeDownCast(this->parseScalarVolume(i));

				if (obj)
				{
					this->outData.append(obj);
				}
				else
				{
					this->cleanUp();
					return "Failed to parse scalar volume!";
				}
			}

			break;

		// This shouldn't happen, since we already check for successful determination
		// of data type above.

		default:

			this->cleanUp();
			return "Unsupported file type!";
	}

	if (progress)
	{
		// Update progress bar
		progress->setValue(75);
		progress->setLabelText("Parsing transformation matrix...");
	}

	// Delete the old transformation matrix
	if (this->transformMatrix)
	{
		this->transformMatrix->Delete();
	}

	this->transformMatrix = NULL;

	// Output VTK matrix
	vtkMatrix4x4 * m = NULL;

	// Temporary matrix array
	double tM[16];

	//If both qform_code and sform_code larger than 0, ask to the user. 
	QString selectedItem("");
	if (this->NiftiImage->qform_code > 0 && this->NiftiImage->sform_code > 0)
	{
		this->userOut->selectItemDialog("Select Transformation Matrix", "Qform-code and Sform-code are positive. Select one of the matrices.", "QForm,SForm",selectedItem);
	}
	
     
	// Use the QForm matrix if available
	if (this->NiftiImage->qform_code > 0 && selectedItem!="SForm")
	{
		// Get the transformation matrix
		tM[ 0] = (double) this->NiftiImage->qto_xyz.m[0][0];	
		tM[ 1] = (double) this->NiftiImage->qto_xyz.m[0][1];	
		tM[ 2] = (double) this->NiftiImage->qto_xyz.m[0][2];	
		tM[ 3] = (double) this->NiftiImage->qto_xyz.m[0][3];	

		tM[ 4] = (double) this->NiftiImage->qto_xyz.m[1][0];	
		tM[ 5] = (double) this->NiftiImage->qto_xyz.m[1][1];	
		tM[ 6] = (double) this->NiftiImage->qto_xyz.m[1][2];	
		tM[ 7] = (double) this->NiftiImage->qto_xyz.m[1][3];	

		tM[ 8] = (double) this->NiftiImage->qto_xyz.m[2][0];	
		tM[ 9] = (double) this->NiftiImage->qto_xyz.m[2][1];	
		tM[10] = (double) this->NiftiImage->qto_xyz.m[2][2];	
		tM[11] = (double) this->NiftiImage->qto_xyz.m[2][3];	

		tM[12] = (double) this->NiftiImage->qto_xyz.m[3][0];	
		tM[13] = (double) this->NiftiImage->qto_xyz.m[3][1];	
		tM[14] = (double) this->NiftiImage->qto_xyz.m[3][2];	
		tM[15] = (double) this->NiftiImage->qto_xyz.m[3][3];	
	}
	
	// Otherwise, use the SForm matrix
	else if (this->NiftiImage->sform_code)
	{
		// Get the transformation matrix
		tM[ 0] = (double) this->NiftiImage->sto_xyz.m[0][0];	
		tM[ 1] = (double) this->NiftiImage->sto_xyz.m[0][1];	
		tM[ 2] = (double) this->NiftiImage->sto_xyz.m[0][2];	
		tM[ 3] = (double) this->NiftiImage->sto_xyz.m[0][3];	

		tM[ 4] = (double) this->NiftiImage->sto_xyz.m[1][0];	
		tM[ 5] = (double) this->NiftiImage->sto_xyz.m[1][1];	
		tM[ 6] = (double) this->NiftiImage->sto_xyz.m[1][2];	
		tM[ 7] = (double) this->NiftiImage->sto_xyz.m[1][3];	

		tM[ 8] = (double) this->NiftiImage->sto_xyz.m[2][0];	
		tM[ 9] = (double) this->NiftiImage->sto_xyz.m[2][1];	
		tM[10] = (double) this->NiftiImage->sto_xyz.m[2][2];	
		tM[11] = (double) this->NiftiImage->sto_xyz.m[2][3];	

		tM[12] = (double) this->NiftiImage->sto_xyz.m[3][0];	
		tM[13] = (double) this->NiftiImage->sto_xyz.m[3][1];	
		tM[14] = (double) this->NiftiImage->sto_xyz.m[3][2];	
		tM[15] = (double) this->NiftiImage->sto_xyz.m[3][3];	
	}

	// Otherwise, just use the voxel spacing
	else
	{
		tM[ 0] = (double) this->NiftiImage->dx;
		tM[ 1] = 0.0;
		tM[ 2] = 0.0;
		tM[ 3] = 0.0;

		tM[ 4] = 0.0;
		tM[ 5] = (double) this->NiftiImage->dy;	
		tM[ 6] = 0.0;
		tM[ 7] = 0.0;

		tM[ 8] = 0.0;	
		tM[ 9] = 0.0;
		tM[10] = (double) this->NiftiImage->dz;	
		tM[11] = 0.0;

		tM[12] = 0.0;	
		tM[13] = 0.0;	
		tM[14] = 0.0;
		tM[15] = 1.0;	
	}

	// Create a VTK matrix
	m = vtkMatrix4x4::New();
	m->DeepCopy(tM);

	this->transformMatrix = m;

	if (this->progress)
	{
		// Finalize the progress bar
		this->progress->setValue(100);
	}

	return "";
}


//--------------------------[ determineDataType ]--------------------------\\

bool bmiaNiftiReader::determineDataType()
{
	// We start by checking if we've got MiND extensions
	int extPos = 0;
	nifti1_extension * ext = this->findExtension(NIFTI_ECODE_MIND_IDENT, extPos);

	if (ext)
	{
		// Discrete sphere function
		if (this->compareMiNDID(ext->edata, (char*) "DISCSPHFUNC", ext->esize - 8))
		{
			this->imageDataType = NDT_DiscreteSphere;
			return true;
		}

		// Spherical harmonics
		if (this->compareMiNDID(ext->edata, (char*) "REALSPHARMCOEFFS", ext->esize - 8))
		{
			this->imageDataType = NDT_SphericalHarm;
			return true;
		}

		// DTI tensors
		if (this->compareMiNDID(ext->edata, (char*) "DTENSOR", ext->esize - 8))
		{
			this->imageDataType = NDT_DTITensors;
			return true;
		}
	}

	// If the image is just 3D, we interpret it as a scalar volume
	if (this->NiftiImage->dim[0] == 3 && this->NiftiImage->dim[5] == 1)
	{
		this->imageDataType = NDT_ScalarVolume;
		return true;
	}

	// Otherwise, if it has a six-element vector per voxel, and we've got either 
	// the "SYMMATRIX" or "VECTOR" intent code, we assume that we've got DTI tensors

	if (this->NiftiImage->dim[0] == 5 && this->NiftiImage->dim[5] == 6 &&
		(this->NiftiImage->intent_code == NIFTI_INTENT_SYMMATRIX || this->NiftiImage->intent_code == NIFTI_INTENT_VECTOR))
	{
		this->imageDataType = NDT_DTITensors;
		return true;
	}

	// Check if we've got triangles
	if (this->NiftiImage->intent_code == NIFTI_INTENT_TRIANGLE)
	{
		this->imageDataType = NDT_Triangles;
		return true;
	}

	// No clear interpretation available, but if it's a 5D image with 1D vectors
	// per voxel, we can still interpret it as a volume of generic vectors.

	if (this->NiftiImage->dim[0] == 5 && this->NiftiImage->dim[5] > 1)
	{
		this->imageDataType = NDT_GenericVector;
		return true;
	}

	// No match found
	this->imageDataType = NDT_Unknown;
	return false;
}


//----------------------------[ compareMiNDID ]----------------------------\\

bool bmiaNiftiReader::compareMiNDID(char * id, char * target, int n)
{
	// We should have at least some characters to compare
	if (n <= 0)
		return false;

	// Loop through all characters
	for (int i = 0; i < n; ++i)
	{
		// Check for the end of the string
		if (id[i] == 0 || target[i] == 0)
			return true;

		// Return false on mismatch
		if (id[i] != target[i])
			return false;
	}

	// No mismatch, so return true
	return true;
}


//----------------------------[ findExtension ]----------------------------\\

nifti1_extension * bmiaNiftiReader::findExtension(int targetID, int & extPos)
{
	// Check if the NIfTI image has been set
	if (!(this->NiftiImage))
		return NULL;

	// Check if the NIfTI image contains MiND extensions
	if(strcmp(this->NiftiImage->intent_name, "MiND") != 0)
		return NULL;

	// Check if the extension index is within range
	if (extPos < 0 || extPos >= this->NiftiImage->num_ext)
		return NULL;

	// Loop from the current extension index to the last extension
	for (int i = extPos; i < this->NiftiImage->num_ext; ++i)
	{
		// Get the current extension
		nifti1_extension * ext = &(this->NiftiImage->ext_list[i]);

		// Check if the code matches
		if (ext->ecode == targetID)
		{
			extPos = i;
			return ext;
		}
	}

	return NULL;
}


//--------------------------[ parseScalarVolume ]--------------------------\\

vtkImageData * bmiaNiftiReader::parseScalarVolume(int component)
{
	// Create an output array with one scalar per voxel
	int arraySize = this->NiftiImage->dim[1] * this->NiftiImage->dim[2] * this->NiftiImage->dim[3];
	double * outDoubleArray = new double[arraySize];

	// Copy the "component"-th component of the input to the array
	switch (this->NiftiImage->datatype)
	{
		case DT_UNSIGNED_CHAR:	createDoubleScalarArrayMacro(unsigned char,  component);		break;
		case DT_SIGNED_SHORT:	createDoubleScalarArrayMacro(unsigned short, component);		break;
		case DT_UINT16:			createDoubleScalarArrayMacro(unsigned short, component);		break;
		case DT_SIGNED_INT:		createDoubleScalarArrayMacro(int,            component);		break;
		case DT_FLOAT:			createDoubleScalarArrayMacro(float,			 component);		break;
		case DT_DOUBLE:			createDoubleScalarArrayMacro(double,		 component);		break;

		default:
			delete [] outDoubleArray;
			return NULL;
	}

	// Create an image using the new array
	return this->createimageData(outDoubleArray, 1, "Scalars");
}


//----------------------------[ parseDTIVolume ]---------------------------\\

vtkImageData * bmiaNiftiReader::parseDTIVolume()
{
	// Default index map. By default, NIfTI stores symmetrical tensors like this:
	//
	// a[0]
	// a[1] a[2]
	// a[3] a[4] a[5],
	//
	// while we use the following format:
	// 
	// b[0] b[1] b[2]
	//      b[3] b[4]
	//           b[5]
	// 
	// The default index map thus converts from "a" to "b".

	int indexMap[6] = {0, 1, 3, 2, 4, 5};

	// Check if we've got a MiND extension
	int extPos = 0;
	nifti1_extension * ext = this->findExtension(NIFTI_ECODE_MIND_IDENT, extPos);
	int firstExtPos = extPos;

	// If not, no big deal, we just use the default index map
	if (ext)
	{
		// If we do have a MiND extension, check if 1) it's identifier is "DTENSOR",
		// and 2) if we've got at least six extensions after the current one.

		if (this->compareMiNDID(ext->edata, (char*) "DTENSOR", ext->esize - 8) && (extPos + 6) < this->NiftiImage->num_ext)
		{
			// If so, loop through the next six extensions
			for (int i = 0; i < 6; ++i)
			{
				ext = &(this->NiftiImage->ext_list[++extPos]);

				// Check if the extension code is correct
				if (ext->ecode != NIFTI_ECODE_DT_COMPONENT)
					continue;

				// The two integer values of the i-th extension after the "DTENSOR"
				// extension determine the 2D tensor position of the i-th vector element.
				// This position starts at one (i.e., D11 is the top-left element
				// of a tensor, NOT D00).

				int * indices = (int *) ext->edata;

				// Swap the bytes of the extension data
				if (this->NiftiImage->byteorder == 2)
				{
					nifti_swap_4bytes(1, &(indices[0]));
					nifti_swap_4bytes(1, &(indices[1]));
				}

				if (indices[0] == 1 && indices[1] == 1)		indexMap[0] = extPos - firstExtPos - 1;
				if (indices[0] == 1 && indices[1] == 2)		indexMap[1] = extPos - firstExtPos - 1;
				if (indices[0] == 2 && indices[1] == 1)		indexMap[1] = extPos - firstExtPos - 1;
				if (indices[0] == 1 && indices[1] == 3)		indexMap[2] = extPos - firstExtPos - 1;
				if (indices[0] == 3 && indices[1] == 1)		indexMap[2] = extPos - firstExtPos - 1;
				if (indices[0] == 2 && indices[1] == 2)		indexMap[3] = extPos - firstExtPos - 1;
				if (indices[0] == 2 && indices[1] == 3)		indexMap[4] = extPos - firstExtPos - 1;
				if (indices[0] == 3 && indices[1] == 2)		indexMap[4] = extPos - firstExtPos - 1;
				if (indices[0] == 3 && indices[1] == 3)		indexMap[5] = extPos - firstExtPos - 1;
			}
		}
	}

	// Create an output array for the six unique tensor elements
	int arraySize = this->NiftiImage->dim[1] * this->NiftiImage->dim[2] * this->NiftiImage->dim[3];
	double * outDoubleArray = new double[arraySize * 6];

	// Copy the input array to the output array, using the specified index mapping
	switch (this->NiftiImage->datatype)
	{
		case DT_UNSIGNED_CHAR:	createDoubleMappedArrayMacro(unsigned char,  6);		break;
		case DT_SIGNED_SHORT:	createDoubleMappedArrayMacro(unsigned short, 6);		break;
		case DT_UINT16:			createDoubleMappedArrayMacro(unsigned short, 6);		break;
		case DT_SIGNED_INT:		createDoubleMappedArrayMacro(int,            6);		break;
		case DT_FLOAT:			createDoubleMappedArrayMacro(float,			 6);		break;
		case DT_DOUBLE:			createDoubleMappedArrayMacro(double,		 6);		break;

		default:
			delete [] outDoubleArray;
			return NULL;
	}

	// Create an image using the new data array
	return this->createimageData(outDoubleArray, 6, "Tensors");
}


//----------------------[ parseDiscreteSphereVolume ]----------------------\\

vtkImageData * bmiaNiftiReader::parseDiscreteSphereVolume()
{
	// Check if we've got a MiND extension
	int extPos = 0;
	nifti1_extension * ext = this->findExtension(NIFTI_ECODE_MIND_IDENT, extPos);

	// The MiND extension is not optional in this case
	if (!ext)
		return NULL;

	// The MiND extension should have the "DISCSPHFUNC" code
	if (!(this->compareMiNDID(ext->edata, (char*) "DISCSPHFUNC", ext->esize - 8)))
		return NULL;

	// Create an array for the spherical angles
	vtkDoubleArray * angleArray = vtkDoubleArray::New();
	angleArray->SetNumberOfComponents(2);
	angleArray->SetName("Spherical Directions");
	
	// Loop through the remainder of the extensions
	while (extPos < this->NiftiImage->num_ext)
	{
		ext = &(this->NiftiImage->ext_list[++extPos]);

		// Break if an extension with an incorrect code is found
		if (ext->ecode != NIFTI_ECODE_SPHERICAL_DIRECTION)
			break;

		// Add the two angles to the array
		float * angles = (float *) ext->edata;

		// Swap the bytes of the input
		if (this->NiftiImage->byteorder == 2)
		{
			nifti_swap_4bytes(1, &(angles[0]));
			nifti_swap_4bytes(1, &(angles[1]));
		}

		angleArray->InsertNextTuple2(angles[0], angles[1]);
	}

	// The number of angle sets should match the vector length of the NIfTI image
	if (angleArray->GetNumberOfTuples() != this->NiftiImage->dim[5])
	{
		angleArray->Delete();
		return NULL;
	}

	// Create a new double array with the correct vector length
	int arraySize = this->NiftiImage->dim[1] * this->NiftiImage->dim[2] * this->NiftiImage->dim[3];
	double * outDoubleArray = new double[arraySize * this->NiftiImage->dim[5]];

	// Copy the input array to the output
	switch (this->NiftiImage->datatype)
	{
		case DT_UNSIGNED_CHAR:	createDoubleVectorArrayMacro(unsigned char,  this->NiftiImage->dim[5]);		break;
		case DT_SIGNED_SHORT:	createDoubleVectorArrayMacro(unsigned short, this->NiftiImage->dim[5]);		break;
		case DT_UINT16:			createDoubleVectorArrayMacro(unsigned short, this->NiftiImage->dim[5]);		break;
		case DT_SIGNED_INT:		createDoubleVectorArrayMacro(int,            this->NiftiImage->dim[5]);		break;
		case DT_FLOAT:			createDoubleVectorArrayMacro(float,		     this->NiftiImage->dim[5]);		break;
		case DT_DOUBLE:			createDoubleVectorArrayMacro(double,		 this->NiftiImage->dim[5]);		break;

		default:
			delete [] outDoubleArray;
			angleArray->Delete();
			return NULL;
	}

	// Append "_topo" to the filename
	QString geoFileName = this->filenameQ.insert(this->filenameQ.lastIndexOf("."), "_topo");

	vtkIntArray * triangleArray = NULL;
	bmiaNiftiReader * geoReader = new bmiaNiftiReader(this->userOut);
	
	// Check if the topology file exists
	if (geoReader->CanReadFile(geoFileName.toLatin1().data()))
	{
		// Try to read the topology file
		QString err = geoReader->readNIfTIFile(geoFileName.toLatin1().data(), false);

		if (err.isEmpty() && geoReader->outData.size() > 0 && geoReader->imageDataType == NDT_Triangles)
		{
			triangleArray = vtkIntArray::SafeDownCast(geoReader->outData.at(0));
		}
	}

	// If the triangle array could not be constructed from the topology file (either
	// because this file does not exist, or because there was an error while reading),
	// we use out own sphere triangulator to compute this array.

	if (triangleArray == NULL)
	{
		triangleArray = vtkIntArray::New();
		triangleArray->SetName("Triangles");
		SphereTriangulator * triangulator = new SphereTriangulator;
		triangulator->triangulateFromAnglesArray(angleArray, triangleArray);
		delete triangulator;
	}

	// Create an image for the sphere radii
	vtkImageData * discreteSphereImage = this->createimageData(outDoubleArray, this->NiftiImage->dim[5], "Vectors");

	// Add the angles array to this image
	discreteSphereImage->GetPointData()->AddArray(angleArray);
	angleArray->Delete();

	// If available, add the triangle array
	if (triangleArray)
	{
		discreteSphereImage->GetPointData()->AddArray(triangleArray);
	}

	// Delete the triangles reader
	delete geoReader;

	// Done!
	return discreteSphereImage;
}


//--------------------[ parseSphericalHarmonicsVolume ]--------------------\\

vtkImageData * bmiaNiftiReader::parseSphericalHarmonicsVolume()
{
	// Check if we've got a MiND extension
	int extPos = 0;
	nifti1_extension * ext = this->findExtension(NIFTI_ECODE_MIND_IDENT, extPos);

	// The MiND extension is not optional in this case
	if (!ext)
		return NULL;

	// The MiND extension should have the "REALSPHARMCOEFFS" code
	if (!(this->compareMiNDID(ext->edata, (char*) "REALSPHARMCOEFFS", ext->esize - 8)))
		return NULL;

	// Get the number of components
	int numberOfComponents = this->NiftiImage->num_ext - extPos - 1;

	// Number of components should match for 0, 2, 4, 6, or 8-th order SH
	if (numberOfComponents !=  1 && numberOfComponents !=  6 && numberOfComponents != 15 &&
		numberOfComponents != 28 && numberOfComponents != 45)
		return NULL;

	// Number of components should match the vector length
	if (numberOfComponents != this->NiftiImage->dim[5])
		return NULL;

	// Allocate the index map
	int * indexMap = new int[numberOfComponents];

	// Index for the index map
	int currentIndex = 0;

	// Loop through the remainder of the extensions
	while (extPos < this->NiftiImage->num_ext - 1)
	{
		ext = &(this->NiftiImage->ext_list[++extPos]);

		// Break if an extension with an incorrect code is found
		if (ext->ecode != NIFTI_ECODE_SHC_DEGREEORDER)
		{
			delete[] indexMap;
			return NULL;
		}

		// Add the two angles to the array
		int * order = (int *) ext->edata;

		// Swap the bytes of the input
		if (this->NiftiImage->byteorder == 2)
		{
			nifti_swap_4bytes(1, &(order[0]));
			nifti_swap_4bytes(1, &(order[1]));
		}

		int targetIndex = 0;

		// Compute starting index for this value of "l" (order[0])
		for (int i = 0; i < order[0]; i += 2)		
			targetIndex += (i * 2) + 1;

		// Convert "m" (order[1]), which is in the range "-l" to "l", to the range
		// 0 to "2 * l + 1", by adding "l", and add this to the starting index.

		targetIndex += order[1] + order[0];

		// Double-check that the index is in the right range
		if (targetIndex < 0 || targetIndex >= numberOfComponents)
		{
			delete[] indexMap;
			return NULL;
		}

		// Set the index
		indexMap[currentIndex++] = targetIndex;
	}

	// Create a new double array with the correct vector length
	int arraySize = this->NiftiImage->dim[1] * this->NiftiImage->dim[2] * this->NiftiImage->dim[3];
	double * outDoubleArray = new double[arraySize * this->NiftiImage->dim[5]];

	// Copy the input array to the output
	switch (this->NiftiImage->datatype)
	{
		case DT_UNSIGNED_CHAR:	createDoubleMappedArrayMacro(unsigned char,  this->NiftiImage->dim[5]);		break;
		case DT_SIGNED_SHORT:	createDoubleMappedArrayMacro(unsigned short, this->NiftiImage->dim[5]);		break;
		case DT_UINT16:			createDoubleMappedArrayMacro(unsigned short, this->NiftiImage->dim[5]);		break;
		case DT_SIGNED_INT:		createDoubleMappedArrayMacro(int,            this->NiftiImage->dim[5]);		break;
		case DT_FLOAT:			createDoubleMappedArrayMacro(float,		     this->NiftiImage->dim[5]);		break;
		case DT_DOUBLE:			createDoubleMappedArrayMacro(double,		 this->NiftiImage->dim[5]);		break;

	default:
		delete [] outDoubleArray;
		delete [] indexMap;
		return NULL;
	}

	// Create an image for the sphere radii
	vtkImageData * shImage = this->createimageData(outDoubleArray, this->NiftiImage->dim[5], "Scalars");

	// Done!
	return shImage;
}


//----------------------------[ parseTriangles ]---------------------------\\

vtkIntArray * bmiaNiftiReader::parseTriangles()
{
	// We should have three elements per vector
	if (this->NiftiImage->dim[5] != 3)
		return NULL;

	// Create an array for the point indices of the triangles (i.e., three indices
	// per triangle, for a total of "dim[1]" triangles.

	int arraySize = this->NiftiImage->dim[1];
	int * outIntArray = new int[arraySize * 3];

	// Copy the input array to the output array

	switch (this->NiftiImage->datatype)
	{
		case DT_UNSIGNED_CHAR:	createIntVectorArrayMacro(unsigned char,  3);		break;
		case DT_SIGNED_SHORT:	createIntVectorArrayMacro(unsigned short, 3);		break;
		case DT_UINT16:			createIntVectorArrayMacro(unsigned short, 3);		break;
		case DT_SIGNED_INT:		createIntVectorArrayMacro(int,			  3);		break;
		case DT_FLOAT:			createIntVectorArrayMacro(float,		  3);		break;
		case DT_DOUBLE:			createIntVectorArrayMacro(double,		  3);		break;

		default:
			delete [] outIntArray;
			return NULL;
	}

	// Create a VTK array for these indices
	vtkIntArray * trianglesArray = vtkIntArray::New();
	trianglesArray->SetNumberOfComponents(3);
	trianglesArray->SetNumberOfTuples(this->NiftiImage->dim[1]);
	trianglesArray->SetArray(outIntArray, arraySize, 1);
	trianglesArray->SetName("Triangles");

	return trianglesArray;
}


//---------------------------[ createimageData ]---------------------------\\

vtkImageData * bmiaNiftiReader::createimageData(double * data, int numberOfComponents, const char * arrayName)
{
	// Create a new volume
	vtkImageData * newVolume = vtkImageData::New();
	newVolume->SetNumberOfScalarComponents(numberOfComponents);
	newVolume->SetScalarTypeToDouble();

	// Spacing is set to 1.0 isotropically for NIfTI images. This is done because
	// in most cases, the spacing is also part of the transformation matrix, and
	// if we were to use the provided spacing values, this spacing would
	// essentially be applied twice. A more elegant solution to this problem
	// should be created in the future.

	newVolume->SetSpacing(1.0, 1.0, 1.0);

	newVolume->SetExtent(	0, this->NiftiImage->dim[1] - 1, 
							0, this->NiftiImage->dim[2] - 1, 
							0, this->NiftiImage->dim[3] - 1);
	newVolume->GetDimensions();

	vtkPointData * newPD = newVolume->GetPointData();

	// Setup the scalar array. We use doubles by default.
	vtkDoubleArray * newScalars = vtkDoubleArray::New();
	newScalars->SetNumberOfComponents(numberOfComponents);
	newScalars->SetNumberOfTuples(this->NiftiImage->dim[1] * this->NiftiImage->dim[2] * this->NiftiImage->dim[3]);
	newScalars->SetArray(data, newScalars->GetNumberOfTuples() * numberOfComponents, 1);
	newScalars->SetName(arrayName);

	newPD->SetScalars(newScalars);

	return newVolume;
}


} // namespace bmia
