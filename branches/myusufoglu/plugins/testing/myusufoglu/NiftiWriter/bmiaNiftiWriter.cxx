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
* bmiaNiftiWriter.cxx
*
* * 2013-03-16   Mehmet Yusufoglu
* - Create the class. Writes the scalar data in Nifti format.
* - 
*/


/** Includes */

#include "bmiaNiftiWriter.h"


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

	bmiaNiftiWriter::bmiaNiftiWriter(UserOutput * rUserOut)
	{
		// Initialize pointers to NULL
		this->NiftiImage		= NULL;
		this->transformMatrix	= NULL;
		this->progress			= NULL;
		this->userOut			= rUserOut;
	}


	//------------------------------[ Destructor ]-----------------------------\\

	bmiaNiftiWriter::~bmiaNiftiWriter()
	{
		// Use the "cleanUp" function. This will 'delete' the VTK data objects generated
		// by this reader, but if the reader was successful, all these data objects 
		// will have been registered to new data sets, which means that they will not 
		// really be deleted in the "cleanUp" function.

		this->cleanUp(); 
	}


	//-------------------------------[ cleanUp ]-------------------------------\\

	void bmiaNiftiWriter::cleanUp()
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

	int bmiaNiftiWriter::CanReadFile(const char * filename)
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






	//--------------------------[ determineDataType ]--------------------------\\

	bool bmiaNiftiWriter::determineDataType()
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

	bool bmiaNiftiWriter::compareMiNDID(char * id, char * target, int n)
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

	nifti1_extension * bmiaNiftiWriter::findExtension(int targetID, int & extPos)
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

	void bmiaNiftiWriter::writeScalarVolume(vtkImageData *image, QString saveFileName, vtkObject * transform)
	{


		nifti_image * m_NiftiImage = new nifti_image;
		m_NiftiImage = nifti_simple_init_nim();
		//print for debug
		image->Print(cout);
		double dataTypeSize = 1.0;
		int dim[3];
		int wholeExtent[6];
		double spacing[3];
		double origin[3];
		image->Update();
		int numComponents = image->GetNumberOfScalarComponents();
		int imageDataType = image->GetScalarType();

		image->GetOrigin(origin);
		image->GetSpacing(spacing);
		image->GetDimensions(dim);
		image->GetWholeExtent(wholeExtent);
		m_NiftiImage->dt = 0;

		m_NiftiImage->ndim = 3;
		m_NiftiImage->dim[1] = wholeExtent[1] + 1;
		m_NiftiImage->dim[2] = wholeExtent[3] + 1;
		m_NiftiImage->dim[3] = wholeExtent[5] + 1;
		m_NiftiImage->dim[4] = 1;
		m_NiftiImage->dim[5] = 1;
		m_NiftiImage->dim[6] = 1;
		m_NiftiImage->dim[7] = 1;
		m_NiftiImage->nx =  m_NiftiImage->dim[1];
		m_NiftiImage->ny =  m_NiftiImage->dim[2];
		m_NiftiImage->nz =  m_NiftiImage->dim[3];
		m_NiftiImage->nt =  m_NiftiImage->dim[4];
		m_NiftiImage->nu =  m_NiftiImage->dim[5];
		m_NiftiImage->nv =  m_NiftiImage->dim[6];
		m_NiftiImage->nw =  m_NiftiImage->dim[7];

		//nhdr.pixdim[0] = 0.0 ;
		m_NiftiImage->pixdim[1] = spacing[0];
		m_NiftiImage->pixdim[2] = spacing[1];
		m_NiftiImage->pixdim[3] = spacing[2];
		m_NiftiImage->pixdim[4] = 0;
		m_NiftiImage->pixdim[5] = 1;
		m_NiftiImage->pixdim[6] = 1;
		m_NiftiImage->pixdim[7] = 1;
		m_NiftiImage->dx = m_NiftiImage->pixdim[1];
		m_NiftiImage->dy = m_NiftiImage->pixdim[2];
		m_NiftiImage->dz = m_NiftiImage->pixdim[3];
		m_NiftiImage->dt = m_NiftiImage->pixdim[4];
		m_NiftiImage->du = m_NiftiImage->pixdim[5];
		m_NiftiImage->dv = m_NiftiImage->pixdim[6];
		m_NiftiImage->dw = m_NiftiImage->pixdim[7];

		int numberOfVoxels = m_NiftiImage->nx;

		if(m_NiftiImage->ny>0){
			numberOfVoxels*=m_NiftiImage->ny;
		}
		if(m_NiftiImage->nz>0){
			numberOfVoxels*=m_NiftiImage->nz;
		}
		if(m_NiftiImage->nt>0){
			numberOfVoxels*=m_NiftiImage->nt;
		}
		if(m_NiftiImage->nu>0){
			numberOfVoxels*=m_NiftiImage->nu;
		}
		if(m_NiftiImage->nv>0){
			numberOfVoxels*=m_NiftiImage->nv;
		}
		if(m_NiftiImage->nw>0){
			numberOfVoxels*=m_NiftiImage->nw;
		}

		m_NiftiImage->nvox = numberOfVoxels;

		if(numComponents==1 || numComponents==6 ){
			switch(imageDataType)
			{
			case VTK_BIT://DT_BINARY:
				m_NiftiImage->datatype = DT_BINARY;
				m_NiftiImage->nbyper = 0;
				dataTypeSize = 0.125;
				break;
			case VTK_UNSIGNED_CHAR://DT_UNSIGNED_CHAR:
				m_NiftiImage->datatype = DT_UNSIGNED_CHAR;
				m_NiftiImage->nbyper = 1;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_SIGNED_CHAR://DT_INT8:
				m_NiftiImage->datatype = DT_INT8;
				m_NiftiImage->nbyper = 1;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_SHORT://DT_SIGNED_SHORT:
				m_NiftiImage->datatype = DT_SIGNED_SHORT;
				m_NiftiImage->nbyper = 2;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_SHORT://DT_UINT16:
				m_NiftiImage->datatype = DT_UINT16;
				m_NiftiImage->nbyper = 2;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_INT://DT_SIGNED_INT:
				m_NiftiImage->datatype = DT_SIGNED_INT;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_INT://DT_UINT32:
				m_NiftiImage->datatype = DT_UINT32;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_FLOAT://DT_FLOAT:
				m_NiftiImage->datatype = DT_FLOAT;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_DOUBLE://DT_DOUBLE:
				m_NiftiImage->datatype = DT_DOUBLE;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_LONG://DT_INT64:
				m_NiftiImage->datatype = DT_INT64;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_LONG://DT_UINT64:
				m_NiftiImage->datatype = DT_UINT64;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			default:
				cout << "cannot handle this type" << endl ;
				break;
			}
		}
		// m_NiftiImage->data = image->GetPointData( // scalar pointer i ekle buraya !!!! yer ac? 
		m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;
		m_NiftiImage->data=const_cast<void *>( image->GetScalarPointer());

		m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
		m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed
		// qform code


		//Transformation and quaternion; quaternion has no scaling but viste uses spacing=1 and scaling parameter of the user transform of the actor is changed with spacing because of the transform.
		if(transform)
		{
			cout << "transform"  << endl;
			vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
			matrix = vtkMatrix4x4::SafeDownCast(transform);
			if(matrix)
			{
				// sform matrix or qform quaternion, which one will be used. Both can also be used if bothcodes are > 0.
				m_NiftiImage->qform_code = 0; // Decided to use only sform code. If this is set > 0 then qform quaternion or sform matrix is used.
				m_NiftiImage->sform_code = 2; // sform matrix is used only if sform_code > 0.


				matrix->Print(cout);
				cout << "transform 1.1"  << endl;
				mat44 matrixf;
				for(int i=0;i<4;i++)
					for(int j=0;j<4;j++)
					{
						if(m_NiftiImage->qform_code > 0)
							matrixf.m[i][j] = matrix->GetElement(i,j);
						cout <<  matrixf.m[i][j] << endl;
						// sform code
						if(m_NiftiImage->sform_code >0 )
							m_NiftiImage->sto_xyz.m[i][j]= matrix->GetElement(i,j);
					}

					// convert transformation matrix to quaternion
					nifti_mat44_to_quatern(matrixf, &( m_NiftiImage->quatern_b), &( m_NiftiImage->quatern_c), &( m_NiftiImage->quatern_d), 
						&( m_NiftiImage->qoffset_x), &(m_NiftiImage->qoffset_y), &(m_NiftiImage->qoffset_z), &(m_NiftiImage->dx) , &(m_NiftiImage->dy) ,&(m_NiftiImage->dz) , &(m_NiftiImage->qfac));


					//cout << m_NiftiImage->quatern_b << " " << m_NiftiImage->quatern_c << " " << m_NiftiImage->quatern_d << " " << m_NiftiImage->qfac << " " << endl;
					//cout << m_NiftiImage->qoffset_x << " " << m_NiftiImage->qoffset_y << " " << m_NiftiImage->qoffset_z <<endl;

					// in case the matrix is not pure transform, quaternion can not include scaling part. Therefore if the matris is not a pure transform matrix use scaling factor in spacing?
					float scaling[3];
					if(matrix->Determinant() != 1 && (m_NiftiImage->qform_code > 0) )
					{
						// If determinant is not 1 find scaling
						vtkTransform *transform = vtkTransform::New();
						transform->SetMatrix(matrix);
						transform->Scale(scaling);

						m_NiftiImage->pixdim[1] = spacing[0]*scaling[0];
						m_NiftiImage->pixdim[2] = spacing[1]*scaling[1];
						m_NiftiImage->pixdim[3] = spacing[2]*scaling[2];
						transform->Delete();
					}
			}
			else {
				cout << "Invalid   matrix \n";
			}
		}
		else
		{
			cout << "Invalid transformation object \n";
		}



		nifti_set_iname_offset(m_NiftiImage);
		// Write the image file
		nifti_image_write( m_NiftiImage );

	}


	// ----------------------------[ writeDTIVolume ]--------------------------- \\
	//Any datastructure with extention
	void bmiaNiftiWriter::writeMindData(vtkImageData *image, QString saveFileName, vtkObject * transform, QString dataStructure)

	{
		//dim[5]= 6 
		//there must be 6 extentions
		cout << "writeMindData" << endl;
		nifti_image * m_NiftiImage = new nifti_image;
		m_NiftiImage = nifti_simple_init_nim();




		image->Print(cout);
		cout << "writeMindData 1.2" << endl;
		double dataTypeSize = 1.0;
		int dim[3];
		int wholeExtent[6];
		double spacing[3];
		double origin[3];
		image->Update();
		int numComponents = image->GetNumberOfScalarComponents();
		int imageDataType = image->GetScalarType();

		image->GetOrigin(origin); 
		image->GetSpacing(spacing);
		image->GetDimensions(dim);
		image->GetWholeExtent(wholeExtent);

		cout << "writeMindData 1.3 image scalar datatype" <<  imageDataType << "pointdata type:" << image->GetPointData()->GetScalars()->GetDataType() << endl;

		m_NiftiImage->byteorder		= nifti_short_order();
		m_NiftiImage->ndim = 5;
		m_NiftiImage->dim[0] = 5;
		m_NiftiImage->dim[1] = wholeExtent[1] + 1;
		m_NiftiImage->dim[2] = wholeExtent[3] + 1;
		m_NiftiImage->dim[3] = wholeExtent[5] + 1;
		m_NiftiImage->dim[4] = 1;
		m_NiftiImage->dim[5] = 1; // Each data sets again below
		m_NiftiImage->dim[6] = 0;
		m_NiftiImage->dim[7] = 0;
		m_NiftiImage->nx =  m_NiftiImage->dim[1];
		m_NiftiImage->ny =  m_NiftiImage->dim[2];
		m_NiftiImage->nz =  m_NiftiImage->dim[3];
		m_NiftiImage->nt =  m_NiftiImage->dim[4];
		m_NiftiImage->nu =  m_NiftiImage->dim[5]; // Each data sets again below
		m_NiftiImage->nv =  m_NiftiImage->dim[6];
		m_NiftiImage->nw =  m_NiftiImage->dim[7];

		m_NiftiImage->pixdim[0] = 0.0 ;
		m_NiftiImage->pixdim[1] = spacing[0];
		m_NiftiImage->pixdim[2] = spacing[1];
		m_NiftiImage->pixdim[3] = spacing[2];
		m_NiftiImage->pixdim[4] = 0;
		m_NiftiImage->pixdim[5] = 0;
		m_NiftiImage->pixdim[6] = 0;
		m_NiftiImage->pixdim[7] = 0;
		m_NiftiImage->dx = m_NiftiImage->pixdim[1];
		m_NiftiImage->dy = m_NiftiImage->pixdim[2];
		m_NiftiImage->dz = m_NiftiImage->pixdim[3];
		m_NiftiImage->dt = m_NiftiImage->pixdim[4];
		m_NiftiImage->du = m_NiftiImage->pixdim[5];
		m_NiftiImage->dv = m_NiftiImage->pixdim[6];
		m_NiftiImage->dw = m_NiftiImage->pixdim[7];
		cout << "writeMindData 1.4" << endl;
		int numberOfVoxels = m_NiftiImage->nx;

		if(m_NiftiImage->ny>0){
			numberOfVoxels*=m_NiftiImage->ny;
		}
		if(m_NiftiImage->nz>0){
			numberOfVoxels*=m_NiftiImage->nz;
		}
		if(m_NiftiImage->nt>0){
			numberOfVoxels*=m_NiftiImage->nt;
		}
		if(m_NiftiImage->nu>0){
			numberOfVoxels*=m_NiftiImage->nu;
		}


		m_NiftiImage->nvox = numberOfVoxels; // Each data sets again below

		if(numComponents!=0 ){
			cout << "writeMindData 1.3 coponenets:" << numComponents<< " voxels: " <<  numberOfVoxels << "imagedatatype"<<  imageDataType << endl;
			switch(imageDataType)
			{
			case VTK_BIT://DT_BINARY:
				m_NiftiImage->datatype = DT_BINARY;
				m_NiftiImage->nbyper = 0;
				dataTypeSize = 0.125;
				break;
			case VTK_UNSIGNED_CHAR://DT_UNSIGNED_CHAR:
				m_NiftiImage->datatype = DT_UNSIGNED_CHAR;
				m_NiftiImage->nbyper = 1;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_SIGNED_CHAR://DT_INT8:
				m_NiftiImage->datatype = DT_INT8;
				m_NiftiImage->nbyper = 1;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_SHORT://DT_SIGNED_SHORT:
				m_NiftiImage->datatype = DT_SIGNED_SHORT;
				m_NiftiImage->nbyper = 2;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_SHORT://DT_UINT16:
				m_NiftiImage->datatype = DT_UINT16;
				m_NiftiImage->nbyper = 2;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_INT://DT_SIGNED_INT:
				m_NiftiImage->datatype = DT_SIGNED_INT;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_INT://DT_UINT32:
				m_NiftiImage->datatype = DT_UINT32;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_FLOAT://DT_FLOAT:
				m_NiftiImage->datatype = DT_FLOAT;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_DOUBLE://DT_DOUBLE:
				m_NiftiImage->datatype = DT_DOUBLE;  
				m_NiftiImage->nbyper = 8; 
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_LONG://DT_INT64:
				m_NiftiImage->datatype = DT_INT64;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_LONG://DT_UINT64:
				m_NiftiImage->datatype = DT_UINT64;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			default:
				cout << "cannot handle this type" << endl ;
				break;
			}
		}

		m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;//??
		cout << " writeMindData 1.7 "<< endl;
		m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
		cout << " 1.71 "<< endl;
		m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed


		cout << " writeMindData 1.8 "<< endl;



		// Using the "Vector" intent code
		m_NiftiImage->intent_code = NIFTI_INTENT_VECTOR;
		m_NiftiImage->intent_p1   = 0.0f;
		m_NiftiImage->intent_p2   = 0.0f;
		m_NiftiImage->intent_p3   = 0.0f;
		// The kernel NIfTI image does not have a transformation matrix
		m_NiftiImage->qform_code		= 0;
		m_NiftiImage->quatern_b			= 0.0f;
		m_NiftiImage->quatern_c			= 0.0f;
		m_NiftiImage->quatern_d			= 0.0f;
		m_NiftiImage->qoffset_x			= 0.0f;
		m_NiftiImage->qoffset_y			= 0.0f;
		m_NiftiImage->qoffset_z			= 0.0f;
		m_NiftiImage->qfac				= 0.0f;




		if(transform)
		{
			cout << "transform"  << endl;
			vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
			matrix = vtkMatrix4x4::SafeDownCast(transform);
			if(matrix)
			{
				// sform matrix or qform quaternion, which one will be used. Both can also be used if bothcodes are > 0.
				m_NiftiImage->qform_code = 0; // Decided to use only sform code. If this is set > 0 then qform quaternion or sform matrix is used.
				m_NiftiImage->sform_code = 1; // sform matrix is used only if sform_code > 0.


				matrix->Print(cout);
				cout << "transform 1.1"  << endl;
				mat44 matrixf;
				for(int i=0;i<4;i++)
					for(int j=0;j<4;j++)
					{
						if(m_NiftiImage->qform_code > 0)
						{
							matrixf.m[i][j] = matrix->GetElement(i,j);
							cout <<  matrixf.m[i][j] << endl;
						}
						// sform code
						if(m_NiftiImage->sform_code >0 )
							m_NiftiImage->sto_xyz.m[i][j]= matrix->GetElement(i,j);
					}

					// convert transformation matrix to quaternion
					if(m_NiftiImage->qform_code > 0)
						nifti_mat44_to_quatern(matrixf, &( m_NiftiImage->quatern_b), &( m_NiftiImage->quatern_c), &( m_NiftiImage->quatern_d), 
						&( m_NiftiImage->qoffset_x), &(m_NiftiImage->qoffset_y), &(m_NiftiImage->qoffset_z), &(m_NiftiImage->dx) , &(m_NiftiImage->dy) ,&(m_NiftiImage->dz) , &(m_NiftiImage->qfac));


					//cout << m_NiftiImage->quatern_b << " " << m_NiftiImage->quatern_c << " " << m_NiftiImage->quatern_d << " " << m_NiftiImage->qfac << " " << endl;
					//cout << m_NiftiImage->qoffset_x << " " << m_NiftiImage->qoffset_y << " " << m_NiftiImage->qoffset_z <<endl;

					// in case the matrix is not pure transform, quaternion can not include scaling part. Therefore if the matris is not a pure transform matrix use scaling factor in spacing?
					float scaling[3];
					if(matrix->Determinant() != 1 && (m_NiftiImage->qform_code > 0) )
					{
						// If determinant is not 1 find scaling
						vtkTransform *transform = vtkTransform::New();
						transform->SetMatrix(matrix);
						transform->Scale(scaling);

						m_NiftiImage->pixdim[1] = spacing[0]*scaling[0];
						m_NiftiImage->pixdim[2] = spacing[1]*scaling[1];
						m_NiftiImage->pixdim[3] = spacing[2]*scaling[2];
						transform->Delete();
					}
			}
			else {
				cout << "Invalid   matrix \n";
			}
		}
		else
		{
			cout << "Invalid transformation object \n";
		}
		// Set the intent name to "MiND"


		// Set the intent name to "MiND"
		char * intentName = (char*) "MiND";
#ifdef _WIN32
		strcpy_s(&(m_NiftiImage->intent_name[0]), 5, intentName);
#else
		strcpy(&(m_NiftiImage->intent_name[0]), intentName);
#endif
		// Initialize the extension list
		m_NiftiImage->num_ext  = 0;
		m_NiftiImage->ext_list = NULL;


		cout << " writeMindData 1.9 "<< endl;


		// Add the main LONI MiND extensions one descriptor N data-related
		// DTI 
		if(dataStructure.contains("DTI")) 
		{ 

			cout << "save dti"<< endl;
			m_NiftiImage->dim[5] = 6; // 6 components of thematrix
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nvox = m_NiftiImage->nx * m_NiftiImage->ny * m_NiftiImage->nz * m_NiftiImage->nt * m_NiftiImage->nu;



			nifti_add_extension(m_NiftiImage, "DTENSOR", 8, NIFTI_ECODE_MIND_IDENT);
			cout << " writeMindData DTI 2.0 "<< endl;
			for(int i=1;  i<=3; i++)
				for(int j=1;j<=3; j++)
					if(j<=i) {
						int index[2];
						index[0]=i;
						index[1]=j; 
						cout << i << "," << j << endl;
						nifti_add_extension(m_NiftiImage, (char *) &(index[0]), 2 * sizeof(int), NIFTI_ECODE_DT_COMPONENT);	 
					}

					vtkDataArray * inTensors = image->GetPointData()->GetArray("Tensors");

					if (!inTensors)
					{
						cout <<"Input data has no tensors!";
						return;
					}

					int indexMap[6] = {0, 1, 3, 2, 4, 5};
					double *outDoubleArray = static_cast<double*>(image->GetPointData()->GetArray("Tensors")->GetVoidPointer(0));

					int arraySize = image->GetPointData()->GetArray("Tensors")->GetNumberOfTuples();
					int comp = image->GetPointData()->GetArray("Tensors")->GetNumberOfComponents();
					cout << arraySize << " " << comp << endl;
					image->GetPointData()->GetArray("Tensors")->Print(cout);
					cout << "size of outDoubleArray:" << sizeof(outDoubleArray) << endl; 
					cout << "transform 1.91"  << endl;
					double * niftiImageData =  new double[arraySize*comp];
					//this->NiftiImage->data = (void *) new double[arraySize*comp+arraySize];
					cout << "transform 1.92"  << endl;
					for (int i = 0; i < arraySize; ++i) 
						for (int j = 0; j < comp; ++j)
						{

							niftiImageData[i + indexMap[j] * arraySize]  = (double) outDoubleArray[j + comp * i];	
						}
						cout << "transform 1.93"  << endl;
						m_NiftiImage->data =  (void *) niftiImageData;

		}

		// Discrete Sphere
		else if(dataStructure.contains("discrete sp")) {
			cout << "Dicrete Sphere"<< endl;
			// overwrite 
			m_NiftiImage->dim[5] = image->GetNumberOfScalarComponents(); //  *2 ??? No data isone scalar for each index set.
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nvox = m_NiftiImage->nx * m_NiftiImage->ny * m_NiftiImage->nz * m_NiftiImage->nt * m_NiftiImage->nu;
			m_NiftiImage->swapsize		= 8;					// ...and the swap size is also 8.
			m_NiftiImage->iname_offset	= 1024;					// Offset for the image name
			image->Print(cout);

			char buffer[24];
			memset(buffer, 0, sizeof(buffer));
			strncpy(buffer, "DISCSPHFUNC", sizeof(buffer));

			nifti_add_extension(m_NiftiImage, buffer, 24, NIFTI_ECODE_MIND_IDENT); // 24 is length of array which inc. DISCSPHFUNC
			vtkDataArray *directionsDoubleArray = image->GetPointData()->GetArray("Spherical Directions") ;
			int numberOfDirections = image->GetPointData()->GetArray("Spherical Directions")->GetNumberOfTuples();
			for(int j=0;j<numberOfDirections; j++)
			{
				float indx[2]; 
				indx[0]= (float) directionsDoubleArray->GetTuple2(j)[0];
				indx[1]= (float) directionsDoubleArray->GetTuple2(j)[1];
				cout << j << endl;
				nifti_add_extension(m_NiftiImage, (char *) &(indx[0]), 2 * sizeof(float), NIFTI_ECODE_SPHERICAL_DIRECTION);	 
			}


			vtkDataArray * inVectors = image->GetPointData()->GetArray("Vectors");
			cout << "Vectors Tuples and Components:" << image->GetPointData()->GetArray("Vectors")->GetNumberOfComponents() << endl;
			cout << image->GetPointData()->GetArray("Vectors")->GetNumberOfTuples() << endl;
			image->GetPointData()->GetArray("Vectors")->Print(cout);
			if (!inVectors)
			{
				cout <<"Input data has no vectors!";
				return;
			}


			double *outDoubleArray = static_cast<double*>(image->GetScalarPointer() );

			int arraySize = image->GetNumberOfPoints();
			int comp = image->GetNumberOfScalarComponents();
			cout << arraySize << " " << comp << endl;



			double * niftiImageData =  new double[arraySize*comp];
			//this->NiftiImage->data = (void *) new double[arraySize*comp+arraySize];
			cout << "transform 1.92"  << endl;
			for (int i = 0; i < arraySize; ++i) 
				for (int j = 0; j < comp; ++j)
				{
					// change from row-major to column major
					niftiImageData[i+ arraySize * j]  = (double) outDoubleArray[j + comp * i];	

				}
				cout << "transform 1.93"  << endl;
				m_NiftiImage->data =  (void *) niftiImageData;

		}


		else if(dataStructure.contains("spherical harmonics")) {
			cout << "Spherical harmonics"<< endl;
			// overwrite 
			m_NiftiImage->dim[5] = image->GetNumberOfScalarComponents(); //  *2 ??? No data isone scalar for each index set.
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nvox = m_NiftiImage->nx * m_NiftiImage->ny * m_NiftiImage->nz * m_NiftiImage->nt * m_NiftiImage->nu;
			m_NiftiImage->swapsize		= 8;					// ...and the swap size is also 8.
			m_NiftiImage->iname_offset	= 1024;					// Offset for the image name
			image->Print(cout);

			char buffer[24];
			memset(buffer, 0, sizeof(buffer));
			strncpy(buffer, "REALSPHARMCOEFFS", sizeof(buffer));

			nifti_add_extension(m_NiftiImage, buffer, 24, NIFTI_ECODE_SHC_DEGREEORDER); // 24 is length of array which inc. DISCSPHFUNC
			vtkDataArray *directionsDoubleArray = image->GetPointData()->GetArray("Spherical Directions") ;
			int numberOfDirections = image->GetPointData()->GetArray("Spherical Directions")->GetNumberOfTuples();
			for(int j=0;j<numberOfDirections; j++)
			{
				float indx[2]; 
				indx[0]= (float) directionsDoubleArray->GetTuple2(j)[0];
				indx[1]= (float) directionsDoubleArray->GetTuple2(j)[1];
				cout << j << endl;
				nifti_add_extension(m_NiftiImage, (char *) &(indx[0]), 2 * sizeof(float), NIFTI_ECODE_SPHERICAL_DIRECTION);	 
			}


			vtkDataArray * inVectors = image->GetPointData()->GetArray("Vectors");
			cout << "Vectors Tuples and Components:" << image->GetPointData()->GetArray("Vectors")->GetNumberOfComponents() << endl;
			cout << image->GetPointData()->GetArray("Vectors")->GetNumberOfTuples() << endl;
			image->GetPointData()->GetArray("Vectors")->Print(cout);
			if (!inVectors)
			{
				cout <<"Input data has no vectors!";
				return;
			}


			double *outDoubleArray = static_cast<double*>(image->GetScalarPointer() );

			int arraySize = image->GetNumberOfPoints();
			int comp = image->GetNumberOfScalarComponents();
			cout << arraySize << " " << comp << endl;



			double * niftiImageData =  new double[arraySize*comp];
			//this->NiftiImage->data = (void *) new double[arraySize*comp+arraySize];
			cout << "transform 1.92"  << endl;
			for (int i = 0; i < arraySize; ++i) 
				for (int j = 0; j < comp; ++j)
				{
					// change from row-major to column major
					niftiImageData[i+ arraySize * j]  = (double) outDoubleArray[j + comp * i];	

				}
				cout << "transform 1.93"  << endl;
				m_NiftiImage->data =  (void *) niftiImageData;

		}


		else {
			qDebug() << "The data type is not suitable to Loni Mind Save Funtion"<< endl;
		}

		// Add the two angles to the NIfTI file as an extension
		nifti_set_iname_offset(m_NiftiImage);
		cout << " writeMindData 2.1 num extentions:"<< m_NiftiImage->num_ext <<  endl;
		nifti_image_write( m_NiftiImage );

	}

	// DTI without MIND extention method
	void bmiaNiftiWriter::writeDTIVolume(vtkImageData *image, QString saveFileName, vtkObject * transform)
	{
		//dim[5]= 6 
		//there must be 6 extentions
		cout << "writeDTIVolume" << endl;
		nifti_image * m_NiftiImage = new nifti_image;
		m_NiftiImage = nifti_simple_init_nim();




		image->Print(cout);
		cout << "writeDTIVolume 1.2" << endl;
		double dataTypeSize = 1.0;
		int dim[3];
		int wholeExtent[6];
		double spacing[3];
		double origin[3];
		image->Update();
		int numComponents = image->GetNumberOfScalarComponents();
		int imageDataType = image->GetScalarType();

		image->GetOrigin(origin);
		image->GetSpacing(spacing);
		image->GetDimensions(dim);  
		image->GetWholeExtent(wholeExtent);

		cout << "writeDTIVolume 1.3 image scalar datatype" <<  imageDataType << "pointdata type:" << image->GetPointData()->GetScalars()->GetDataType() << endl;
		//cin.get();
		//m_NiftiImage->swapsize  = 16; 
		//		= nifti_short_order();
		m_NiftiImage->ndim = 5;
		m_NiftiImage->dim[0] = 5;
		m_NiftiImage->dim[1] = wholeExtent[1] + 1;
		m_NiftiImage->dim[2] = wholeExtent[3] + 1;
		m_NiftiImage->dim[3] = wholeExtent[5] + 1;
		m_NiftiImage->dim[4] = 1;
		m_NiftiImage->dim[5] = 6; // 6 *2 ???
		m_NiftiImage->dim[6] = 0;
		m_NiftiImage->dim[7] = 0;
		m_NiftiImage->nx =  m_NiftiImage->dim[1];
		m_NiftiImage->ny =  m_NiftiImage->dim[2];
		m_NiftiImage->nz =  m_NiftiImage->dim[3];
		m_NiftiImage->nt =  m_NiftiImage->dim[4];
		m_NiftiImage->nu =  m_NiftiImage->dim[5];
		m_NiftiImage->nv =  m_NiftiImage->dim[6];
		m_NiftiImage->nw =  m_NiftiImage->dim[7];
		//m_NiftiImage->cal_max = 0.00747808;
		//m_NiftiImage->cal_min

		//	m_NiftiImage->pixdim[0] = -1 ;
		m_NiftiImage->pixdim[1] = spacing[0]; cout << "spacing[0] " << spacing[0] << endl;
		m_NiftiImage->pixdim[2] = spacing[1]; cout << "spacing[1] " << spacing[1] << endl;
		m_NiftiImage->pixdim[3] = spacing[2];cout << "spacing[2] " << spacing[2] << endl;
		m_NiftiImage->pixdim[4] = 1;
		m_NiftiImage->pixdim[5] = 1;
		m_NiftiImage->pixdim[6] = 0;
		m_NiftiImage->pixdim[7] = 0;
		m_NiftiImage->dx = m_NiftiImage->pixdim[1];
		m_NiftiImage->dy = m_NiftiImage->pixdim[2];
		m_NiftiImage->dz = m_NiftiImage->pixdim[3];
		m_NiftiImage->dt = m_NiftiImage->pixdim[4];
		m_NiftiImage->du = m_NiftiImage->pixdim[5];
		m_NiftiImage->dv = m_NiftiImage->pixdim[6];
		m_NiftiImage->dw = m_NiftiImage->pixdim[7];
		cout << "writeDTIVolume 1.4" << endl;
		cin.get();
		int numberOfVoxels = m_NiftiImage->nx;

		if(m_NiftiImage->ny>0){
			numberOfVoxels*=m_NiftiImage->ny;
		}
		if(m_NiftiImage->nz>0){
			numberOfVoxels*=m_NiftiImage->nz;
		}
		if(m_NiftiImage->nt>0){
			numberOfVoxels*=m_NiftiImage->nt;
		}
		if(m_NiftiImage->nu>0){
			numberOfVoxels*=m_NiftiImage->nu;
		}


		m_NiftiImage->nvox = numberOfVoxels;

		if(numComponents==6 ){
			cout << "writeDTIVolume 1.3 coponenets:" << numComponents<< " voxels: " <<  numberOfVoxels << "imagedatatype"<<  imageDataType << endl;
			switch(imageDataType)
			{
			case VTK_BIT://DT_BINARY:
				m_NiftiImage->datatype = DT_BINARY;
				m_NiftiImage->nbyper = 0;
				dataTypeSize = 0.125;
				break;
			case VTK_UNSIGNED_CHAR://DT_UNSIGNED_CHAR:
				m_NiftiImage->datatype = DT_UNSIGNED_CHAR;
				m_NiftiImage->nbyper = 1;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_SIGNED_CHAR://DT_INT8:
				m_NiftiImage->datatype = DT_INT8;
				m_NiftiImage->nbyper = 1;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_SHORT://DT_SIGNED_SHORT:
				m_NiftiImage->datatype = DT_SIGNED_SHORT;
				m_NiftiImage->nbyper = 2;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_SHORT://DT_UINT16:
				m_NiftiImage->datatype = DT_UINT16;
				m_NiftiImage->nbyper = 2;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_INT://DT_SIGNED_INT:
				m_NiftiImage->datatype = DT_SIGNED_INT;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_INT://DT_UINT32:
				m_NiftiImage->datatype = DT_UINT32;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_FLOAT://DT_FLOAT:
				m_NiftiImage->datatype = DT_FLOAT;
				m_NiftiImage->nbyper = 4;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_DOUBLE://DT_DOUBLE:
				m_NiftiImage->datatype = DT_DOUBLE;  
				m_NiftiImage->nbyper = 8; 
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_LONG://DT_INT64:
				m_NiftiImage->datatype = DT_INT64;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			case VTK_UNSIGNED_LONG://DT_UINT64:
				m_NiftiImage->datatype = DT_UINT64;
				m_NiftiImage->nbyper = 8;
				dataTypeSize = m_NiftiImage->nbyper;
				break;
			default:
				cout << "cannot handle this type" << endl ;
				break;
			}
		}

		m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;//??
		cout << " writeDTIVolume1.7 "<< endl;
		m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
		cout << " 1.71 "<< endl;
		m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed

		cout << " writeDTIVolume1.8 "<< endl;



		// Using the "Vector" intent code
		m_NiftiImage->intent_code = NIFTI_INTENT_SYMMATRIX;
		m_NiftiImage->intent_p1   = 0.0f; // CHECK !!!!
		m_NiftiImage->intent_p2   = 0.0f;
		m_NiftiImage->intent_p3   = 0.0f;
		//m_NiftiImage->scl_slope = 1;
		//m_NiftiImage->scl_inter = 0;

		if(transform)
		{
			cout << "transform"  << endl;
			vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
			matrix = vtkMatrix4x4::SafeDownCast(transform);
			if(matrix)
			{
				// sform matrix or qform quaternion, which one will be used. Both can also be used if bothcodes are > 0.
				m_NiftiImage->qform_code = 0; // Decided to use only sform code. If this is set > 0 then qform quaternion or sform matrix is used.
				m_NiftiImage->sform_code = 1; // sform matrix is used only if sform_code > 0.


				matrix->Print(cout);
				cout << "transform 1.1"  << endl;
				mat44 matrixf;
				for(int i=0;i<4;i++)
					for(int j=0;j<4;j++)
					{
						if(m_NiftiImage->qform_code > 0)
						{
							matrixf.m[i][j] = matrix->GetElement(i,j);
							cout <<  matrixf.m[i][j] << endl;
						}
						// sform code
						if(m_NiftiImage->sform_code >0 )
							m_NiftiImage->sto_xyz.m[i][j]= matrix->GetElement(i,j);
					}

					// convert transformation matrix to quaternion
					if(m_NiftiImage->qform_code > 0)
						nifti_mat44_to_quatern(matrixf, &( m_NiftiImage->quatern_b), &( m_NiftiImage->quatern_c), &( m_NiftiImage->quatern_d), 
						&( m_NiftiImage->qoffset_x), &(m_NiftiImage->qoffset_y), &(m_NiftiImage->qoffset_z), &(m_NiftiImage->dx) , &(m_NiftiImage->dy) ,&(m_NiftiImage->dz) , &(m_NiftiImage->qfac));


					//cout << m_NiftiImage->quatern_b << " " << m_NiftiImage->quatern_c << " " << m_NiftiImage->quatern_d << " " << m_NiftiImage->qfac << " " << endl;
					//cout << m_NiftiImage->qoffset_x << " " << m_NiftiImage->qoffset_y << " " << m_NiftiImage->qoffset_z <<endl;

					// in case the matrix is not pure transform, quaternion can not include scaling part. Therefore if the matris is not a pure transform matrix use scaling factor in spacing?
					float scaling[3];
					if(matrix->Determinant() != 1 && (m_NiftiImage->qform_code > 0) )
					{
						// If determinant is not 1 find scaling
						vtkTransform *transform = vtkTransform::New();
						transform->SetMatrix(matrix);
						transform->Scale(scaling);

						m_NiftiImage->pixdim[1] = spacing[0]*scaling[0];
						m_NiftiImage->pixdim[2] = spacing[1]*scaling[1];
						m_NiftiImage->pixdim[3] = spacing[2]*scaling[2];
						transform->Delete();
					}
			}
			else {
				cout << "Invalid   matrix \n";
			}
		}
		else
		{
			cout << "Invalid transformation object \n";
		}
		// Set the intent name to "MiND"
		char * intentName = (char*) "DTI";
#ifdef _WIN32
		strcpy_s(&(m_NiftiImage->intent_name[0]), 4, intentName);
#else
		strcpy(&(m_NiftiImage->intent_name[0]), intentName);
#endif
		// Initialize the extension list
		//	m_NiftiImage->num_ext  = 0;
		//	m_NiftiImage->ext_list = NULL;

		if (!vtkAbstractArray::SafeDownCast(image->GetPointData()->GetArray("Tensors")))
		{
			cout << "ERROR: Tensors array missing or not converted to int" << endl;
		}
		int indexMap[6] = {0, 1, 3, 2, 4, 5};
		int arraySize = image->GetPointData()->GetArray("Tensors")->GetNumberOfTuples();
		int comp = image->GetPointData()->GetArray("Tensors")->GetNumberOfComponents();
		cout << arraySize << " " << comp << endl;
		double *outDoubleArray = static_cast<double*>(image->GetPointData()->GetArray("Tensors")->GetVoidPointer(0));

		image->GetPointData()->GetArray("Tensors")->Print(cout);
		cout << "size of outDoubleArray:" << sizeof(outDoubleArray) << endl; 
		cout << "transform 1.91"  << endl;
		double * niftiImageData =  new double[arraySize*comp];
		//this->NiftiImage->data = (void *) new double[arraySize*comp+arraySize];
		cout << "transform 1.92"  << endl;
		for (int i = 0; i < arraySize; ++i) 
			for (int j = 0; j < comp; ++j)
			{

				niftiImageData[i + indexMap[j] * arraySize]  = (double) outDoubleArray[j + comp * i];	
			}
			cout << "transform 1.93"  << endl;
			m_NiftiImage->data =  (void *) niftiImageData;
			//m_NiftiImage->data= (double *) calloc(image->GetPointData()->GetArray("Tensors")->GetNumberOfTuples(), sizeof(double)*6);
			//m_NiftiImage->data = (double *) vtkDoubleArray::SafeDownCast(image->GetPointData()->GetArray("Tensors"))->GetDa=
			//C_TYPE * inArrayCasted = (C_TYPE *) this->NiftiImage->data;				 
			//		for (int i = 0; i < arraySize; ++i) 									 
			//			outDoubleArray[i] = (double) inArrayCasted[i + COMP * arraySize];	 	

			cout << " writeDTIVolume 2.1 num extentions:"<< m_NiftiImage->num_ext <<  endl;
			//int num=image->GetPointData()->GetArray("Tensors")->GetNumberOfTuples();
			//m_NiftiImage->data = outDoubleArray; 
			nifti_image_write( m_NiftiImage );
			delete[]  outDoubleArray;
			delete[] m_NiftiImage;
	}


	//----------------------[ writeDiscreteSphereVolume ]----------------------\\

	void bmiaNiftiWriter::writeDiscreteSphereVolume()
	{


	}


	//--------------------[ writeSphericalHarmonicsVolume ]--------------------\\

	void bmiaNiftiWriter::writeSphericalHarmonicsVolume()
	{


	}


	//----------------------------[ writeTriangles ]---------------------------\\

	void bmiaNiftiWriter::writeTriangles()
	{






	} // namespace bmia
}