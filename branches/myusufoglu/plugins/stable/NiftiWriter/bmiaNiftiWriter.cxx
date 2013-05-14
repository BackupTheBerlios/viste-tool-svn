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


//m_NiftiImage must have been defined before the call of this macro
#define fillNiftiDataStringMacro(C_TYPE)								\
	{																			\
	C_TYPE *outDoubleArray = static_cast<C_TYPE*>(image->GetPointData()->GetArray("Tensors")->GetVoidPointer(0));		\
	C_TYPE * niftiImageData =  new C_TYPE[arraySize*comp];   								\
	for (int i = 0; i < arraySize; ++i)   \
	for (int j = 0; j < comp; ++j)    \
	niftiImageData[i + indexMap[j] * arraySize]  = outDoubleArray[j + comp * i];	 \
	m_NiftiImage->data =  (void *) niftiImageData;  \
	}

//m_NiftiImage must have been defined before calling this macro
#define createArrayMacro(C_TYPE)								\
	{          \
	C_TYPE *niftiImageData =  new C_TYPE[arraySize*comp];   \
	for (int i = 0; i < arraySize; ++i)           \
	for (int j = 0; j < comp; ++j)       \
	niftiImageData[i+ arraySize * j]  = outDoubleArray[j + comp * i];	\
	m_NiftiImage->data =  (void *) niftiImageData; \
  } 

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


	//--------------------------[ writeScalarVolume ]--------------------------\\

	void bmiaNiftiWriter::writeScalarVolume(vtkImageData *image, QString saveFileName, vtkObject * transform)
	{


		nifti_image * m_NiftiImage = new nifti_image;
		m_NiftiImage = nifti_simple_init_nim();
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
		// if start index is 0 wholeExtent[1] below is enough, but if the image is a cropped image etc.
		m_NiftiImage->dim[1] = wholeExtent[1]-wholeExtent[0] + 1;
		m_NiftiImage->dim[2] = wholeExtent[3]-wholeExtent[2] + 1;
		m_NiftiImage->dim[3] = wholeExtent[5]-wholeExtent[4] + 1;
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
		m_NiftiImage->pixdim[1] =  spacing[0];
		m_NiftiImage->pixdim[2] =  spacing[1];
		m_NiftiImage->pixdim[3] =  spacing[2];
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

		// scalar but when it is a product of DTI, remains 6, solve the problem in production of FA etc. from DTI
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
				qDebug() << "cannot handle this type" << endl ;
				break;
			}
		}

		m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;
		m_NiftiImage->data=const_cast<void *>( image->GetScalarPointer());

		m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
		m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed


		//Transformation and quaternion; quaternion has no scaling but viste uses spacing=1 and scaling parameter of the user transform of the actor is changed with spacing because of the transform.
		if(transform)
		{
			vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
			matrix = vtkMatrix4x4::SafeDownCast(transform);

			if(!matrix)
			{

				qDebug() << "Invalid transformation  matrix \n";
				matrix =  vtkMatrix4x4::New();
				matrix->Identity();
			}


			// if extent does not start from zero if it is a cropped image for example:
			// sform matrix or qform quaternion, which one will be used. Both can also be used if bothcodes are > 0.
			m_NiftiImage->qform_code = 0; // Decided to use only sform code. If this is set > 0 then qform quaternion or sform matrix is used.
			m_NiftiImage->sform_code = 1; // sform matrix is used only if sform_code > 0.


			//matrix->Print(cout);
			mat44 matrixf;
			for(int i=0;i<4;i++)
				for(int j=0;j<4;j++)
				{
					if(m_NiftiImage->qform_code > 0)
						matrixf.m[i][j] = matrix->GetElement(i,j);
					// sform code
					if(m_NiftiImage->sform_code >0 )
						m_NiftiImage->sto_xyz.m[i][j]= matrix->GetElement(i,j);
				}

				// convert transformation matrix to quaternion
				nifti_mat44_to_quatern(matrixf, &( m_NiftiImage->quatern_b), &( m_NiftiImage->quatern_c), &( m_NiftiImage->quatern_d), 
					&( m_NiftiImage->qoffset_x), &(m_NiftiImage->qoffset_y), &(m_NiftiImage->qoffset_z), &(m_NiftiImage->dx) , &(m_NiftiImage->dy) ,&(m_NiftiImage->dz) , &(m_NiftiImage->qfac));
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
		else
		{
			cout << "Invalid transformation object \n";
		}



		nifti_set_iname_offset(m_NiftiImage);
		// Write the image file
		nifti_image_write( m_NiftiImage );

	}


	// ----------------------------[ writeMindData ]--------------------------- \\
	//Any datastructure with extention
	void bmiaNiftiWriter::writeMindData(vtkImageData *image, QString saveFileName, vtkObject * transform, QString dataStructure)

	{
		//there must be 6 extentions
		nifti_image * m_NiftiImage = new nifti_image;
		m_NiftiImage = nifti_simple_init_nim();
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

		m_NiftiImage->byteorder		= nifti_short_order();
		m_NiftiImage->ndim = 5;
		m_NiftiImage->dim[0] = 5;
		m_NiftiImage->dim[1] = wholeExtent[1]-wholeExtent[0] + 1;// if start index is 0 wholeExtent[1] is enough
		m_NiftiImage->dim[2] = wholeExtent[3]-wholeExtent[2] + 1;
		m_NiftiImage->dim[3] = wholeExtent[5]-wholeExtent[4] + 1;
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
		m_NiftiImage->pixdim[1] =  spacing[0];
		m_NiftiImage->pixdim[2] =  spacing[1];
		m_NiftiImage->pixdim[3] =  spacing[2];
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
				qDebug() << "cannot handle this type" << endl ;
				break;
			}
		}

		m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;//
		m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
		m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed

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
			vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
			matrix = vtkMatrix4x4::SafeDownCast(transform);



			if(!matrix)
			{

				qDebug() << "Invalid transformation  matrix \n";
				matrix =  vtkMatrix4x4::New();
				matrix->Identity();
			}

			// sform matrix or qform quaternion, which one will be used. Both can also be used if bothcodes are > 0.
			m_NiftiImage->qform_code = 0; // Decided to use only sform code. If this is set > 0 then qform quaternion or sform matrix is used.
			m_NiftiImage->sform_code = 1; // sform matrix is used only if sform_code > 0.


			mat44 matrixf;
			for(int i=0;i<4;i++)
				for(int j=0;j<4;j++)
				{
					if(m_NiftiImage->qform_code > 0)
					{
						matrixf.m[i][j] = matrix->GetElement(i,j);
					}
					// use only sform code, give the matrix itself
					if(m_NiftiImage->sform_code >0 )
						m_NiftiImage->sto_xyz.m[i][j]= matrix->GetElement(i,j);
				}

				// convert transformation matrix to quaternion
				if(m_NiftiImage->qform_code > 0)
					nifti_mat44_to_quatern(matrixf, &( m_NiftiImage->quatern_b), &( m_NiftiImage->quatern_c), &( m_NiftiImage->quatern_d), 
					&( m_NiftiImage->qoffset_x), &(m_NiftiImage->qoffset_y), &(m_NiftiImage->qoffset_z), &(m_NiftiImage->dx) , &(m_NiftiImage->dy) ,&(m_NiftiImage->dz) , &(m_NiftiImage->qfac));

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
		else
		{
			qDebug() << "Invalid transformation object \n";
		}

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

		// Add the main LONI MiND extensions one descriptor N data-related
		// DTI 
		if(dataStructure.contains("DTI")) 
		{ 

			m_NiftiImage->dim[5] = 6; // 6 components of thematrix
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nvox = m_NiftiImage->nx * m_NiftiImage->ny * m_NiftiImage->nz * m_NiftiImage->nt * m_NiftiImage->nu;

			nifti_add_extension(m_NiftiImage, "DTENSOR", 8, NIFTI_ECODE_MIND_IDENT);
			for(int i=1;  i<=3; i++)
				for(int j=1;j<=3; j++)
					if(j<=i) {
						int index[2];
						index[0]=i;
						index[1]=j; 
						nifti_add_extension(m_NiftiImage, (char *) &(index[0]), 2 * sizeof(int), NIFTI_ECODE_DT_COMPONENT);	 
					}

					vtkDataArray * inTensors = image->GetPointData()->GetArray("Tensors");

					if (!inTensors)
					{
						qDebug() <<"Input data has no tensors!";
						return;
					}

					int indexMap[6] = {0, 1, 3, 2, 4, 5};
					double *outDoubleArray = static_cast<double*>(image->GetPointData()->GetArray("Tensors")->GetVoidPointer(0));

					int arraySize = image->GetPointData()->GetArray("Tensors")->GetNumberOfTuples();
					int comp = image->GetPointData()->GetArray("Tensors")->GetNumberOfComponents();
					double * niftiImageData =  new double[arraySize*comp];
					//this->NiftiImage->data = (void *) new double[arraySize*comp+arraySize];
					for (int i = 0; i < arraySize; ++i) 
						for (int j = 0; j < comp; ++j)
						{
							niftiImageData[i + indexMap[j] * arraySize]  = (double) outDoubleArray[j + comp * i];	
						}
						m_NiftiImage->data =  (void *) niftiImageData;

		}

		// Discrete Sphere
		else if(dataStructure.contains("discrete sphere")) {
			// overwrite 
			m_NiftiImage->dim[5] = image->GetNumberOfScalarComponents(); //   
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nvox = m_NiftiImage->nx * m_NiftiImage->ny * m_NiftiImage->nz * m_NiftiImage->nt * m_NiftiImage->nu;
			m_NiftiImage->swapsize		= 8;					// ...and the swap size is also 8.
			m_NiftiImage->iname_offset	= 1024;					// Offset for the image name

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
				nifti_add_extension(m_NiftiImage, (char *) &(indx[0]), 2 * sizeof(float), NIFTI_ECODE_SPHERICAL_DIRECTION);	 
			}


			vtkDataArray * inVectors = image->GetPointData()->GetArray("Vectors");
			//image->GetPointData()->GetArray("Vectors")->Print(cout);
			if (!inVectors)
			{
				cout <<"Input data has no vectors!";
				return;
			}
			double *outDoubleArray = static_cast<double*>(image->GetScalarPointer() );
			int arraySize = image->GetNumberOfPoints();
			int comp = image->GetNumberOfScalarComponents();
			double * niftiImageData =  new double[arraySize*comp];

			for (int i = 0; i < arraySize; ++i) 
				for (int j = 0; j < comp; ++j)
				{
					// change from row-major to column major
					niftiImageData[i+ arraySize * j]  = (double) outDoubleArray[j + comp * i];	
				}
				m_NiftiImage->data =  (void *) niftiImageData;

		}
		// Spherical Harmonics
		else if(dataStructure.contains("spherical harmonics")) {
			// overwrite 
			//m_NiftiImage->dim[5] = image->GetNumberOfScalarComponents(); //   Works for loaded nifti but not sharm
			m_NiftiImage->dim[5] = image->GetPointData()->GetArray(0)->GetNumberOfComponents(); // works for sharm loaded
			m_NiftiImage->nu =  m_NiftiImage->dim[5];
			m_NiftiImage->nvox = m_NiftiImage->nx * m_NiftiImage->ny * m_NiftiImage->nz * m_NiftiImage->nt * m_NiftiImage->nu;
			m_NiftiImage->swapsize		= 8;					// ...and the swap size is also 8.
			m_NiftiImage->iname_offset	= 1024;					// Offset for the image name
			char buffer[24];
			memset(buffer, 0, sizeof(buffer));
			strncpy(buffer, "REALSPHARMCOEFFS", sizeof(buffer));
			nifti_add_extension(m_NiftiImage, buffer, 24, NIFTI_ECODE_MIND_IDENT); // 24 is length of array which inc. DISCSPHFUNC


			int shOrder;

			// Get the SH order, based on the number of coefficients
			switch( image->GetPointData()->GetScalars()->GetNumberOfComponents())
			{
			case 1:		shOrder = 0;	break;
			case 6:		shOrder = 2;	break;
			case 15:	shOrder = 4;	break;
			case 28:	shOrder = 6;	break;
			case 45:	shOrder = 8;	break;

			default:
				qDebug() << "Number of SH coefficients are not present!" << endl;
				return;
			}


			for(int j=0;j<=shOrder; j++)
			{
				if( j%2 == 0) //even
				{
					for(int i = -1*j; i<=j; i++)
					{ 
						int indx[2]; 
						indx[0]= (int)  j;
						indx[1]= (int) i;
						nifti_add_extension(m_NiftiImage, (char *) &(indx[0]), 2 * sizeof(int), NIFTI_ECODE_SHC_DEGREEORDER);	 
					}
				}
			}


			double *outDoubleArray = static_cast<double*>(image->GetScalarPointer() );

			int arraySize = image->GetNumberOfPoints(); 
			int comp = image->GetNumberOfScalarComponents();
			comp=image->GetPointData()->GetNumberOfComponents(); // for all data, 72dirs and others

			//m_NiftiImage must have been defined before calling this macro
			switch (m_NiftiImage->datatype)
			{
			case DT_FLOAT:  createArrayMacro(float); break;
			case DT_DOUBLE:  createArrayMacro(double); break;
			default:
				createArrayMacro(double); break;

			}	

		}

		else {
			qDebug() << "The data type is not suitable to Loni Mind Save Funtion"<< endl;
		}

		// Add the two angles to the NIfTI file as an extension
		nifti_set_iname_offset(m_NiftiImage);
		nifti_image_write( m_NiftiImage );

	}

	// DTI without MIND extention method
	void bmiaNiftiWriter::writeDTIVolume(vtkImageData *image, QString saveFileName, vtkObject * transform)
	{

		//there must be 6 extentions
		nifti_image * m_NiftiImage = new nifti_image;
		m_NiftiImage = nifti_simple_init_nim();
		//image->Print(cout);
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

		m_NiftiImage->ndim = 5;
		m_NiftiImage->dim[0] = 5;
		m_NiftiImage->dim[1] = wholeExtent[1]-wholeExtent[0] + 1;// if start index is 0 wholeExtent[1] is enough
		m_NiftiImage->dim[2] = wholeExtent[3]-wholeExtent[2] + 1;
		m_NiftiImage->dim[3] = wholeExtent[5]-wholeExtent[4] + 1;
		m_NiftiImage->dim[4] = 1;
		m_NiftiImage->dim[5] = 6; 
		m_NiftiImage->dim[6] = 0;
		m_NiftiImage->dim[7] = 0;
		m_NiftiImage->nx =  m_NiftiImage->dim[1];
		m_NiftiImage->ny =  m_NiftiImage->dim[2];
		m_NiftiImage->nz =  m_NiftiImage->dim[3];
		m_NiftiImage->nt =  m_NiftiImage->dim[4];
		m_NiftiImage->nu =  m_NiftiImage->dim[5];
		m_NiftiImage->nv =  m_NiftiImage->dim[6];
		m_NiftiImage->nw =  m_NiftiImage->dim[7];

		m_NiftiImage->pixdim[1] = spacing[0];  
		m_NiftiImage->pixdim[2] = spacing[1];  
		m_NiftiImage->pixdim[3] = spacing[2]; 
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
				qDebug() << "cannot handle this type" << endl ;
				break;
			}
		}

		m_NiftiImage->nifti_type = NIFTI_FTYPE_NIFTI1_1;//??
		m_NiftiImage->fname = nifti_makehdrname( saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0);
		m_NiftiImage->iname = nifti_makeimgname(saveFileName.toStdString().c_str(), m_NiftiImage->nifti_type,false,0); // 0 is compressed

		// Using the "Vector" intent code
		m_NiftiImage->intent_code = NIFTI_INTENT_SYMMATRIX;
		m_NiftiImage->intent_p1   = 0.0f; // CHECK !!!!
		m_NiftiImage->intent_p2   = 0.0f;
		m_NiftiImage->intent_p3   = 0.0f;


		if(transform)
		{

			vtkMatrix4x4 *matrix =  vtkMatrix4x4::New();
			matrix = vtkMatrix4x4::SafeDownCast(transform);


			if(!matrix)
			{

				qDebug() << "Invalid transformation  matrix \n";
				matrix =  vtkMatrix4x4::New();
				matrix->Identity();
			}



			// sform matrix or qform quaternion, which one will be used. Both can also be used if bothcodes are > 0.
			m_NiftiImage->qform_code = 0; // Decided to use only sform code. If this is set > 0 then qform quaternion or sform matrix is used.
			m_NiftiImage->sform_code = 1; // sform matrix is used only if sform_code > 0.


			//matrix->Print(cout);
			mat44 matrixf;
			for(int i=0;i<4;i++)
				for(int j=0;j<4;j++)
				{
					if(m_NiftiImage->qform_code > 0)
					{
						matrixf.m[i][j] = matrix->GetElement(i,j);
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
		else
		{
			qDebug() << "Invalid transformation object \n";
		}
		// Set the intent name to "MiND"
		char * intentName = (char*) "DTI";
#ifdef _WIN32
		strcpy_s(&(m_NiftiImage->intent_name[0]), 4, intentName);
#else
		strcpy(&(m_NiftiImage->intent_name[0]), intentName);
#endif

		if (!vtkAbstractArray::SafeDownCast(image->GetPointData()->GetArray("Tensors")))
		{
			qDebug() << "ERROR: Tensors array missing or not converted to int" << endl;
		}
		int indexMap[6] = {0, 1, 3, 2, 4, 5};
		int arraySize = image->GetPointData()->GetArray("Tensors")->GetNumberOfTuples();
		int comp = image->GetPointData()->GetArray("Tensors")->GetNumberOfComponents();


		switch (m_NiftiImage->datatype)
		{

		//m_NiftiImage must have been defined before calling this macro
		case DT_FLOAT:  fillNiftiDataStringMacro(float); break;
		case DT_DOUBLE:  fillNiftiDataStringMacro(double); break;

		default: 
			fillNiftiDataStringMacro(double); break;
		}	

		nifti_image_write( m_NiftiImage );
		//delete[]  outDoubleArray;
		delete[] m_NiftiImage;
	}


	//----------------------[ writeDiscreteSphereVolume ]----------------------\\

	void bmiaNiftiWriter::writeDiscreteSphereVolume()
	{
		// This part is on WriteMINDData() function

	}


	//--------------------[ writeSphericalHarmonicsVolume ]--------------------\\

	void bmiaNiftiWriter::writeSphericalHarmonicsVolume()
	{
		// This part is on WriteMINDData() function

	}


	//----------------------------[ writeTriangles ]---------------------------\\

	void bmiaNiftiWriter::writeTriangles()
	{

	} // namespace bmia
}