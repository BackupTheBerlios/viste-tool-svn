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


#ifndef bmia_NiftiWriterPlugin_bmiaNiftiWriter_h
#define bmia_NiftiWriterPlugin_bmiaNiftiWriter_h


/** Includes - VTK */

#include <vtkImageReader2.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkShortArray.h>
#include <vtkIntArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>

/** Includes - Qt */

#include <QFile>
#include <QtDebug>
#include <QMessageBox>
#include <QList>
#include <QString>
#include <QProgressDialog>

/** Includes - NIfTI */

#include "NIfTI/nifti1_io.h"

/** Includes - Custom Files */

#include "HARDI/SphereTriangulator.h"
#include "core/UserOutput.h"

namespace bmia {


/** Class for reading files in NIfTI format. The reader supports several different
	types of NIfTI data. The type of the input file is determined based on the 
	intent code, the dimensionality of the image, and/or the MiND extensions,
	if available. Since NIfTI files can contain data that cannot be expressed as
	an image volume - such as a triangle array, which is represented with a
	"vtkIntArray" object - a data output can be any child of "vtkObject". It is
	up the "NiftiWriterPlugin" class to create the correct type of data set for
	the output objects. 
*/

class bmiaNiftiWriter
{
	public:

		/** Constructor */

		bmiaNiftiWriter(UserOutput * rUserOut);

		/** Destructor */

		~bmiaNiftiWriter();

		/** Check if we can read the input file. Returns one if we can, zero otherwise.
			filename	Input filename. */

		virtual int CanReadFile(const char * filename);

	 
		 

		/** Return supported file extensions. */

		virtual const char * GetFileExtensions()
		{
			return ".nii .nii.gz";
		}

		/** Get the transformation matrix read from the NIfTI file. */
	
		vtkMatrix4x4 * getTransformMatrix()
		{
			return transformMatrix;
		};

		/** Delete all temporary objects. */

		void cleanUp();

		/** Enumeration for all supported data types. To add support for a new
			type of data: 1) Add a value to this enumeration; 2) Modify the
			"determineDataType" function, so that it can correctly detect NIfTI
			files of this data type; 3) Implement a parsing function for the data
			type; 4) Add support for the new data type in "readNIfTIFile". */

		enum NIfTIDataType
		{
			NDT_Unknown = 0,		/**< Unknown data type. */
			NDT_ScalarVolume,		/**< Single-component scalar volume. */
			NDT_DTITensors,			/**< Second-order tensors with six unique elements. */
			NDT_DiscreteSphere,		/**< Discrete sphere function, incl. sphere directions and (optionally) topology. */
			NDT_SphericalHarm,		/**< Spherical harmonics. */
			NDT_Triangles,			/**< Triangles, used to describe topology. */
			NDT_GenericVector		/**< Vector of undetermined purpose. Will be split into N scalar volumes. */
		};

		/** Data type of the NIfTI image. */

		NIfTIDataType imageDataType;

		/** List of the output objects, all cast to "vtkObject" pointers. Most pointers
			will be to "vtkImageData" objects, but other data types are also possible
			(like "vtkIntArray" for "NDT_Triangles"). The final size of the list depends
			on "imageDataType": For most types it will be one, but some it will
			be more (e.g., for "NDT_GenericVectors", we will have N output objects). */

		QList<vtkObject *> outData;


		/** Write the scalar data passed as vtkimagedata using filename and the tranformation matrix. 
			@param image	Input Scalar component. 
			@param saveFileName The file name including the extention.
			@param attObject Transformation info including rotation and translation.
			*/

		void writeScalarVolume( vtkImageData *image, QString saveFileName, vtkObject * attObject);
		
		/** Write the scalar data passed as vtkimagedata using filename and the tranformation object. 
			@param image	Input Scalar component. 
			@param saveFileName The file name including the extention.
			@param attObject Transformation info including rotation and translation.
			*/

		void writeScalarVolume(int component, vtkImageData *image, QString saveFileName, vtkObject * attObject);

	/** Write image containing second-order DTI tensors.  DTI saved as standart nifti, without MIND extentions 
	     DTI data passed as vtkimagedata and saved using filename and the tranformation object. 
			@param image	Input DTI data. 
			@param saveFileName The file name including the extention.
			@param attObject Transformation info including rotation and translation.
			*/

		void writeDTIVolume(vtkImageData *image, QString saveFileName, vtkObject * transform);

		/** Save the image containing second-order DTI tensors, spherical harmonics or discrete sphere.  
		 Data saved as Loni-Mind nifti, ie. with MIND extentions.
		    @param image	Input data as DTI or Spherical Harmonics or Discrete Sphere. 
			@param saveFileName The file name including the extention.
			@param attObject Transformation info including rotation and translation.
			@param dataStructure DTI, Discrete Sphereor Spherical Harmonics.
		 
		 */

		void writeMindData(vtkImageData *image, QString saveFileName,vtkObject * transform, QString dataStructure);


	protected:

		/** The NIfTI image object constructed when writing the ".nii" file. */

		nifti_image * NiftiImage;

	private:
    
		/** Transformation matrix of the NIfTI file (if any). */
	
		vtkMatrix4x4 * transformMatrix;

		/** Determine the data type of the NIfTI image by looking at its intent code,
			its dimensionality, and/or its MiND extensions. */

		bool determineDataType();

		
	

		/** Create an image containing, per voxel, the radius for each of the
			spherical directions. These spherical directions, which are read
			from a MiND extension, are stored in a scalar array named 
			"Spherical Directions", which is attached to the output image.
			The topology of the output glyphs can be read from a separate NIfTI
			file (with the same name as the current file, appended with "_topo"), 
			or, failing that, constructed here. In either case, an array describing 
			the topology (triangles) is also attached to the output image. */

		void writeDiscreteSphereVolume();

		/** Create an image containing, for each voxel, a set of Spherical Harmonics
			coefficients. The number of coefficients (i.e., the vector length) should
			be 1 (0th order), 6 (2nd), 15 (4th), 28 (6th) or 45 (8th). The coefficients
			are stored in ascending order: First the coefficient for l = 0, then the
			five coefficients for l = 2, and so on. Uses the MiND extensions. */

		void writeSphericalHarmonicsVolume();

		/** Create an array of 3-element integer vectors. These integers represent
			point indices of the vertices of a discrete sphere function. Each set
			of three point indices describes a triangle; these triangles can later
			be used to create the geometry glyphs. */

		void writeTriangles();

		/** Compare one string to a target string. Used for the MiND identifiers. 
			@param id			Input string.
			@param target		Target string.
			@param n			Maximum length of the comparison. */

		bool compareMiNDID(char * id, char * target, int n);

		/** Find an extension with an "ecode" identifier equal to the target ID.
			Returns a pointer to the extension on success, and NULL otherwise.
			Searching starts at the "extPos" position in the extension list, and
			on success, the position of the target extension is copied to "extPos".
			@param targetID		Desired "ecode" value.
			@param extPos		Starting location for search and output index. */

		nifti1_extension * findExtension(int targetID, int & extPos);

	 
		/** Copy of the input filename. */

		QString filenameQ;

		/** Progress dialog. */

		QProgressDialog * progress;

		UserOutput * userOut;									// User output (for s_form q_form selection)

}; // class bmiaNiftiWriter


} // namespace bmia


#endif // bmia_NiftiWriterPlugin_bmiaNiftiWriter_h
