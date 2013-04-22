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
 * vtkImageMask2.cxx
 *
 * 2011-07-08	Evert van Aart
 * - First version. 
 * 
 */


#ifndef bmia_RayCastPlugin_ImageMask2_h
#define bmia_RayCastPlugin_ImageMask2_h


/** Includes - VTK */

#include <vtkMatrix4x4.h>
#include <vtkDataArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkCell.h>
#include <vtkInformation.h>


namespace bmia {


/** This class applies an image mask to a source image. Both inputs and the output
	are of type "vtkImageData". The sizes of the two input images do not need to be
	the same; additionally, the filter supports transformed images. For example, it
	is not uncommon for an image mask to consist of only the subvolume of interest,
	which is placed into the correct region of the source image by the transformation
	matrix. If this case, the output image has the same size as the masking image,
	and the transformation matrix of the masking image is applied to the output. If
	both images are the same size, the output will also be that size, and the
	transformation matrix of the source image is used. */

class vtkImageMask2
{
	public:

		/** Constructor */

		vtkImageMask2();

		/** Destructor */

		~vtkImageMask2();

		/** Reset the stored transformation matrices to the identity matrices. */

		void clearMatrices();

		/** Set the transformation matrix of the source image. If the input matrix
			is NULL, the source matrix is set to the identity matrix.
			@param m		Input matrix. */

		void setSourceMatrix(vtkMatrix4x4 * m);

		/** Set the transformation matrix of the masking image. If the input matrix
			is NULL, the mask matrix is set to the identity matrix.
			@param m		Input matrix. */

		void setMaskMatrix(vtkMatrix4x4 * m);

		/** Determine whether or not to invert the mask. For non-inverted masks,
			all source values corresponding to a non-zero mask value are sent to 
			the output, while the other values are set to zero. Inverted masks,
			on the other hand, only output the source value if the mask is zero.
			@param rInvert	Turn inverting on or off (Default = off). */

		void setInvert(bool rInvert)
		{
			invert = rInvert;
		}

		/** Get the matrix that should be applied to the output. This will either
			be "MSource" (if both images have the same size), or "MMask" (different
			sizes). The class calling this function should copy the matrix before
			using it. */

		vtkMatrix4x4 * getOutputMatrix()
		{
			return MOut;
		}

		/** Run the filter. */

		void Update();

		/** Set the first input (source image). 
			@param i		Source image. */

		void setInput0(vtkImageData * i)
		{
			inputImage0 = i;
		}

		/** Set the first input (masking image). 
			@param i		Masking image. */

		void setInput1(vtkImageData * i)
		{
			inputImage1 = i;
		}

		/** Return the output of the filter. */

		vtkImageData * getOutput()
		{
			return outputImage;
		}

	protected:

		/** Transformation matrix of the source image. */

		vtkMatrix4x4 * MSource;

		/** Transformation matrix of the masking image. */

		vtkMatrix4x4 * MMask;

		/** The matrix that should be applied to the output. This will either be 
			"MSource" (if both images have the same size), or "MMask" (different
			sizes). */

		vtkMatrix4x4 * MOut;
		
		/** Determines whether or not to invert the mask. For non-inverted masks,
			all source values corresponding to a non-zero mask value are sent to 
			the output, while the other values are set to zero. Inverted masks,
			on the other hand, only output the source value if the mask is zero. */
	
		bool invert;

		/** Source image. */

		vtkImageData * inputImage0;

		/** Masking image. */

		vtkImageData * inputImage1;

		/** Output image. */

		vtkImageData * outputImage;

}; // class vtkImageMask2


} // namespace bmia


#endif // bmia_RayCastPlugin_ImageMask2_h