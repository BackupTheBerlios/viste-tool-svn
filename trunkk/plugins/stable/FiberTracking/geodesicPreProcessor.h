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
 * geodesicPreProcessor.h
 *
 * 2011-05-25	Evert van Aart
 * - First version. 
 *
 */


#ifndef bmia_FiberTrackingPlugin_geodesicPreprocessor_h
#define bmia_FiberTrackingPlugin_geodesicPreprocessor_h


/** Includes - VTK */

#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>


namespace bmia {


/** This class is used to pre-process the DTI tensors before they are used by the
	geodesic fiber-tracking algorithm. It performs two actions: The Pre-Processing
	itself, which consists of applying a constant gain and/or sharpening the tensors;
	and inverting tensors. It contains optional arrays which contain the pre-processed
	or inverted tensor for each voxel in the image. If a pre-processed or inverted
	tensor is requested (usually by the "geodesicNeighborhood" class), and the
	corresponding array has been constructed (by the "preProcessFullImage" or
	"invertFullImage" function), the output tensor is fetched from this array; 
	otherwise, it is computed on the spot. This allows us to choose between fast
	performance with high memory requirements (pre-processing and inverting the
	full image before we start tracking), and lower performance with lower memory
	requirements (computing everything when we need it). 
*/

class geodesicPreProcessor
{
	public:

		/** Constructor. */

		geodesicPreProcessor();

		/** Destructor. */

		~geodesicPreProcessor();

		/** Small value for comparisons to zero, to avoid rounding error. */

		static double PP_CLOSE_TO_ZERO;

		/** Methods for tensor sharpening. */

		enum SharpeningMethod
		{
			SM_None = 0,		/**< Do not sharpen tensors. */
			SM_Exponent,		/**< Exponentiate tensors by integer exponent. */
			SM_TraceDivision,	/**< Divide all tensor elements by tensor trace. */
			SM_TraceDivAndExp	/**< Exponentiation followed by division by trace. */
		};

		/** Set the input DTI image. Checks the image to see if it contains suitable
			data; if so, set the "inTensorImage" and "inTensorArray" pointers, and
			return true; return false otherwise.
			@param rImage		Input DTI image. */

		bool setInputImage(vtkImageData * rImage);

		/** Set the scalar image, which is used for the sharpening threshold. Checks
			the input image to see if it contains suitable data; if so, set the 
			"scalarArray" pointer and return true; return false otherwise.
			@param rImage		Input scalar image. */

		bool setScalarImage(vtkImageData * rImage);

		/** Pre-process the entire image, and store the pre-processed tensors in
			"fullPPTensorArray". The "setInputImage" and "setScalarImage" functions
			should be called before this function. */

		void preProcessFullImage();

		/** Inverts all tensors of the pre-processed tensor image. "preProcessFullImage"
			should be called before calling this function. Inverted tensors are stored
			in "fullInvTensorArray". */

		void invertFullImage();

		/** Pre-process a single tensor. If "fullPPTensorArray" does not exist (i.e.,
			"preProcessFullImage" was never called), this fetches the input tensor 
			from "inTensorArray", and writes the pre-processed tensor to "outTensor".
			If the full image has been pre-processed, get the pre-processed tensor
			from the array.
			@param pointId		Index of the target voxel.
			@param outTensor	Output pre-processed tensor. */

		void preProcessSingleTensor(vtkIdType pointId, double * outTensor);

		/** Inverts a single tensor. If "fullInvTensorArray" exists, the inverted
			tensor will be fetched  from this array; otherwise, the input tensor
			will be inverted directly. 
			@param pointId		Index of the target voxel.
			@param ppTensor		Input pre-processed tensor.
			@param outTensor	Output inverted tensor. */

		void invertSingleTensor(vtkIdType pointId, double * ppTensor, double * outTensor);

		/** Enable or disable pre-processing.
			@param rEnable		Enable or disable pre-processing. */

		void setEnable(bool rEnable)
		{
			enablePP = rEnable;
		}

		/** Set the tensor gain.
			@param rGain		Desired gain. */

		void setTensorGain(int rGain)
		{
			gain = rGain;
		}

		/** Set the desired method for sharpening the tensors. 
			@param rMethod		Desired sharpening method. */

		void setSharpeningMethod(SharpeningMethod rMethod)
		{
			sharpenMethod = rMethod;
		}

		/** Set the threshold for sharpening the tensors. If the voxel scalar value
			(from "scalarArray") is lower than this value, the tensor will be
			sharpened; otherwise, it will remain unchanged (except for the gain).
			@param rThreshold	Desired scalar threshold. */

		void setSharpenThreshold(double rThreshold)
		{
			sharpenThreshold = rThreshold;
		}

		/** Set the exponent used for sharpening the tensors. 
			@param rExponent	Desired tensor exponent. */

		void setSharpenExponent(int rExponent)
		{
			exponent = rExponent;
		}

	private:
	
		/** Image containing the input DTI tensors. */

		vtkImageData * inTensorImage;

		/** Array containing the input DTI tensors. */

		vtkDataArray * inTensorArray;

		/** Scalar array. Scalars are compared to "sharpenThreshold" to determine
			whether or not the tensor should be sharpened. */

		vtkDataArray * scalarArray;

		/** Array for pre-processed tensors of the entire image. Filled when the
			function "preProcessFullImage" is called. */

		vtkDataArray * fullPPTensorArray;

		/** Array for inverted tensors of the entire image. Filled when the function
			"invertFullImage" is called. */

		vtkDataArray * fullInvTensorArray;

		/** Enable or disable pre-processing. */

		bool enablePP;

		/** Gain factor for the tensors. */

		int gain;

		/** Sharpening method used to sharpen the tensors. */

		SharpeningMethod sharpenMethod;

		/** Threshold for sharpening. Tensors will only be sharpened if the scalar
			value at that voxel (from "scalarArray") is less than this threshold. */

		double sharpenThreshold;

		/** Exponent for sharpening the tensors. Only integer exponents are supported. */

		int exponent;

		/** Exponentiate an input tensor. Overwrites the input tensor. 
			@param T	Input and output tensor. */

		void powerInputTensor(double * T);

		/** Inverts a tensor. If the tensor cannot be inverted, the output will be
			an identity tensor. Input and output cannot be the same. 
			@param iT	Input tensor (pre-processed or original). 
			@param oT	Output tensor (inverted). */

		void invertTensor(double * iT, double * oT);

}; // class geodesicPreProcessor


} // namespace bmia


#endif // bmia_FiberTrackingPlugin_geodesicPreprocessor_h