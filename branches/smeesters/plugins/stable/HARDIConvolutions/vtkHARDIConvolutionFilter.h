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
 * vtkHARDIConvolutionFilter.h
 *
 * 2011-08-01	Evert van Aart
 * - First version
 *
 */


#ifndef bmia_HARDIConvolutionsPlugin_vtkHARDIConvolutionFilter_h
#define bmia_HARDIConvolutionsPlugin_vtkHARDIConvolutionFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>

/** Includes - Qt */

#include <QStringList>

/** Includes - Custom Files */

#include "KernelNIfTIReader.h"
#include "KernelGenerator.h"


namespace bmia {


class vtkHARDIConvolutionFilter : public vtkSimpleImageToImageFilter
{
	public:

		static vtkHARDIConvolutionFilter * New();

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

		void setGenerator(KernelGenerator * rGen)
		{
			generator = rGen;
			niftiFileNames = NULL;
		}

		void setNIfTIFileNames(QStringList * rNames)
		{
			niftiFileNames = rNames;
			generator = NULL;
		}

		void setThreshold(double rT, bool rRelative)
		{
			Threshold = rT;
			useRelativethreshold = rRelative;
		}

		void setNumberOfIterations(int rIter)
		{
			numberOfIterations = rIter;
		}

	protected:

		vtkHARDIConvolutionFilter();

		~vtkHARDIConvolutionFilter();

		struct MapInfo
		{
			vtkIdType pointId;
			unsigned short dirId;
		};

		KernelGenerator * generator;

		QStringList * niftiFileNames;

		vtkIdType * neighborhood;
		double * kernelValues;
		MapInfo * kernelMask;
		int maskSize;

		vtkImageData * inputImage;
		vtkDoubleArray * imageValues;

		double Threshold;
		bool useRelativethreshold;

		int numberOfIterations;

	private:

		void getPositionNeighbourhood(int * ijkBase, int * kernelDim, int * imageDim);

		void maskKernel(int kernelSize, int numberOfDirections);

}; // class vtkHARDIConvolutionFilter

} // namespace


#endif // bmia_HARDIConvolutionsPlugin_vtkHARDIConvolutionFilter_h
