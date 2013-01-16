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
 * KernelNIfTIWriter.h
 *
 * 2011-07-26	Evert van Aart
 * - First version.
 *
 * 2012-03-12	Ralph Brecheisen
 * - Inclusion of malloc.h is now conditional because on Mac OSX
 *   this file does not exist and can be ignored.
 */


#ifndef bmia_HARDIConvolutionsPlugin_KernelNIfTIWriter_h
#define bmia_HARDIConvolutionsPlugin_KernelNIfTIWriter_h


/** Includes - Custom Files */

#include "NIfTI/nifti1.h"
#include "NIfTI/nifti1_io.h"

/** Includes - Qt */

#include <QString>
#include <QByteArray>

/** Includes - C++ */

#if !defined(__APPLE__)
#include <malloc.h>
#endif

#include <vector>


namespace bmia {


/** Very simple NIfTI writer, used to write the kernels generated by "KernelGenerator".
	Writes the data to file with the "Vector" intent code, which contains the directions
	as spherical coordinates in MiND extensions. Generated files can either be opened
	by the user through the regular NIfTI reader (for manual inspection of the kernels),
	or by "KernelNIfTIReader" when applying the convolutions.
*/

class KernelNIfTIWriter
{
	public:

		/** Constructor */

		KernelNIfTIWriter();

		/** Destructor */

		~KernelNIfTIWriter();

		/** Store the desired file name.
			@param rName		Output file name. */

		void setFileName(QString rName)
		{
			fileName = rName;
		}

		/** Store the dimensions of the kernel.
			@param rDim			Kernel dimensions. */

		void setDimensions(int * rDim)
		{
			dim[0] = rDim[0];
			dim[1] = rDim[1];
			dim[2] = rDim[2];
		}

		/** Store a pointer to the list of directions of the discrete sphere function.
			Directions are stored as 3D unit vectors. This class does not modify
			this vector of directions. 
			@param rDirections	Spherical directions. */

		void setDirections(std::vector<double *> * rDirections)
		{
			directions = rDirections;
		}

		/** Write a complete kernel image to a NIfTI file. The input array contains
			all values for all voxels, so a total of "N * M" values, where N is the
			number of kernel voxels, and M is the number of directions per voxel. 
			Note that the directions are the outermost dimension; in other words,
			in the data array, all N values for one direction are grouped together,
			followed by the N values for the second direction, and so on.
			@param kernelData	Output kernel values. */

		bool writeKernel(double * kernelData);

	private:

		/** Full path of the output file. */

		QString fileName;

		/** Kernel dimensions. */

		int dim[3];

		/** Array of discrete sphere function directions (unit vectors). */

		std::vector<double *> * directions;

}; // KernelNIfTIWriter


} // namespace bmia


#endif // bmia_HARDIConvolutionsPlugin_KernelNIfTIWriter_h