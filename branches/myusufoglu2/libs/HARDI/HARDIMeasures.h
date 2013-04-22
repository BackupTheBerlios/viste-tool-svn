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
 * HARDIMeasures.h
 *
 * 2008-10-22	Vesna Prckovska
 * - First version
 *
 * 2009-01-13	Tim Peeters
 * - Second version
 *
 * 2010-12-07	Evert van Aart
 * - Made compatible with DTITool3. 
 * - Removed dependence on GSL. Visualization of HARDI glyphs depends on this
 *   class, and we want basic visualization to be independent of GSL.
 *
 * 2010-12-16	Evert van Aart
 * - Removed the "Classify Voxel" measure, since it wasn't actually used.
 *
 */


#ifndef bmia_HARDIMeasures_h
#define bmia_HARDIMeasures_h


/** Includes - Custom Files*/

#include "HARDITransformationManager.h"
#include "sphereTesselator.h"

/** Includes - C++ */

#include <assert.h>
#include <math.h>
#include <string.h>


namespace bmia {


class HARDIMeasures 
{
	public:

		/** Enumeration of all supported HARDI measures. */

		enum HARDIMeasureType
		{  
			GA = 0,			// General Anisotropy
			V,				// Variance
			GFA,			// General Fractional Anisotropy
			FMI,			// Fractional Multi-Fiber Index
			R0,				// Rank 0
			R2,				// Rank 2
			Ri,				// Rank i
			Iso,			// Isotropic component
			SE,				// ShannonEntropy
			CRE,			// Cumulative Residual Entropy
			NM				// Number of Maxima
		};
   
		/** Number of measures. Should always match the "HARDIMeasureType" enumeration. */

		static const int numberOfMeasures = 11;

		/** Array of long measure names. Order and quantity should always
			match the "HARDIMeasureType" enumeration shown above. */

		static const char * longNames[];

		/** Array of short measure names. Order and quantity should always
			match the "HARDIMeasureType" enumeration shown above. */

		static const char * shortNames[];

		/** Constructor */

		HARDIMeasures();

		/** Destructor */

		~HARDIMeasures();

		/** Return the value of the specified measure with the given coefficients. 
			The input "measure" should be at least zero and less then "numberOfMeasures".
			@param measure		Desired HARDI measure.
			@param coeff		Spherical harmonics coefficients.
			@param l			Order of SH coefficients. */
   
		double HARDIMeasure(int measure, double * coeff, int l);

		/** Return the long/short name of the specified measure.
			@param measure		Desired HARDI measure. */
  
		static const char * GetLongName(int measure);
		static const char * GetShortName(int measure);

		/** Compute the measure, given a set of spherical harmonics coefficients
			"coeff" of order "l". The first eight measures (V, GA, GFA, FMI, R0,
			R2, Ri and Iso) are relatively easy to compute, since they only depend 
			on the coefficients themselves. Therefore, they can be used as statics.
			@param coeff	SH coefficients.
			@param l		Order of SH coefficients. */

		static double Variance(double * coeff, int l);
		static double GeneralAnisotropy(double * coeff, int l);
		static double GeneralFractionalAnisotropy(double * coeff, int l);
		static double FractionalMultifiberIndex(double * coeff, int l);
		static double Rank0(double * coeff, int l);
		static double Rank2(double * coeff, int l);
		static double RankI(double * coeff, int l);
		static double IsotropicComponent(double * coeff, int l);	

		/** Compute the measure, given a set of spherical harmonics coefficients
			"coeff" of order "l". Three measures (NM, SE and CRE) require a tessellated
			sphere. In previous versions, tessellation was performed again and again for
			each voxel, which was wildly inefficient. Now, we only compute the tessellation
			once, which should speed things up for these measures. Since we store the
			result of the tessellation as a class variable, these functions are not static. 
			@param coeff	SH coefficients.
			@param l		Order of SH coefficients. */

		double NumberMaxima(double * coeff, int l);
		double ShannonEntropy(double * coeff, int l);
		double CummulativeResidualEntropy(double * coeff, int l);

/** TODO: We do not currently need this function. When we do, we should evaluate whether
	the HARDIMeasures class is the right place for it, and what the input arguments should be. 
		static double CalculateFTestForVoxelClassiffication(int ngrad, double ** gradDirrInSphericalCoordinates, double * ADCtrue, int lOld, int lCurrent, double * SHcoeffOld, double * SHcoeffCurrent);
*/

	private:

		/** Create a tessellation with a specific order, using the "sphereTessellator"
			class. The tessellation points are stored as spherical coordinates. 
			@param tessOrder	Desired tessellation order. */

		void getTessellation(int tessOrder = 5);

		/** Delete all spherical coordinate pairs in "tessellationPointsInSC". */

		void clearTessellation();

		/** Return the number of deformator elements higher than "delta_i", divided
			by the total number of deformator elements. 
			@param deformator	Vector containing deformator elements.
			@param delta_i		Threshold. */

		double getProbability(std::vector<double> * deformator, double delta_i);

		/** List of tessellation points. Computed once the first time the measure NM, SE or
			CRE is requested; on subsequent calls, the values stored in this vector are reused. */

		std::vector<double *> tessellationPointsInSC;

}; // class HARDIMeasures


} // namespace bmia


#endif // bmia_HARDIMeasures_h
