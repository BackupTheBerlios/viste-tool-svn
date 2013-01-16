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
 * vtkScalarVolumeToSeedPoints.h
 *
 * 2011-05-10	Evert van Aart
 * - First version
 *
 */


#ifndef bmia_SeedingPlugin_vtkScalarVolumeToSeedPoints_h
#define bmia_SeedingPlugin_vtkScalarVolumeToSeedPoints_h


/** Includes - VTK */

#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkObjectFactory.h>


namespace bmia {


/** Simple filter that generates a seed point for every voxel with a scalar value
	that lies between a lower and an upper threshold value. Used for volume seeding.
*/

class vtkScalarVolumeToSeedPoints : public vtkDataSetToUnstructuredGridFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkScalarVolumeToSeedPoints, vtkDataSetToUnstructuredGridFilter);

		/** Constructor Call */

		static vtkScalarVolumeToSeedPoints * New();

		/** Set the lower threshold.
			@param rT			Desired lower threshold. */

		void setMinThreshold(double rT)
		{
			minThreshold = rT;
		}

		/** Set the upper threshold.
			@param rT			Desired upper threshold. */

		void setMaxThreshold(double rT)
		{
			maxThreshold = rT;
		}

	protected:

		/** Constructor */

		vtkScalarVolumeToSeedPoints();

		/** Destructor */

		~vtkScalarVolumeToSeedPoints();

		/** Main entry point for the execution of the filter. */

		void Execute();

		/** Lower threshold. */

		double minThreshold;

		/** Upper threshold. */

		double maxThreshold;

}; // class vtkScalarVolumeToSeedPoints


} // namespace bmia


#endif // bmia_SeedingPlugin_vtkScalarVolumeToSeedPoints_h