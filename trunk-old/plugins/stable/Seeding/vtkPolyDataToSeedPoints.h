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
 * vtkPolyDataToSeedPoints.h
 *
 * 2011-04-18	Evert van Aart
 * - First Version.
 *
 */


#ifndef bmia_SeedingPlugin_vtkPolyDataToSeedPoints_h
#define bmia_SeedingPlugin_vtkPolyDataToSeedPoints_h


/** Includes - VTK */

#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkObjectFactory.h>


namespace bmia {


/** Very simple class that takes a "vtkPolyData" object as input, and uses it to
	generate a "vtkUnstructuredGrid" object with all points of the polydata. There
	are easier ways to do this, but using this filter has the advantage that the
	pipeline will automatically update if the fibers change. 
*/

class vtkPolyDataToSeedPoints : public vtkDataSetToUnstructuredGridFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkPolyDataToSeedPoints, vtkDataSetToUnstructuredGridFilter);

		/** Constructor Call */

		static vtkPolyDataToSeedPoints * New();

		/** Constructor */

		vtkPolyDataToSeedPoints()
		{
			
		};

		/** Destructor */

		~vtkPolyDataToSeedPoints()
		{
			
		};

	protected:

		/** Main point of entry of the filter. */

		void Execute();

};
		

} // namespace


#endif // bmia_SeedingPlugin_vtkPolyDataToSeedPoints_h
