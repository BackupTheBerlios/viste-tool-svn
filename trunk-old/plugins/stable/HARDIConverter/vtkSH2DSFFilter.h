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
 * vtkSH2DSFFilter.h
 *
 * 2011-08-05	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_HARDIConverterPlugin_vtkSH2DSFFilter_h
#define bmia_HARDIConverterPlugin_vtkSH2DSFFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkMath.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>

/** Includes - Custom Files */

#include "HARDI/sphereTesselator.h"
#include "HARDI/SphereTriangulator.h"
#include "HARDI/HARDITransformationManager.h"


namespace bmia {


/** This filter converts a volume containing Spherical Harmonics data to one 
	containing Discrete Sphere Functions (DSF). The DSF image has three arrays:
	One array containing the spherical angles, one containing the values per 
	voxel and angle (i.e., the radius for that angle), and one containing the
	triangulation of the computed glyphs. 
*/

class vtkSH2DSFFilter : public vtkSimpleImageToImageFilter
{
	public:

		/** Constructor Call */

		static vtkSH2DSFFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkSH2DSFFilter, vtkSimpleImageToImageFilter);

		/** Execute the filter.
			@param input	Input SH image.
			@param output	Output DSF image. */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

		/** Set the order of tessellation.
			@param rOder	Desired tessellation order. */

		void setTessOrder(int rOrder)
		{
			tessOrder = rOrder;
		}

	protected:

		/** Constructor. */

		vtkSH2DSFFilter();

		/** Destructor. */

		~vtkSH2DSFFilter();

	private:

		/** Tessellation order. */

		int tessOrder;


}; // vtkSH2DSFFilter


} // namespace bmia


#endif // bmia_HARDIConverterPlugin_vtkSH2DSFFilter_h