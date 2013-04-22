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
 * vtkCHGlyphMapper.h
 *
 * 2008-11-20	Tim Peeters
 * - First version
 *
 * 2011-01-07	Evert van Aart
 * - Made compatible with DTITool3, refactored some code.
 *
 * 2011-06-20	Evert van Aart
 * - Added support for LUTs.
 * - Removed support for measure range clamping and square rooting, as both things
 *   can be achieved with the Transfer Functions now.
 * - Measure values are pre-computed once now, which should smoothen camera movements
 *   when rendering SH glyphs.
 *
 * 2011-07-07	Evert van Aart
 * - After rendering the glyphs, put GL options that were disabled/enabled during
 *   rendering back to their original state. In particular, failing to re-enable
 *   blending cause problems with text rendering.
 *
 */


#ifndef bmia_vtkCHGlyphMapper_h
#define bmia_vtkCHGlyphMapper_h


/** Includes - Custom Files */

#include "vtkSHGlyphMapper.h"
#include "HWShading/vtkMyShaderProgram.h"
#include "HWShading/vtkMyShaderProgramReader.h"
#include "HWShading/vtkUniformVec3.h"
#include "HWShading/vtkUniformFloat.h"
#include "HWShading/vtkUniformInt.h"
#include "HWShading/vtkUniformVec3.h"
#include "HARDI/HARDIMeasures.h"

/** Includes - VTK */

#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkPointSet.h>
#include <vtkPointData.h>
#include <vtkMath.h>
#include <vtkCamera.h>

/** Includes - C++ */

#include <assert.h>


namespace bmia {


/** Mapper for GPU rendering of spherical harmonics glyphs using MvA's 
	Cylindrical Harmonics polynomials. Inherits from the default SH
	mapper, "vtkSHGlyphMapper".
*/

class vtkCHGlyphMapper : public vtkSHGlyphMapper
{
	public:
  
		/** Constructor Call */

		static vtkCHGlyphMapper * New();

	protected:
  
		/** Constructor */

		vtkCHGlyphMapper();

		/** Destructor */
  
		~vtkCHGlyphMapper();

		/** Read the shader program. */

		virtual void ReadShaderProgram();

		/** Draw glyphs at each seed point. */

	  virtual void DrawPoints();

	private:

	/** Maximum value of rho for each combination of "l" (order) and "m" (coefficient index
		within order). Multiply each amplitude "a_lm" with the returned coefficients and 
		add the result to obtain "r_max". 
		@param l			Order. 
		@param m			Coefficient index. */

	static float CylinderRhoMax(int l, int m);

	/** Maximum value of Z for each combination of "l" (order) and "m" (coefficient index
		within order). Multiply each amplitude "a_lm" with the returned coefficients and 
		add the result to obtain "z_max".
		@param l			Order. 
		@param m			Coefficient index. */

	static float CylinderZMax(int l, int m);

	/** Compute the maximum bounding cylinder radius for given coefficients. 
		@param coefficients	SH Coefficients. */

	float CylinderRMax(double * coefficients);

	/** Compute the maximum Z-range of the bounding cylinder for given coefficients.
		@param coefficients	SH Coefficients. */

	float CylinderZMax(double * coefficients);

}; // class vtkCHGlyphMapper


} // namespace bmia


#endif // bmia_vtkCHGlyphMapper
