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
 * vtkDTIGlyphMapperVA.h
 *
 * 2008-09-01	Tim Peeters
 * - First version
 *
 * 2011-02-08	Evert van Aart
 * - Added support for coloring the glyphs using LUT or weighted RGB.
 * - Cleaned up the code, added comments.
 *
 * 2011-07-07	Evert van Aart
 * - After rendering the glyphs, put GL options that were disabled/enabled during
 *   rendering back to their original state. In particular, failing to re-enable
 *   blending cause problems with text rendering.
 *
 */


#ifndef bmia_vtkDTIGlyphMapperVA_h
#define bmia_vtkDTIGlyphMapperVA_h


/** Includes - Custom Files */

#include "vtkGlyphMapperVA.h"
#include "HWShading/vtkUniformSampler.h"
#include "HWShading/vtkMyShaderProgram.h"
#include "HWShading/vtkMyShaderProgramReader.h"
#include "HWShading/vtkUniformFloat.h"

/** Includes - VTK */

#include <vtkgl.h>
#include <vtkObjectFactory.h>
#include <vtkPointSet.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkMath.h>


namespace bmia {


/** Mapper for GPU rendering of DTI glyphs. The input image is assumed to contain
	the eigenvectors and eigenvalues of the DTI tensors, which are used for 
	constructing and coloring the glyphs. The glyphs can be colored using regular
	RGB (depending on the orientation), weighted RGB (where the weight affects 
	either the lightness or the saturation), and using a Look-Up Table. 
*/

class vtkDTIGlyphMapperVA : public vtkGlyphMapperVA
{
	public:
  
		/** Constructor Call */

		static vtkDTIGlyphMapperVA * New();

		/** Methods for coloring the DTI glyphs. */

		enum ColoringMethod
		{
			CM_RGB = 0,		/**< RGB Coloring (Orientation). */
			CM_WRGBA,		/**< Weighted RGB (Saturation). */
			CM_WRGBB,		/**< Weighted RGB (Lightness). */
			CM_LUT			/**< Look-Up Table. */
		};

		/** Set the coloring method.
			@param rM	 New coloring method. */

		void setColoringMethod(int rM)
		{
			currentColoringMethod = (ColoringMethod) rM;
		}
  
		/** Set the coloring method.
			@param rM	 New coloring method. */

		void setColoringMethod(ColoringMethod rM)
		{
			currentColoringMethod = rM;
		}

	protected:
  
		/** Constructor */

		vtkDTIGlyphMapperVA();

		/** Destructor */

		~vtkDTIGlyphMapperVA();

		/** Draw the DTI glyphs. */

		virtual void DrawPoints();

	private:

		/** Currently selected coloring method. */

		ColoringMethod currentColoringMethod;

		/** Hard-coded value equal to the square root of one third;
			used for saturation-based color weighing. */

		double SQRT13;

}; // class vtkDTIGlyphMapperVA


} // namespace bmia


#endif // bmia_vtkDTIGlyphMapperVA_h
