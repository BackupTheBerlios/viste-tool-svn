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
