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
