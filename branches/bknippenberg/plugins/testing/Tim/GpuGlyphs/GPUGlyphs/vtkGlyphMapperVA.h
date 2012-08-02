/*
 *  vtkGlyphMapperVA.h
 *
 *  2008-02-27	Tim Peeters
 *  - First version of vtkGlyphMapper.
 *
 *  2008-09-03	Tim Peeters
 *  - First version of vtkGlyphMapperVA which uses vertex attributes to
 *    pass data to the shaders instead of textures.
 *
 * 2011-02-08	Evert van Aart
 * - Added support for transforming and coloring tensors.
 * - Cleaned up code, added comments.
 *
 * 2011-02-18	Evert van Aart
 * - Improved error handling for non-supported GPUs.
 *
 */


#ifndef bmia_vtkGlyphMapperVA_h
#define bmia_vtkGlyphMapperVA_h


/** Includes - Qt */

#include <QMessageBox>
#include <qdatastream.h>

/** Includes - VTK */

#include <vtkVolumeMapper.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPointSet.h>
#include <vtkOpenGLExtensionManager.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCamera.h>
#include <vtkScalarsToColors.h>
#include <vtkTimerLog.h>

/** Includes - Custom Files */

#include "HWShading/vtkUniformFloat.h"
#include "HWShading/vtkUniformIvec3.h"
#include "HWShading/vtkUniformVec3.h"
#include "HWShading/vtkMyShaderProgram.h"
#include "HWShading/vtkFBO.h"
#include "HWShading/vtkUniformSampler.h"
#include "HWShading/vtkUniformBool.h"

/** Includes - C++ */

#include <assert.h>


namespace bmia {


/** Superclass for GPU-based glyph rendering methods. As opposed to "vtkGlyphMapper", 
	volume data is not kept on GPU memory as a texture, but passed per vertex using vertex 
	attributes. The mappers for DTI glyphs and Spherical Harmonics HARDI glyphs inherit
	from this class. 
*/

class vtkGlyphMapperVA : public vtkVolumeMapper
{
	public:

		/** Render the GPU glyphs to the screen.
			@param ren		VTK renderer of the main window.
			@param vol		VTK actor for the glyphs. */

		virtual void Render(vtkRenderer * ren, vtkVolume * vol);

		/** Set the point set that defines the seed points.
			@param point	New seed point set. */
   
		void SetSeedPoints(vtkPointSet * points);

		/** Get the current seed point set. */
  
		vtkGetObjectMacro(SeedPoints, vtkPointSet);

		/** Set the maximum radius of the glyphs in any direction. This is used 
			for constructing the bounding boxes around the glyphs in "DrawPoints".
			@param r		Desired radius. */
  
		void SetMaxGlyphRadius(float r);

		/** Get the maximum glyph radius. */
  
		vtkGetMacro(MaxGlyphRadius, float);

		/** Set the scale of the glyphs. 
			@param scale	Desired scale. */
  
		void SetGlyphScaling(float scale);

		/** Get the glyph scale. */
  
		vtkGetMacro(GlyphScaling, float);

		/** Set the transformation matrix used to transform the glyphs.
			@param m		New matrix. */

		void setTransformationMatrix(vtkMatrix4x4 * m)
		{
			transformationMatrix = m;
		}

		/** Set the Anisotropy Index image, which can be used to color the glyphs. 
			@param rI		New AI image. */

		void setAIImage(vtkImageData * rI)
		{
			aiImage = rI;
		}

		/** Set the Look-Up Table (Transfer Function), used for coloring the glyphs. 
			@param rLUT		NEW LUT. */

		void setLUT(vtkScalarsToColors * rLUT)
		{
			lut = rLUT;
		}

	protected:
  
		/** Constructor */

		vtkGlyphMapperVA();

		/** Destructor */

		~vtkGlyphMapperVA();

		/** Initializes the shader uniforms. Called in the subclasses of this class, 
			after the shader programs have been set up. */
  
		virtual void SetupShaderUniforms();

		/** Update the uniforms of the shaders. Called before activating the shader 
			program, and before drawing points. Implemented in the subclasses. */
  
		virtual void UpdateShaderUniforms() 
		{

		};

		/** Check for the required OpenGL extensions and load them. When done, set 
			"this->Initialized" to true. The render window pointer is used to initialize
			the OpenGL extension manager, which is necessary on some systems.
			@param renwin		VTK render window. */
  
		void Initialize(vtkRenderWindow * renwin);

		/** Draw the points of the input data. This method draws bounding boxes around the 
			points such that the glyphs, with maximum radius "MaxGlyphRadius", will always fit
			in. In some cases (e.g., with lines), this function can be overridden to use a 
			more restricted bounding box. */
  
		virtual void DrawPoints() = 0;

		/** Transform a 3D vector using the transformation matrix stored in this class. The
			"translate" parameter indicated whether or not we should apply the translation 
			(fourth column of the matrix); this should be true for 3D coordinates, and false
			for vectors (e.g., eigenvectors). 
			@param p			3D vector.
			@param translate	Apply translation if true. */

		void transformPoint(double * p, bool translate = true);

		/** Shader Program used to draw the glyphs. */
  
		vtkMyShaderProgram * SP;

		/** Maximum radius of the glyphs, used to create bounding boxes. */
  
		double MaxGlyphRadius;

		/** Position of the camera. */

		double * EyePosition;

		/** Position of the camera (shader uniform). */
  
		vtkUniformVec3 * UEyePosition;

		/** Position of the light (shader uniform). */

		vtkUniformVec3 * ULightPosition;

		/** Maximum radius of the glyphs (shader uniform). */
		vtkUniformFloat * UMaxGlyphRadius;

		/** Scale of the glyphs  (shader uniform). */

		vtkUniformFloat * UGlyphScaling;

		/** Current camera of the active render window. This is set here in "Render", 
			and is later used in "vtkCHGlyphMapper", in "DrawPoints". */
  
		vtkCamera * CurrentCamera;

		/** Current transformation matrix. */
  
		vtkMatrix4x4 * transformationMatrix;

		/** Anisotropy Index image, used for coloring the glyphs. */

		vtkImageData * aiImage;

		/** Look-Up Table, used for coloring the glyphs. */

		vtkScalarsToColors * lut;

	private:

		/** Scale of the DTI glyphs. */

		double GlyphScaling;
  
		/** Set of points defining the positions of the glyphs. */

		vtkPointSet* SeedPoints;

		/** True if the "Initialize" function has been successfully called, 
			false otherwise. */

		bool Initialized;

		/** True if initialization of the mapper fails; this means that the GPU
			does not support the necessary components, making it impossible to render
			the GPU glyphs. 
		*/

		bool notSupported;

}; // class vtkGlyphMapperVA


} // namespace bmia


#endif // bmia_vtkGlyphMapperVA_h
