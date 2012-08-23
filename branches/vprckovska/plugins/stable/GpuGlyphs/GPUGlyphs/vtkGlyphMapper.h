/**
 *  vtkGlyphMapper.h
 *  by Tim Peeters
 *
 *  2008-02-27	Tim Peeters
 *  - First version
 */

#ifndef bmia_vtkGlyphMapper_h
#define bmia_vtkGlyphMapper_h

#include <vtkVolumeMapper.h>
class vtkPointSet;

namespace bmia {

class vtkMyShaderProgram;
class vtkUniformFloat;
class vtkUniformIvec3;
class vtkUniformVec3;
class vtkFBO;
class vtkUniformSampler;
class vtkUniformBool;

/**
 * Superclass for GPU-based glyph rendering methods.
 * Volume data is stored in GPU memory using 3D textures.
 */
class vtkGlyphMapper : public vtkVolumeMapper
{
public:

  virtual void Render(vtkRenderer *ren, vtkVolume *vol);

  /**
   * Set/Get the PointSet that defines the seed points
   * for the glyphs to render.
   */
  void SetSeedPoints(vtkPointSet* points);
  vtkGetObjectMacro(SeedPoints, vtkPointSet);

  /**
   * Set/Get the maximum radius of the glyphs in any direction.
   * This is used for constructing the bounding boxes around the
   * glyphs in DrawPoints.
   */
  void SetMaxGlyphRadius(float r);
  vtkGetMacro(MaxGlyphRadius, float);

  void SetNegateX(bool negate);
  void SetNegateY(bool negate);
  void SetNegateZ(bool negate);
  bool GetNegateX();
  bool GetNegateY();
  bool GetNegateZ();

  void PrintNumberOfSeedPoints();

  void SetGlyphScaling(float scale);
  vtkGetMacro(GlyphScaling, float);

protected:
  vtkGlyphMapper();
  ~vtkGlyphMapper();
  /**
   * To be called in subclasses after the shader programs have
   * been set-up.
   */
  virtual void SetupShaderUniforms();

  /**
   * Check for the needed OpenGL extensions and load them.
   * When done, set this->Initialized.
   */
  void Initialize();

  /**
   * Load all needed textures.
   * This function can call LoadTexture(...).
   */
  virtual void LoadTextures() = 0;

 /**
  * Load texture and bind it to index texture_index.
  * The texture data comes from the input ImageData in array
  * with name array_name.
  */  
  void LoadTexture(int texture_index, const char* array_name);

  /**
   * Draw the points of the input data.
   * This method draws bounding boxes around the points such that
   * the glyphs, with maximum radius MaxGlyphRadius will always fit
   * in. In some cases (e.g. with lines), this function can be
   * overridden to use a more restricted bounding box.
   */
  virtual void DrawPoints();

  /**
   * Shader program for rendering to depth buffer.
   * No lighting calculations need to be done here.
   */
  vtkMyShaderProgram* SPDepth;

  /**
   * Shader program for rendering the final scene to the screen.
   */
  vtkMyShaderProgram* SPLight;

  /**
   * Call this after activating the shader program!
   */
  virtual void SetTextureLocations() {};
  void SetTextureLocation(vtkMyShaderProgram* sp, vtkUniformSampler* sampler); //, const char* texture_name);

  /**
   * Draw screen-filling quad ;)
   */
//  void DrawScreenFillingQuad(int width, int height);
  void DrawScreenFillingQuad(int viewport[4]);

private:

  double GlyphScaling;
  double MaxGlyphRadius;
  vtkPointSet* SeedPoints;

  vtkUniformVec3* UEyePosition;
  vtkUniformVec3* ULightPosition;
  vtkUniformIvec3* UTextureDimensions;
  vtkUniformVec3* UTextureSpacing;
  vtkUniformFloat* UMaxGlyphRadius;
  vtkUniformFloat* UGlyphScaling;

//  vtkShadowMappingHelper* ShadowMappingHelper;
  vtkFBO* FBODepth;
  vtkFBO* FBOShadow;

  bool Initialized;

  vtkUniformBool* UNegateX;
  vtkUniformBool* UNegateY;
  vtkUniformBool* UNegateZ;

}; // class vtkGlyphMapper
} // namespace bmia
#endif // bmia_vtkGlyphMapper_h
