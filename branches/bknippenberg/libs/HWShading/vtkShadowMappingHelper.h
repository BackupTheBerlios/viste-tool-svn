/**
 * vtkShadowMappigHelper.h
 * by Tim Peeters
 *
 * 2005-09-12	Tim Peeters
 * - First version
 *
 * 2005-11-22	Tim Peeters
 * - Added functions for getting&setting this->ShaderProgram.
 *
 * 2006-01-30	Tim Peeters
 * - Replaced #include <GL/glew.h> by inclusion of vtkgl.h.
 *
 * 2006-03-06	Tim Peeters
 * - Added functions SetupTextureMatrix() and RestoreTextureMatrix() and
 *   GLdouble StoredTextureMatrix[16] variable.
 */

#ifndef bmia_vtkShadowMappingHelper_h
#define bmia_vtkShadowMappingHelper_h

#include <vtkObject.h>

#include <vtkgl.h> // for GLuint

class vtkCamera;

namespace bmia {

class vtkUniformSampler;
class vtkMyShaderProgram;

/**
 * Helper class for shadow mapping. This class can be used for initializing
 * and rendering to the shadow map. The actual rendering must be done in a
 * seperate class (such as vtkShadowRenderer).
 */
class vtkShadowMappingHelper : public vtkObject {

public:
  static vtkShadowMappingHelper* New();

  /**
   * Initialize texture for storing&reading shadow map
   */
  void InitializeShadowMap();

  vtkGetMacro(ShadowTextureInitialized, bool);
  vtkGetMacro(ShadowTexture, GLuint);

  /**
   * The functions to call before and after rendering of the scene to the
   * shadow map to set up and bind the texture etc. Between the calling of these
   * functions the scene must be rendered as normal.
   */
  void PreShadowMapRender(vtkCamera* lightCamera);
  void PostShadowMapRender();

  vtkGetObjectMacro(ShadowMapSampler, vtkUniformSampler);

  // TODO: change the implementation of the stuff below.
  // if the shadow map width or height is changed, the shadow map must be
  // initialized again!!!
  // XXX: perhaps just set ShadowTextureInitialized to false. and make sure
  // the old shadow map is thrown away.
  vtkSetClampMacro(ShadowMapWidth, GLuint, 2, 2096);
  vtkSetClampMacro(ShadowMapHeight, GLuint, 2, 2096);

  /**
   * Can be used to replace the shader program used to render
   * the shadow map.
   */
  void SetShaderProgram(vtkMyShaderProgram* program);
  vtkGetObjectMacro(ShaderProgram, vtkMyShaderProgram);

  /**
   * Set-up the texture matrix such that the texture matrix can be used for
   * conversion between camera and light coordinates. This function should
   * only be called some time after {Pre,Post}ShadowMapRender were called.
   */
  void SetupTextureMatrix(vtkCamera* cam);

  /**
   * Restore the texture matrix to the state it was in before calling
   * SetupTextureMatrix().
   */
  void RestoreTextureMatrix();

protected:
  vtkShadowMappingHelper();
  ~vtkShadowMappingHelper();

  GLuint ShadowMapWidth;
  GLuint ShadowMapHeight;

private:
  /**
   * The number of the texture that is used for the shadow map.
   * TODO: use a FBO in the future.
   */
  GLuint ShadowTexture;
  GLuint ShadowFBO;
  GLuint ShadowRBO;
  GLuint ColorTexture;
  GLuint ColorRBO;
  GLuint DepthRBO;

  /**
   * Used to store the old viewport before changing it for the shadowmap.
   */
  GLint WindowViewport[4];

  GLclampf ColorClearValue[4];

  /**
   * Used by PreRenderShadowMap() to store a texture matrix that is later
   * used in SetupTextureMatrix().
   */
  GLdouble StoredTextureMatrix[16];

  /**
   * Shader program for rendering depth values to shadow map.
   */
  vtkMyShaderProgram* ShaderProgram;

  /**
   * True if the texture shadow map has been initialized, and false otherwise.
   */
  bool ShadowTextureInitialized;

  /**
   * The sampler that is used for the shadow map in the shaders.
   */
  vtkUniformSampler* ShadowMapSampler;

}; // class vtkShadowMappingHelper
} // namespace bmia

#endif // bmia_vtkShadowMappingHelper_h
