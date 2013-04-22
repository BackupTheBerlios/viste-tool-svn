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

/**
 * vtkFBO.h
 * by Tim Peeters
 *
 * 2008-03-02	Tim Peeters
 * - First version, based on vtkFBO.h
 */

#ifndef bmia_vtkFBO_h
#define bmia_vtkFBO_h

#include <vtkObject.h>
#include <vtkgl.h> // for GLuint

class vtkCamera;

namespace bmia {

class vtkUniformSampler;
class vtkMyShaderProgram;

/**
 * Helper class for rendering to Framebuffer objects (FBOs).
 */
class vtkFBO : public vtkObject {

public:
  static vtkFBO* New();

  /**
   * Initialize texture for storing&reading shadow map
   */
  void Initialize();

  vtkGetMacro(Initialized, bool);
//  vtkGetMacro(Texture, GLuint);
  GLuint GetTexture() { return this->GetTexture1(); };
  GLuint GetTexture1() { return this->Textures[0]; };
  GLuint GetTexture2() { return this->Textures[1]; };

  /**
   * The functions to call before and after rendering of the scene to the
   * shadow map to set up and bind the texture etc. Between the calling of these
   * functions the scene must be rendered as normal.
   */
  void PreRender(vtkCamera* lightCamera);
  void PostRender();

  vtkGetObjectMacro(Sampler, vtkUniformSampler);
  vtkGetObjectMacro(Sampler2, vtkUniformSampler);

  // TODO: change the implementation of the stuff below.
  // if the shadow map width or height is changed, the shadow map must be
  // initialized again!!!
  // XXX: perhaps just set ShadowTextureInitialized to false. and make sure
  // the old shadow map is thrown away.
//  vtkSetClampMacro(Width, GLuint, 2, 4096);
//  vtkSetClampMacro(Height, GLuint, 2, 4096);
//  vtkGetMacro(Width, GLuint);
//  vtkGetMacro(Height, GLuint);

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

//  void SetTwoTextures(bool twotextures);
  vtkSetMacro(TwoTextures, bool);
  vtkGetMacro(TwoTextures, bool);
  vtkBooleanMacro(TwoTextures, bool);  

  void ActiveTextures();

  GLuint* GetViewport()
    {
    return (GLuint*)(&(this->Viewport));
    }

  void SetViewport(int v[4])
    {
    for (int i=0; i < 4; i++)
      {
      this->Viewport[i] = v[i];
      }
    this->Modified();
    }

protected:
  vtkFBO();
  ~vtkFBO();

//  GLuint Width;
//  GLuint Height;
  GLuint Viewport[4];

  virtual void Clean();

private:
  /**
   * The number of the texture that is used for the shadow map.
   * TODO: use a FBO in the future.
   */
  GLuint Textures[2];
  GLuint FBO;
  GLuint RBO;
  GLuint ColorTexture;
  GLuint ColorRBO;
  GLuint DepthRBO;

  // Render to two textures instead of one.
  bool TwoTextures;

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
   * True if the texture shadow map has been initialized, and false otherwise.
   */
  bool Initialized;

  /**
   * The sampler that is used for the shadow map in the shaders.
   */
  vtkUniformSampler* Sampler;
  vtkUniformSampler* Sampler2;

  /**
   * Called by Initialize().
   */
  void TextureParameters();

}; // class vtkFBO
} // namespace bmia

#endif // bmia_vtkFBO_h
