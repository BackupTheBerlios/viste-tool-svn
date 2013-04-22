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
 * vtkShadowRenderer.h
 * by Tim Peeters
 *
 * 2005-07-19	Tim Peeters
 * - First version
 *
 * 2005-08-08	Tim Peeters
 * - Renamed ShowShadowMap() to DrawShadowMap().
 * - Added bool ShowShadowMap variable and functions to get/set it.
 *
 * 2006-01-30	Tim Peeters
 * - Removed #include <GL/glew.h>
 */

#ifndef bmia_vtkShadowRenderer_h
#define bmia_vtkShadowRenderer_h

#include <vtkOpenGLRenderer.h>
//#include <GL/glew.h>

#include <vtkTimeStamp.h>

class vtkCamera;

namespace bmia {

//class vtkShadowMappingSP;
class vtkMyShaderProgram;
class vtkUniformSampler;
class vtkShadowMappingHelper;

/**
 * Class for rendering scenes with shadowing.
 * WARNING: glewInit() must be called after the render window was
 * initialized and before this renderer renders.
 */
class vtkShadowRenderer : public vtkOpenGLRenderer {

public:
  static vtkShadowRenderer* New();

  /**
   * Get/Set the rendering of shadows by this renderer.
   */
  vtkSetMacro(RenderShadows, bool);
  vtkGetMacro(RenderShadows, bool);
  vtkBooleanMacro(RenderShadows, bool);
  void SetShadows(bool shadows) { this->SetRenderShadows(shadows); };
  bool GetShadows() { return this->GetRenderShadows(); };
  vtkBooleanMacro(Shadows, bool);

  /**
   * Specifies whether the shadow map must be drawn on the screen or
   * the scene itself. This option only has an effect if RenderShadows
   * is true. If RenderShadows is false, the scene is always drawn with
   * no shadows, and no shadowmap is generated or displayed.
   */
  vtkSetMacro(ShowShadowMap, bool);
  vtkGetMacro(ShowShadowMap, bool);
  vtkBooleanMacro(ShowShadowMap, bool);

  // TODO: make this protected? it's protected in vtkShadowRenderer
  // from Sandia.. why?
  virtual void DeviceRender();

protected:
  vtkShadowRenderer();
  ~vtkShadowRenderer();

  
private:
  /**
   * Helper for rendering to the shadow map and creating shadow mapping
   * texture.
   */
  vtkShadowMappingHelper* ShadowMappingHelper;

  /**
   * Specifies whether shadows must be rendered or not.
   */
  bool RenderShadows;

  /**
   * Initialize texture for storing&reading shadow map.
   */
  void InitializeShadowMap();

  /**
   * Render to the shadow map from the light.
   */
  void RegenerateShadowMap();

  /**
   * Renders the scene using the shadow map for shadows.
   */
  void RenderWithShadows();

  /**
   * Display the shadow map.
   */
  void DrawShadowMap();

  /**
   * Shader program used for the shadow mapping.
   */
//  vtkShadowMappingSP* ShadowMappingSP;
  vtkMyShaderProgram* ShadowMappingSP;

  /**
   * The last time the shadow map was (re)generated.
   */
  vtkTimeStamp ShadowMapMTime;

  virtual unsigned long GetShadowMapMTime();

  /**
   * When true, the shadowmap is shown by the DeviceRender() function
   * instead of the scene itself.
   */
  bool ShowShadowMap;

};

} // namespace bmia

#endif // bmia_vtkShadowRenderer_h
