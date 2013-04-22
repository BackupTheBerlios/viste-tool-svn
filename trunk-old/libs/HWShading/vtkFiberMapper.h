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
 * vtkFiberMapper.h
 * by Tim Peeters
 *
 * 2005-06-22	Tim Peeters
 * - First version
 *
 * 2005-06-29	Tim Peeters
 * - Made GetShaderProgram() public so that parameters can be set directly
 *   on the shader program.
 *
 * 2005-07-18	Tim Peeters
 * - Renamed class from vtkOpenGLFiberMapper to vtkFiberMapper.
 *
 * 2005-09-13	Tim Peeters
 * - Private variable ShaderProgram is now of type
 *   vtkAnisoLiShadowMapSP instead of vtkAnisotropicLightingSP
 * - Added private ShadowMappingHelper variable.
 * - Added RegenerateShadowMap()
 *
 * 2005-10-19	Tim Peeters
 * - Added variable bool Shadowing and functions SetShadowing(),
 *   ShadowingOn() and ShadowingOff() so that it is possible to switch off
 *   (and on) shadows.
 *
 * 2006-02-22	Tim Peeters
 * - Added Initialize() function and Initialized variable.
 * - Added GetInitialized() function.
 * - Added IsRenderSupported() function.
 *
 * 2006-03-05	Tim Peeters
 * - Removed GetShaderProgram() and GetShaderProgramShadowS() functions.
 * - Added functions for setting lighting parameters. This could first
 *   be done by getting the shader programs of this mapper and then
 *   setting the parameters of those shading programs. This is no
 *   longer supported.
 *
 * 2006-12-26	Tim Peeters
 * - Add support for tone shading.
 *
 * 2010-07-23	Tim Peeters
 * - Add TypeMacro so that vtkFiberMapper::SafeDownCast can be called.
 */

#ifndef bmia_vtkFiberMapper_h
#define bmia_vtkFiberMapper_h

#include <vtkOpenGLPolyDataMapper.h>

class vtkRenderer;
class vtkActor;
class vtkPoints;
class vtkCellArray;
class vtkRenderWindow;

namespace bmia {

class vtkAnisotropicLightingSP;
class vtkAnisoLiShadowMapSP;
class vtkMyShaderProgram;
class vtkShadowMappingHelper;

/**
 * Renders fibers in a better way.
 * Inherits from vtkOpenGLPolyDataMapper and overrides Draw() to
 * do the drawing of lines in a "better" way.
 * TODO: describe in *which* way.
 */
class vtkFiberMapper : public vtkOpenGLPolyDataMapper
{
public:
  static vtkFiberMapper *New();
  vtkTypeMacro(bmia::vtkFiberMapper,vtkOpenGLPolyDataMapper);

  // Description:
  // Draw method for OpenGL.
  //virtual int Draw(vtkRenderer *ren, vtkActor *a);
  virtual void Render(vtkRenderer *ren, vtkActor *a);

  /**
   * Enable/Disable anisotropic lighting for the lines. If anisotropic
   * lighting is disabled, all lines will be drawn without any lighting.
   * Switching lighting off also disables shadowing!
   */
  void SetLighting(bool light);
  vtkGetMacro(Lighting, bool);
  vtkBooleanMacro(Lighting, bool);

  /**
   * Enable/Disable shadowing.
   */
  void SetShadowing(bool render_shadows);
  vtkGetMacro(Shadowing, bool);
  vtkBooleanMacro(Shadowing, bool);

  /**
   * Enable/Disable tone shading.
   */
  void SetToneShading(bool tone_shading);
  vtkGetMacro(ToneShading, bool);
  vtkBooleanMacro(ToneShading, bool);

  void SetCoolColor(float r, float g, float b);
  void SetWarmColor(float r, float g, float b);

  /**
   * Set the thickness in pixels in the shadow map of a shadow
   * cast by a fiber.
   */
  vtkSetMacro(ShadowLineWidth, float);
  vtkGetMacro(ShadowLineWidth, float);

//  vtkGetObjectMacro(ShaderProgram, vtkAnisotropicLightingSP);
//  vtkGetObjectMacro(ShaderProgramShadow, vtkAnisoLiShadowMapSP);

  /**
   * Set/Get the (anisotropic) lighting parameters.
   */
  void SetAmbientContribution(float contribution);
  void SetDiffuseContribution(float contribution);
  void SetSpecularContribution(float contribution);
  void SetSpecularPower(float power);
  void SetAmbientContributionShadow(float contribution);
  void SetDiffuseContributionShadow(float contribution);
  void SetSpecularContributionShadow(float contribution);
//  void SetSpecularPowerShadow(float power);
  float GetAmbientContribution();
  float GetDiffuseContribution();
  float GetSpecularContribution();
  float GetSpecularPower();
  float GetAmbientContributionShadow();
  float GetDiffuseContributionShadow();
  float GetSpecularContributionShadow();
//  float GetSpecularPowerShadow();

  void SetRGBColoring(bool coloring);
  vtkGetMacro(RGBColoring, bool);
  vtkBooleanMacro(RGBColoring, bool);

  vtkGetMacro(Initialized, bool);

  /**
   * Returns true if the OpenGL extensions required by this mapper are
   * available, and false otherwise.
   */
  bool IsRenderSupported();

protected:
  vtkFiberMapper();
  ~vtkFiberMapper();

  float ShadowLineWidth;

  /**
   * Draw linestrips.
   *
   * @param points The points in the linestrips.
   * @param lineStrips The linestrips to draw.
   * @param ren The renderer to use for checking the abort status of the render window.
   * @param noAbort Abort status.
   */
  virtual void DrawLines(vtkPoints* points, vtkCellArray* lineStrips,
			vtkRenderer* ren, int &noAbort);

  bool Initialized;
  void Initialize(vtkRenderWindow * renwin);

private:

  /**
   * Specifies whether shadowing must be enabled or disabled.
   */
  bool Shadowing;

  vtkAnisotropicLightingSP* ShaderProgram;
  vtkAnisoLiShadowMapSP* ShaderProgramShadow;

  /**
   * True if anisotropic lighting must be enabled, and false otherwise.
   */
  bool Lighting;

  /**
   * Specifies whether tone shading must be enabled or disabled.
   */
  bool ToneShading;

  bool RGBColoring;

  /**
   * Helps with the rendering of the shadow map.
   */
  vtkShadowMappingHelper* ShadowMappingHelper;

  /**
   * Render the scene to the shadow map.
   */
  //void RegenerateShadowMap();
  virtual void RegenerateShadowMap(vtkPoints* points, vtkCellArray* lineStrips,
			vtkRenderer* ren, int &noAbort);

public:
  static void DrawShadowMap(); //GLfloat windowWidth, GLfloat windowHeight);

//  vtkFiberMapper(const vtkFiberMapper&); // Not implemented
//  void operator=(const vtkFiberMapper&); // Not Implemented

}; // class vtkFiberMapper

} // namespace bmia

#endif
