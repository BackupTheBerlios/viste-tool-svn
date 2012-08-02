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
 * vtkAnisotropicLightingSP.h
 * by Tim Peeters
 *
 * 2005-06-28	Tim Peeters
 * - First version
 *
 * 2006-12-26	Tim Peeters
 * - Add support for tone shading
 */

#ifndef bmia_vtkAnisotropicLightingSP_h
#define bmia_vtkAnisotropicLightingSP_h

#include "vtkMyShaderProgram.h"

namespace bmia {

class vtkUniformFloat;
class vtkUniformBool;
class vtkUniformVec3;
class vtkVertexShader;
class vtkFragmentShader;

/**
 * Shader program that applies anisotropic lighting realistic rendering
 * of lines. To be used by special OpenGL mappers that render lines.
 * Instead of passing a normal to OpenGL using glNormal(), glNormal() must
 * be used to pass the tangent direction of the line for each vertex.
 * The shader program will then itself compute the appropriate normal for
 * the given light(s) and eye positions and locations.
 * Currently only one light source is supported.
 */
class vtkAnisotropicLightingSP : public vtkMyShaderProgram {

public:
  static vtkAnisotropicLightingSP *New();

  /**
   * Get/Set the diffuse and specular contributions to the final lighting of
   * the line. Values must be between 0 and 1.
   * Note: VTK usually uses doubles, but in OpenGL 2.0 the uniforms for
   * shaders are floats, so floats are also used here.
   */
  void SetDiffuseContribution(float contribution);
  float GetDiffuseContribution() { return this->DiffuseContribution; }
  void SetSpecularContribution(float contribution);
  float GetSpecularContribution() { return this->SpecularContribution; }
  void SetAmbientContribution(float contribution);
  float GetAmbientContribution() { return this->AmbientContribution; }

  /**
   * Get/Set the specular power component. Can be used to generate sharper
   * or less-sharp highlights.
   * Note: VTK usually uses doubles, but in OpenGL 2.0 the uniforms for
   * shaders are floats, so floats are also used here.
   */
  void SetSpecularPower(float power);
  float GetSpecularPower() { return this->SpecularPower; }

  void SetRGBColoring(bool coloring);
  bool GetRGBColoring() { return this->RGBColoring; }

  void SetToneShading(bool tone);
  bool GetToneShading() { return this->ToneShading; }
  void SetWarmColor(double* rgb);
  void SetWarmColor(double red, double green, double blue);
  void SetCoolColor(double* rgb);
  void SetCoolColor(double red, double green, double blue);
  void GetWarmColor(double rgb[3]);
  void GetCoolColor(double rgb[3]);

protected:
  vtkAnisotropicLightingSP();
  ~vtkAnisotropicLightingSP();

  float SpecularPower;
  float DiffuseContribution;
  float SpecularContribution;
  float AmbientContribution;

  vtkUniformFloat* SpecularPowerUniform;
  vtkUniformFloat* DiffuseContributionUniform;
  vtkUniformFloat* SpecularContributionUniform;
  vtkUniformFloat* AmbientContributionUniform;

  /**
   * RGB encoding of local direction.
   */
  bool RGBColoring;
  vtkUniformBool* RGBColoringUniform;

  bool ToneShading;
  vtkUniformBool* ToneShadingUniform;
  vtkUniformVec3* WarmColorUniform;
  vtkUniformVec3* CoolColorUniform;

  vtkVertexShader* VertexShader;
  vtkFragmentShader* FragmentShader;
  vtkFragmentShader* ShaderFunctions;

private:

}; // class vtkAnisotropicLightingSP

} // namespace bmia

#endif // bmia_vtkAnisotropicLightingSP_h
