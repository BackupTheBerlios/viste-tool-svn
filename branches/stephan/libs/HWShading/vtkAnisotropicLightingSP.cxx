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
 * vtkAnisotropicLightingSP.cxx
 * by Tim Peeters
 *
 * 2005-06-28	Tim Peeters
 * - First version
 *
 * 2005-09-13	Tim Peeters
 * - Use fragment shader for lighting calculatins instead of vertex shader.
 *
 * 2006-12-26	Tim Peeters
 * - Add support for tone shading.
 */

#include "vtkAnisotropicLightingSP.h"
#include "AnisotropicLightingVertexText.h"
#include "AnisotropicLightingFragmentText.h"
#include "AnisotropicLightingFunctionsText.h"

#include "vtkVertexShader.h"
#include "vtkFragmentShader.h"
#include "vtkUniformFloat.h"
#include "vtkUniformBool.h"
#include "vtkUniformVec3.h"

//#include "vtkShaderDir.h"

#include <vtkObjectFactory.h>
#include <vtkstd/string>

namespace bmia {

vtkStandardNewMacro(vtkAnisotropicLightingSP);

vtkAnisotropicLightingSP::vtkAnisotropicLightingSP()
{
  // float variables
  this->SpecularPower = 30.0;
  this->AmbientContribution = 0.2;
  this->DiffuseContribution = 0.6;
  this->SpecularContribution = 0.4;

  this->VertexShader = vtkVertexShader::New();
  this->ShaderFunctions = vtkFragmentShader::New();
  this->FragmentShader = vtkFragmentShader::New();

  this->VertexShader->SetSourceText(AnisotropicLightingVertexText);
  this->ShaderFunctions->SetSourceText(AnisotropicLightingFunctionsText);
  this->FragmentShader->SetSourceText(AnisotropicLightingFragmentText);

  this->AddShaderObject(this->VertexShader);
  this->AddShaderObject(this->ShaderFunctions);
  this->AddShaderObject(this->FragmentShader);

  // Initialize uniforms
  this->SpecularPowerUniform = vtkUniformFloat::New();
  this->SpecularPowerUniform->SetName("SpecularPower");
  this->SpecularPowerUniform->SetValue(this->SpecularPower);
  this->AddShaderUniform(this->SpecularPowerUniform);

  this->DiffuseContributionUniform = vtkUniformFloat::New();
  this->DiffuseContributionUniform->SetName("DiffuseContribution");
  this->DiffuseContributionUniform->SetValue(this->DiffuseContribution);
  this->AddShaderUniform(this->DiffuseContributionUniform);

  this->SpecularContributionUniform = vtkUniformFloat::New();
  this->SpecularContributionUniform->SetName("SpecularContribution");
  this->SpecularContributionUniform->SetValue(this->SpecularContribution);
  this->AddShaderUniform(this->SpecularContributionUniform);

  this->AmbientContributionUniform = vtkUniformFloat::New();
  this->AmbientContributionUniform->SetName("AmbientContribution");
  this->AmbientContributionUniform->SetValue(this->AmbientContribution);
  this->AddShaderUniform(this->AmbientContributionUniform);

  this->RGBColoring = false;
  this->RGBColoringUniform = vtkUniformBool::New();
  this->RGBColoringUniform->SetName("RGBColoring");
  this->RGBColoringUniform->SetValue(this->RGBColoring);
  this->AddShaderUniform(this->RGBColoringUniform);

  this->ToneShading = false;
  this->ToneShadingUniform = vtkUniformBool::New();
  this->ToneShadingUniform->SetName("ToneShading");
  this->ToneShadingUniform->SetValue(this->ToneShading);
  this->AddShaderUniform(this->ToneShadingUniform);

  this->WarmColorUniform = vtkUniformVec3::New();
  this->WarmColorUniform->SetName("WarmColor");
  this->WarmColorUniform->SetValue(1.0, 0.8, 0.0);
  this->AddShaderUniform(this->WarmColorUniform);
  this->CoolColorUniform = vtkUniformVec3::New();
  this->CoolColorUniform->SetName("CoolColor");
  this->CoolColorUniform->SetValue(0.0, 0.0, 0.8);
  this->AddShaderUniform(this->CoolColorUniform);
}

vtkAnisotropicLightingSP::~vtkAnisotropicLightingSP()
{
  this->SpecularPowerUniform->Delete();
  this->SpecularPowerUniform = NULL;
  this->DiffuseContributionUniform->Delete();
  this->DiffuseContributionUniform = NULL;
  this->SpecularContributionUniform->Delete();
  this->SpecularContributionUniform = NULL;
  this->AmbientContributionUniform->Delete();
  this->AmbientContributionUniform = NULL;
  this->RGBColoringUniform->Delete();
  this->RGBColoringUniform = NULL;
  this->ToneShadingUniform->Delete();
  this->ToneShadingUniform = NULL;
  this->WarmColorUniform->Delete();
  this->WarmColorUniform = NULL;
  this->CoolColorUniform->Delete();
  this->CoolColorUniform = NULL;

  this->VertexShader->Delete();
  this->VertexShader = NULL;
  this->FragmentShader->Delete();
  this->FragmentShader = NULL;
  this->ShaderFunctions->Delete();
  this->ShaderFunctions = NULL;
}

void vtkAnisotropicLightingSP::SetSpecularPower(float power)
{
  vtkDebugMacro(<<"Setting specular power to "<<power);
  if (power != this->SpecularPower)
    {
    this->SpecularPower = power;
    this->SpecularPowerUniform->SetValue(this->SpecularPower);
    if (this->IsLinked())
      {
      this->SetGlUniform(this->SpecularPowerUniform);
      }
    // commented Modified() out otherwise the shader program is being re-linked
    // when this value is changed.
    //this->Modified();
    }
}

void vtkAnisotropicLightingSP::SetDiffuseContribution(float contribution)
{
  vtkDebugMacro(<<"Setting diffuse contribution to "<<contribution);
  if (contribution != this->DiffuseContribution)
    {
    this->DiffuseContribution = contribution;
    this->DiffuseContributionUniform->SetValue(this->DiffuseContribution);
    if (this->IsLinked())
      {
      this->SetGlUniform(this->DiffuseContributionUniform);
      }
    //this->Modified();
    }
}

void vtkAnisotropicLightingSP::SetSpecularContribution(float contribution)
{
  vtkDebugMacro(<<"Setting specular contribution to "<<contribution);
  if (contribution != this->SpecularContribution)
    {
    this->SpecularContribution = contribution;
    this->SpecularContributionUniform->SetValue(this->SpecularContribution);
    if (this->IsLinked())
      {
      this->SetGlUniform(this->SpecularContributionUniform);
      }
    //this->Modified();
    }
}

void vtkAnisotropicLightingSP::SetAmbientContribution(float contribution)
{
  vtkDebugMacro(<<"Setting ambient contribution to "<<contribution);
  if (contribution != this->AmbientContribution)
    {
    this->AmbientContribution = contribution;
    this->AmbientContributionUniform->SetValue(this->AmbientContribution);
    if (this->IsLinked())
      {
      this->SetGlUniform(this->AmbientContributionUniform);
      }
    // this->Modified();
  } // if
}

void vtkAnisotropicLightingSP::SetRGBColoring(bool coloring)
{
  vtkDebugMacro(<<"Setting RGB coloring to "<<coloring);
  if (coloring != this->RGBColoring)
    {
    this->RGBColoring = coloring;
    this->RGBColoringUniform->SetValue(this->RGBColoring);
    if (this->IsLinked())
      {
      this->SetGlUniform(this->RGBColoringUniform);
      } // if
    } // if
}

void vtkAnisotropicLightingSP::SetToneShading(bool tone)
{
  vtkDebugMacro(<<"Setting tone shading to "<<tone);
  if (tone != this->ToneShading)
    {
    this->ToneShading = tone;
    this->ToneShadingUniform->SetValue(this->ToneShading);
    if (this->IsLinked())
      {
      this->SetGlUniform(this->ToneShadingUniform);
      } // if
    } // if
}

void vtkAnisotropicLightingSP::SetWarmColor(double red, double green, double blue)
{
  vtkDebugMacro(<<"Setting warm color to "<<red<<", "<<green<<", "<<blue<<".");
  this->WarmColorUniform->SetValue(red, green, blue);
  if (this->IsLinked())
    {
    this->SetGlUniform(this->WarmColorUniform);
    } // if
}

void vtkAnisotropicLightingSP::SetWarmColor(double* rgb)
{
  this->SetWarmColor(rgb[0], rgb[1], rgb[2]);
}

void vtkAnisotropicLightingSP::GetWarmColor(double rgb[3])
{
  float* val = this->WarmColorUniform->GetValue();
  for (int i=0; i < 3; i++) rgb[i] = (double) val[i];
}

void vtkAnisotropicLightingSP::SetCoolColor(double red, double green, double blue)
{
  vtkDebugMacro(<<"Setting cool color to "<<red<<", "<<green<<", "<<blue<<".");
  this->CoolColorUniform->SetValue(red, green, blue);
  if (this->IsLinked())
    {
    this->SetGlUniform(this->CoolColorUniform);
    } // if
}

void vtkAnisotropicLightingSP::SetCoolColor(double* rgb)
{
  this->SetCoolColor(rgb[0], rgb[1], rgb[2]);
}

void vtkAnisotropicLightingSP::GetCoolColor(double rgb[3])
{
  float* val = this->CoolColorUniform->GetValue();
  for (int i=0; i < 3; i++) rgb[i] = (double) val[i];
}


} // namespace bmia
