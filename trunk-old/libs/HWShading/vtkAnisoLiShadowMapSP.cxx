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
 * vtkAnisoLiShadowMap.cxx
 * by Tim Peeters
 *
 * 2005-09-13	Tim Peeters
 * - First version
 *
 * 2005-10-19	Tim Peeters
 * - Because this class is now a subclass of vtkAnisotropicLightingSP,
 *   functions and variables already in that class were removed here.
 *   It is almost empty now. Except for the two lines in the constructor.
 */

#include "vtkAnisoLiShadowMapSP.h"
#include "AnisotropicLightingShadowMappingVertexText.h"
#include "AnisotropicLightingShadowMappingFragmentText.h"

#include "vtkVertexShader.h"
#include "vtkFragmentShader.h"

#include "vtkUniformFloat.h"

#include <vtkObjectFactory.h>
//#include <vtkstd/string>

namespace bmia {

vtkStandardNewMacro(vtkAnisoLiShadowMapSP);

vtkAnisoLiShadowMapSP::vtkAnisoLiShadowMapSP()
{
  this->VertexShader->SetSourceText(AnisotropicLightingShadowMappingVertexText);
  this->FragmentShader->SetSourceText(AnisotropicLightingShadowMappingFragmentText);

  this->AmbientContributionShadow = 0.5*this->AmbientContribution;
  this->DiffuseContributionShadow = 0.5*this->DiffuseContribution;
  this->SpecularContributionShadow = 0.0;

  this->AmbientContributionShadowUniform = vtkUniformFloat::New();
  this->AmbientContributionShadowUniform->SetName("AmbientContributionShadow");
  this->AmbientContributionShadowUniform->SetValue(this->AmbientContributionShadow);
  this->AddShaderUniform(this->AmbientContributionShadowUniform);

  this->DiffuseContributionShadowUniform = vtkUniformFloat::New();
  this->DiffuseContributionShadowUniform->SetName("DiffuseContributionShadow");
  this->DiffuseContributionShadowUniform->SetValue(this->DiffuseContributionShadow);
  this->AddShaderUniform(this->DiffuseContributionShadowUniform);

  this->SpecularContributionShadowUniform = vtkUniformFloat::New();
  this->SpecularContributionShadowUniform->SetName("SpecularContributionShadow");
  this->SpecularContributionShadowUniform->SetValue(this->SpecularContributionShadow);
  this->AddShaderUniform(this->SpecularContributionShadowUniform);
}

vtkAnisoLiShadowMapSP::~vtkAnisoLiShadowMapSP()
{
  this->AmbientContributionShadowUniform->Delete();
  this->AmbientContributionShadowUniform = NULL;
  this->DiffuseContributionShadowUniform->Delete();
  this->DiffuseContributionShadowUniform = NULL;
  this->SpecularContributionShadowUniform->Delete();
  this->SpecularContributionShadowUniform = NULL;
}

void vtkAnisoLiShadowMapSP::SetAmbientContributionShadow(float contribution)
{
  vtkDebugMacro(<<"Setting ambient contribution shadow to "<<contribution);
  if (contribution != this->AmbientContributionShadow)
    {
    this->AmbientContributionShadow = contribution;
    this->AmbientContributionShadowUniform->SetValue(this->AmbientContributionShadow);
    if (this->IsLinked())
      {
//      this->SetGlUniform(this->AmbientContributionShadowUniform);
      }
    // this->Modified();
  } // if
}

void vtkAnisoLiShadowMapSP::SetDiffuseContributionShadow(float contribution)
{
  vtkDebugMacro(<<"Setting diffuse contribution shadow to "<<contribution);
  if (contribution != this->DiffuseContributionShadow)
    {
    this->DiffuseContributionShadow = contribution;
    this->DiffuseContributionShadowUniform->SetValue(this->DiffuseContributionShadow);
    if (this->IsLinked())
      {
//      this->SetGlUniform(this->DiffuseContributionShadowUniform);
      }
    //this->Modified();
    }
}

void vtkAnisoLiShadowMapSP::SetSpecularContributionShadow(float contribution)
{
  vtkDebugMacro(<<"Setting specular contribution shadow to "<<contribution);
  if (contribution != this->SpecularContributionShadow)
    {
    this->SpecularContributionShadow = contribution;
    this->SpecularContributionShadowUniform->SetValue(this->SpecularContributionShadow);
    if (this->IsLinked())
      {
//      this->SetGlUniform(this->SpecularContributionShadowUniform);
      }
    //this->Modified();
    }
}

} // namespace bmia
