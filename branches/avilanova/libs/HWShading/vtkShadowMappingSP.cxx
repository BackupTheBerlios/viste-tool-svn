/**
 * vtkShadowMappingSP.cxx
 * by Tim Peeters
 *
 * 2005-07-26	Tim Peeters
 * - First version
 *
 * 2005-09-12	Tim Peeters
 * - Included new shader that contains lighting calculation functions used
 *   in the fragment shader.
 */

#include "vtkShadowMappingSP.h"
#include "ShadowMappingVertexText.h"
#include "ShadowMappingFragmentText.h"
#include "SpotlightFunctionsText.h"

#include "vtkVertexShader.h"
#include "vtkFragmentShader.h"
//#include "vtkUniformFloat.h"

#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkShadowMappingSP)

vtkShadowMappingSP::vtkShadowMappingSP()
{
  this->VertexShader = vtkVertexShader::New();
  this->FragmentShader = vtkFragmentShader::New();
  this->SpotlightFuncShader = vtkFragmentShader::New();

  this->VertexShader->SetSourceText(ShadowMappingVertexText);
  this->FragmentShader->SetSourceText(ShadowMappingFragmentText);
  this->SpotlightFuncShader->SetSourceText(SpotlightFunctionsText);

  this->AddShaderObject(this->VertexShader);
  this->AddShaderObject(this->SpotlightFuncShader);
  this->AddShaderObject(this->FragmentShader);
}

vtkShadowMappingSP::~vtkShadowMappingSP()
{
  this->VertexShader->Delete(); this->VertexShader = NULL;
  this->FragmentShader->Delete(); this->FragmentShader = NULL;
  this->SpotlightFuncShader->Delete(); this->SpotlightFuncShader = NULL;
}

} // namespace bmia
