/**
 * vtkShadowMappingHelperLines.cxx
 * by Tim Peeters
 *
 * 2005-11-22	Tim Peeters
 * - First version
 */

#include "vtkShadowMappingHelperLines.h"
#include <vtkObjectFactory.h>

#include "vtkMyShaderProgram.h"
#include "vtkFragmentShader.h"
#include "vtkVertexShader.h"

#include "BuildShadowMapLinesVertexText.h"
#include "BuildShadowMapLinesFragmentText.h"

namespace bmia {

vtkStandardNewMacro(vtkShadowMappingHelperLines);

vtkShadowMappingHelperLines::vtkShadowMappingHelperLines()
{
  vtkMyShaderProgram* prog = vtkMyShaderProgram::New();
  prog->SetAttrib(2, "LineID");
  vtkVertexShader* vert = vtkVertexShader::New();
  vert->SetSourceText(BuildShadowMapLinesVertexText);
  vtkFragmentShader* frag = vtkFragmentShader::New();
  frag->SetSourceText(BuildShadowMapLinesFragmentText);
  prog->AddShaderObject(vert);
  prog->AddShaderObject(frag);
  vert->Delete(); vert = NULL;
  frag->Delete(); frag = NULL;
  this->SetShaderProgram(prog);
  prog->Delete(); prog = NULL;
}

vtkShadowMappingHelperLines::~vtkShadowMappingHelperLines()
{
  // nothing to do.
}

} // namespace bmia
