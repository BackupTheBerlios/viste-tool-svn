/**
 * vtkShaderUniform.cxx
 * by Tim Peeters
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Switch to OpenGL 2.0
 *
 * 2006-01-30	Tim Peeters
 * - Switch from using glew to vtkOpenGLExtensionManager and vtkgl.h
 */

#include "vtkShaderUniform.h"

namespace bmia {

vtkShaderUniform::vtkShaderUniform()
{
  this->Name = NULL;
}

vtkShaderUniform::~vtkShaderUniform()
{
  if (this->Name)
    {
    delete [] this->Name;
    this->Name = NULL;
    }
}

bool vtkShaderUniform::SetGlUniform()
{
  if (!this->GetHandleValid())
    {
    this->Location = -1;
    vtkErrorMacro(<<"No handle set!");
    return false;
    }

  if (!this->Name)
    {
    this->Location = -1;
    vtkWarningMacro(<<"Uniform has no name!");
    return false;
    }

  this->Location = vtkgl::GetUniformLocation(this->GetHandle(), this->Name);
  vtkDebugMacro(<<"Location of uniform "<<this->Name<<" with handle "<<this->GetHandle()<<" is: "<<this->Location);

  if (this->Location == -1)
    {
    vtkWarningMacro(<<"Location of uniform with name "<<this->Name<<" could not be determined!");
    return false;
    }

  // this->Location has a valid value.
  this->SetGlUniformSpecific();
  return true;
}

} // namespace bmia
