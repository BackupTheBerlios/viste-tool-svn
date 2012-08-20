/**
 * vtkShaderBaseHandle.cxx
 * by Tim Peeters
 *
 * 2005-05-17	Tim Peeters
 * - First version, based on the (old) vtkShaderBase.cxx
 *
 * 2005-06-06	Tim Peeters
 * - Switch to OpenGL 2.0
 * - Removed glDelete()
 */

#include "vtkShaderBaseHandle.h"

namespace bmia {

vtkShaderBaseHandle::vtkShaderBaseHandle()
{
  this->HandleValid = false;
  // this->Handle is not initialized here
}

vtkShaderBaseHandle::~vtkShaderBaseHandle()
{
  // nothing was created by this class, so don't destroy anything either.
  // do that in the subclasses that were creating things.
}

GLuint vtkShaderBaseHandle::GetHandle()
{
  if (!this->GetHandleValid())
    {
    vtkErrorMacro(<<"Calling GetHandle() without a valid handle!");
    }
  return this->Handle;
}

void vtkShaderBaseHandle::SetHandle(GLuint newhandle)
{
  this->Handle = newhandle;
  this->HandleValid = true;
}

void vtkShaderBaseHandle::UnsetHandle()
{
  this->HandleValid = false;
}

bool vtkShaderBaseHandle::GetHandleValid()
{
  return this->HandleValid;
}

} // namespace bmia
