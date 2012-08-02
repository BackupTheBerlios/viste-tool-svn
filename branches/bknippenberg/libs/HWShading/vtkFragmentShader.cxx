/**
 * vtkFragmentShader.cxx
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Switched to OpenGL 2.0 (removed ARB in OpenGL types and functions)
 * - Renamed glCreate() to CreateGlShader()
 *
 * 2006-01-30	Tim Peeters
 * - Switch from glew to vtkOpenGLExtensionManager and vtkgl.h
 */

#include "vtkFragmentShader.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkFragmentShader);

vtkFragmentShader::vtkFragmentShader()
{
  // Nothing to do. Everything was already done in vtkShaderObject.
}

vtkFragmentShader::~vtkFragmentShader()
{
  // Nothing.
}

bool vtkFragmentShader::CreateGlShader()
{
  if (this->GetHandleValid())
    {
    // no need to create a new handle.
    return true;
    }

  vtkDebugMacro("Calling glCreateShader(GL_FRAGMENT_SHADER)");
  GLuint handle = vtkgl::CreateShader(vtkgl::FRAGMENT_SHADER);
  vtkDebugMacro("glCreateShader() returned handle "<<handle<<".");
  this->SetHandle(handle);
  return true;
}

} // namespace bmia
