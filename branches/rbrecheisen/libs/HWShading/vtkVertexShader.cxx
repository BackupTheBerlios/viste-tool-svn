/**
 * vtkVertexShader.cxx
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 * - Renamed glCreateObject() to CreateGlShader()
 *
 * 2006-01-30	Tim Peeters
 * - Switch from glew to VTK OpenGL extension manager.
 *   Use vtkgl::CreateShader(vtkgl::VERTEX_SHADER) instead of
 *   glCreateShader(GL_VERTEX_SHADER).
 */

#include "vtkVertexShader.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkVertexShader);

vtkVertexShader::vtkVertexShader()
{
  // Nothing to do. Everything was already done in vtkShaderObject.
}

vtkVertexShader::~vtkVertexShader()
{
  // Nothing.
}

bool vtkVertexShader::CreateGlShader()
{
  if (this->GetHandleValid())
    {
    // no need to create a new handle.
    return true;
    }

  vtkDebugMacro("Calling glCreateShader(GL_VERTEX_SHADER)");
  GLuint handle = vtkgl::CreateShader(vtkgl::VERTEX_SHADER);
  vtkDebugMacro("glCreateShader() returned handle "<<handle<<".");
  this->SetHandle(handle);
  return true;
}

} // namespace bmia
