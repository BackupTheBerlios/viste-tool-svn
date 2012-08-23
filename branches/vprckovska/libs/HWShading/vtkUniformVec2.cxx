/**
 * vtkUniformVec2.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 *
 * 2006-01-30	Tim Peeters
 * - Use vtkgl::Uniform2f() instead of glUniform2f().
 */

#include "vtkUniformVec2.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformVec2);

vtkUniformVec2::vtkUniformVec2()
{
  this->Value[0] = 0.0;
  this->Value[1] = 0.0;
}

vtkUniformVec2::~vtkUniformVec2()
{
  // nothing to do.
}

void vtkUniformVec2::SetGlUniformSpecific()
{
  vtkgl::Uniform2f(this->Location, this->Value[0], this->Value[1]);
}

} // namespace bmia
