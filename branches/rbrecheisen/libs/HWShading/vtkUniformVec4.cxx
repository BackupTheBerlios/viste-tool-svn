/**
 * vtkUniformVec4.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 *
 * 2006-01-30	Tim Peeters
 * - Use vtkgl::Uniform4f() instead of glUniform4f().
 */

#include "vtkUniformVec4.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformVec4);

vtkUniformVec4::vtkUniformVec4()
{
  this->Value[0] = 0.0;
  this->Value[1] = 0.0;
  this->Value[2] = 0.0;
  this->Value[3] = 0.0;
}

vtkUniformVec4::~vtkUniformVec4()
{
  // nothing to do.
}

void vtkUniformVec4::SetGlUniformSpecific()
{
  vtkgl::Uniform4f(this->Location, this->Value[0], this->Value[1], this->Value[2], this->Value[3]);
}

} // namespace bmia
