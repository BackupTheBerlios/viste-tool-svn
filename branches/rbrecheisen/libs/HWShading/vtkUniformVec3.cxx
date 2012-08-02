/**
 * vtkUniformVec3.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 *
 * 2006-01-30	Tim Peeters
 * - Use vtkgl::Uniform3f() instead of glUniform3f().
 */

#include "vtkUniformVec3.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformVec3);

vtkUniformVec3::vtkUniformVec3()
{
  this->Value[0] = 0.0;
  this->Value[1] = 0.0;
  this->Value[2] = 0.0;
}

vtkUniformVec3::~vtkUniformVec3()
{
  // nothing to do.
}

void vtkUniformVec3::SetGlUniformSpecific()
{
  vtkDebugMacro(<<"Calling glUniform3f("<<this->Location<<", "<<this->Value[0]
	<<", "<<this->Value[1]<<", "<<this->Value[2]<<").");
  vtkgl::Uniform3f(this->Location, this->Value[0], this->Value[1], this->Value[2]);
}

} // namespace bmia
