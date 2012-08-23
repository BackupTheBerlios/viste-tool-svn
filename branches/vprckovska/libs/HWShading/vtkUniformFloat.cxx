/**
 * vtkUniformFloat.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 *
 * 2006-01-30	Tim Peeters
 * - Use vtkgl::Uniform1f() instead of glUniform1f()
 */

#include "vtkUniformFloat.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformFloat);

vtkUniformFloat::vtkUniformFloat()
{
  this->Value = 0.0;
}

vtkUniformFloat::~vtkUniformFloat()
{
  // nothing to do.
}

void vtkUniformFloat::SetGlUniformSpecific()
{
  vtkDebugMacro(<<"Calling glUniform1f("<<this->Location<<", "
		<<this->Value<<").");
  vtkgl::Uniform1f(this->Location, this->Value);
}

} // namespace bmia
