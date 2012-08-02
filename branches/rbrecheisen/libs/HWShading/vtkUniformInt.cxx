/**
 * vtkUniformInt.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 *
 * 2006-01-30	Tim Peeters
  - Use vtkgl::Uniform1i() instead of glUniform1i().
 */

#include "vtkUniformInt.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformInt);

vtkUniformInt::vtkUniformInt()
{
  this->Value = 0;
}

vtkUniformInt::~vtkUniformInt()
{
  // nothing to do.
}

void vtkUniformInt::SetGlUniformSpecific()
{
  vtkgl::Uniform1i(this->Location, this->Value);
}

} // namespace bmia
