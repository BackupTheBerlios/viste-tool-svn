/**
 * vtkUniformBool.cxx
 *
 * 2005-05-17	Tim Peeters
 * - First version
 *
 * 2005-06-06	Tim Peeters
 * - Use OpenGL 2.0
 *
 * 2006-01-30	Tim PEeters
 * - Use vtkgl::Unifrom1i() instead of glUniform1i().
 */

#include "vtkUniformBool.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformBool);

vtkUniformBool::vtkUniformBool()
{
  this->Value = false;
}

vtkUniformBool::~vtkUniformBool()
{
  // nothing to do.
}

void vtkUniformBool::SetGlUniformSpecific()
{
  // Bools may be passed as either integers or floats where 0 or 0.0f is
  // equivealent to false, and other values are equivalent to true.
  // Here we use an integer value to pass a bool. 
  if (this->Value)
    {
    vtkgl::Uniform1i(this->Location, 1);
    }
  else
    {
    vtkgl::Uniform1i(this->Location, 0);
    }
}

} // namespace bmia
