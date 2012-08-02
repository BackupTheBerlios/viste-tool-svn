/**
 * vtkUniformIvec3.cxx
 *
 * 2007-10-25	Tim Peeters
 * - First version. Based on vtkUniformVec3.cxx
 */

#include "vtkUniformIvec3.h"
#include <vtkObjectFactory.h>

namespace bmia {

vtkStandardNewMacro(vtkUniformIvec3);

vtkUniformIvec3::vtkUniformIvec3()
{
  this->Value[0] = 0;
  this->Value[1] = 0;
  this->Value[2] = 0;;
}

vtkUniformIvec3::~vtkUniformIvec3()
{
  // nothing to do.
}

void vtkUniformIvec3::SetGlUniformSpecific()
{
  vtkDebugMacro(<<"Calling glUniform3i("<<this->Location<<", "<<this->Value[0]
	<<", "<<this->Value[1]<<", "<<this->Value[2]<<").");
  vtkgl::Uniform3i(this->Location, this->Value[0], this->Value[1], this->Value[2]);
}

} // namespace bmia
