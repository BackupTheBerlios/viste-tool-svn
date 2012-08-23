/**
 * vtkUniformIvec3.h
 *
 * 2007-10-25	Tim Peeters
 * - First version, based on vtkUniformVec3.h
 */

#ifndef bmia_vtkUniformIvec3_h
#define bmia_vtkUniformIvec3_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform ivec3 variables.
 */
class vtkUniformIvec3: public vtkShaderUniform
{
public:
  static vtkUniformIvec3* New();

  vtkSetVector3Macro(Value, int);
  vtkGetVector3Macro(Value, int);

protected:
  vtkUniformIvec3();
  ~vtkUniformIvec3();

  virtual void SetGlUniformSpecific();

private:
  int Value[3];
};

} // namespace bmia

#endif
