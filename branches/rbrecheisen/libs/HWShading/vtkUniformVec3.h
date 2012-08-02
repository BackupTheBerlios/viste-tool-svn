/**
 * vtkUniformVec3.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkUniformVec3_h
#define bmia_vtkUniformVec3_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform vec3 variables.
 */
class vtkUniformVec3: public vtkShaderUniform
{
public:
  static vtkUniformVec3* New();

  vtkSetVector3Macro(Value, float);
  vtkGetVector3Macro(Value, float);

protected:
  vtkUniformVec3();
  ~vtkUniformVec3();

  virtual void SetGlUniformSpecific();

private:
  float Value[3];
};

} // namespace bmia

#endif
