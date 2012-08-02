/**
 * vtkUniformVec4.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkUniformVec4_h
#define bmia_vtkUniformVec4_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform vec4 variables.
 */
class vtkUniformVec4: public vtkShaderUniform
{
public:
  static vtkUniformVec4* New();

  vtkSetVector4Macro(Value, float);
  vtkGetVector4Macro(Value, float);

protected:
  vtkUniformVec4();
  ~vtkUniformVec4();

  virtual void SetGlUniformSpecific();

private:
  float Value[4];
};

} // namespace bmia

#endif
