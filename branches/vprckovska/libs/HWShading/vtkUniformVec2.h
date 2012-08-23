/**
 * vtkUniformVec2.h
 *
 * 2005-05-17	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkUniformVec2_h
#define bmia_vtkUniformVec2_h

#include "vtkShaderUniform.h"

namespace bmia {

/**
 * Class for representing uniform vec2 variables.
 */
class vtkUniformVec2: public vtkShaderUniform
{
public:
  static vtkUniformVec2* New();

  vtkSetVector2Macro(Value, float);
  vtkGetVector2Macro(Value, float);

protected:
  vtkUniformVec2();
  ~vtkUniformVec2();

  virtual void SetGlUniformSpecific();

private:
  float Value[2];
};

} // namespace bmia

#endif
