/**
 * vtkFragmentShader.h
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters
 * -First version
 *
 * 2005-06-06	Tim Peeters
 * - Renamed glCreateObject() to CreateGlShader()
 */

#ifndef bmia_vtkFragmentShader_h
#define bmia_vtkFragmentShader_h

#include "vtkShaderObject.h"

namespace bmia {

/**
 * GLSL Fragment Shader Object
 */
class vtkFragmentShader : public vtkShaderObject
{
public:
  static vtkFragmentShader* New();

protected:
  vtkFragmentShader();
  ~vtkFragmentShader();

  virtual bool CreateGlShader();

private:

};

} // namespace bmia

#endif
