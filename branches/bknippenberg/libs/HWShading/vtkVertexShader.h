/**
 * vtkVertexShader.h
 * by Tim Peeters
 *
 * 2005-05-03	Tim Peeters	First version
 */

#ifndef bmia_vtkVertexShader_h
#define bmia_vtkVertexShader_h

#include "vtkShaderObject.h"

namespace bmia {

/**
 * GLSL Vertex Shader Object
 */
class vtkVertexShader : public vtkShaderObject
{
public:
  static vtkVertexShader* New();

protected:
  vtkVertexShader();
  ~vtkVertexShader();

  virtual bool CreateGlShader();

private:

};

} // bmia

#endif // bmia_vtkVertexShader_h
