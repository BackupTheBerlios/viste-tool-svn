/**
 * vtkShadowMappingSP.h
 * by Tim Peeters
 *
 * 2005-07-26	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkShadowMappingSP_h
#define bmia_vtkShadowMappingSP_h

#include "vtkMyShaderProgram.h"

namespace bmia {

class vtkVertexShader;
class vtkFragmentShader;

/**
 * Shader program that renders a scene with shadows. A shadow map must
 * have been generated and supplied to this shader program.
 */
class vtkShadowMappingSP : public vtkMyShaderProgram {

public:
  static vtkShadowMappingSP* New();

protected:
  vtkShadowMappingSP();
  ~vtkShadowMappingSP();

private:
  vtkVertexShader* VertexShader;
  vtkFragmentShader* FragmentShader;
  vtkFragmentShader* SpotlightFuncShader;

}; // class vtkShadowMappingSP

} // namespace bmia

#endif // bmia_vtkShadowMappingSP_h
