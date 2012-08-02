/**
 * vtkAnisoLiShadowMapSP.h
 * by Tim Peeters
 *
 * 2005-09-13	Tim Peeters
 * - First version
 *
 * 2005-10-19	Tim Peeters
 * - Made this class a subclass of vtkAnisotropicLightingSP.
 * - Remove functions already in vtkAnisotropicLightingSP.
 *
 * 2005-12-08	Tim Peeters
 * - Added support for parameters {Ambient,Diffuse,Specular}ContributionShadow.
 */

#ifndef bmia_vtkAnisoLiShadowMapSP_h
#define bmia_vtkAnisoLiShadowMapSP_h

#include "vtkAnisotropicLightingSP.h"

namespace bmia {

/**
 * Shader program that combines anisotropic lighting and shadow mapping.
 */
class vtkAnisoLiShadowMapSP : public vtkAnisotropicLightingSP {

public:
  static vtkAnisoLiShadowMapSP *New();

  void SetDiffuseContributionShadow(float contribution);
  float GetDiffuseContributionShadow() { return this->DiffuseContributionShadow; }
  void SetSpecularContributionShadow(float contribution);
  float GetSpecularContributionShadow() { return this->SpecularContributionShadow; }
  void SetAmbientContributionShadow(float contribution);
  float GetAmbientContributionShadow() { return this->AmbientContributionShadow; }

protected:
  vtkAnisoLiShadowMapSP();
  ~vtkAnisoLiShadowMapSP();

  float AmbientContributionShadow;
  float DiffuseContributionShadow;
  float SpecularContributionShadow;

  vtkUniformFloat* AmbientContributionShadowUniform;
  vtkUniformFloat* DiffuseContributionShadowUniform;
  vtkUniformFloat* SpecularContributionShadowUniform;

private:

}; // class vtkAnisotropicLightingSP

} // namespace bmia

#endif // bmia_vtkAnisoLiShadowMapSP_h
