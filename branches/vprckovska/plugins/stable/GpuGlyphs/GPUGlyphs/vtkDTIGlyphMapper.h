/**
 * vtkDTIGlyphMapper.h
 * by Tim Peeters
 *
 * 2008-02-28	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkDTIGlyphMapper_h
#define bmia_vtkDTIGlyphMapper_h

#include "vtkGlyphMapper.h"
#include <vtkgl.h>

namespace bmia {

class vtkUniformSampler;

/**
 * Mapper for GPU rendering of glyphs for DTI
 */
class vtkDTIGlyphMapper : public vtkGlyphMapper
{
public:
  static vtkDTIGlyphMapper* New();

  virtual void ReloadTextures();
protected:
  vtkDTIGlyphMapper();
  ~vtkDTIGlyphMapper();

  virtual void LoadTextures();
  virtual void SetTextureLocations(); //vtkMyShaderProgram* program);

  virtual void UnloadTextures();

private:

  vtkUniformSampler* UTextureEV1;
  vtkUniformSampler* UTextureEV2;
  GLuint Textures[2];

}; // class vtkDTIGlyphMapper
} // namespace bmia

#endif // bmia_vtkDTIGlyphMapper_h
