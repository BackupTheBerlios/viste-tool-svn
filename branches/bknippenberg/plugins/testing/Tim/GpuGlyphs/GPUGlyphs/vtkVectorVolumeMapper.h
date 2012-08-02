/**
 * vtkVectorVolumeMapper.h
 * by Tim Peeters
 *
 * 2007-10-22	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkVectorVolumeMapper_h
#define bmia_vtkVectorVolumeMapper_h

#include <vtkVolumeMapper.h>

class vtkPlaneWidget;
class vtkTimerLog;

namespace bmia {

class vtkMyShaderProgram;
class vtkUniformFloat;
class vtkUniformVec3;
class vtkUniformIvec3;
class vtkUniformBool;

/**
 * Class for GPU-based rendering of lines to represent
 * a vector field.
 */
class vtkVectorVolumeMapper : public vtkVolumeMapper
{
public:
  static vtkVectorVolumeMapper* New();

  virtual void Render(vtkRenderer *ren, vtkVolume *vol);

  vtkGetMacro(Initialized, bool);

  /**
   * The widget used to select the current volume slice
   * to visualize.
   */
  vtkGetObjectMacro(PlaneWidget, vtkPlaneWidget);
  void SetPlaneWidget(vtkPlaneWidget* PlaneWidget);

  /**
   * This function overrides the SetInputConnection from vtkAlgorithm
   * to always set the input for the PlaneWidget, and then call the
   * superclass SetInputConnection.
   * The idea is that this function is always called when the input
   * data for the vtkVectorVolumeMapper is called.
   */
  virtual void SetInputConnection(int port, vtkAlgorithmOutput* input);
  virtual void SetInputConnection(vtkAlgorithmOutput* input);

  // TODO: SetH and GetH functions.
  
  /**
   * The distance between seed points.
   */
  void SetSeedDistance(float d);
  vtkGetMacro(SeedDistance, float);

  /**
   * The length of the short lines to render
   */
  void SetLineLength(float length);
  vtkGetMacro(LineLength, float);

  /**
   * Set the thickness of the line segments
   */
  void SetLineThickness(float thickness);
  vtkGetMacro(LineThickness, float);

  /**
   * Enable/disable the use of shadows
   */
  void SetUseShadows(bool shadows);

protected:		
  vtkVectorVolumeMapper();
  ~vtkVectorVolumeMapper();

  bool Initialized;
  void Initialize();

  //bool TextureLoaded;
  // void LoadTexture()
  // void UnloadTexture();
  
  void RenderBoundingBox();
  virtual void LoadTexture();
  virtual void LoadTextures() { this->LoadTexture(); }
  void LoadTexture(int texture_index, const char* array_name);

  float SeedDistance;
  float LineLength;
  float LineThickness;

  vtkMyShaderProgram* GetShaderProgram()
    {
    return this->ShaderProgram;
    }

  void SetShaderProgram(vtkMyShaderProgram* sp);
  void SetupShaderUniforms();

private:

  vtkPlaneWidget* PlaneWidget;
  vtkMyShaderProgram* ShaderProgram;

  vtkUniformVec3* EyePosition;
  vtkUniformVec3* LightPosition;
  //  vtkUniformVec3 EyeDir;
  // vtkUniformVec3 LightDir?
  vtkUniformIvec3* TextureDimensions;
  vtkUniformVec3* TextureSpacing;
  vtkUniformFloat* UniformLineLength;
  vtkUniformFloat* UniformSeedDistance;
  vtkUniformFloat* UniformLineThickness;
  vtkUniformBool* UniformUseShadows;

  vtkUniformVec3* UniformPoiOrigin;
  vtkUniformVec3* UniformPoiPoint1;
  vtkUniformVec3* UniformPoiPoint2;
  vtkUniformVec3* Step1;
  vtkUniformVec3* Step2;

  vtkTimerLog* Timer;
  int TimerSteps;

}; // class vtkVectorVolumeMapper
} // namespace bmia
#endif // bmia_vtkVectorVolumeMapper_h
