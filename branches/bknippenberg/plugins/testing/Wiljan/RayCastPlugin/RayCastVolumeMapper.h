/*
 * RayCastVolumeMapper.h
 *
 * 2010-03-12	Wiljan van Ravensteijn
 * - First version
 *
 * 2010-11-15	Evert van Aart
 * - Using an object of type "vtkOpenGLExtensionManager" without first calling 
 *   "SetRenderWindow" caused crashes on some systems. Fixed this by moving
 *   the code related to the extension manager to a new function called 
 *   "Initialize", which is executed the first time "render" is called. The
 *   renderer passed to the "render" function is used to get the required
 *   "vtkRenderWindow" pointer.
 *
 * 2011-01-17	Evert van Aart
 * - In "InitShaders", we now select the correct shader as the current one using
 *   the "renderMethod" class variable. This ensures correct rendering on the first
 *   render pass.
 *
 * 2011-03-02	Evert van Aart
 * - Fixed rendering for viewports that do not start at (0, 0).
 *
 */

#ifndef RAYCASTVOLUMEMAPPER_H
#define RAYCASTVOLUMEMAPPER_H

#include <vtkVolumeMapper.h>
#include <vtkRenderWindow.h>
#include <QTimer>
#include <QObject>

class vtkShaderBase;
class vtkShaderDVR;
class vtkShaderMIP;
class vtkShaderIsosurface;
class vtkShaderToon;
class vtkShaderRayDirections;
class vtkClippingPlane;
class vtkRenderer;
class vtkMatrix4x4;

enum RENDERMETHOD {DVR, MIP, ISOSURFACE, TOON};

class RayCastVolumeMapper :  public QObject,public vtkVolumeMapper
{
  Q_OBJECT

public:
  //! Creates a new instance of the volume mapper.
  static RayCastVolumeMapper * New();

  //! Renders the volume.
  void Render( vtkRenderer * pRen, vtkVolume * pVol );

  //! Informs vtk with what data is expected.
  int FillInputPortInformation( int iPort, vtkInformation * pInfo );

  //! called when the transferfunction is changed.
  void transferFunctionChanged();

  //! called when the volume is changed.
  void volumeChanged();

  //! Use this to enable or disable a clipping plane
  //! @param plane One of six [0..5] clipping planes.
  //! @param enabled Whether to enable or disable the plane.
  void setEnableClippingPlane(int plane, bool enabled);

  //! Set the render method that will be used.
  //! @param renderMethod One of the render methods.
  void setRenderMethod(RENDERMETHOD rRenderMethod);

  //! Set the distance of between raycasting samples. When having interaction
  //! with the raycaster, interactive stepsize is used. Only when no interaction
  //! is noticed, stepsize will be used.
  //! @param stepSize The size of the step.
  void setStepsize(float stepSize);

  //! Set the distance of between raycasting samples when interacting with the
  //! raycaster.
  //! @param stepSize The size of the step.
  void setInteractiveStepSize(float interactiveStepsize);

  //! Set the isovalue that is used when renderingmethod is isovalue.
  //! @param isovalue The isovalue that is used.
  void setIsoValue(float isovalue);

  //! Set the opacity of the object when renderingmethod is isovalue.
  //! @param isovalueOpacity The opacity that is used.
  void setIsoValueOpacity(float isovalueOpacity);

  //! Set the range which will be considered transparent when rendering
  //! grayscale values on the clipping planes.
  //! @param min The minimum value of the range.
  //! @param max The minimum value of the range.
  void setClippingPlanesThreshold(float min, float max);

  //! Set whether to use grayscale-values on the clipping planes.
  //! @param value Whether to use grayscale-values.
  void setUseGrayScaleValues(bool value);

  //! Set a transformationMatrix that is applied after any vtk transformations.
  //! @param pMatrix A pointer to a vtkMatrix
  void setExternalTransformationMatrix(vtkMatrix4x4* pMatrix);

  //! Retrieve any clipping plane
  //! @param index Index of the clipping plane in the range [0..5]
  vtkClippingPlane* getPlane(int index);

  //! Reinitialize the clipping planes
  void reInitClippingPlanes();

  //! Return whether the required gpu extensions are supported
  bool extensionsSupported();

  void setIsoValueColor(float red, float green, float blue);


protected:
  RENDERMETHOD renderMethod;
  QTimer timer;
  bool m_bShadersInitialized;
  bool m_pClippingPlanesInitialized;
  bool m_bInteractiveModeEnabled;
  bool m_bUseGraysScaleValues;
  bool m_bExtensionsSupported;
  unsigned int m_iFboId;
  unsigned int m_iGeometryTxId;
  unsigned int m_iDepthTxId;
  unsigned int m_iStartTxId;
  unsigned int m_iStopTxId;
  unsigned int m_iRayDirTxId;
  unsigned int m_iFinalImageTxId;
  unsigned int m_iTextureId;
  unsigned int m_iColormapTextureId;
  unsigned int m_fDepthTextureId;
  unsigned char * m_pTable;
  bool clipState[6];
  int m_iNrBits;
  int m_iWindowWidth;
  int m_iWindowHeight;
  int m_iResolution;
  int m_iDimensions[3];
  double m_dSpacing[3];
  float m_fStepSize;
  float m_fInteractiveStepSize;
  float m_fInternalStepSize;
  float m_fIsoValue;
  float m_fIsoValueOpacity;
  float m_fClippingPlaneMinThreshold;
  float m_fClippingPlaneMaxThreshold;
      float m_fIsovalueColor[3];
  vtkMatrix4x4* m_pExternalTransformationMatrix;
  vtkShaderBase * m_pActiveShader;
  vtkShaderRayDirections * m_pRayDirections;
  vtkShaderDVR * m_pDVR;
  vtkShaderMIP * m_pMIP;
  vtkShaderIsosurface * m_pIsosurface;
  vtkShaderToon * m_pToon;
  vtkClippingPlane * m_pPlane[6];

  //! Constructor.
  RayCastVolumeMapper();

  //! Destructor.
  ~RayCastVolumeMapper();

  //! Initializes a set of textures that are used for offscreen rendering. It
  //! also initializes a Framebuffer Object for this purpose.
  //! @param iWidth the width of the current render window
  //! @param iHeight the height of the current render window
  void InitBuffers( int iWidth, int iHeight );

  //! Initializes a set of shader programs that handle different ways of
  //! rendering of the volume.
  void InitShaders();

  //! Initializes the clipping planes by placing them just outside the
  //! scene bounds and disabling them.
  void InitClippingPlanes( vtkVolume * pVolume );

  //! Loads the texture into GPU memory and assigns the texture ID.
  int LoadTexture(vtkVolume * pVol);

  //! Loads the transferfunction as a texture into GPU memory
  void InitTransferFunctionTexture(vtkVolume* pVol);

  //! Descritezes the transferfunction
  void DiscretizePiecewiseLinear(vtkVolume* pVol);

  //! Renders a bounding box with the given dimensions in world space units. Normally,
  //! these are computed by taking the volume resolution in the X, Y and Z direction,
  //! and multiplying these values by the voxel size in each direction.
  //! @param fX the X dimension (in mm)
  //! @param fY the Y dimension (in mm)
  //! @param fZ the Z dimension (in mm)
  void RenderBBox( float fX, float fY, float fZ );

  void RenderTransparentBBox( float fX, float fY, float fZ );

  //! Renders color-coded planes at the position of the clipping planes, if
  //! they are enabled. Thanks to these planes the ray start positions are
  //! correctly obtained.
  //! @param fX the X dimension (in mm)
  //! @param fY the Y dimension (in mm)
  //! @param fZ the Z dimension (in mm)
  //! @param iTxId the volume texture ID
  void RenderClippingPlanes( float fX, float fY, float fZ, unsigned int iTxId );

  //! Render a color-code plane at the position of the near clipping plane.
  //! This causes the rays to start at the near clipping plane instead
  //! behind the camera.
  //! Not implemented yet. Not functioning properly.
  //! @param pRen the renderer
  //! @param fX the X dimension (in mm)
  //! @param fY the Y dimension (in mm)
  //! @param fZ the Z dimension (in mm)
  void RenderNearClippingPlane( vtkRenderer * pRen, float fX, float fY, float fZ );

  void worldToObject(vtkMatrix4x4* pMatrix, double point[3]);

  // Initialize the mapper
  void Initialize(vtkRenderWindow * renwin);

  // True if "Initialized" has completed succesfully
  bool Initialized;

protected slots:
  //! Feedback function from the timer. Renders the scene again. And emits
  //! the signal render. A less uglier solution would call the rerender from
  //! within the mapper.
  void timeout();

signals:
  //! Signal emitted when a rerender is required.
  void render();


};

#endif // RAYCASTVOLUMEMAPPER_H
