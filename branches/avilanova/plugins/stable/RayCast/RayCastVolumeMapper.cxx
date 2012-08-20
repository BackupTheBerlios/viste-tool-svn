/*
 * RayCastVolumeMapper.cxx
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
 * 2011-08-17	Evert van Aart
 * - Added Ralph's changes related to texture rendering options. Changes are marked
 *   with "Ralph" in the comments.
 *
 */

#if defined (_WIN32) || defined (_WIN64) || defined (__linux__)
#define BMIA_TEXTURE_3D (vtkgl::TEXTURE_3D)
#define bmiaTexImage3D(a, b, c, d, e, f, g, h, i, j) (vtkgl::TexImage3D(a, b, c, d, e, f, g, h, i, j))
#endif

#if defined (__APPLE__) || defined (__MACH__)
#define BMIA_TEXTURE_3D (GL_TEXTURE_3D)
#define bmiaTexImage3D(a, b, c, d, e, f, g, h, i, j) (glTexImage3D(a, b, c, d, e, f, g, h, i, j))
#endif

#include "RayCastVolumeMapper.h"
#include "vtkShaderDVR.h"
#include "vtkShaderMIP.h"
#include "vtkShaderIsosurface.h"
#include "vtkShaderToon.h"
#include "vtkShaderRayDirections.h"
#include "vtkClippingPlane.h"
#include <vtkObjectFactory.h>
#include <vtkVolume.h>
#include <vtkImageData.h>
#include <vtkInformation.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
#include <vtkRenderWindow.h>
#include <vtkMatrix4x4.h>
#include <vtkRenderer.h>
#include <vtkWindow.h>
#include <vtkPlaneCollection.h>
#include <vtkCamera.h>
#include <vtkFloatArray.h>
#include <vtkRenderWindow.h>
#include "vtkOpenGLExtensionManager.h"
#include <vtkgl.h>


vtkStandardNewMacro( RayCastVolumeMapper )

        RayCastVolumeMapper::RayCastVolumeMapper()
{
    this->renderMethod = MIP;
    this->SetNumberOfInputPorts(1);
    this->m_iFboId = 0;
    this->m_iDepthTxId = 0;
    this->m_iStartTxId = 0;
    this->m_iStopTxId = 0;
    this->m_iRayDirTxId = 0;
    this->m_iGeometryTxId = 0;
    this->m_iFinalImageTxId = 0;
    this->m_iTextureId = 0;
    this->m_pTable = 0;
    this->m_iResolution = 32;
    this->m_iColormapTextureId = 0;
    this->m_fDepthTextureId = 0;
    this->m_iNrBits = 0;
    this->m_iWindowWidth = 0;
    this->m_iWindowHeight = 0;
    this->m_bShadersInitialized = false;
    this->m_bInteractiveModeEnabled = false;
    this->m_pClippingPlanesInitialized = false;
    this->m_fStepSize = 1.0f;
    this->m_fInternalStepSize =  m_fStepSize;
    this->m_fInteractiveStepSize = 8.0f;
    this->m_fIsoValue = 0.0f;
    this->m_fIsoValueOpacity = 1.0f;
    this->m_pRayDirections = 0;
    this->m_pDVR = 0;
    this->m_pMIP = 0;
    this->m_pIsosurface = 0;
    this->m_pToon = 0;
    this->m_pActiveShader = 0;
    this->timer.setInterval(500);
    this->timer.setSingleShot(true);
    this->m_fClippingPlaneMinThreshold = 0.0;
    this->m_fClippingPlaneMaxThreshold = 0.0;
    this->m_bUseGraysScaleValues = false;
    this->m_pExternalTransformationMatrix = 0;
    this->m_fIsovalueColor[0] = 0.625f;
    this->m_fIsovalueColor[1] = 0.625f;
    this->m_fIsovalueColor[2] = 0.625f;

    for( int i = 0; i < 6; i++ )
    {
        m_pPlane[i] = vtkClippingPlane::New();
        m_pPlane[i]->SetId( i );
        m_pPlane[i]->Disable();
    }

    m_pPlane[0]->SetOrigin( 0, 0, 0 );
    m_pPlane[0]->SetNormal( 1, 0, 0 );
    m_pPlane[1]->SetOrigin( 0, 0, 0 );
    m_pPlane[1]->SetNormal( -1, 0, 0 );
    m_pPlane[2]->SetOrigin( 0, 0, 0 );
    m_pPlane[2]->SetNormal( 0, 1, 0 );
    m_pPlane[3]->SetOrigin( 0, 0, 0 );
    m_pPlane[3]->SetNormal( 0, -1, 0 );
    m_pPlane[4]->SetOrigin( 0, 0, 0 );
    m_pPlane[4]->SetNormal( 0, 0, 1 );
    m_pPlane[5]->SetOrigin( 0, 0, 0 );
    m_pPlane[5]->SetNormal( 0, 0, -1 );

    for( int i = 0; i < 6; i++ )
    {
        AddClippingPlane( m_pPlane[i] );
    }

	// "Initialized" is false by default
	this->Initialized = false;

    connect(&this->timer,SIGNAL(timeout()),this,SLOT(timeout()));

}

void RayCastVolumeMapper::Initialize(vtkRenderWindow * renwin)
{
    vtkOpenGLExtensionManager *extensions = vtkOpenGLExtensionManager::New();
	extensions->SetRenderWindow(renwin);

	this->m_bExtensionsSupported = true;
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_VERSION_1_3");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_VERSION_2_0");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_ARB_vertex_shader");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_ARB_shader_objects");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_ARB_fragment_shader");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_ARB_multitexture");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_EXT_framebuffer_object");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_ARB_texture_non_power_of_two");
    this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_ARB_texture_float");

#if defined (_WIN32) || defined (_WIN64) || defined (__linux__)
	this->m_bExtensionsSupported = this->m_bExtensionsSupported && extensions->ExtensionSupported("GL_EXT_texture3D");
#endif

    if (this->m_bExtensionsSupported)
    {
        extensions->LoadExtension("GL_VERSION_1_3");
        extensions->LoadExtension("GL_VERSION_2_0");
        extensions->LoadExtension("GL_ARB_vertex_shader");
        extensions->LoadExtension("GL_ARB_shader_objects");
        extensions->LoadExtension("GL_ARB_fragment_shader");
        extensions->LoadExtension("GL_EXT_framebuffer_object");
        extensions->LoadExtension("GL_ARB_multitexture");
        extensions->LoadExtension( "GL_ARB_texture_float" );

#if defined (_WIN32) || defined (_WIN64) || defined (__linux__)
		extensions->LoadCorePromotedExtension("GL_EXT_texture3D");
#endif
	}

    extensions->Delete();

    if (!m_bExtensionsSupported)
        return;

    // init shaders

    if( ! m_bShadersInitialized)
    {
        InitShaders();
        m_bShadersInitialized = true;
    }

	// Successfully initialized
	this->Initialized = true;
}


RayCastVolumeMapper::~RayCastVolumeMapper()
{
    if( m_iDepthTxId )
        glDeleteTextures( 1, & m_iDepthTxId );
    if( m_iGeometryTxId )
        glDeleteTextures( 1, & m_iGeometryTxId );
    if( m_iStartTxId )
        glDeleteTextures( 1, & m_iStartTxId );
    if( m_iStopTxId )
        glDeleteTextures( 1, & m_iStopTxId );
    if( m_iRayDirTxId )
        glDeleteTextures( 1, & m_iRayDirTxId );
    if( m_iFinalImageTxId )
        glDeleteTextures( 1, & m_iFinalImageTxId );
    if( m_fDepthTextureId )
        glDeleteTextures( 1, & m_fDepthTextureId );
    if( m_iFboId )
        vtkgl::DeleteFramebuffersEXT( 1, & m_iFboId );
    if( m_pRayDirections )
        m_pRayDirections->Delete();
    if( m_pDVR )
        m_pDVR->Delete();
    if( m_pMIP )
        m_pMIP->Delete();
    if( m_pIsosurface )
        m_pIsosurface->Delete();
    if( m_pToon )
        m_pToon->Delete();

    if( m_iTextureId )
        glDeleteTextures( 1, & m_iTextureId );
    if( m_pTable )
        delete [] m_pTable;
    if( m_iColormapTextureId )
        glDeleteTextures( 1, & m_iColormapTextureId );
}

int RayCastVolumeMapper::FillInputPortInformation( int iPort, vtkInformation * pInfo )
{
    if( iPort == 0 )
    {
        pInfo->Set( vtkAlgorithm::INPUT_IS_REPEATABLE(), 1 );
        pInfo->Set( vtkAlgorithm::INPUT_IS_OPTIONAL(), 1 );
        pInfo->Set( vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "ImageData" );
    }
    return 0;
}

void RayCastVolumeMapper::Render( vtkRenderer * pRen, vtkVolume * pVol )
{
	if (!(this->Initialized))
		this->Initialize(pRen->GetRenderWindow());

    if (!m_bExtensionsSupported)
        return;

    if( ! m_pClippingPlanesInitialized )
    {
        InitClippingPlanes( pVol );
        m_pClippingPlanesInitialized = true;

        pRen->ResetCameraClippingRange();
    }
    else
    {
        for( int i = 0; i < 6; i++ )
        {
            this->m_pPlane[i]->SetEnabled(this->clipState[i]);
        }
    }

	pRen->ResetCameraClippingRange();

	if (!(this->m_pTable) && pVol->GetProperty())
        this->InitTransferFunctionTexture(pVol);

    // init buffers

    int * size = pRen->GetVTKWindow()->GetSize();
    if( size[0] != m_iWindowWidth || size[1] != m_iWindowHeight )
    {
        m_iWindowWidth = size[0];
        m_iWindowHeight = size[1];
        InitBuffers( m_iWindowWidth, m_iWindowHeight );
    }

	// Get the viewport of the 3D subcanvas
	double viewport[4];
	pRen->GetViewport(viewport);

	// Translate the viewport values from the range 0-1 to pixel values
	viewport[2] = (viewport[2] - viewport[0]) * size[0];
	viewport[3] = (viewport[3] - viewport[1]) * size[1];
	viewport[0] *= size[0];
	viewport[1] *= size[1];

    //-load texture if not loaded

    if( !(this->m_iTextureId) )
        this->LoadTexture(pVol);

    //-calculate world dimensions

    float worldDims[3], worldDiagonal;
    worldDims[0] = (this->m_iDimensions[0] - 1) * this->m_dSpacing[0];
	worldDims[1] = (this->m_iDimensions[1] - 1) * this->m_dSpacing[1];
	worldDims[2] = (this->m_iDimensions[2] - 1) * this->m_dSpacing[2];
	worldDiagonal = sqrtf(
            worldDims[0]*worldDims[0] + worldDims[1]*worldDims[1] + worldDims[2]*worldDims[2] );

    glPushAttrib(
            GL_ENABLE_BIT         |
            GL_COLOR_BUFFER_BIT   |
            GL_STENCIL_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT   |
            GL_POLYGON_BIT        |
			GL_LIGHTING_BIT       |
            GL_TEXTURE_BIT );

    // Setup the rendering context and transformations.
    glClearColor( 0, 0, 0, 0 );
    glClearDepth( 1 );

    glEnable   ( GL_DEPTH_TEST );
    glEnable   ( GL_CULL_FACE );
    glDisable  ( GL_LIGHTING );
    glDisable  ( GL_BLEND );
    glDisable  ( GL_ALPHA_TEST );
    glFrontFace( GL_CCW );


    // Set the transformations correct for rendering the scene.
    vtkMatrix4x4 * matrix = vtkMatrix4x4::New();
    pVol->GetMatrix( matrix );

    if (m_pExternalTransformationMatrix != 0)
    {
        vtkMatrix4x4::Multiply4x4(matrix,this->m_pExternalTransformationMatrix,matrix);
    }
    matrix->Transpose();
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
    glMultMatrixd(matrix->Element[0]);

    for( int i = 0; i < 6; i++ )
    {
        this->m_pPlane[i]->Update();
    }

    GLenum buffers[] = { vtkgl::COLOR_ATTACHMENT0_EXT};


    // camera to object space
    double position[3];
    pRen->GetActiveCamera()->GetPosition(position);
    this->worldToObject(matrix,position);

    //store depth buffer such that we can do depth checking.

    glBindTexture(GL_TEXTURE_2D, m_fDepthTextureId);
	glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, size[0], size[1]);

    //store prior scene already in final image;

    glBindTexture(GL_TEXTURE_2D, m_iFinalImageTxId);
    glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,size[0], size[1]);


    /////////////////////////////////////////
    // RAY STARTING POSITIONS

    // Here we render the ray starting positions to the color texture 0. We get the correct
    // starting positions by combining rendering of the color-coded bounding box with the
    // color-coded near clipping plane. This should ensure that whenever the user moves the
    // camera inside the volume, the rays will start at the near clipping plane instead of
    // the bounding box geometry, which lies behind the camera in that case.
    glBindTexture( GL_TEXTURE_2D, 0 );
    vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, m_iFboId );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_iStartTxId, 0 );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, m_iDepthTxId, 0 );
    vtkgl::DrawBuffers( 2, buffers );

    glDepthFunc( GL_LESS );
    glCullFace ( GL_BACK );

    glClearColor( 0, 0, 0, 0 );
    glClearDepth( 1 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    RenderClippingPlanes( worldDims[0], worldDims[1], worldDims[2], 0 );
    RenderBBox( worldDims[0], worldDims[1], worldDims[2] );
    vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, 0 );

    /////////////////////////////////////////
    // RAY ENDING POSITIONS
    glBindTexture( GL_TEXTURE_2D, 0 );
    vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, m_iFboId );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_iStopTxId, 0 );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, m_iDepthTxId, 0 );
    vtkgl::DrawBuffers( 2, buffers );

    // Make sure we get the bounding box backfaces.
    glDepthFunc( GL_GREATER );
    glCullFace ( GL_FRONT );

    glClearColor( 0, 0, 0, 0 );
    glClearDepth( 0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    RenderClippingPlanes( worldDims[0], worldDims[1], worldDims[2], 0 );
    RenderBBox( worldDims[0], worldDims[1], worldDims[2] );

    vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, 0);


    /////////////////////////////////////////
    // RAY DIRECTION AND LENGTH

    // Here we use the framebuffer object to render the ray directions and lengths into
    // the offscreen color texture 1. We do this by culling the front faces of the bounding
    // box. This will result in the backfaces of the bounding box, where each fragment is
    // color coded with the view space position. Inside the shader program we subtract this
    // position from the color coded positions in the ray start position texture.
    glBindTexture( GL_TEXTURE_2D, 0 );
    vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, m_iFboId );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_iRayDirTxId, 0 );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, m_iDepthTxId, 0 );
    vtkgl::DrawBuffers( 2, buffers );

    // Make sure we get the bounding box backfaces.
    glDepthFunc( GL_LESS );
	glCullFace( GL_BACK );

    glClearColor( 0, 0, 0, 0 );
    glClearDepth( 1 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );


    // Activate the ray directions shader and pass it the color texture containing color coded
    // ray starting positions.
    m_pRayDirections->Activate();

    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, m_iStartTxId );
    m_pRayDirections->SetInt1( 0, "frontBuffer" );

    vtkgl::ActiveTexture( vtkgl::TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, m_iStopTxId );
    m_pRayDirections->SetInt1( 1, "backBuffer" );

    m_pRayDirections->SetFloat1( 1.0f / m_iWindowWidth, "dx" );
    m_pRayDirections->SetFloat1( 1.0f / m_iWindowHeight, "dy" );

    RenderClippingPlanes( worldDims[0], worldDims[1], worldDims[2], 0 );
    RenderBBox( worldDims[0], worldDims[1], worldDims[2] );

    vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, 0);
    m_pRayDirections->Deactivate();

    /////////////////////////////////////////
    // RAYCASTING

    // Here we start rendering the actual volume to the offscreen color texture 0. We use
    // the framebuffer object again and activate the appropriate shader program that
    // handles the GPU rendering.
    glBindTexture( GL_TEXTURE_2D, 0 );
    vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, m_iFboId );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_iFinalImageTxId, 0 );
    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, m_iDepthTxId, 0 );
    vtkgl::DrawBuffers( 2, buffers );

    // If clipping is enabled we need to do backface culling, otherwise it will not work.
    // It's unclear yet why. If no clipping is enabled, we can do front face culling so
    // we can fly through the volume.
	glDepthFunc( GL_LESS );
    glCullFace( GL_BACK );

    glClearColor( 0, 0, 0, 0 );
    glClearDepth( 1 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Compute the stepsize. If the interactive mode is enabled we need to use the
    // largest stepsize to improve rendering performance. If not, we use the step
    // size set according to the quality level.
    float stepSize = m_bInteractiveModeEnabled ? m_fInteractiveStepSize : m_fInternalStepSize;

    // Active the currently selected shader. We pass the shader all the default parameters
    // that most shaders share, such as window scaling factors, raycasting stepsize,the
    // volume texture and the ray length texture.

    m_pActiveShader->Activate();
    m_pActiveShader->SetFloat1( 1.0f / m_iWindowWidth, "dx" );
    m_pActiveShader->SetFloat1( 1.0f / m_iWindowHeight, "dy" );
    m_pActiveShader->SetFloat1( stepSize, "stepSize" );
    m_pActiveShader->SetFloat1( worldDiagonal, "diagonal" );

    m_pActiveShader->SetFloat3( worldDims[0], worldDims[1], worldDims[2], "worldDimensions" );

	// Set the viewport
	m_pActiveShader->SetFloat4(viewport[0], viewport[1], viewport[2], viewport[3], "viewport");

    // use clipping planes for rendering opaque clipping planes

    double range[2];
    this->GetInput()->GetScalarRange(range);

    float minClipThreshold = (this->m_fClippingPlaneMinThreshold - range[0]) /
                             (range[1] -range[0]);
    float maxClipThreshold = (this->m_fClippingPlaneMaxThreshold - range[0]) /
                             (range[1] -range[0]);

    m_pActiveShader->SetFloat1(minClipThreshold, "clippingMinThreshold");
    m_pActiveShader->SetFloat1(maxClipThreshold, "clippingMaxThreshold");
    m_pActiveShader->SetBool1(this->m_bUseGraysScaleValues, "useGrayScaleClippingPlanes");
    if (this->m_pPlane[0]->IsEnabled() && this->m_bUseGraysScaleValues)
    {
        double origin[3];
        this->m_pPlane[0]->GetOrigin(origin);
        m_pActiveShader->SetFloat1(origin[0] / worldDims[0], "clippingX1");

    }
    else
    {
        m_pActiveShader->SetFloat1(-99999, "clippingX1");
    }

    if (this->m_pPlane[1]->IsEnabled() && this->m_bUseGraysScaleValues)
    {
        double origin[3];
        this->m_pPlane[1]->GetOrigin(origin);
        m_pActiveShader->SetFloat1(origin[0] / worldDims[0], "clippingX2");
    }
    else
    {
        m_pActiveShader->SetFloat1(99999, "clippingX2");
    }

    if (this->m_pPlane[2]->IsEnabled()  && this->m_bUseGraysScaleValues)
    {
        double origin[3];
        this->m_pPlane[2]->GetOrigin(origin);
        m_pActiveShader->SetFloat1(origin[1] / worldDims[1], "clippingY1");
    }
    else
    {
        m_pActiveShader->SetFloat1(99999, "clippingY1");
    }

    if (this->m_pPlane[3]->IsEnabled()  && this->m_bUseGraysScaleValues)
    {
        double origin[3];
        this->m_pPlane[3]->GetOrigin(origin);
        m_pActiveShader->SetFloat1(origin[1] / worldDims[1], "clippingY2");
    }
    else
    {
        m_pActiveShader->SetFloat1(99999, "clippingY2");
    }

    if (this->m_pPlane[4]->IsEnabled() && this->m_bUseGraysScaleValues)
    {
        double origin[3];
        this->m_pPlane[4]->GetOrigin(origin);
        m_pActiveShader->SetFloat1(origin[2] / worldDims[2], "clippingZ1");
    }
    else
    {
        m_pActiveShader->SetFloat1(99999, "clippingZ1");
    }

    if (this->m_pPlane[5]->IsEnabled() && this->m_bUseGraysScaleValues)
    {
        double origin[3];
        this->m_pPlane[5]->GetOrigin(origin);
        m_pActiveShader->SetFloat1(origin[2] / worldDims[2], "clippingZ2");
    }
    else
    {
        m_pActiveShader->SetFloat1(99999, "clippingZ2");
    }

    m_pActiveShader->SetFloat3(position[0],position[1],position[2],"cameraposition");

    vtkgl::ActiveTexture( vtkgl::TEXTURE4 );
    glBindTexture( GL_TEXTURE_2D, this->m_fDepthTextureId);
    m_pActiveShader->SetInt1( 4, "depthBuffer" );

    vtkgl::ActiveTexture( vtkgl::TEXTURE2 );
	glBindTexture( BMIA_TEXTURE_3D, this->m_iTextureId );
    m_pActiveShader->SetInt1( 2, "volumeBuffer" );

    vtkgl::ActiveTexture( vtkgl::TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, m_iStartTxId );
    m_pActiveShader->SetInt1( 1, "frontBuffer" );

    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, m_iRayDirTxId );
    m_pActiveShader->SetInt1( 0, "rayBuffer" );

    if( m_pActiveShader == m_pMIP )
    {
    }
    else
        if( m_pActiveShader == m_pDVR )
        {
        // Pass the transfer function texture to the DVR shader.
        vtkgl::ActiveTexture(  vtkgl::TEXTURE3 );
        glBindTexture( GL_TEXTURE_1D, this->m_iColormapTextureId );
        m_pActiveShader->SetInt1( 3, "tfBuffer" );
    }
    else
        if( m_pActiveShader == m_pIsosurface )
        {
        // The isosurface shader requires an isovalue and opacity.
        this->GetInput()->GetScalarRange(range);
        float isoValue = (this->m_fIsoValue - range[0]) / (range[1] - range[0]);
        m_pActiveShader->SetFloat1( isoValue, "isoValue" );
        m_pActiveShader->SetFloat1( m_fIsoValueOpacity, "isoValueOpacity" );

        m_pActiveShader->SetFloat3(m_fIsovalueColor[0],m_fIsovalueColor[1],m_fIsovalueColor[2],"isoValueColor");

        // Overwrite stepsize parameter. We increase it significantly because this will
        // speed up performance. Because we're using hitpoint refinement this will not
        // affect rendering quality.
        m_pActiveShader->SetFloat1( 2.0f, "stepSize" );
    }
    else
        if( m_pActiveShader == m_pToon)
        {
        // The isosurface shader requires an isovalue and opacity.
        this->GetInput()->GetScalarRange(range);
        float isoValue = (this->m_fIsoValue - range[0]) / (range[1] - range[0]);
        m_pActiveShader->SetFloat1( isoValue, "isoValue" );
        m_pActiveShader->SetFloat1( m_fIsoValueOpacity, "isoValueOpacity" );

        m_pActiveShader->SetFloat3(m_fIsovalueColor[0],m_fIsovalueColor[1],m_fIsovalueColor[2],"isoValueColor");

        // Overwrite stepsize parameter. We increase it significantly because this will
        // speed up performance. Because we're using hitpoint refinement this will not
        // affect rendering quality.
        m_pActiveShader->SetFloat1( 2.0f, "stepSize" );
    }

    // end of specific info

    RenderClippingPlanes( worldDims[0], worldDims[1], worldDims[2], 0 );
    RenderTransparentBBox( worldDims[0], worldDims[1], worldDims[2] );

    vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, 0);
    m_pActiveShader->Deactivate();

    glPopMatrix();

    /////////////////////////////////////////
    // RENDER IMAGE TO SCREEN

    // The color texture now contains the rendered image of our volume. We should take
    // this texture and dump it to screen.
    glViewport( 0, 0, m_iWindowWidth, m_iWindowHeight );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    // gluOrtho2D( 0, m_iWindowWidth, 0, m_iWindowHeight );
    glOrtho(0,m_iWindowWidth,0,m_iWindowHeight,-1,1);
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    for( int i = 0; i < 6; i++ )
    {
        this->m_pPlane[i]->Disable();
    }

    //  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glCullFace( GL_BACK );

    glDisable( GL_DEPTH_TEST );

    glEnable( GL_BLEND );

    //check blending function (first one might be GL_ONE).
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    glEnable( GL_TEXTURE_2D );
    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );

	// Ralph: set mode to GL_REPLACE in case other plugins use textures that
	// require a different mode (such as GL_MODULATE). Since the texture
	// environment mode is part of global state (and not the texture object)
	// we have to make sure it is set explicitly here
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );

	glBindTexture( GL_TEXTURE_2D, m_iFinalImageTxId );

    glBegin( GL_QUADS );
    glTexCoord2f( 0, 0 ); glVertex2f( 0, 0 );
    glTexCoord2f( 1, 0 ); glVertex2f( m_iWindowWidth, 0 );
    glTexCoord2f( 1, 1 ); glVertex2f( m_iWindowWidth, m_iWindowHeight );
    glTexCoord2f( 0, 1 ); glVertex2f( 0, m_iWindowHeight );
    glEnd();


    glDisable(GL_TEXTURE_2D);

    glEnable( GL_DEPTH_TEST );
    glEnable( GL_LIGHTING );

    glPopAttrib();

    glFinish();

    for( int i = 0; i < 6; i++ )
        this->m_pPlane[i]->SetEnabled( clipState[i] );

    if (this->m_bInteractiveModeEnabled)
        this->timer.start();
    this->m_bInteractiveModeEnabled = true;

    for( int i = 0; i < 6; i++ )
        this->m_pPlane[i]->SetEnabled( false );
}

void RayCastVolumeMapper::transferFunctionChanged()
{
    if( m_pTable )
        delete [] m_pTable;
    this->m_pTable = 0;

    if( this->m_iColormapTextureId )
    {
        glDeleteTextures( 1, & this->m_iColormapTextureId );
        this->m_iColormapTextureId = 0;
    }
}

void RayCastVolumeMapper::volumeChanged()
{
    if( m_iTextureId > 0 )
        glDeleteTextures( 1, & m_iTextureId );
    m_iTextureId = 0;
}

void RayCastVolumeMapper::reInitClippingPlanes()
{
    m_pClippingPlanesInitialized = false;
}

void RayCastVolumeMapper::InitBuffers( int width, int height )
{
    if( m_iGeometryTxId > 0 )
        glDeleteTextures( 1, & m_iGeometryTxId );
    if( m_iDepthTxId > 0 )
        glDeleteTextures( 1, & m_iDepthTxId );
    if( m_iStartTxId > 0 )
        glDeleteTextures( 1, & m_iStartTxId );
    if( m_iStopTxId > 0 )
        glDeleteTextures( 1, & m_iStopTxId );
    if( m_iRayDirTxId > 0 )
        glDeleteTextures( 1, & m_iRayDirTxId );
    if( m_iFinalImageTxId > 0 )
        glDeleteTextures( 1, & m_iFinalImageTxId );
    if( m_fDepthTextureId > 0 )
        glDeleteTextures( 1, & m_fDepthTextureId );
    if( m_iFboId > 0 )
        vtkgl::DeleteFramebuffersEXT( 1, & m_iFboId );

    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );

    glGenTextures( 1, & m_iGeometryTxId );
    glBindTexture( GL_TEXTURE_2D, m_iGeometryTxId );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures( 1, & m_iDepthTxId );
    glBindTexture( GL_TEXTURE_2D, m_iDepthTxId );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glGenTextures( 1, & m_iStartTxId );
    glBindTexture( GL_TEXTURE_2D, m_iStartTxId );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures( 1, & m_iStopTxId );
    glBindTexture( GL_TEXTURE_2D, m_iStopTxId );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures( 1, & m_iRayDirTxId );
    glBindTexture( GL_TEXTURE_2D, m_iRayDirTxId );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures( 1, & m_iFinalImageTxId );
    glBindTexture( GL_TEXTURE_2D, m_iFinalImageTxId );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
    glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures(1,&m_fDepthTextureId);
    glBindTexture(GL_TEXTURE_2D, m_fDepthTextureId);
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, vtkgl::DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT,GL_FLOAT, 0);

    vtkgl::GenFramebuffersEXT( 1, & m_iFboId );
    vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, 0 );
}

void RayCastVolumeMapper::InitShaders()
{
    if( m_pRayDirections )
        m_pRayDirections->Delete();
    m_pRayDirections = vtkShaderRayDirections::New();
    m_pRayDirections->Load();
    m_pRayDirections->Deactivate();

    if( m_pMIP )
        m_pMIP->Delete();
    m_pMIP = vtkShaderMIP::New();
    m_pMIP->Load();
    m_pMIP->Deactivate();

    if( m_pDVR )
        m_pDVR->Delete();
    m_pDVR = vtkShaderDVR::New();
    m_pDVR->Load();
    m_pDVR->Deactivate();

    if( m_pIsosurface )
        m_pIsosurface->Delete();
    m_pIsosurface = vtkShaderIsosurface::New();
    m_pIsosurface->Load();
    m_pIsosurface->Deactivate();

    if( m_pToon )
        m_pToon->Delete();
    m_pToon = vtkShaderToon::New();
    m_pToon->Load();
    m_pToon->Deactivate();

	switch(this->renderMethod)
	{
	case DVR:
		{
			this->m_pActiveShader = this->m_pDVR;
			break;
		}
	case MIP:
		{
			this->m_pActiveShader = this->m_pMIP;
			break;
		}
	case ISOSURFACE:
		{
			this->m_pActiveShader = this->m_pIsosurface;
			break;
		}
	case TOON:
		{
			this->m_pActiveShader = this->m_pToon;
		}
	}
}

void RayCastVolumeMapper::InitClippingPlanes( vtkVolume * pVolume  )
{
    double bounds[6];
    pVolume->Modified();
    pVolume->GetBounds(bounds);

    if( GetClippingPlanes() )
    {
//        m_pPlane[0]->SetOrigin( bounds[0], 0, 0 );
        m_pPlane[0]->Disable();
        this->clipState[0] = false;
//        m_pPlane[1]->SetOrigin( bounds[1], 0, 0 );
        m_pPlane[1]->Enable();
        this->clipState[1] = false;
//        m_pPlane[2]->SetOrigin( 0, bounds[2], 0 );
        m_pPlane[2]->Disable();
        this->clipState[2] = false;
//        m_pPlane[3]->SetOrigin( 0, bounds[3], 0 );
        m_pPlane[3]->Disable();
        this->clipState[3] = false;
//        m_pPlane[4]->SetOrigin( 0, 0, bounds[4] );
        m_pPlane[4]->Disable();
        this->clipState[4] = false;
//        m_pPlane[5]->SetOrigin( 0, 0, bounds[5] );
        m_pPlane[5]->Disable();
        this->clipState[5] = false;
    }
}

int RayCastVolumeMapper::LoadTexture(vtkVolume * pVol)
{
    int dims[3];
    this->GetInput()->GetDimensions(dims);

    double voxelSize[3];
    this->GetInput()->GetSpacing( voxelSize );

    if( m_iTextureId > 0 )
        glDeleteTextures( 1, & m_iTextureId );

    int scalarSize = this->GetInput()->GetScalarSize();

    m_iNrBits = (scalarSize == 1) ? 8 : 16;
    m_iNrBits = (scalarSize == 4) ? 32: m_iNrBits;
    m_iNrBits = (scalarSize == 8) ? 32: m_iNrBits;

    glGenTextures( 1, & m_iTextureId );
    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );

	glBindTexture( BMIA_TEXTURE_3D, m_iTextureId );
	glTexParameteri( BMIA_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( BMIA_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( BMIA_TEXTURE_3D, GL_TEXTURE_WRAP_S, vtkgl::CLAMP_TO_EDGE );
	glTexParameteri( BMIA_TEXTURE_3D, GL_TEXTURE_WRAP_T, vtkgl::CLAMP_TO_EDGE );
	glTexParameteri( BMIA_TEXTURE_3D, vtkgl::TEXTURE_WRAP_R, vtkgl::CLAMP_TO_EDGE );
	bmiaTexImage3D( BMIA_TEXTURE_3D, 0, GL_ALPHA16, dims[0], dims[1], dims[2],0, GL_ALPHA, GL_FLOAT, 0);

    void * data = this->GetInput()->GetPointData()->GetScalars()->GetVoidPointer( 0 );
    float * dataFloat = NULL;

    if( scalarSize == 1 )
    {
        double range[2];
        this->GetInput()->GetScalarRange(range);
        double newValue;
        double oldValue;
        unsigned char * floatData = (unsigned char*) data;
        dataFloat = (float *) calloc(dims[0] * dims[1] * dims[2], sizeof(float) );
        for(int i = 0; i < dims[0] * dims[1] * dims[2]; i++)
        {
            oldValue = floatData[i];
            newValue = (oldValue - range[0]) / (range[1] - range[0]);
            dataFloat[i] = (float) newValue;
        }

		bmiaTexImage3D( BMIA_TEXTURE_3D, 0, GL_ALPHA16, dims[0], dims[1], dims[2], 0, GL_ALPHA, GL_FLOAT, dataFloat );

        if (dataFloat != NULL)
            delete [] dataFloat;
    }
    else if( scalarSize == 2 )
    {
        double range[2];
        this->GetInput()->GetScalarRange(range);
        double newValue;
        double oldValue;
        unsigned short * floatData = (unsigned short*) data;
        dataFloat = (float *) calloc(dims[0] * dims[1] * dims[2], sizeof(float) );
        for(int i = 0; i < dims[0] * dims[1] * dims[2]; i++)
        {
            oldValue = floatData[i];
            newValue = (oldValue - range[0]) / (range[1] - range[0]);
            dataFloat[i] = (float) newValue;
        }

		bmiaTexImage3D( BMIA_TEXTURE_3D, 0, GL_ALPHA16, dims[0], dims[1], dims[2], 0, GL_ALPHA, GL_FLOAT, dataFloat );

        if (dataFloat != NULL)
            delete [] dataFloat;
    }
    else if( scalarSize == 4 )
    {
        double range[2];
        this->GetInput()->GetScalarRange(range);
        float newValue;
        float oldValue;
        float * floatData = (float*) data;
        dataFloat = (float *) calloc(dims[0] * dims[1] * dims[2], sizeof(float) );
        for(int i = 0; i < dims[0] * dims[1] * dims[2]; i++)
        {
            oldValue = floatData[i];
            newValue = (oldValue - range[0]) / (range[1] - range[0]);
            dataFloat[i] = (float) newValue;
        }

		bmiaTexImage3D( BMIA_TEXTURE_3D, 0, GL_ALPHA16, dims[0], dims[1], dims[2], 0, GL_ALPHA, GL_FLOAT, dataFloat );

        if (dataFloat != NULL)
            delete [] dataFloat;
    }
    else if (scalarSize == 8 )
    {
        double range[2];
        this->GetInput()->GetScalarRange(range);
        double newValue;
        double oldValue;
        double * floatData = (double*) data;
        dataFloat = (float *) calloc(dims[0] * dims[1] * dims[2], sizeof(float) );
        for(int i = 0; i < dims[0] * dims[1] * dims[2]; i++)
        {
            oldValue = floatData[i];
            newValue = (oldValue - range[0]) / (range[1] - range[0]);
            dataFloat[i] = (float) newValue;
        }

       bmiaTexImage3D( BMIA_TEXTURE_3D, 0, GL_ALPHA16, dims[0], dims[1], dims[2], 0, GL_ALPHA, GL_FLOAT, dataFloat );

        if (dataFloat != NULL)
            delete [] dataFloat;
    }
    else
    {
        std::cout << "RayCastVolumeMapper::LoadTexture() could not load texture" << std::endl;
    }

	glBindTexture( BMIA_TEXTURE_3D, 0 );

    // Store volume dimensions and voxel spacing.
    m_iDimensions[0] = dims[0];
    m_iDimensions[1] = dims[1];
    m_iDimensions[2] = dims[2];

    double spacing[3];
    this->GetInput()->GetSpacing( spacing );
    m_dSpacing[0] = spacing[0];
    m_dSpacing[1] = spacing[1];
    m_dSpacing[2] = spacing[2];
    return 1;
}

void RayCastVolumeMapper::InitTransferFunctionTexture(vtkVolume* pVol)
{
    if( ! m_pTable )
        DiscretizePiecewiseLinear(pVol);

    if( m_iColormapTextureId )
        glDeleteTextures( 1, & m_iColormapTextureId );

    glGenTextures( 1, & m_iColormapTextureId );
    glBindTexture( GL_TEXTURE_1D, m_iColormapTextureId );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexImage1D( GL_TEXTURE_1D, 0, GL_RGBA, m_iResolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_pTable );
    glBindTexture( GL_TEXTURE_1D, 0 );
}

void RayCastVolumeMapper::DiscretizePiecewiseLinear(vtkVolume* pVol)
{
    if( m_pTable )
        delete [] m_pTable;

    m_pTable = new unsigned char[4 * m_iResolution];
    for( int i = 0; i < 4 * m_iResolution; i++ )
        m_pTable[i] = 0;

    double range[2];
    this->GetInput()->GetScalarRange(range);

    int k = 0;
    double step;
    float color[4] = { 0, 0, 0, 0 };
    float xRange = static_cast<float>(range[1] - range[0]);

    for( int i = 0; i < m_iResolution; i++ )
    {
        step = i / (double)(m_iResolution - 1);
        step = step * xRange;
        color[0] = pVol->GetProperty()->GetRGBTransferFunction()->GetRedValue(step);
        color[1] = pVol->GetProperty()->GetRGBTransferFunction()->GetGreenValue(step);
        color[2] = pVol->GetProperty()->GetRGBTransferFunction()->GetBlueValue(step);
        color[3] = pVol->GetProperty()->GetScalarOpacity()->GetValue(step);

        // Add color to the color table
        m_pTable[k]   = static_cast<int>(255 * color[0]);
        m_pTable[k+1] = static_cast<int>(255 * color[1]);
        m_pTable[k+2] = static_cast<int>(255 * color[2]);
        m_pTable[k+3] = static_cast<int>(255 * color[3]);
        k += 4;
    }
}

void RayCastVolumeMapper::RenderBBox( float fX, float fY, float fZ )
{
    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
    glBegin( GL_QUADS );

    glNormal3f( 0, 0, -1 );
    glTexCoord3f( 0, 0, 0 ); glColor3f( 0, 0, 0 ); glVertex3f( 0, 0, 0 );
    glTexCoord3f( 0, 1, 0 ); glColor3f( 0, 1, 0 ); glVertex3f( 0, fY, 0 );
    glTexCoord3f( 1, 1, 0 ); glColor3f( 1, 1, 0 ); glVertex3f( fX, fY, 0 );
    glTexCoord3f( 1, 0, 0 ); glColor3f( 1, 0, 0 ); glVertex3f( fX, 0, 0 ); // back

    glNormal3f( 0, 0, 1 );
    glTexCoord3f( 0, 0, 1 ); glColor3f( 0, 0, 1 ); glVertex3f( 0, 0, fZ );
    glTexCoord3f( 1, 0, 1 ); glColor3f( 1, 0, 1 ); glVertex3f( fX, 0, fZ );
    glTexCoord3f( 1, 1, 1 ); glColor3f( 1, 1, 1 ); glVertex3f( fX, fY, fZ );
    glTexCoord3f( 0, 1, 1 ); glColor3f( 0, 1, 1 ); glVertex3f( 0, fY, fZ ); // front

    glNormal3f( 0, 1, 0 );
    glTexCoord3f( 0, 1, 0 ); glColor3f( 0, 1, 0 ); glVertex3f( 0, fY, 0 );
    glTexCoord3f( 0, 1, 1 ); glColor3f( 0, 1, 1 ); glVertex3f( 0, fY, fZ );
    glTexCoord3f( 1, 1, 1 ); glColor3f( 1, 1, 1 ); glVertex3f( fX, fY, fZ );
    glTexCoord3f( 1, 1, 0 ); glColor3f( 1, 1, 0 ); glVertex3f( fX, fY, 0 ); // top

    glNormal3f( 0, -1, 0 );
    glTexCoord3f( 0, 0, 0 ); glColor3f( 0, 0, 0 ); glVertex3f( 0, 0, 0 );
    glTexCoord3f( 1, 0, 0 ); glColor3f( 1, 0, 0 ); glVertex3f( fX, 0, 0 );
    glTexCoord3f( 1, 0, 1 ); glColor3f( 1, 0, 1 ); glVertex3f( fX, 0, fZ );
    glTexCoord3f( 0, 0, 1 ); glColor3f( 0, 0, 1 ); glVertex3f( 0, 0, fZ ); // bottom

    glNormal3f( -1, 0, 0 );
    glTexCoord3f( 0, 0, 0 ); glColor3f( 0, 0, 0 ); glVertex3f( 0, 0, 0 );
    glTexCoord3f( 0, 0, 1 ); glColor3f( 0, 0, 1 ); glVertex3f( 0, 0, fZ );
    glTexCoord3f( 0, 1, 1 ); glColor3f( 0, 1, 1 ); glVertex3f( 0, fY, fZ );
    glTexCoord3f( 0, 1, 0 ); glColor3f( 0, 1, 0 ); glVertex3f( 0, fY, 0 ); // left

    glNormal3f( 1, 0, 0 );
    glTexCoord3f( 1, 0, 0 ); glColor3f( 1, 0, 0 ); glVertex3f( fX, 0, 0 );
    glTexCoord3f( 1, 1, 0 ); glColor3f( 1, 1, 0 ); glVertex3f( fX, fY, 0 );
    glTexCoord3f( 1, 1, 1 ); glColor3f( 1, 1, 1 ); glVertex3f( fX, fY, fZ );
    glTexCoord3f( 1, 0, 1 ); glColor3f( 1, 0, 1 ); glVertex3f( fX, 0, fZ ); // right

    glEnd();
}

void RayCastVolumeMapper::RenderTransparentBBox( float fX, float fY, float fZ )
{
    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
    glColor4f(0,0,0,0);
    glBegin( GL_QUADS );

    glNormal3f( 0, 0, -1 );
    glTexCoord3f( 0, 0, 0 ); glVertex3f( 0, 0, 0 );
    glTexCoord3f( 0, 1, 0 ); glVertex3f( 0, fY, 0 );
    glTexCoord3f( 1, 1, 0 ); glVertex3f( fX, fY, 0 );
    glTexCoord3f( 1, 0, 0 ); glVertex3f( fX, 0, 0 ); // back

    glNormal3f( 0, 0, 1 );
    glTexCoord3f( 0, 0, 1 ); glVertex3f( 0, 0, fZ );
    glTexCoord3f( 1, 0, 1 ); glVertex3f( fX, 0, fZ );
    glTexCoord3f( 1, 1, 1 ); glVertex3f( fX, fY, fZ );
    glTexCoord3f( 0, 1, 1 ); glVertex3f( 0, fY, fZ ); // front

    glNormal3f( 0, 1, 0 );
    glTexCoord3f( 0, 1, 0 ); glVertex3f( 0, fY, 0 );
    glTexCoord3f( 0, 1, 1 ); glVertex3f( 0, fY, fZ );
    glTexCoord3f( 1, 1, 1 ); glVertex3f( fX, fY, fZ );
    glTexCoord3f( 1, 1, 0 ); glVertex3f( fX, fY, 0 ); // top

    glNormal3f( 0, -1, 0 );
    glTexCoord3f( 0, 0, 0 ); glVertex3f( 0, 0, 0 );
    glTexCoord3f( 1, 0, 0 ); glVertex3f( fX, 0, 0 );
    glTexCoord3f( 1, 0, 1 ); glVertex3f( fX, 0, fZ );
    glTexCoord3f( 0, 0, 1 ); glVertex3f( 0, 0, fZ ); // bottom

    glNormal3f( -1, 0, 0 );
    glTexCoord3f( 0, 0, 0 ); glVertex3f( 0, 0, 0 );
    glTexCoord3f( 0, 0, 1 ); glVertex3f( 0, 0, fZ );
    glTexCoord3f( 0, 1, 1 ); glVertex3f( 0, fY, fZ );
    glTexCoord3f( 0, 1, 0 ); glVertex3f( 0, fY, 0 ); // left

    glNormal3f( 1, 0, 0 );
    glTexCoord3f( 1, 0, 0 ); glVertex3f( fX, 0, 0 );
    glTexCoord3f( 1, 1, 0 ); glVertex3f( fX, fY, 0 );
    glTexCoord3f( 1, 1, 1 ); glVertex3f( fX, fY, fZ );
    glTexCoord3f( 1, 0, 1 ); glVertex3f( fX, 0, fZ ); // right

    glEnd();
}


void RayCastVolumeMapper::RenderClippingPlanes( float fX, float fY, float fZ, unsigned int iTxId )
{
    double origin[3];
    double normal[3];
    glDisable( GL_CULL_FACE );

    if( this->m_pPlane[0]->IsEnabled() )
    {
        m_pPlane[0]->GetOrigin( origin );
        m_pPlane[0]->GetNormal( normal );
        m_pPlane[0]->Disable();
        glBegin( GL_QUADS );
        glColor3f( origin[0] / fX, 0, 0 ); glVertex3f( origin[0], 0,  0 );
        glColor3f( origin[0] / fX, 0, 1 ); glVertex3f( origin[0], 0,  fZ );
        glColor3f( origin[0] / fX, 1, 1 ); glVertex3f( origin[0], fY, fZ );
        glColor3f( origin[0] / fX, 1, 0 ); glVertex3f( origin[0], fY, 0 );
        glEnd();
        m_pPlane[0]->Enable();
    }

    if( this->m_pPlane[1]->IsEnabled() )
    {
        this->m_pPlane[1]->GetOrigin( origin );
        this->m_pPlane[1]->GetNormal( normal );
        this->m_pPlane[1]->Disable();
        glBegin( GL_QUADS );
        glColor3f( origin[0] / fX, 0, 0 ); glVertex3f( origin[0], 0,  0 );
        glColor3f( origin[0] / fX, 0, 1 ); glVertex3f( origin[0], 0,  fZ );
        glColor3f( origin[0] / fX, 1, 1 ); glVertex3f( origin[0], fY, fZ );
        glColor3f( origin[0] / fX, 1, 0 ); glVertex3f( origin[0], fY, 0 );
        glEnd();
        this->m_pPlane[1]->Enable();
    }

    if( this->m_pPlane[2]->IsEnabled() )
    {
        this->m_pPlane[2]->GetOrigin( origin );
        this->m_pPlane[2]->GetNormal( normal );
        this->m_pPlane[2]->Disable();
        glBegin( GL_QUADS );
        glColor3f( 0, origin[1] / fY, 0 ); glVertex3f( 0,  origin[1], 0 );
        glColor3f( 0, origin[1] / fY, 1 ); glVertex3f( 0,  origin[1], fZ );
        glColor3f( 1, origin[1] / fY, 1 ); glVertex3f( fX, origin[1], fZ );
        glColor3f( 1, origin[1] / fY, 0 ); glVertex3f( fX, origin[1], 0 );
        glEnd();
        this->m_pPlane[2]->Enable();
    }

    if( this->m_pPlane[3]->IsEnabled() )
    {
        this->m_pPlane[3]->GetOrigin( origin );
        this->m_pPlane[3]->GetNormal( normal );
        this->m_pPlane[3]->Disable();
        glBegin( GL_QUADS );
        glColor3f( 0, origin[1] / fY, 0 ); glVertex3f( 0,  origin[1], 0 );
        glColor3f( 0, origin[1] / fY, 1 ); glVertex3f( 0,  origin[1], fZ );
        glColor3f( 1, origin[1] / fY, 1 ); glVertex3f( fX, origin[1], fZ );
        glColor3f( 1, origin[1] / fY, 0 ); glVertex3f( fX, origin[1], 0 );
        glEnd();
        this->m_pPlane[3]->Enable();
    }

    if( this->m_pPlane[4]->IsEnabled() )
    {
        this->m_pPlane[4]->GetOrigin( origin );
        this->m_pPlane[4]->GetNormal( normal );
        this->m_pPlane[4]->Disable();
        glBegin( GL_QUADS );
        glColor3f( 0, 0, origin[2] / fZ ); glVertex3f( 0,  0,  origin[2] );
        glColor3f( 1, 0, origin[2] / fZ ); glVertex3f( fX, 0,  origin[2] );
        glColor3f( 1, 1, origin[2] / fZ ); glVertex3f( fX, fY, origin[2] );
        glColor3f( 0, 1, origin[2] / fZ ); glVertex3f( 0,  fY, origin[2] );
        glEnd();
        this->m_pPlane[4]->Enable();
    }

    if( this->m_pPlane[5]->IsEnabled() )
    {
        this->m_pPlane[5]->GetOrigin( origin );
        this->m_pPlane[5]->GetNormal( normal );
        this->m_pPlane[5]->Disable();
        glBegin( GL_QUADS );
        glColor3f( 0, 0, origin[2] / fZ ); glVertex3f( 0,  0,  origin[2] );
        glColor3f( 1, 0, origin[2] / fZ ); glVertex3f( fX, 0,  origin[2] );
        glColor3f( 1, 1, origin[2] / fZ ); glVertex3f( fX, fY, origin[2] );
        glColor3f( 0, 1, origin[2] / fZ ); glVertex3f( 0,  fY, origin[2] );
        glEnd();
        this->m_pPlane[5]->Enable();
    }

    glEnable( GL_CULL_FACE );

}

vtkClippingPlane* RayCastVolumeMapper::getPlane(int index)
{
    return this->m_pPlane[index];
}

void RayCastVolumeMapper::timeout()
{
    this->m_bInteractiveModeEnabled = false;
    emit render();
}

void RayCastVolumeMapper::setRenderMethod(RENDERMETHOD rRenderMethod)
{
    switch(rRenderMethod)
    {
    case DVR:
        {
            this->m_pActiveShader = this->m_pDVR;
            break;
        }
    case MIP:
        {
            this->m_pActiveShader = this->m_pMIP;
            break;
        }
    case ISOSURFACE:
        {
            this->m_pActiveShader = this->m_pIsosurface;
            break;
        }
    case TOON:
        {
            this->m_pActiveShader = this->m_pToon;
        }
    }

	this->renderMethod = rRenderMethod;
}

void RayCastVolumeMapper::setStepsize(float stepSize)
{
    this->m_fInternalStepSize = stepSize;
}

void RayCastVolumeMapper::setInteractiveStepSize(float interactiveStepsize)
{
    this->m_fInteractiveStepSize = interactiveStepsize;
}

void RayCastVolumeMapper::setIsoValue(float isovalue)
{
    this->m_fIsoValue = isovalue;
}

void RayCastVolumeMapper::setIsoValueOpacity(float isovalueOpacity)
{
    this->m_fIsoValueOpacity = isovalueOpacity;
}

void RayCastVolumeMapper::setClippingPlanesThreshold(float min, float max)
{
    this->m_fClippingPlaneMinThreshold = min;
    this->m_fClippingPlaneMaxThreshold = max;
}

void RayCastVolumeMapper::setUseGrayScaleValues(bool value)
{
    this->m_bUseGraysScaleValues = value;
}

void RayCastVolumeMapper::setEnableClippingPlane(int plane, bool enabled)
{
    Q_ASSERT((plane < 6) && (plane >= 0));
    this->clipState[plane] = enabled;
}

void RayCastVolumeMapper::RenderNearClippingPlane(vtkRenderer * pRen, float fX, float fY, float fZ )
{
    int * size = pRen->GetSize();
    double dispLL[] = { 0.0,     0.0,     1 };
    double dispLR[] = { size[0], 0.0,     1 };
    double dispUL[] = { 0.0,     size[1], 1 };
    double dispUR[] = { size[0], size[1], 1 };

    double worldLL[4];
    double worldLR[4];
    double worldUL[4];
    double worldUR[4];

    pRen->SetDisplayPoint( dispLL[0], dispLL[1], dispLL[2] );
    pRen->DisplayToWorld();
    pRen->GetWorldPoint( worldLL );

    pRen->SetDisplayPoint( dispLR[0], dispLR[1], dispLR[2] );
    pRen->DisplayToWorld();
    pRen->GetWorldPoint( worldLR );

    pRen->SetDisplayPoint( dispUL[0], dispUL[1], dispUL[2] );
    pRen->DisplayToWorld();
    pRen->GetWorldPoint( worldUL );

    pRen->SetDisplayPoint( dispUR[0], dispUR[1], dispUR[2] );
    pRen->DisplayToWorld();
    pRen->GetWorldPoint( worldUR );

    double texLL[] = { worldLL[0] / fX, worldLL[1] / fY, worldLL[2] / fZ };
    double texLR[] = { worldLR[0] / fX, worldLR[1] / fY, worldLR[2] / fZ };
    double texUL[] = { worldUL[0] / fX, worldUL[1] / fY, worldUL[2] / fZ };
    double texUR[] = { worldUR[0] / fX, worldUR[1] / fY, worldUR[2] / fZ };

    glBegin( GL_QUADS );
    glColor3f( texLL[0], texLL[1], texLL[2] ); glVertex3f( worldLL[0], worldLL[1], worldLL[2] );
    glColor3f( texLR[0], texLR[1], texLR[2] ); glVertex3f( worldLR[0], worldLR[1], worldLR[2] );
    glColor3f( texUR[0], texUR[1], texUR[2] ); glVertex3f( worldUR[0], worldUR[1], worldUR[2] );
    glColor3f( texUL[0], texUL[1], texUL[2] ); glVertex3f( worldUL[0], worldUL[1], worldUL[2] );
    glEnd();
}

void RayCastVolumeMapper::worldToObject(vtkMatrix4x4* pMatrix, double point[3])
{
    double positionH[4];
    double positionH2[4];
    positionH[0] = point[0];
    positionH[1] = point[1];
    positionH[2] = point[2];
    positionH[3] = 1.0f;
    pMatrix->MultiplyPoint(positionH,positionH2);
    point[0] = positionH2[0];
    point[1] = positionH2[1];
    point[2] = positionH2[2];
}

void RayCastVolumeMapper::setExternalTransformationMatrix(vtkMatrix4x4* pMatrix)
{
    this->m_pExternalTransformationMatrix = pMatrix;
}

bool RayCastVolumeMapper::extensionsSupported()
{
    return this->m_bExtensionsSupported;
}

void RayCastVolumeMapper::setIsoValueColor(float red, float green, float blue)
{
    m_fIsovalueColor[0] = red;
    m_fIsovalueColor[1] = green;
    m_fIsovalueColor[2] = blue;
}


