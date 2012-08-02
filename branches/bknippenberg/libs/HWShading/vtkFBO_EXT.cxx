/**
 * vtkFBO.cxx
 * by Tim Peeters
 *
 * 2008-03-01	Tim Peeters
 * - First version, based on vtkFBO.cxx
 */

#include "vtkFBO.h"
#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>    // for light camera
#include <vtkMatrix4x4.h> // for light camera

#include "vtkUniformSampler.h"
#include "vtkShaderObjectCollection.h"

#include <assert.h>

namespace bmia {

#define CHECK_FRAMEBUFFER_STATUS() \
{ \
GLenum status; \
status = glCheckFramebufferStatusEXT(vtkgl::FRAMEBUFFER_EXT); \
switch(status) { \
case vtkgl::FRAMEBUFFER_COMPLETE_EXT: \
vtkDebugMacro(<<"Framebuffer objects supported."); \
break; \
case vtkgl::FRAMEBUFFER_UNSUPPORTED_EXT: \
/* choose different formats */ \
vtkErrorMacro(<<"Framebuffer objects not supported!"); \
break; \
default: \
/* programming error; will fail on all hardware */ \
vtkErrorMacro(<<"FBO programming error; will fail on all hardware!"); \
} \
}

vtkStandardNewMacro(vtkFBO);

vtkFBO::vtkFBO() {
  this->Initialized = false;
  this->Sampler = vtkUniformSampler::New();
  this->Sampler->SetName("OutputTexture1"); // TODO: assign the sampler from the outside?
  this->Sampler2 = vtkUniformSampler::New();
  this->Sampler2->SetName("OutputTexture2");;

//  this->Width = 1024;
//  this->Height = 1024;
  this->Viewport[0] = this->Viewport[1] = 0;
  this->Viewport[2] = this->Viewport[3] = 1024;

  this->TwoTextures = false;
  this->Textures[0] = 999;
  this->Textures[1] = 999;
}

vtkFBO::~vtkFBO() {
  this->Sampler->Delete(); this->Sampler = NULL;
  this->Sampler2->Delete(); this->Sampler = NULL;

  if (this->Textures[0] != 999) this->Clean();
}

void vtkFBO::Initialize()
{
  if (this->Textures[0] != 999) this->Clean();


  // TODO: check whether needed extensions are supported
  //
  GLint maxbuffers;
  glGetIntegerv(vtkgl::MAX_COLOR_ATTACHMENTS_EXT, &maxbuffers);
//  cout<<"maxbuffers = "<<maxbuffers<<endl;
  assert(maxbuffers >= 2);

  // =========================
  // Create the shadow texture
  // =========================

  // Generate one texture name:
  if (this->TwoTextures)
    {
    glGenTextures(2, this->Textures);
    }
  else
    {
    glGenTextures(1, this->Textures);
    }

  //glGenTextures(1, &(this->Texture));

//  vtkgl::ActiveTexture(vtkgl::TEXTURE0);
  // Bind texture to texturing target: 
  glBindTexture(vtkgl::TEXTURE_RECTANGLE_ARB, this->Textures[0]);
  this->TextureParameters();

//  vtkgl::ActiveTexture(vtkgl::TEXTURE1);
  glBindTexture(vtkgl::TEXTURE_RECTANGLE_ARB, this->Textures[1]);
  this->TextureParameters();

  // create FBO
  vtkgl::GenFramebuffersEXT(1, &(this->FBO));
  vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, this->FBO);

//  glDrawBuffer(GL_NONE);
//  glReadBuffer(GL_NONE);

  // bind texture to FBO
  // 0 is the number of mipmap levels. We want none.
  vtkgl::FramebufferTexture2DEXT(vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT, vtkgl::TEXTURE_RECTANGLE_ARB, this->Textures[0], 0);
  if (this->TwoTextures)
    {
    vtkgl::FramebufferTexture2DEXT(vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT1_EXT, vtkgl::TEXTURE_RECTANGLE_ARB, this->Textures[1], 0);
    GLenum buffers[] = { vtkgl::COLOR_ATTACHMENT0_EXT, vtkgl::COLOR_ATTACHMENT1_EXT };
    vtkgl::DrawBuffers(2, buffers);
//    glDrawBuffer(vtkgl::COLOR_ATTACHMENT0_EXT);
    }
  else
    {
    glDrawBuffer(vtkgl::COLOR_ATTACHMENT0_EXT);
    }

  // the render buffer for storing intermediate depth values.
  vtkgl::GenRenderbuffersEXT(1, &(this->DepthRBO));
  vtkgl::BindRenderbufferEXT(vtkgl::RENDERBUFFER_EXT, this->DepthRBO);
  vtkgl::RenderbufferStorageEXT(vtkgl::RENDERBUFFER_EXT, vtkgl::DEPTH_COMPONENT24, this->Viewport[2], this->Viewport[3]); //Width, this->Height);
  vtkgl::FramebufferRenderbufferEXT(vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT, vtkgl::RENDERBUFFER_EXT, this->DepthRBO);

 // glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_RECTANGLE_ARB, this->ShadowTexture, 0);
//  CHECK_FRAMEBUFFER_STATUS();

  GLenum status;
  status = vtkgl::CheckFramebufferStatusEXT(vtkgl::FRAMEBUFFER_EXT);

  switch(status) {
  case vtkgl::FRAMEBUFFER_COMPLETE_EXT:
    vtkDebugMacro(<<"Framebuffer objects supported.");
    break;
  case vtkgl::FRAMEBUFFER_UNSUPPORTED_EXT:
    // choose different formats
    vtkErrorMacro(<<"Framebuffer objects not supported!");
    break;
  case vtkgl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
    vtkErrorMacro(<<"FBO: Incomplete attachment!");
    break;
  case vtkgl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
    vtkErrorMacro(<<"GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT");
    break;
//  case vtkgl::FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
//    vtkErrorMacro(<<"FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT");
//    break;
  case vtkgl::FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
    vtkErrorMacro(<<"FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT");
    break;
  case vtkgl::FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
    vtkErrorMacro(<<"FRAMEBUFFER_INCOMPLETE_FORMATS_EXT");
    break;
  case vtkgl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
    vtkErrorMacro(<<"FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT");
    break;
  case vtkgl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
    vtkErrorMacro(<<"FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT");
    break;
//  case vtkgl::FRAMEBUFFER_STATUS_ERROR_EXT:
//    vtkErrorMacro(<<"FBO: status error!");
//    break;
  default:
    // programming error; will fail on all hardware
    vtkErrorMacro(<<"FBO programming error; will fail on all hardware!");
    //assert(0);
  }
  vtkDebugMacro(<<"Shadowmap texture initialized.");

  // TODO: check what the 0 here is. maybe I need different values if I use multiple FBOs
//  vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, 0); 
  this->Initialized = true;
  vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, 0);
}

void vtkFBO::Clean()
{
  vtkgl::DeleteFramebuffersEXT(1, &(this->FBO));
  vtkgl::DeleteRenderbuffersEXT(1, &(this->DepthRBO));
  if (this->TwoTextures)
    {
    glDeleteTextures(2, this->Textures);
    }
  else
    { // only one texture
    glDeleteTextures(1, this->Textures);
    }
}

// TODO: speed this up. Generation of shadow map is too slow. This can
// be tested by making the call to RegenerateShadowMap in DeviceRender()
// unconditional.
void vtkFBO::PreRender(vtkCamera* lightCamera)
{
  // first, store the matrices that I am going to change temporarily
  // for rendering the shadow map. These will be restored in
  // PostRenderShadowMap().
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glMatrixMode(GL_TEXTURE);
  glPushMatrix();
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();

  // TODO: leave scissor test enabled, but choose correct range for scissoring.
  glDisable(GL_SCISSOR_TEST);
  vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, this->FBO);
//glDrawBuffer(vtkgl::COLOR_ATTACHMENT0_EXT);
//vtkgl::ActiveTexture(vtkgl::TEXTURE1);

  glGetIntegerv(GL_VIEWPORT, this->WindowViewport);
//  glViewport(0, 0, this->Width, this->Height);
//  glScissor(0, 0, this->Width, this->Height);
//  glViewport(this->Viewport[0], this->Viewport[1], this->Viewport[2], this->Viewport[3]);
  glViewport(0, 0, this->Viewport[2], this->Viewport[3]);
//  glScissor(this->Viewport[0], this->Viewport[1], this->Viewport[0]+this->Viewport[2], this->Viewport[1]+this->Viewport[3]); // its probably not even enabled.
  glScissor(0, 0, this->Viewport[2], this->Viewport[3]);

  vtkMatrix4x4* matrix = vtkMatrix4x4::New();
  vtkDebugMacro(<<"Clear depth buffer");
  GLbitfield  clear_mask = 0;

  // store the clear color to set it back later.
  glGetFloatv(GL_COLOR_CLEAR_VALUE, this->ColorClearValue);
  // set the new clear color.
  glClearColor(0, 0, 0, 0);
//  glClearColor(1, 1, 0, 1);
//  glClearColor(
//	((double)rand())/(double)RAND_MAX,
//	((double)rand())/(double)RAND_MAX,
//	((double)rand())/(double)RAND_MAX,
//	1);

  clear_mask |= GL_COLOR_BUFFER_BIT;

  glClearDepth( (GLclampd)( 1.0 ) );
  clear_mask |= GL_DEPTH_BUFFER_BIT;
  glClear(clear_mask);
  // Set the viewport size to the shadow map size:
  // XXX: COMMENTED OUT BECAUSE IT MESSES UP THE VIEW IN vtkFiberMapper, but not in vtkShadowRenderer.
  //glViewport(0, 0, 1024, 1024);	//TODO: find out how I can use this without messing up the view.
//  glViewport(0, 0, 800, 550);

  // Deactivate writing in the color buffer (for better performance):
  //glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  // Set up projection parameters:
//  glMatrixMode(GL_PROJECTION);
//  matrix->DeepCopy(lightCamera->GetPerspectiveTransformMatrix(1, 0, 1)); //TODO: replace first 1 by aspect ratio
//  matrix->Transpose();
//  glLoadMatrixd(matrix->Element[0]);

/*
  // Also add this to the texture matrix.
  vtkgl::ActiveTexture(vtkgl::TEXTURE0);
  glMatrixMode(GL_TEXTURE);
  glLoadMatrixd(matrix->Element[0]);
  // Set up modelview parameters
  glMatrixMode(GL_MODELVIEW);
  matrix->DeepCopy(lightCamera->GetViewTransformMatrix());
  matrix->Transpose();
  glLoadMatrixd(matrix->Element[0]);

  // Also add this to the texture matrix
  glMatrixMode(GL_TEXTURE);
  glMultMatrixd(matrix->Element[0]);

  // store the texture matrix because it will be used later in SetupTextureMatrix.
  glGetDoublev(GL_TEXTURE_MATRIX, this->StoredTextureMatrix);
*/
  glMatrixMode(GL_MODELVIEW);
  //glGetDoublev(GL_TEXTURE_MATRIX, this->StoredTextureMatrix);
  matrix->Delete(); matrix = NULL;
  //glPolygonOffset(po_scale, po_bias);
  //glEnable(GL_POLYGON_OFFSET_FILL);
  //glPolygonOffset(5.0, 2.0);
}

// Draw geometry:
// ren->UpdateGeometry(); // TODO: check if this does nothing with the matrices and/or
			  // switch from modelview to other matrices?

void vtkFBO::PostRender()
{
//  glViewport(0, 0, this->ShadowMapWidth, this->ShadowMapHeight);

  // restore the projection matrix:
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  // restore the viewport of the window.
  glViewport(this->WindowViewport[0], this->WindowViewport[1], this->WindowViewport[2], this->WindowViewport[3]);
  //glScissor(this->WindowViewport[0], this->WindowViewport[1], this->WindowViewport[2], this->WindowViewport[3]);

  vtkgl::BindFramebufferEXT(vtkgl::FRAMEBUFFER_EXT, 0);
  // restore the texture and modelview matrices:
  glMatrixMode(GL_TEXTURE);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  // restore the original clear color.
  glClearColor(this->ColorClearValue[0], this->ColorClearValue[1], this->ColorClearValue[2], this->ColorClearValue[3]);

  // XXX: is glActiveTexture(GL_TEXTURE0); needed?
  // appears not.. what is glActivateTexture useful for then?
//  vtkgl::ActiveTexture(vtkgl::TEXTURE0);
//  this->Sampler->SetValue(0); // is this needed???
  // XXX: make sure actors w/ textures etc don't mess things up.
  // see how different textures for different actors are handled by VTK.

  // Store the depth buffer in the shadow texture:
//  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, this->Textures[0]);

//  vtkgl::ActiveTexture(vtkgl::TEXTURE1);
//  this->Sampler2->SetValue(1);
//  glBindTexture(GL_TEXTURE_RECTANGLE_ARB, this->Textures[1]);

//  vtkgl::ActiveTexture(vtkgl::TEXTURE0);

  // disable the offset again
  //glDisable(GL_POLYGON_OFFSET_FILL);

  // Activate writing in the color buffer
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

  glMatrixMode(GL_MODELVIEW);
}

void vtkFBO::ActiveTextures()
{
  vtkgl::ActiveTexture(vtkgl::TEXTURE0);
  glBindTexture(vtkgl::TEXTURE_RECTANGLE_ARB, this->Textures[0]);
  this->Sampler->SetValue(0);

  vtkgl::ActiveTexture(vtkgl::TEXTURE1);
  glBindTexture(vtkgl::TEXTURE_RECTANGLE_ARB, this->Textures[1]);
  this->Sampler2->SetValue(1);

  vtkgl::ActiveTexture(vtkgl::TEXTURE0);
}

void vtkFBO::SetupTextureMatrix(vtkCamera* cam)
{
  // first, store the old matrix so that it can be restored in RestoreTextureMatrix().
  glMatrixMode(GL_TEXTURE);
  glPushMatrix();
  GLdouble* m = this->StoredTextureMatrix;
  //cout<<"Setting texture matrix to "<<m[0]<<", "<<m[1]<<", "<<m[2]<<", "<<m[3]<<", "<<m[4]<<", "<<m[5]<<", "<<m[6]<<", "<<m[7]<<", "<<m[8]<<", "<<m[9]<<", "<<m[10]
	//	<<", "<<m[11]<<", "<<m[12]<<", "<<m[13]<<", "<<m[14]<<", "<<m[15]<<"."<<endl;
  glLoadMatrixd(this->StoredTextureMatrix);

  // use the texture matrix for conversion between camera and light coordinates
  vtkMatrix4x4* viewTransformMatrix = cam->GetViewTransformMatrix();
  vtkMatrix4x4* inverseViewTransformMatrix = vtkMatrix4x4::New();
  inverseViewTransformMatrix->DeepCopy(viewTransformMatrix);
  inverseViewTransformMatrix->Invert();
  inverseViewTransformMatrix->Transpose();
  //glMatrixMode(GL_TEXTURE);
  //glPushMatrix();

  glMultMatrixd(inverseViewTransformMatrix->Element[0]);  
  inverseViewTransformMatrix->Delete();
  inverseViewTransformMatrix = NULL;
  viewTransformMatrix = NULL;

  glMatrixMode(GL_MODELVIEW);
}

void vtkFBO::RestoreTextureMatrix()
{
  glMatrixMode(GL_TEXTURE);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void vtkFBO::TextureParameters()
{
  // Specify a 2D texture
  // Levels of detail: 0 (no mipmap)
  // Internal components: Depth component
  // Width, Height: 512, 512
  // Border: 0
  // Format: Depth component
  // Type: unsigned byte
  // Image data pointer: NULL
//  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA32F_ARB, this->Width, this->Height, 0, GL_RGBA, GL_FLOAT, NULL);
  glTexImage2D(vtkgl::TEXTURE_RECTANGLE_ARB, 0, vtkgl::RGBA32F_ARB, this->Viewport[2], this->Viewport[3], 0, GL_RGBA, GL_FLOAT, NULL);
//  glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, this->ShadowMapWidth, this->ShadowMapHeight, 0, GL_RGB, GL_SHORT, NULL);

  // Linear interpolation for pixel values when pixel is > or <= one
  // texture element:
  glTexParameteri(vtkgl::TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(vtkgl::TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  //glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  //glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Clamp (and not repeat) parameters for texture coordinates:
  glTexParameteri(vtkgl::TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(vtkgl::TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP);
} 

} // namespace bmia
