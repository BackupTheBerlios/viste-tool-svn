/**
 * vtkGlyphMapper.cxx
 * by Tim Peeters
 *
 * 2008-02-27	Tim Peeters
 * - First version
 */

#include "vtkGlyphMapper.h"
#include <assert.h>
#include <vtkObjectFactory.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkPointSet.h>
#include <vtkOpenGLExtensionManager.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCamera.h>

#include "vtkUniformFloat.h"
#include "vtkUniformIvec3.h"
#include "vtkUniformVec3.h"
#include "vtkMyShaderProgram.h"
#include "vtkFBO.h"
#include "vtkFiberMapper.h"
#include "vtkUniformSampler.h"

#include "vtkUniformBool.h"
//#include "glext.h"

namespace bmia {

vtkGlyphMapper::vtkGlyphMapper()
{
  this->Initialized = false;
  this->SPDepth = NULL;
  this->SPLight = NULL;
  this->SeedPoints = NULL;
  this->MaxGlyphRadius = 0.5;
  this->FBODepth = vtkFBO::New();
  this->FBODepth->GetSampler()->SetName("IntersectionTexture");
  this->FBODepth->GetSampler2()->SetName("GlyphCenterTexture");
  this->FBODepth->TwoTexturesOn();

  this->GlyphScaling = 1.0;
}

void vtkGlyphMapper::SetupShaderUniforms()
{
  this->UEyePosition = vtkUniformVec3::New();
  this->UEyePosition->SetName("EyePosition");
  if (this->SPDepth) this->SPDepth->AddShaderUniform(this->UEyePosition);
  if (this->SPLight) this->SPLight->AddShaderUniform(this->UEyePosition);

  this->ULightPosition = vtkUniformVec3::New();
  this->ULightPosition->SetName("LightPosition");
  if (this->SPLight) this->SPLight->AddShaderUniform(this->ULightPosition); 

  this->UTextureDimensions = vtkUniformIvec3::New();
  this->UTextureDimensions->SetName("TextureDimensions");
  if (this->SPDepth) this->SPDepth->AddShaderUniform(this->UTextureDimensions);
  if (this->SPLight) this->SPLight->AddShaderUniform(this->UTextureDimensions);

  this->UTextureSpacing = vtkUniformVec3::New();
  this->UTextureSpacing->SetName("TextureSpacing");
  if (this->SPDepth) this->SPDepth->AddShaderUniform(this->UTextureSpacing);
  if (this->SPLight) this->SPLight->AddShaderUniform(this->UTextureSpacing);

  this->UMaxGlyphRadius = vtkUniformFloat::New();
  this->UMaxGlyphRadius->SetName("MaxGlyphRadius");
//  if (this->SPDepth) this->SPDepth->AddShaderUniform(this->UMaxGlyphRadius);
//  if (this->SPLight) this->SPLight->AddShaderUniform(this->UMaxGlyphRadius);

  if (this->SPLight)
    {
    this->SPLight->AddShaderUniform(this->FBODepth->GetSampler());
    this->SPLight->AddShaderUniform(this->FBODepth->GetSampler2());
//    this->SPLight->AddShaderUniform(this->FBOShadow->GetSampler());
    }

  this->UNegateX = vtkUniformBool::New();
  this->UNegateX->SetName("NegateX");
  this->UNegateX->SetValue(false);
  this->UNegateY = vtkUniformBool::New();
  this->UNegateY->SetName("NegateY");
  this->UNegateY->SetValue(false);
  this->UNegateZ = vtkUniformBool::New();
  this->UNegateZ->SetName("NegateZ");
  this->UNegateZ->SetValue(false);
  if (this->SPDepth)
    {
    this->SPDepth->AddShaderUniform(this->UNegateX);
    this->SPDepth->AddShaderUniform(this->UNegateY);
    this->SPDepth->AddShaderUniform(this->UNegateZ);
    }
  if (this->SPLight)
    {
    this->SPLight->AddShaderUniform(this->UNegateX);
    this->SPLight->AddShaderUniform(this->UNegateY);
    this->SPLight->AddShaderUniform(this->UNegateZ);
    }

  this->UGlyphScaling = vtkUniformFloat::New();
  this->UGlyphScaling->SetName("GlyphScaling");
  this->UGlyphScaling->SetValue(this->GlyphScaling);
  if (this->SPDepth)
    {
    this->SPDepth->AddShaderUniform(this->UGlyphScaling);
    }
// For now, scaling is not needed in lighting calculations.
//  if (this->SPLight)
//    {
//    this->SPLight->AddShaderUniform(this->UGlyphScaling);
//    }
}

vtkGlyphMapper::~vtkGlyphMapper()
{
  if (this->SPDepth)
    {
    this->SPDepth->Delete();
    this->SPDepth = NULL;
    }

  if (this->SPLight)
    {
    this->SPLight->Delete();
    this->SPLight = NULL;
    }

  this->UEyePosition->Delete(); this->UEyePosition = NULL;
  this->ULightPosition->Delete(); this->ULightPosition = NULL;
  this->UTextureDimensions->Delete(); this->UTextureDimensions = NULL;
  this->UTextureSpacing->Delete(); this->UTextureSpacing = NULL;
  this->UMaxGlyphRadius->Delete(); this->UMaxGlyphRadius = NULL;

  this->UNegateX->Delete(); this->UNegateX = NULL;
  this->UNegateY->Delete(); this->UNegateY = NULL;
  this->UNegateZ->Delete(); this->UNegateZ = NULL;
  
//  this->FBOShadow->Delete(); this->FBOShadow = NULL;
}

void vtkGlyphMapper::Initialize()
{
  cout<<"Initializing!"<<endl;
  assert( this->Initialized == false );

  vtkOpenGLExtensionManager* extensions = vtkOpenGLExtensionManager::New();

  int supports_GL_VERSION_2_0			= extensions->ExtensionSupported("GL_VERSION_2_0");
  int supports_GL_EXT_framebuffer_object 	= extensions->ExtensionSupported("GL_EXT_framebuffer_object");
  int supports_GL_ARB_texture_rectangle		= extensions->ExtensionSupported("GL_ARB_texture_rectangle");
  int supports_GL_ARB_texture_float		= extensions->ExtensionSupported("GL_ARB_texture_float");

  if (supports_GL_VERSION_2_0)
    {
    extensions->LoadExtension("GL_VERSION_2_0");
    }
  else
    {
    vtkWarningMacro(<<"GL_VERSION_2_0 is not supported!");
    return;
    }

  if (supports_GL_EXT_framebuffer_object)
    {
    extensions->LoadExtension("GL_EXT_framebuffer_object");
    }
  else
    {
    vtkWarningMacro(<<"GL_EXT_framebuffer_object not supported!");
    return;
    }

  if (supports_GL_ARB_texture_rectangle)
    {
    extensions->LoadExtension("GL_ARB_texture_rectangle");
    }
  else
    {
    vtkWarningMacro(<<"GL_ARB_texture_rectangle not supported!");
    return;
    }

/*
  if (supports_GL_ARB_texture_float)
    {
    extensions->LoadExtension("GL_ARB_texture_float");
    }
  else
    {
    vtkWarningMacro(<<"GL_ARB_texture_float is not supported!");
    return;
    }
*/
  extensions->Delete(); extensions = NULL;
cout<<"Initializing .."<<endl;
//  glActiveTexture(GL_TEXTURE0);
  this->FBODepth->Initialize();
//  this->FBODepth->GetSampler()->SetValue(0);
//  glActiveTexture(GL_TEXTURE1);
//  this->FBOShadow->Initialize();
//  this->FBOShadow->GetSampler()->SetValue(1);
cout<<"Shadow map initialized!"<<endl;
  this->Initialized = true;
}

void vtkGlyphMapper::SetMaxGlyphRadius(float r)
{
  vtkDebugMacro(<< this->GetClassName() << " ( " <<this<< " ): "<<"setting MaxGlyphRadius to "<<r);
  if (this->MaxGlyphRadius != r)
    {
    this->MaxGlyphRadius = r;
    this->UMaxGlyphRadius->SetValue(this->MaxGlyphRadius);
    this->Modified();
    }
}

vtkCxxSetObjectMacro(vtkGlyphMapper, SeedPoints, vtkPointSet);

void vtkGlyphMapper::LoadTexture(int texture_index, const char* array_name)
{
  // First, get the volume from the input:
  vtkImageData* input = this->GetInput();
  assert(input);
  int dims[3]; double spacing[3];
  input->GetDimensions(dims); input->GetSpacing(spacing);

  // Set the texture parameters in the shaders according to the input:
  this->UTextureDimensions->SetValue(dims);
  this->UTextureSpacing->SetValue((float)spacing[0], (float)spacing[1], (float)spacing[2]);

  // bind 3D texture target
  glBindTexture( vtkgl::TEXTURE_3D, texture_index );

  // set texture parameters
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(vtkgl::TEXTURE_3D, vtkgl::TEXTURE_WRAP_R, GL_CLAMP);
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  // linear interpolation gives errors!
  //glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  //glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  vtkPointData* pd = input->GetPointData();
  assert(pd);
  // XXX: Assuming 4-component float data here. Make this flexible?
  vtkFloatArray* vectors = vtkFloatArray::SafeDownCast(pd->GetScalars(array_name));
  assert(vectors);

  //float* testarray = vectors->GetPointer(0);
  //for (int i=0;i<=dims[0]*dims[1]*dims[2];i++)
  //cout<<testarray[i]<<endl;

  // Load the 3D volume texture to graphics card memory
  vtkgl::TexImage3D(vtkgl::TEXTURE_3D, 0, GL_RGBA,
//	  glTexImage3D(vtkgl::TEXTURE_3D, 0, GL_RGBA,
		    dims[0], dims[1], dims[2],
			0, GL_RGBA, GL_FLOAT,
			(float*) vectors->GetPointer(0));

  // that's it.
}

void vtkGlyphMapper::PrintNumberOfSeedPoints()
{
  vtkPointSet* pointset = this->GetSeedPoints();
  if (!pointset)
    {
    vtkWarningMacro(<<"No point set to render!");
    return;
    }

  vtkIdType num_points = pointset->GetNumberOfPoints();
  cout<<"The number of seed points is: "<<num_points<<endl;
}

void vtkGlyphMapper::DrawPoints()
{
  glDisable(GL_TEXTURE_2D);
  vtkPointSet* pointset = this->GetSeedPoints();
  if (!pointset)
    {
    vtkWarningMacro(<<"No point set to render!");
    return;
    }

  vtkIdType num_points = pointset->GetNumberOfPoints();
  vtkDebugMacro(<<"Going to render "<<num_points<<" points.");

  glPointSize(5.0);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
//  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);

  double p[3];
  double BB[8][3];
  double r = this->MaxGlyphRadius;

  glBegin(GL_QUADS);
  for (vtkIdType i = 0; i < num_points; i++)
    {
    pointset->GetPoint(i, p);
    // TODO: use a display-list for rendering the cubes. 
    // also, not all coordinates need to be recomputed here. camera can be moved.
    BB[0][0] = p[0] - r; BB[0][1] = p[1] - r; BB[0][2] = p[2] - r;
    BB[1][0] = p[0] + r; BB[1][1] = p[1] - r; BB[1][2] = p[2] - r;
    BB[2][0] = p[0] + r; BB[2][1] = p[1] + r; BB[2][2] = p[2] - r;
    BB[3][0] = p[0] - r; BB[3][1] = p[1] + r; BB[3][2] = p[2] - r;
    BB[4][0] = p[0] - r; BB[4][1] = p[1] - r; BB[4][2] = p[2] + r;
    BB[5][0] = p[0] + r; BB[5][1] = p[1] - r; BB[5][2] = p[2] + r;
    BB[6][0] = p[0] + r; BB[6][1] = p[1] + r; BB[6][2] = p[2] + r;
    BB[7][0] = p[0] - r; BB[7][1] = p[1] + r; BB[7][2] = p[2] + r;
//    glVertex3dv(point);
    glTexCoord3dv(p);
    // front
    glVertex3dv(BB[0]); glVertex3dv(BB[3]); glVertex3dv(BB[2]); glVertex3dv(BB[1]);
    // left
    glVertex3dv(BB[0]); glVertex3dv(BB[4]); glVertex3dv(BB[7]); glVertex3dv(BB[3]);
    // bottom
    glVertex3dv(BB[0]); glVertex3dv(BB[1]); glVertex3dv(BB[5]); glVertex3dv(BB[4]);
    // right
    glVertex3dv(BB[1]); glVertex3dv(BB[2]); glVertex3dv(BB[6]); glVertex3dv(BB[5]);
    // top
    glVertex3dv(BB[3]); glVertex3dv(BB[7]); glVertex3dv(BB[6]); glVertex3dv(BB[2]);
    // back
    glVertex3dv(BB[5]); glVertex3dv(BB[6]); glVertex3dv(BB[7]); glVertex3dv(BB[4]);

    }
  glEnd(); // GL_QUADS

//  glEnable(GL_TEXTURE_2D);
}

void vtkGlyphMapper::Render(vtkRenderer* ren, vtkVolume* vol)
{
  // Initialize:
  ren->GetRenderWindow()->MakeCurrent();

  if (!this->Initialized)
    {
    this->Initialize();
    this->LoadTextures();
    } // if
  this->LoadTextures();
  vtkCamera* cam = ren->GetActiveCamera();
  double* camera_position = cam->GetPosition();
  this->UEyePosition->SetValue((float)camera_position[0], (float)camera_position[1], (float)camera_position[2]);

  // Set-up the light for the lighting calculations.
  // Its not exactly the same as the camera position to make the shadows visible.
  // TODO: see what is needed of the stuff below. Maybe I just need to change the position a bit.
  // XXX: actually in the shaders I only use the light position.
  double vec[3];
  vtkCamera* light = vtkCamera::New();
  cam->GetPosition(vec);
  light->SetPosition(vec);
  cam->GetFocalPoint(vec);
  light->SetFocalPoint(vec);
  cam->GetViewUp(vec);
  light->SetViewUp(vec);
  light->SetViewAngle(cam->GetViewAngle());
  light->SetClippingRange(cam->GetClippingRange());

  // move the light a bit away from the camera; otherwise the shadows are not visible.
  // Off to one side
  light->Roll(-10);
  // Elevate
  light->Elevation(20);
  // Zoom out
  light->Zoom(0.5);
  light->GetPosition(vec);
  this->ULightPosition->SetValue((float)vec[0], (float)vec[1], (float)vec[2]);

  // Render to the shadow map
//  this->FBO->SetShaderProgram(this->SPDepth);

/*
  if (this->SPDepth) this->SPDepth->Activate();
//  if (this->SPDepth) this->SetTextureLocations(this->SPDepth);
  this->FBOShadow->PreRender(light);
  this->DrawPoints();
  this->FBOShadow->PostRender();
  if (this->SPDepth) this->SPDepth->Deactivate();
  light->Delete(); light = NULL;
*/
//  glEnable(GL_DEPTH_TEST);

  this->SetTextureLocations();

  int* viewportSize = ren->GetSize();
//cout<<"viewport size = "<<viewportSize[0]<<"x"<<viewportSize[1]<<endl;
  int* viewportOrigin = ren->GetOrigin();
  int viewport[4];
  viewport[0] = viewportOrigin[0];
  viewport[1] = viewportOrigin[1];
  viewport[2] = viewportSize[0];
  viewport[3] = viewportSize[1];

  GLuint* FBOviewport = FBODepth->GetViewport();
  bool viewport_equal = true;
  for (int i=0; i < 4; i++) if (FBOviewport[i] != viewport[i]) viewport_equal = false;

  if (!viewport_equal)
    {
    this->FBODepth->SetViewport(viewport);
cout<<"----- setting viewport to "<<viewport[0]<<", "<<viewport[1]<<", "<<viewport[2]<<", "<<viewport[3]<<endl;
    this->FBODepth->Initialize();
    }
/*
  if (this->FBODepth->GetWidth() != (viewportSize[0]) || (this->FBODepth->GetHeight() != viewportSize[1]))
    {
    this->FBODepth->SetWidth(viewportSize[0]);
    this->FBODepth->SetHeight(viewportSize[1]);
    this->FBODepth->Initialize();
    }
*/
  // 2nd render pass also to a depth buffer
  if (this->SPDepth) this->SPDepth->Activate();
  this->FBODepth->PreRender(cam);
  this->DrawPoints();
  this->FBODepth->PostRender();
  if (this->SPDepth) this->SPDepth->Deactivate();
  //vtkFiberMapper::DrawShadowMap(); // for testing
  cam = NULL;

  // TODO: name these 2 functions the same. They do the same.
  this->FBODepth->ActiveTextures();
  this->SetTextureLocations();

  // now on the screen
  if (this->SPLight) this->SPLight->Activate();
  this->DrawScreenFillingQuad(viewport);
  if (this->SPLight) this->SPLight->Deactivate();
}

// TODO: delete this function.
void vtkGlyphMapper::SetTextureLocation(vtkMyShaderProgram* sp, vtkUniformSampler* sampler) //, const char* texture_name)
{
  GLuint program = sp->GetHandle();
  GLuint textLoc = vtkgl::GetUniformLocation(program, sampler->GetName());
  sampler->SetValue(textLoc);
}

//void vtkGlyphMapper::DrawScreenFillingQuad(int width, int height)
void vtkGlyphMapper::DrawScreenFillingQuad(int viewport[4])
{
//  cout<<"--------------------- d - r - a - w - i - n - g ----- a ----- q - u - a - d ----------------- !!!! ---"<<endl;
  // store the projection matrix so that I can restore it after rendering the quad.
int width = viewport[2];
int height = viewport[3];

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  //glOrtho(0, width, 0, height, -1, 1);
//  glOrtho(viewport[0], viewport[0]+viewport[2], viewport[1], viewport[1]+viewport[3], -1, 1);
  glOrtho(0, 1, 0, 1, -1, 1);
//  glDisable(GL_DEPTH_TEST);
//  glEnable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
//  glViewport(0, 0, width, height);
//  glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
//  glViewport(0, 0, 1, 1);
  glMatrixMode(GL_TEXTURE);
  glPushMatrix();
  glLoadIdentity();

//  glDisable(GL_LIGHTING);
  //glDisable(GL_TEXTURE_RECTANGLE_ARB)

//  glColor4f(1, 0, 0, 1);
  glEnable(vtkgl::TEXTURE_RECTANGLE_ARB);
  glBegin(GL_QUADS);

    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0);

    glTexCoord2f(viewport[2], 0.0f);
    glVertex2f(1.0f, 0.0f);

    glTexCoord2f(viewport[2], viewport[3]);
    glVertex2f(1.0f, 1.0f);

    glTexCoord2f(0.0f, viewport[3]);
    glVertex2f(0.0f, 1.0f);
  glEnd(); // GL_QUADS
  glDisable(vtkgl::TEXTURE_RECTANGLE_ARB);

// glEnable(GL_LIGHTING);
//  glDisable(GL_LIGHTING);
  glMatrixMode(GL_TEXTURE);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
//  glEnable(GL_DEPTH_TEST);
}

void vtkGlyphMapper::SetNegateX(bool negate)
{
  this->UNegateX->SetValue(negate);
}

void vtkGlyphMapper::SetNegateY(bool negate)
{
  this->UNegateY->SetValue(negate);
}

void vtkGlyphMapper::SetNegateZ(bool negate)
{
  this->UNegateZ->SetValue(negate);
}

bool vtkGlyphMapper::GetNegateX()
{
  return this->UNegateX->GetValue();
}

bool vtkGlyphMapper::GetNegateY()
{
  return this->UNegateY->GetValue();
}

bool vtkGlyphMapper::GetNegateZ()
{
  return this->UNegateZ->GetValue();
}

void vtkGlyphMapper::SetGlyphScaling(float scale)
{
  this->GlyphScaling = scale;
  this->UGlyphScaling->SetValue(this->GlyphScaling);
//  this->SetMaxGlyphRadius(0.5* this->GlyphScaling);
  this->SetMaxGlyphRadius(this->GlyphScaling);
}

} // namespace bmia
