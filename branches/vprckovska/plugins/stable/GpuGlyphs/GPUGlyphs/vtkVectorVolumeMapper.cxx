/**
 * vtkVectorVolumeMapper.cxx
 * by Tim Peeters
 *
 * 2007-10-22	Tim Peeters
 * - First version
 *
 * 2008-02-11	Tim Peeters
 * - Add support for loading multiple volumes as textures
 */

#include <assert.h>

#include "vtkVectorVolumeMapper.h"
#include <vtkObjectFactory.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkVolume.h>
#include <vtkOpenGLExtensionManager.h>
#include <vtkPlaneWidget.h>

#include <vtkAlgorithmOutput.h>
#include <vtkImageData.h>
#include <vtkCamera.h>

#include <vtkMath.h>
#include <vtkgl.h>

#include "vtkMyShaderProgram.h"
#include "vtkMyShaderProgramReader.h"
#include "vtkUniformFloat.h"
#include "vtkUniformVec3.h"
#include "vtkUniformIvec3.h"
#include "vtkUniformBool.h"

#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkPointData.h>

#include <vtkTimerLog.h>

#define ATTRIBUTE_POI_ORIGIN 1
#define ATTRIBUTE_POI_POINT1 2
//XXX: when I put 3 on the following line instead of 4, something goes wrong
#define ATTRIBUTE_POI_POINT2 4 

namespace bmia {

vtkStandardNewMacro(vtkVectorVolumeMapper);
vtkCxxSetObjectMacro(vtkVectorVolumeMapper, ShaderProgram, vtkMyShaderProgram);

vtkVectorVolumeMapper::vtkVectorVolumeMapper()
{
  this->Timer = vtkTimerLog::New();
  this->TimerSteps = 0;

  this->ShaderProgram = NULL;

  // read the shader program from text file:
  vtkMyShaderProgramReader* sp_reader = vtkMyShaderProgramReader::New();
  sp_reader->SetFileName("vecvol.prog");
  sp_reader->Execute();
//  this->MyShaderProgram = sp_reader->GetOutput();
//  this->MyShaderProgram->Register(this);
  this->SetShaderProgram(sp_reader->GetOutput());
  sp_reader->Delete();
  sp_reader = NULL;

  this->EyePosition = NULL;
  this->LightPosition = NULL;
  this->TextureDimensions = NULL;
  this->TextureSpacing = NULL;
  this->UniformLineLength = NULL;
  this->UniformLineThickness = NULL;
  this->UniformSeedDistance = NULL;
  this->UniformPoiOrigin = NULL;
  this->UniformPoiPoint1 = NULL;
  this->UniformPoiPoint2 = NULL;
  this->UniformUseShadows = NULL;
  this->Step1 = NULL;
  this->Step2 = NULL;

  this->SetupShaderUniforms();

  // initialize this->Initialized and this->PlaneWidget:
  this->Initialized = false;
  this->PlaneWidget = vtkPlaneWidget::New();
  //this->PlaneWidget->SetRepresentationToWireframe();
}

void vtkVectorVolumeMapper::SetupShaderUniforms()
{
  // set-up all uniform variables that I need:
  if (this->EyePosition) this->EyePosition->Delete();
  this->EyePosition = vtkUniformVec3::New();
  this->EyePosition->SetName("EyePosition");
  this->ShaderProgram->AddShaderUniform(this->EyePosition);

  if (this->LightPosition) this->LightPosition->Delete();
  this->LightPosition = vtkUniformVec3::New();
  this->LightPosition->SetName("LightPosition");
  this->ShaderProgram->AddShaderUniform(this->LightPosition);

  if (this->TextureDimensions) this->TextureDimensions->Delete();
  this->TextureDimensions = vtkUniformIvec3::New();
  this->TextureDimensions->SetName("TextureDimensions");
  this->ShaderProgram->AddShaderUniform(this->TextureDimensions);

  if (this->TextureSpacing) this->TextureSpacing->Delete();
  this->TextureSpacing = vtkUniformVec3::New();
  this->TextureSpacing->SetName("TextureSpacing");
  this->ShaderProgram->AddShaderUniform(this->TextureSpacing);

  this->LineLength = 2.0;
  if (this->UniformLineLength) this->UniformLineLength->Delete();
  this->UniformLineLength = vtkUniformFloat::New();
  this->UniformLineLength->SetName("LineLength");
  this->UniformLineLength->SetValue(this->LineLength);
  this->ShaderProgram->AddShaderUniform(this->UniformLineLength);
  this->SeedDistance = 1.0;
  if (this->UniformSeedDistance) this->UniformSeedDistance->Delete();
  this->UniformSeedDistance = vtkUniformFloat::New();
  this->UniformSeedDistance->SetName("SeedDistance");
  this->UniformSeedDistance->SetValue(this->SeedDistance);
  this->ShaderProgram->AddShaderUniform(this->UniformSeedDistance);
  this->LineThickness = 3.0;
  this->UniformLineThickness = vtkUniformFloat::New();
  this->UniformLineThickness->SetName("LineThickness");
  this->UniformLineThickness->SetValue(this->LineThickness);
  this->ShaderProgram->AddShaderUniform(this->UniformLineThickness);

  if (this->UniformPoiOrigin) this->UniformPoiOrigin->Delete();
  this->UniformPoiOrigin = vtkUniformVec3::New();
  this->UniformPoiOrigin->SetName("PoiOrigin");
  this->ShaderProgram->AddShaderUniform(this->UniformPoiOrigin);
  if (this->UniformPoiPoint1) this->UniformPoiPoint1->Delete();
  this->UniformPoiPoint1 = vtkUniformVec3::New();
  this->UniformPoiPoint1->SetName("PoiPoint1");
  this->ShaderProgram->AddShaderUniform(this->UniformPoiPoint1);
  if (this->UniformPoiPoint2) this->UniformPoiPoint2->Delete();
  this->UniformPoiPoint2 = vtkUniformVec3::New();
  this->UniformPoiPoint2->SetName("PoiPoint2");
  this->ShaderProgram->AddShaderUniform(this->UniformPoiPoint2);

  this->UniformUseShadows = vtkUniformBool::New();
  this->UniformUseShadows->SetName("UseShadows");
  this->UniformUseShadows->SetValue(true);
  this->ShaderProgram->AddShaderUniform(this->UniformUseShadows);

  if (this->Step1) this->Step1->Delete();
  this->Step1 = vtkUniformVec3::New();
  this->Step1->SetName("Step1");
  this->ShaderProgram->AddShaderUniform(this->Step1);
  if (this->Step2) this->Step2->Delete();
  this->Step2 = vtkUniformVec3::New();
  this->Step2->SetName("Step2");
  this->ShaderProgram->AddShaderUniform(this->Step2);
}

vtkVectorVolumeMapper::~vtkVectorVolumeMapper()
{
  // destroy everything that I created in the constructor.
  this->EyePosition->Delete(); this->EyePosition = NULL;
  this->LightPosition->Delete(); this->LightPosition = NULL;
  this->TextureDimensions->Delete(); this->TextureDimensions = NULL;
  this->ShaderProgram->Delete(); this->ShaderProgram = NULL;
  this->UniformLineLength->Delete(); this->UniformLineLength = NULL;
  this->UniformSeedDistance->Delete(); this->UniformSeedDistance = NULL;
  this->UniformLineThickness->Delete(); this->UniformLineThickness = NULL;
  this->UniformUseShadows->Delete(); this->UniformUseShadows = NULL;

  this->UniformPoiOrigin->Delete(); this->UniformPoiOrigin = NULL;
  this->UniformPoiPoint1->Delete(); this->UniformPoiPoint1 = NULL;
  this->UniformPoiPoint2->Delete(); this->UniformPoiPoint2 = NULL;
  this->Step1->Delete(); this->Step1 = NULL;
  this->Step2->Delete(); this->Step2 = NULL;

  this->SetShaderProgram(NULL);

  if (this->PlaneWidget)
    {
    this->PlaneWidget->UnRegister(this);
    this->PlaneWidget = NULL;
    }
}

vtkCxxSetObjectMacro(vtkVectorVolumeMapper, PlaneWidget, vtkPlaneWidget);

void vtkVectorVolumeMapper::SetInputConnection(int port, vtkAlgorithmOutput* input)
{
  assert(input);
  vtkAlgorithm* algorithm = input->GetProducer();
  vtkDataObject* dataobject = algorithm->GetOutputDataObject(0);
  assert(dataobject);
  vtkDataSet* dataset = vtkDataSet::SafeDownCast(dataobject);
  assert(dataset);

  vtkDebugMacro("Setting input of PlaneWidget "<<this->PlaneWidget<<" to "<<dataset);
  this->PlaneWidget->SetInput(dataset);
  this->Superclass::SetInputConnection(port, input);
}

void vtkVectorVolumeMapper::SetInputConnection(vtkAlgorithmOutput* input)
{
  this->SetInputConnection(0,input);
}

void vtkVectorVolumeMapper::Initialize()
{
  this->Initialized = false;

  vtkOpenGLExtensionManager* extensions = vtkOpenGLExtensionManager::New();
  extensions->SetRenderWindow(NULL); // set render window to current render window
  
  int supports_GL_VERSION_2_0			= extensions->ExtensionSupported("GL_VERSION_2_0");
  //int supports_GL_EXT_framebuffer_object	= extensions->ExtensionSupported("GL_EXT_framebuffer_object");

  if (supports_GL_VERSION_2_0)
    {
    extensions->LoadExtension("GL_VERSION_2_0");
    }
  else
    {
    vtkWarningMacro(<<"GL_VERSION_2_0 not supported!");
    return;
    }
  /*
  if (supports_GL_EXT_framebuffer_object)
    {
    extensions->LoadExtension("GL_EXT_framebuffer_object");
    }
  else
    {
    vtkWarningMacro(<<"GL_EXT_framebuffer_object not supported!");
    return;
    }
  */
  extensions->Delete(); extensions = NULL;
  this->Initialized = true;
}

/*
bool vtkVectorVolumeMapper::IsRenderSupported()
{
  if (!this->Initialized)
    {
    this->Initialize();
    }

  return this->Initialized;
}
*/

void vtkVectorVolumeMapper::Render(vtkRenderer* ren, vtkVolume* vol)
{
  if (this->TimerSteps == 0) this->Timer->StartTimer();

  // Initialize:
  ren->GetRenderWindow()->MakeCurrent();
  
  if ( !this->Initialized )
    {
    this->Initialize();
    // Make sure the volume is loaded on the GPU as a 3D texture
    this->LoadTextures();
    }

  // Enable shader program
  vtkCamera* cam = ren->GetActiveCamera();
  double* camera_position = cam->GetPosition();
  this->EyePosition->SetValue((float)camera_position[0], (float)camera_position[1], (float)camera_position[2]);

  // Set-up the light for the lighting calculations.
  // Its not exactly the same as the camera position to make the shadows visible.
  // TODO: see what is needed of the stuff below. Maybe I just need to change the position a bit.
  double vec[3];
  vtkCamera* light = vtkCamera::New();
  cam->GetPosition(vec);
//cout<<"camera position = "<<vec[0]<<", "<<vec[1]<<", "<<vec[2]<<endl;
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
  //light->Zoom(0.8);

  light->GetPosition(vec);
//cout<<"light position = "<<vec[0]<<", "<<vec[1]<<", "<<vec[2]<<endl;
  this->LightPosition->SetValue((float)vec[0], (float)vec[1], (float)vec[2]);
  light->Delete(); light = NULL;

  // pass the points of the slicing plane to the shader as uniform variables:
  double O[3]; double P1[3]; double P2[3];
  this->PlaneWidget->GetOrigin(O);
  this->PlaneWidget->GetPoint1(P1);
  this->PlaneWidget->GetPoint2(P2);
  this->UniformPoiOrigin->SetValue((float)O[0], (float)O[1], (float)O[2]);
  this->UniformPoiPoint1->SetValue((float)P1[0], (float)P1[1], (float)P1[2]);
  this->UniformPoiPoint2->SetValue((float)P2[0], (float)P2[1], (float)P2[2]);

  // computations below are needed to compute the distance from the center of the plane
  // to the eye. This is needed if the seed distance should depend on the distance
  // to the plane (see further down). At the moment it is not used.
  //
  // compute the center of the plane.
  // this is the average of the points O, P1, P2, O+(P1-O)+(P2-O).
  double plane_center[3];
  double eye_to_plane[3]; // the vector pointing from camera to the center of the plane.
  for (int i=0; i < 3; i++)
    {
    plane_center[i] = (P1[i]+P2[i]) / 2.0;
    eye_to_plane[i] = plane_center[i] - camera_position[i];
    } // for i
  double plane_eye_distance = vtkMath::Norm(eye_to_plane)/200.0;

  // Compute the steps between the various seed points:
  float s1[3]; float s2[3];
  for (int i=0; i < 3; i++)
	{
	s1[i] = (float)(P1[i] - O[i]);
	s2[i] = (float)(P2[i] - O[i]);
	}
  vtkMath::Normalize(s1);
  vtkMath::Normalize(s2);

  for (int i=0; i < 3; i++)
    {
    // Note: on the following lines it is possible to change the seed distance
    // depending on the distance of the center of the plane to the camera.
    // But its not really useful. The distance would become larger when the plane
    // is further away. But performance for plane far away is already good.
    //
    // Visually, this seems good for fibers, but bad for glyphs.
    // XXX: I'm working on glyphs now so I commented it out. But make it an
    // 		option.
    s1[i] *= this->SeedDistance;// *sqrt(plane_eye_distance);
    s2[i] *= this->SeedDistance;// *sqrt(plane_eye_distance);
    }

  this->Step1->SetValue(s1);
  this->Step2->SetValue(s2);

  // The number of seed point to take into account depends on
  // The direction of the viewing ray, the plane, and on the seed length
  // XXX: compute in vertex shader.

  // activate the shader program
  this->ShaderProgram->Activate();

  // Render bounding box
  this->RenderBoundingBox();

  // Disable shader program
  this->ShaderProgram->Deactivate();
  // Disable 3D texture mapping
  //glDisable(GL_TEXTURE_3D);

  // compute performance in FPS. XXX: Commented stuff gives incorrect results.
  //float t = ren->GetLastRenderTimeInSeconds();
  //float t = ((vtkRenderer*) this->GetRenderWindow()->GetRenderers()->GetItemAsObject(0))->GetLastRenderTimeInSeconds();

  // This always gives about 40 FPS.. even if the interaction is so slow that it becomes unusable.
  // So I guess its incorrect.. :s
  //if (t == 0) cout<<"Infinite FPS, "<<endl;
  //else cout <<1.0/t<<" FPS, "<<endl;
  //cout<<"Viewport size = "<<this->GetRenderWindow()->GetSize()[0]<<"x"<<this->GetRenderWindow()->GetSize()[1]<<endl;
  if (this->TimerSteps == 10)
	{
	  this->Timer->StopTimer();
	 cout<<10.0/this->Timer->GetElapsedTime()<<" FPS!"<<endl;
	 this->TimerSteps = -1;
	}
  this->TimerSteps++;
}

void vtkVectorVolumeMapper::RenderBoundingBox()
{
  int i;
  assert(this->PlaneWidget);

  // Get the points defining the current plane of interest (POI):
  double O[3]; double P1[3]; double P2[3];
  this->PlaneWidget->GetOrigin(O);
  this->PlaneWidget->GetPoint1(P1);
  this->PlaneWidget->GetPoint2(P2);

  // Compute the directions from origin to the points      
  double OP1[3]; double OP2[3]; double nOP1[3]; double nOP2[3];
  double norm1; double norm2;

  for  (i=0; i < 3; i++)
    {
    OP1[i] = P1[i] - O[i];
    OP2[i] = P2[i] - O[i];
    nOP1[i] = OP1[i];
    nOP2[i] = OP2[i];
    } // for i
  norm1 = vtkMath::Normalize(nOP1);
  norm2 = vtkMath::Normalize(nOP2);

  // Compute a vector OQ orthogonal to OP1 and OP2.

  // NOTE: We are assuming here that the plane is a rectangle, so that
  // OP1 is orthogonal to OP2.

  double OQ[3];
  vtkMath::Cross(OP1, OP2, OQ);
  //XXX: is it already normalized? Then the following line can be removed.
  vtkMath::Normalize(OQ);

  // Compute the vectors used for computing the points to render:
  // XXX: in which coordinate space should h be defined? World? Model?
  double HOP1[3]; double HOP2[3];
  for (i=0; i < 3; i++)
    {
    HOP1[i] = nOP1[i]*this->LineLength;
    HOP2[i] = nOP2[i]*this->LineLength;
    OQ[i] = OQ[i] * this->LineLength;
    } // for

  // Determine the points of the bounding box to render:
  double BB[8][3];
  for (int i=0; i < 3; i++)
    {
    BB[0][i] = O[i] - HOP1[i] - HOP2[i] - OQ[i];
    BB[1][i] = O[i] - HOP1[i] - HOP2[i] + OQ[i];

    BB[2][i] = O[i] + OP1[i] + HOP1[i] - HOP2[i] + OQ[i];
    BB[3][i] = O[i] + OP1[i] + HOP1[i] - HOP2[i] - OQ[i];

    BB[4][i] = O[i] + OP2[i] - HOP1[i] + HOP2[i] - OQ[i];
    BB[5][i] = O[i] + OP2[i] - HOP1[i] + HOP2[i] + OQ[i];

    BB[6][i] = O[i] + OP2[i] + OP1[i] + HOP1[i] + HOP2[i] + OQ[i];
    BB[7][i] = O[i] + OP2[i] + OP1[i] + HOP1[i] + HOP2[i] - OQ[i];
    } // for i

  // Cull back faces so that only front faces are rendered.
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  // glFrontFace(...);

  // No standard OpenGL lighting. I have my own lighting :)
  glDisable(GL_LIGHTING);

  glBegin(GL_QUADS);
  // the origin and point1 and point2 of the POI are the same
  // for all quads/vertices rendered. So only set them once:
  // XXX: this was commented out because the points are now
  // unniform variables, not attributes.
  //vtkgl::VertexAttrib3dv(ATTRIBUTE_POI_ORIGIN, O);
  //vtkgl::VertexAttrib3dv(ATTRIBUTE_POI_POINT1, P1);
  //vtkgl::VertexAttrib3dv(ATTRIBUTE_POI_POINT2, P2);	

  // I render the points in the order such that they are clockwise
  // when you look along the normal of the quad pointing outward
  // of the block
  // Set a color. Just for debugging. The shader overrides the color.
  glColor4f(1, 1, 0, 1);

  // render all the faces of the bounding box:
  // front
  glVertex3dv(BB[0]);
  glVertex3dv(BB[3]);
  glVertex3dv(BB[2]);
  glVertex3dv(BB[1]);

  // bottom
  glVertex3dv(BB[0]);
  glVertex3dv(BB[4]);
  glVertex3dv(BB[7]);
  glVertex3dv(BB[3]);

  // left
  glVertex3dv(BB[0]);
  glVertex3dv(BB[1]);
  glVertex3dv(BB[5]);
  glVertex3dv(BB[4]);

  // right
  glVertex3dv(BB[2]);
  glVertex3dv(BB[3]);
  glVertex3dv(BB[7]);
  glVertex3dv(BB[6]);

  // back
  glVertex3dv(BB[4]);
  glVertex3dv(BB[5]);
  glVertex3dv(BB[6]);
  glVertex3dv(BB[7]);

  // top
  glVertex3dv(BB[1]);
  glVertex3dv(BB[2]);
  glVertex3dv(BB[6]);
  glVertex3dv(BB[5]);

  glEnd(); // GL_QUADS
}

void vtkVectorVolumeMapper::LoadTexture()
{
  this->LoadTexture(0, "Eigenvector 1");
}

void vtkVectorVolumeMapper::LoadTexture(int texture_index, const char* array_name)
{
  // First, get the volume from the input:
  vtkImageData* input = this->GetInput();
  assert(input);
  int dims[3]; double spacing[3];
  input->GetDimensions(dims); input->GetSpacing(spacing);
  // Set the texture parameters in the shader according to the input:
  this->TextureDimensions->SetValue(dims);
  this->TextureSpacing->SetValue((float)spacing[0], (float)spacing[1], (float)spacing[2]);
  // Enable 3D texture mapping:
//  glEnable(GL_TEXTURE_3D);
	
  // Active Texture
  // BindTexture(index)
  // TexImage3D(data)
  // HIER VERDER

  // code copied from Real-Time Volume Graphics book, p. 64
  // bind 3D texture target
  glBindTexture( vtkgl::TEXTURE_3D, texture_index );
  // set texture parameters such as wrap mode and filtering
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(vtkgl::TEXTURE_3D, vtkgl::TEXTURE_WRAP_R, GL_CLAMP);
//  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(vtkgl::TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  //glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);

  vtkPointData* pd = input->GetPointData();
  assert(pd);
  //vtkFloatArray* vectors = vtkFloatArray::SafeDownCast(pd->GetVectors());
  vtkFloatArray* vectors = vtkFloatArray::SafeDownCast(pd->GetScalars(array_name));
  assert(vectors);

  // upload the 3D volume texture to local graphics memory
  // TODO: Check for the input data type. Now its float. But other data types
  // may be possible..
  vtkgl::TexImage3D(vtkgl::TEXTURE_3D, 0, GL_RGBA,
  //glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA,
			dims[0], dims[1], dims[2],
			0, GL_RGBA, GL_FLOAT,
			(float*) vectors->GetPointer(0));
}

void vtkVectorVolumeMapper::SetLineLength(float length)
{
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting LineLength to " << length);
  if (this->LineLength != length)
    {
    this->LineLength = length;
    this->UniformLineLength->SetValue(this->LineLength);
    this->Modified();
    } // if
}

void vtkVectorVolumeMapper::SetLineThickness(float thickness)
{
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting LineThickness to " << thickness);
  if (this->LineThickness != thickness)
    {
    this->LineThickness = thickness;
    this->UniformLineThickness->SetValue(this->LineThickness);
    this->Modified();
    }
}

void vtkVectorVolumeMapper::SetSeedDistance(float d)
{
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting SeedDistance to " << d);
  if (this->SeedDistance != d)
    {
    this->SeedDistance = d;
    this->UniformSeedDistance->SetValue(this->SeedDistance);
    this->Modified();
    } // if
}

void vtkVectorVolumeMapper::SetUseShadows(bool shadows)
{
  vtkDebugMacro(<< this->GetClassName() << " (" << this << "): setting UseShadows to " << shadows);
  this->UniformUseShadows->SetValue(shadows);
  this->Modified();
}

} // namespace bmia
