/**
 * vtkDTIGlyphMapper.cxx
 * by Tim Peeters
 *
 * 2008-02-28	Tim Peeters
 * - First version
 */

#include "vtkDTIGlyphMapper.h"
#include <vtkObjectFactory.h>
//#include <vtkgl.h>
#include "vtkUniformSampler.h"
#include "vtkMyShaderProgram.h"
#include "vtkMyShaderProgramReader.h"

namespace bmia {

vtkStandardNewMacro(vtkDTIGlyphMapper);

vtkDTIGlyphMapper::vtkDTIGlyphMapper()
{
  // Create shader programs
  vtkMyShaderProgramReader* reader1 = vtkMyShaderProgramReader::New();
  reader1->SetFileName("dtidepth.prog");
  reader1->Execute();
  this->SPDepth = reader1->GetOutput();
  this->SPDepth->Register(this);
  reader1->Delete();

  vtkMyShaderProgramReader* reader2 = vtkMyShaderProgramReader::New();
  reader2->SetFileName("dtilight.prog");
  reader2->Execute();
  this->SPLight = reader2->GetOutput();
  this->SPLight->Register(this);
  reader2->Delete();

  // the following uniform variables will be initialized
  // in LoadTextures().
  this->UTextureEV1 = NULL;
  this->UTextureEV2 = NULL;

  this->SetupShaderUniforms();
}

vtkDTIGlyphMapper::~vtkDTIGlyphMapper()
{
  this->UnloadTextures();
  assert(this->SPDepth);
  this->SPDepth->Delete(); this->SPDepth = NULL;
  assert(this->SPLight);
  this->SPLight->Delete(); this->SPLight = NULL;
}

void vtkDTIGlyphMapper::LoadTextures()
{
//  GLuint textures[2];
  glGenTextures(2, this->Textures);

  vtkDebugMacro(<<"Generated textures with indices "<<this->Textures[0]<<" and "<<this->Textures[1]);

  // input data should have 4-component data arrays with names "Eigenvector 1"`
  // and "Eigenvector 2".
//  glActiveTexture(GL_TEXTURE2);
  this->LoadTexture(this->Textures[0], "Eigenvector 1");
//  glActiveTexture(GL_TEXTURE3);
  this->LoadTexture(this->Textures[1], "Eigenvector 2");

  if (this->UTextureEV1 || this->UTextureEV2)
    {
    vtkErrorMacro(<<"LoadTextures() should only be called once!"
			<<"And then the texture uniforms should be NULL!");
    } // if

  this->UTextureEV1 = vtkUniformSampler::New();
  this->UTextureEV2 = vtkUniformSampler::New();
  this->UTextureEV1->SetName("Texture1");
  this->UTextureEV2->SetName("Texture2");
  this->UTextureEV1->SetValue(2);
  this->UTextureEV2->SetValue(3);

  if (this->SPDepth)
    {
    this->SPDepth->AddShaderUniform(this->UTextureEV1);
    this->SPDepth->AddShaderUniform(this->UTextureEV2);
    }
  if (this->SPLight)
    {
    this->SPLight->AddShaderUniform(this->UTextureEV1);
    this->SPLight->AddShaderUniform(this->UTextureEV2);
    }
}

void vtkDTIGlyphMapper::UnloadTextures()
{
  // XXX: I have no idea whether this is correct.
  glDeleteTextures(2, this->Textures);

  if (this->SPDepth)
    {
    this->SPDepth->RemoveShaderUniform(this->UTextureEV1);
    this->SPDepth->RemoveShaderUniform(this->UTextureEV2);
    }
  if (this->SPLight)
    {
    this->SPLight->RemoveShaderUniform(this->UTextureEV1);
    this->SPLight->RemoveShaderUniform(this->UTextureEV2);
    }

  this->UTextureEV1->Delete(); this->UTextureEV1 = NULL;
  this->UTextureEV2->Delete(); this->UTextureEV2 = NULL;
}

void vtkDTIGlyphMapper::ReloadTextures()
{
  if (!this->UTextureEV1) return;
  this->UnloadTextures();
  this->LoadTextures();
}

void vtkDTIGlyphMapper::SetTextureLocations() //vtkMyShaderProgram* program)
{
//  assert(program);
//  this->SetTextureLocation(program, this->UTextureEV1);
//  this->SetTextureLocation(program, this->UTextureEV2);

//  cout<<"Texture1 location for "<<program->GetHandle()<<" = "<<this->UTextureEV1->GetValue()<<endl;
//  cout<<"Texture2 location for "<<program->GetHandle()<<" = "<<this->UTextureEV2->GetValue()<<endl;

  vtkgl::ActiveTexture(vtkgl::TEXTURE2);
  glBindTexture(vtkgl::TEXTURE_3D, this->Textures[0]);
  this->UTextureEV1->SetValue(2);
  vtkgl::ActiveTexture(vtkgl::TEXTURE3);
  glBindTexture(vtkgl::TEXTURE_3D, this->Textures[1]);
  this->UTextureEV2->SetValue(3);
  vtkgl::ActiveTexture(vtkgl::TEXTURE0);
   
}

} // namespace bmia
