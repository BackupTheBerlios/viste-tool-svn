/**
 * TestShader.cxx
 * by Tim Peeters
 *
 * 2005-05-09	Tim Peeters
 * - First version
 *
 * 2006-01-30	Tim Peeters
 * - Remove glew stuff.
 * - Use InitializeExtensions function to setup OpenGL2 context.
 */

#include <vtkShaderBase.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkCylinderSource.h>
//#include <vtkConeSource.h> // don't use this for shader testing because it has no normals defined.
#include <vtkSphereSource.h>

#include "vtkMyShaderProgram.h"
#include "vtkMyShaderProgramReader.h"
#include "ExtensionInitialize.h"

using namespace bmia;

int main(int argc, char **argv) {
  cout<<"======================================================================"<<endl;

  cout<<argc<<" arguments."<<endl;
  for (int i=0; i < argc; i++)
    {
    cout<<"-- argument "<<i<<" == "<<argv[i]<<endl;
    }

  if (argc < 2) {
    cout<<"Usage: "<<argv[0]<<" SHADERFILE "<<endl;
    exit(0);
  }

  //assert(argv[1] != NULL);

  const char* shaderfile = argv[1];

  vtkRenderer* r = vtkRenderer::New();
  vtkRenderWindow* rw = vtkRenderWindow::New();
  rw->AddRenderer(r);
  vtkRenderWindowInteractor* rwi = vtkRenderWindowInteractor::New();
  rwi->SetRenderWindow(rw);

  vtkCylinderSource * s1 = vtkCylinderSource::New();
  vtkSphereSource* s2 = vtkSphereSource::New();
  s1->SetResolution(15);
  s2->SetThetaResolution(15);
  s2->SetPhiResolution(15);
  vtkPolyDataMapper * m1 = vtkPolyDataMapper::New();
  vtkPolyDataMapper * m2 = vtkPolyDataMapper::New();
  m1->SetInput(s1->GetOutput());
  m2->SetInput(s2->GetOutput());
  s1->Delete(); s2->Delete();
  vtkActor* a1 = vtkActor::New();
  vtkActor* a2 = vtkActor::New();
  a1->SetMapper(m1);
  a2->SetMapper(m2);
  a2->SetPosition(1.5,0,0);
  m1->Delete(); m2->Delete();
  r->AddActor(a1);
  r->AddActor(a2);
  a1->Delete(); a2->Delete();
  r->SetBackground(0,0,0.5);

  rw->SetSize(400,400);
  rwi->Initialize();
  //glewInit();
  InitializeExtensions(rw);

  vtkMyShaderProgramReader* spreader = vtkMyShaderProgramReader::New();
  spreader->SetFileName(shaderfile);
  spreader->Execute();
  spreader->GetOutput()->Activate();

  rwi->Start();

  spreader->GetOutput()->Deactivate();
  spreader->Delete();
  
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}
