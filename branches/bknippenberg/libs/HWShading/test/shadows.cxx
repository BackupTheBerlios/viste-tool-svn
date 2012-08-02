/**
 * shadows.cxx
 * by Tim Peeters
 *
 * 2005-07-19	Tim Peeters
 * - First version, based on NoShader.cxx
 *
 * 2006-01-30	Tim Peeters
 * - Replace glew by VTK OpenGL extension manager.
 */

#include "vtkShadowRenderer.h"
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
//#include <GL/glew.h>

#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkCylinderSource.h>
#include <vtkSphereSource.h>

#include <vtkOpenGLRenderWindow.h>
#include "ExtensionInitialize.h"

using namespace bmia;

int main(int argc, char **argv) {
  cout<<"=================================S=H=A=D=O=W=S=================================="<<endl;
  vtkShadowRenderer* r = vtkShadowRenderer::New();
  r->ShadowsOn();
  r->ShowShadowMapOn();
  vtkRenderWindow* rw = vtkRenderWindow::New();

  vtkRenderWindowInteractor* rwi = vtkRenderWindowInteractor::New();
//  rwi->SetRenderWindow(rw);

  rw->SetInteractor(rwi);

  vtkCylinderSource * s1 = vtkCylinderSource::New();
  vtkSphereSource* s2 = vtkSphereSource::New();
  s1->SetResolution(200);
  s2->SetThetaResolution(100);
  s2->SetPhiResolution(100);
  s2->SetRadius(0.3);
  vtkPolyDataMapper * m1 = vtkPolyDataMapper::New();
  vtkPolyDataMapper * m2 = vtkPolyDataMapper::New();
  m1->SetInput(s1->GetOutput());
  m2->SetInput(s2->GetOutput());
  s1->Delete(); s2->Delete();
  vtkActor* a1 = vtkActor::New();
  vtkActor* a2 = vtkActor::New();
  a1->SetMapper(m1);
  a2->SetMapper(m2);
  a2->SetPosition(1.2,0,0);
  m1->Delete(); m2->Delete();
  r->AddActor(a1); r->AddActor(a2);
  a1->Delete(); a2->Delete();
  r->SetBackground(0,0,0.5);

  rw->SetSize(800, 400);
//  rwi->Initialize();
  //glewInit();

  rwi->Initialize();

  cout<<"Calling InitializeExtensions("<<rw<<");"<<endl;
  if (!InitializeExtensions(rw))
    {
    cout<<"ERROR!"<<endl;
    return 0;
    }
  cout<<"Extensions loaded!"<<endl;

  rw->AddRenderer(r);
  r->ResetCamera();

  rwi->Start();
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}
