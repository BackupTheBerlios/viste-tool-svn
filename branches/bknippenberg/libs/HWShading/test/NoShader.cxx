/**
 * NoShader.cxx
 * by Tim Peeters
 *
 * 2005-04-29	Tim Peeters
 * - First version
 *
 * 2005-05-05	Tim Peeters
 * - Removed some unneeded stuff to make it compile on Windows.
 */

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkCylinderSource.h>
#include <vtkConeSource.h>
#include <vtkSphereSource.h>

#include <vtkOpenGLRenderWindow.h>

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <sys/types.h>

int main(int argc, char **argv) {
  cout<<"======================================================================"<<endl;
  vtkRenderer* r = vtkRenderer::New();
  vtkRenderWindow* rw = vtkRenderWindow::New();
  rw->AddRenderer(r);
  vtkRenderWindowInteractor* rwi = vtkRenderWindowInteractor::New();
  rwi->SetRenderWindow(rw);

  vtkCylinderSource * s1 = vtkCylinderSource::New();
  vtkSphereSource* s2 = vtkSphereSource::New();
  s1->SetResolution(20);
  s2->SetThetaResolution(20);
  s2->SetPhiResolution(20);
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
  r->AddActor(a1); r->AddActor(a2);
  a1->Delete(); a2->Delete();
  r->SetBackground(0,0,0.5);

  rw->SetSize(400,400);
  rwi->Initialize();

  rwi->Start();
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}
