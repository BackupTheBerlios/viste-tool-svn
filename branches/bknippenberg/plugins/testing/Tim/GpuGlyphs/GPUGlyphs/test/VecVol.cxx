/**
 * VecVol.cxx
 * by Tim Peeters
 *
 * 2007-10-22	Tim Peeters
 * - First version.
 */

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageData.h>
#include "vtkDTIDataManager.h"
#include "vtkVectorVolumeMapper.h"
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPlaneWidget.h>
#include "vtkEigensystemToGPU.h"

int main(int argc, char **argv) {
  cout<<"======================================================================"<<endl;

  if (argc < 2) {
    cout<<"Usage: "<<argv[0]<<" INPUT_FILE"<<endl;
    exit(0);
  }
  const char* filename = argv[1];

  vtkRenderer* r = vtkRenderer::New();
  vtkRenderWindow* rw = vtkRenderWindow::New();
  rw->AddRenderer(r);
  vtkRenderWindowInteractor* rwi = vtkRenderWindowInteractor::New();
  rwi->SetRenderWindow(rw);
  vtkInteractorStyle* is = vtkInteractorStyleTrackballCamera::New();
  rwi->SetInteractorStyle(is);
  is->Delete(); is = NULL;

  bmia::vtkDTIDataManager* manager = bmia::vtkDTIDataManager::New();
  manager->SetFileName(filename);

  /*
  bmia::vtkMEVColoringFilter* color = bmia::vtkMEVColoringFilter::New();
  color->ShiftValuesOn();
  color->SetInputConnection(manager->GetEigensystemOutputPort());

  color->Update();
  */

  bmia::vtkVectorVolumeMapper* mapper = bmia::vtkVectorVolumeMapper::New();
//  mapper->DebugOn();

  cout<<"Let's begin.."<<endl;
//  bmia::vtkTensorToEigensystemFilterGPU* eigen = bmia::vtkTensorToEigensystemFilterGPU::New();
  bmia::vtkEigensystemToGPU* eigen = bmia::vtkEigensystemToGPU::New();
//  eigen->DebugOn();
  eigen->SetInputConnection(manager->GetOutputPort());
  eigen->GetOutput()->Update();
  cout<<"///////////// Computing eigensystem volume..."<<endl;
  //mapper->SetInputConnection(manager->GetEigensystemOutputPort());
  mapper->SetInputConnection(eigen->GetOutputPort());
//  manager->GetEigensystemOutput()->Update();
  cout<<"///////////// Computation of eigensystem volume done. Let's visualize something!"<<endl;

//  manager->GetEigensystemOutput()->Print(cout);
  eigen->GetOutput()->Print(cout);

  vtkVolume* volume = vtkVolume::New();
  volume->SetMapper(mapper);

  manager->SetFileName(filename);

//  mapper->SetSeedDistance(0.5);//675);
//  mapper->SetLineLength(2);

  mapper->SetSeedDistance(0.5);
  mapper->SetLineLength(2.0);
  mapper->SetLineThickness(3.5);

  rw->SetSize(800,600);
//  rw->SetSize(1024,768);

  r->AddViewProp(volume);

  vtkPlaneWidget* widget = mapper->GetPlaneWidget();
//  widget->SetDefaultRenderer(r);
  widget->SetInteractor(rwi);
  widget->On();
  widget->NormalToZAxisOn();
  widget->PlaceWidget();
  rwi->Initialize();

  r->ResetCamera();
  rwi->Start();

  volume->Delete();
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}

