/**
 * dtiglyphs.cxx
 * by Tim Peeters
 *
 * 2008-02-28	Tim Peeters
 * - First version
 */

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageData.h>
#include "vtkDTIDataManager.h"

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPlaneWidget.h>
#include "vtkEigensystemToGPU.h"
#include "vtkDTIGlyphMapper.h"

#include <vtkCommand.h>
#include <vtkPlaneWidget.h>
#include <vtkPointSet.h>
#include <vtkPolyData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkMath.h>
#include <assert.h>

class vtkWidgetCallback : public vtkCommand
{
public:
  static vtkWidgetCallback * New() { return new vtkWidgetCallback; }
  virtual void Execute(vtkObject* caller, unsigned long, void*)
    {
    int i;
    vtkPoints* seedPoints = vtkPoints::New();
    double origin[3]; double point1[3]; double point2[3];
    double direction1[3]; double direction2[3]; double norm1; double norm2;
    this->PlaneWidget->GetOrigin(origin);
    this->PlaneWidget->GetPoint1(point1);
    this->PlaneWidget->GetPoint2(point2);

    for  (i=0; i < 3; i++)
      {
      direction1[i] = point1[i] - origin[i];
      direction2[i] = point2[i] - origin[i];
      } // for i
    norm1 = vtkMath::Normalize(direction1);
    norm2 = vtkMath::Normalize(direction2);

    assert(this->SeedPointDistance > 0.0);
    for (i=0; i < 3; i++)
      {
      direction1[i] *= this->SeedPointDistance;
      direction2[i] *= this->SeedPointDistance;
      } // for i

    int num_steps1 = (int) (norm1/this->SeedPointDistance);
    int num_steps2 = (int) (norm2/this->SeedPointDistance);

    for (int x=0; x < num_steps1; x++)
      {
      for (i=0; i < 3; i++) point1[i] = origin[i];
      for (int y = 0; y < num_steps2; y++)
        {
        seedPoints->InsertNextPoint(point1);
        for (i=0; i < 3; i++) point1[i] += direction2[i];
        } // for y
      for (i=0; i < 3; i++) origin[i] += direction1[i];
      } // for x

    assert(this->PointSet);
    this->PointSet->SetPoints(seedPoints);
    seedPoints->Delete();

    }

  vtkPlaneWidget* PlaneWidget;
  vtkPointSet* PointSet;
  vtkRenderer* Renderer;
  double SeedPointDistance;
}; // class vtkWidgetCallback

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

 // bmia::vtkTensorVolumeMapper* mapper = bmia::vtkTensorVolumeMapper::New();
  bmia::vtkDTIGlyphMapper* mapper = bmia::vtkDTIGlyphMapper::New();
  mapper->DebugOn();

  bmia::vtkEigensystemToGPU* eigen = bmia::vtkEigensystemToGPU::New();
  eigen->SetInputConnection(manager->GetOutputPort());
  eigen->GetOutput()->Update();
  cout<<"///////////// Computing eigensystem volume..."<<endl;
  //mapper->SetInputConnection(manager->GetEigensystemOutputPort());
  mapper->SetInputConnection(eigen->GetOutputPort());
//  manager->GetEigensystemOutput()->Update();
  cout<<"///////////// Computation of eigensystem volume done. Let's visualize something!"<<endl;

//  manager->GetEigensystemOutput()->Print(cout);
  eigen->GetOutput()->Print(cout);

  vtkPointSet* pointset = vtkUnstructuredGrid::New();

  // create a widget
  vtkPlaneWidget* planeWidget = vtkPlaneWidget::New();
  planeWidget->SetInteractor(rwi);
  planeWidget->SetInput(eigen->GetOutput());

  vtkWidgetCallback* myCallback = vtkWidgetCallback::New();
  myCallback->PlaneWidget = planeWidget;
  myCallback->Renderer = r;
  myCallback->PointSet = pointset;
  myCallback->SeedPointDistance = 1.0; // Seedpoint Distance in mm (so not in voxels)
  planeWidget->AddObserver(vtkCommand::InteractionEvent, myCallback);

  planeWidget->SetDefaultRenderer(r);
  planeWidget->On();
  planeWidget->NormalToZAxisOn();
  planeWidget->PlaceWidget();

  myCallback->Execute(NULL, 1, NULL);

  mapper->SetMaxGlyphRadius(1.0);
  vtkVolume* volume = vtkVolume::New();
  volume->SetMapper(mapper);

  manager->SetFileName(filename);
  mapper->SetSeedPoints(pointset);
 
//  mapper->SetSeedDistance(1.0);//675);
  //mapper->SetLineLength(0.3);
  mapper->SetGlyphScaling(1.0);

  rw->SetSize(800,600);
  rw->SetSize(1024,768);

  r->AddViewProp(volume);

//  vtkPlaneWidget* widget = mapper->GetPlaneWidget();
////  widget->SetDefaultRenderer(r);
//  widget->SetInteractor(rwi);
//  widget->On();
//  widget->NormalToZAxisOn();
//  widget->PlaceWidget();
  rwi->Initialize();

  r->ResetCamera();
  rwi->Start();

  volume->Delete();
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}

