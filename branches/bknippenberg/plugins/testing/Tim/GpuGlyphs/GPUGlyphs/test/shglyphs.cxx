/**
 * sh.cxx
 * by Tim Peeters
 *
 * 2008-09-16	Tim Peeters
 * - First version
 */

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageData.h>
//#include "vtkDTIDataManager.h"
//#include "vtkTensorToEigensystemFilter.h"
#include "vtkSHReader.h"

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPlaneWidget.h>
#include "vtkSHGlyphMapper.h"

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

//  bmia::vtkDTIDataManager* manager = bmia::vtkDTIDataManager::New();
//  manager->SetFileName(filename);
  bmia::vtkSHReader* reader = bmia::vtkSHReader::New();
  reader->SetFileName(filename);

  bmia::vtkSHGlyphMapper* mapper = bmia::vtkSHGlyphMapper::New();
//  mapper->DebugOn();

//  bmia::vtkTensorToEigensystemFilter* eigen = bmia::vtkTensorToEigensystemFilter::New();
//  eigen->SetInputConnection(manager->GetOutputPort());
//  eigen->GetOutput()->Update();
//  mapper->SetInputConnection(eigen->GetOutputPort());
//  mapper->SetInputConnection(reader->GetOutputPort());
//  mapper->SetInput(reader->GetOutput());
  mapper->SetInputConnection(reader->GetOutput()->GetProducerPort());

//  eigen->GetOutput()->Print(cout);

  vtkPointSet* pointset = vtkUnstructuredGrid::New();

  // create a widget
  vtkPlaneWidget* planeWidget = vtkPlaneWidget::New();
  planeWidget->SetInteractor(rwi);
//  planeWidget->SetInput(eigen->GetOutput());
  planeWidget->SetInput(reader->GetOutput());

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

  vtkVolume* volume = vtkVolume::New();
  volume->SetMapper(mapper);

//  manager->SetFileName(filename);
  mapper->SetSeedPoints(pointset);
 
  mapper->SetGlyphScaling(0.5);

  rw->SetSize(800,600);
  rw->SetSize(1024,768);

  r->AddViewProp(volume);

  rwi->Initialize();

  r->ResetCamera();
  rwi->Start();

  volume->Delete();
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}

