/**
 * SHGlyphsGUI.cxx
 * by Tim Peeters
 *
 * 2008-10-03	Tim Peeters
 * - First version
 */

#include "SHGlyphsGUI.h"
#include <vtkCommand.h>
#include <vtkPoints.h>
#include <vtkPlaneWidget.h>
#include <vtkPointSet.h>
#include <vtkUnstructuredGrid.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkMath.h>
#include <assert.h>
#include "vtkSHGlyphMapper.h"
#include <vtkVolume.h>

#include <vtkConeSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>

namespace bmia {

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


SHGlyphsGUI::SHGlyphsGUI(QWidget* parent) : QMainWindow(parent)
{
  this->setupUi(this);
  this->SHGlyphMapper = vtkSHGlyphMapper::New();
  vtkRenderWindow* rw = this->vtkWidget->GetRenderWindow();
  this->Renderer = vtkRenderer::New();
  this->Renderer->SetRenderWindow(rw); //rw->GetRenderer();
  rw->AddRenderer(this->Renderer);

  vtkPointSet* pointset = vtkUnstructuredGrid::New();
  // create a widget
  this->PlaneWidget = vtkPlaneWidget::New();
  this->PlaneWidget->SetInteractor(this->vtkWidget->GetInteractor());
//  planeWidget->SetInput(reader->GetOutput());

  this->Callback = vtkWidgetCallback::New();
  this->Callback->PlaneWidget = this->PlaneWidget;
  this->Callback->Renderer = this->Renderer;
  this->Callback->PointSet = pointset;
  this->Callback->SeedPointDistance = 1.0; //2.0; // Seedpoint Distance in mm (so not in voxels)
  this->PlaneWidget->AddObserver(vtkCommand::InteractionEvent, this->Callback);


  this->PlaneWidget->SetDefaultRenderer(this->Renderer);
  //this->PlaneWidget->On();
  //this->PlaneWidget->NormalToYAxisOn();
  //this->PlaneWidget->PlaceWidget();

  this->Callback->Execute(NULL, 1, NULL);

/*
  vtkConeSource* cone = vtkConeSource::New();
  vtkPolyDataMapper * coneMapper = vtkPolyDataMapper::New();
  coneMapper->SetInput(cone->GetOutput());
  vtkActor* coneActor = vtkActor::New();
  coneActor->SetMapper(coneMapper);
  r->AddActor(coneActor);
  r->SetBackground(1.0, 0.0, 1.0);
*/

  this->SHGlyphMapper->SetSeedPoints(pointset);
  this->SHGlyphMapper->SetGlyphScaling(0.5);

  vtkVolume* volume = vtkVolume::New();
  volume->SetMapper(this->SHGlyphMapper);
  this->Renderer->AddViewProp(volume);

  connect(this->scalingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setGlyphScaling(double)));
  connect(this->sharpeningSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setA0Scaling(double)));
  connect(this->localScalingCheckBox, SIGNAL(toggled(bool)), this, SLOT(setLocalScaling(bool)));
  connect(this->stepSizeSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setStepSize(double)));
  connect(this->refinementSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setNumRefineSteps(int)));
  connect(this->zRotationSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setZRotation(int)));
  connect(this->yRotationSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setYRotation(int)));
  connect(this->seedDistanceSpinBox, SIGNAL(valueChanged(double)), this, SLOT(setSeedPointDistance(double)));
}

SHGlyphsGUI::~SHGlyphsGUI()
{
  this->Callback->Delete(); this->Callback = NULL;
  this->SHGlyphMapper->Delete(); this->SHGlyphMapper = NULL;
}

void SHGlyphsGUI::setSeedPointDistance(double distance)
{
  this->Callback->SeedPointDistance = distance;
  this->Callback->Execute(NULL, 1, NULL);
  this->vtkWidget->GetRenderWindow()->Render();
}

void SHGlyphsGUI::setGlyphScaling(double scale)
{
  this->SHGlyphMapper->SetGlyphScaling(scale);
  this->vtkWidget->GetRenderWindow()->Render();
}

void SHGlyphsGUI::setA0Scaling(double scale)
{
  this->SHGlyphMapper->SetSharpening(scale);
  this->vtkWidget->GetRenderWindow()->Render();
}

void SHGlyphsGUI::setLocalScaling(bool local)
{
  cout<<"Setting local scaling to "<<local<<endl;
  this->SHGlyphMapper->SetLocalScaling(local);
  this->vtkWidget->GetRenderWindow()->Render();
}

void SHGlyphsGUI::setStepSize(double step)
{
  if (step > 0.0)
    {
    this->SHGlyphMapper->SetStepSize((float)step);
    this->vtkWidget->GetRenderWindow()->Render();
    }
}

void SHGlyphsGUI::setNumRefineSteps(int num)
{
  this->SHGlyphMapper->SetNumRefineSteps(num);
  this->vtkWidget->GetRenderWindow()->Render();
}

void SHGlyphsGUI::setZRotation(int angle)
{
  // compute the angle in rad instead of deg and pass it to the mapper.
  //cout<<"setting rotation angle to "<<angle<<" degrees."<<endl;
  this->SHGlyphMapper->SetZRotationAngle(0.0174533 * (double)angle);
  this->vtkWidget->GetRenderWindow()->Render();
}

void SHGlyphsGUI::setYRotation(int angle)
{
  // compute the angle in rad instead of deg and pass it to the mapper.
  //cout<<"setting rotation angle to "<<angle<<" degrees."<<endl;
  this->SHGlyphMapper->SetYRotationAngle(0.0174533 * (double)angle);
  this->vtkWidget->GetRenderWindow()->Render();
}

} // namespace bmia
