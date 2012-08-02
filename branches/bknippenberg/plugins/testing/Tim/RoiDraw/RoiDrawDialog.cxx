/*
 * RoiDrawDialog
 *
 * 2010-11-16	Tim Peeters
 * - First version
 *
 * 2010-12-14	Evert van Aart
 * - Hotfix solution for the bug in Windows, which causes errors when
 *   switching windows. This is an improvised solution, a better 
 *   solution should be made in the near future.
 *
 * 2011-01-27	Evert van Aart
 * - Added support for transformed planes.
 *
 */

#include "RoiDrawDialog.h"
#include "RoiDrawPlugin.h"
#include "Helpers/vtkImageSliceActor.h"
#include "data/DataSet.h"
#include "Helpers/vtkImageTracerWidget2.h"

#include <vtkCamera.h>
#include <vtkGlyphSource2D.h>
#include <vtkImageData.h>
//#include <vtkImageTracerWidget.h>
#include <vtkInteractorStyleImage.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkIdList.h>
#include <vtkCellArray.h>
#include <vtkMath.h>
#include <vtkMatrix4x4.h>

#include <QtDebug>

namespace bmia {

RoiDrawDialog::RoiDrawDialog(RoiDrawPlugin* pplugin, QWidget* parent) : QDialog(parent)
{
    this->plugin = pplugin;
    Q_ASSERT(this->plugin);

    this->setupUi(this);

    this->renderer = vtkRenderer::New();
    this->vtkWidget->GetRenderWindow()->AddRenderer(this->renderer);
    vtkRenderWindowInteractor* i = this->vtkWidget->GetInteractor();
    vtkInteractorStyleImage* istyle = vtkInteractorStyleImage::New();
    i->SetInteractorStyle(istyle);
    istyle->Delete(); istyle = NULL;

    Q_ASSERT(this->sliceDataSets.size() == 0);
    this->selectedData = -1;
    this->sliceInput = NULL;

    this->tracerWidget = vtkImageTracerWidget2::New();
    this->tracerWidget->SetInteractor(i);
    // TODO: callback (zie AbstractWidgetSeeding::setupWidget) en ImageTracerWidgetPolygon constructor
    this->tracerWidget->SetDefaultRenderer(this->renderer);
    this->tracerWidget->GetLineProperty()->SetColor(1.0, 1.0, 1.0);
    this->tracerWidget->GetLineProperty()->SetLineWidth(2.0);
    this->tracerWidget->GetGlyphSource()->SetColor(1, 0, 0); // make the handles red
    this->tracerWidget->GetGlyphSource()->SetGlyphTypeToThickCross();
    this->tracerWidget->GetGlyphSource()->SetScale(5.0); // TODO: make scale dependent on data/viewport size.
    this->tracerWidget->ProjectToPlaneOn();
//    this->tracerWidget->SnapToImageOn();

	//tracer_widget->SnapToImageOn(); // commented out because it gives the following warning:
	//Generic Warning: In /home/tim/dl/VTK/Widgets/vtkImageTracerWidget2.cxx, line 1403 SetInput with type vtkImageData first

	//tracer_widget->SetProjectionPosition(0);
    this->tracerWidget->AutoCloseOn();
	// use a capture radius of 1000 pixels to make it always close.
    this->tracerWidget->SetCaptureRadius(1000.0);

    //this->tracerWidget->On();


    connect(this->sliceComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(selectSliceData(int)));
    connect(this->applyButton, SIGNAL(clicked()), this, SLOT(apply()));
    connect(this->closeButton, SIGNAL(clicked()), this, SLOT(close()));
}

RoiDrawDialog::~RoiDrawDialog()
{
    this->tracerWidget->Delete(); this->tracerWidget = NULL;
    this->renderer->Delete(); this->renderer = NULL;
    // TODO
    // remove all data sets that were added?
    this->plugin = NULL;
}

void RoiDrawDialog::addSliceData(data::DataSet* ds)
{
    vtkImageSliceActor* actor = this->getSliceActorFromData(ds);

    this->sliceDataSets.append(ds);
    this->sliceComboBox->addItem(ds->getName());
}

vtkImageSliceActor* RoiDrawDialog::getSliceActorFromData(data::DataSet* ds)
{
    Q_ASSERT(ds);
    Q_ASSERT(ds->getKind() == "sliceActor");
    vtkObject* o = ds->getVtkObject();
    Q_ASSERT(o);
    vtkAssembly* assembly = vtkAssembly::SafeDownCast(o);
    Q_ASSERT(assembly);
    vtkImageSliceActor* actor = static_cast<vtkImageSliceActor*>(o);
    Q_ASSERT(actor);
    return actor;
}

void RoiDrawDialog::selectSliceData(int index)
{
    if (this->selectedData != -1)
	{
	if (this->tracerWidget->GetEnabled()) this->tracerWidget->Off();
	this->tracerWidget->SetViewProp(NULL);
	data::DataSet* sdata = this->sliceDataSets.at(this->selectedData);
	vtkImageSliceActor* actor = this->getSliceActorFromData(sdata);
	this->renderer->RemoveActor(actor);
	actor = NULL; sdata = NULL;
	this->sliceInput = NULL;
	}

    if (index != -1)
    	{ // select a slice data
	Q_ASSERT(index < this->sliceDataSets.size());
	data::DataSet* sdata = this->sliceDataSets.at(index);
	vtkImageSliceActor* actor = this->getSliceActorFromData(sdata);
	this->sliceInput = actor->GetInput();
	// Q_ASSERT(sliceInput); // input may be NULL!
	this->renderer->AddActor(actor);
	this->selectedData = index;
	this->ResetCamera(actor);

	vtkRenderWindowInteractor* i = this->vtkWidget->GetInteractor();

	if (this->sliceInput)
	    {
	    this->tracerWidget->SetInput(this->sliceInput);
	    this->tracerWidget->SetViewProp(actor);
	    this->tracerWidget->On();
	    this->tracerWidget->PlaceWidget(); 
	    }
	actor = NULL; sdata = NULL;
	}

	this->turnPickingOff();
	this->makeCurrent();
//    this->vtkWidget->GetRenderWindow()->Render();
}

void RoiDrawDialog::sliceDataChanged(data::DataSet* ds)
{
    if (this->selectedData == -1) return;

    data::DataSet* sdata = this->sliceDataSets.at(this->selectedData);
    if (ds == sdata)
	{
	this->selectSliceData(this->selectedData);
/*
	vtkImageSliceActor* actor = this->getSliceActorFromData(sdata);
	if (this->sliceInput != actor->GetInput())
	    {
	    this->sliceInput = actor->GetInput();
	    this->ResetCamera(actor);
	    } // input was updated
	actor = NULL;
	this->vtkWidget->GetRenderWindow()->Render();
*/
	} // data was updated
}

void RoiDrawDialog::ResetCamera(vtkImageSliceActor* actor)
{
    Q_ASSERT(actor);
    int axis = actor->GetSliceOrientation();
    //this->renderer->SetViewport(0.0, 1.0/3.0*(float)axis, 0.2, 1.0/3.0*(float)(axis+1));
    
    Q_ASSERT(this->renderer);
    Q_ASSERT((axis >= 0) && (axis < 3));

    vtkImageData* input = actor->GetInput();
    if (!input) return;

    vtkCamera* camera = vtkCamera::New();
    camera->ParallelProjectionOn();

    double bounds[6];
    input->GetBounds(bounds);

    double center[3];
    double normal[3];
    actor->GetPlaneCenter(center);
    actor->GetPlaneNormal(normal);

	// "View up" vector for the camera
	double viewUp[3];

	// The view direction depends on the axis
	switch (axis)
	{
	case 0:
	case 1:
		viewUp[0] = 0.0;
		viewUp[1] = 0.0;
		viewUp[2] = 1.0;
		break;

	case 2:
		viewUp[0] = 0.0;
		viewUp[1] = 1.0;
		viewUp[2] = 0.0;
		break;
	}

	// Get the user transformation matrix from the plane actor
	vtkMatrix4x4 * m = actor->GetUserMatrix();

	// Check if the matrix exists
	if (m)
	{
		// Transform the center of the plane (including translation)
		double center4[4] = {center[0], center[1], center[2], 1.0};
		m->MultiplyPoint(center4, center4);
		center[0] = center4[0];
		center[1] = center4[1];
		center[2] = center4[2];

		// Transform the plane normal (excluding translation)
		double normal4[4] = {normal[0], normal[1], normal[2], 0.0};
		m->MultiplyPoint(normal4, normal4);
		normal[0] = normal4[0];
		normal[1] = normal4[1];
		normal[2] = normal4[2];

		// Transform the view vector (excluding translation)
		double viewUp4[4] = {viewUp[0], viewUp[1], viewUp[2], 0.0};
		m->MultiplyPoint(viewUp4, viewUp4);
		viewUp[0] = viewUp4[0];
		viewUp[1] = viewUp4[1];
		viewUp[2] = viewUp4[2];
	}

	// Normalize the normal
	if (vtkMath::Norm(normal) == 0.0)
	{
		return;
	}

	vtkMath::Normalize(normal);

	// Normalize the view vector
	if (vtkMath::Norm(viewUp) == 0.0)
	{
		return;
	}

	vtkMath::Normalize(viewUp);

	// Set the position of the camera. We start at the center of the plane, and move 
	// along its normal to ensure head-on projection for rotated planes. The distance
	// moved along the normal is equal to the image size along the selected axis, to
	// ensure that the camera is placed outside of the volume.

	camera->SetPosition(	center[0] - (bounds[1] - bounds[0]) * normal[0], 
							center[1] - (bounds[3] - bounds[2]) * normal[1], 
							center[2] - (bounds[5] - bounds[4]) * normal[2]);

	// Set the view vector for the camera
	camera->SetViewUp(viewUp);

	// Set the center of the plane as the camera's focal point
	camera->SetFocalPoint(center);

	/*
  switch (axis)
    {
    case 0:
      camera->SetPosition(-1, center[1], center[2]);
      camera->SetViewUp(0, 0, 1);
      break;
    case 1:
      camera->SetPosition(center[0], -1, center[2]);
      camera->SetViewUp(0, 0, 1);
      break;
    case 2:
      camera->SetPosition(center[0], center[1], -1);
      camera->SetViewUp(0, 1, 0);
      break;
    }
 
  camera->SetFocalPoint(center);
*/
  double min = bounds[0];
  if (bounds[2]<min) min = bounds[2];
  if (bounds[4]<min) min = bounds[4];
  double max = bounds[1];
  if (bounds[3]>max) max = bounds[3];
  if (bounds[5]>max) max = bounds[5];

  camera->Dolly(-max);
  camera->SetParallelScale((max-min)/2);

//  cam->SetClippingRange(min, 2*max+1);
  this->renderer->SetActiveCamera(camera);
  // XXX: this works. but it's not very nice.
  // does it always work? negative voxel size (or would that be ridiculous?)
  // min always seems to be 0.
//  int extent[6];
//  this->Input->GetExtent(extent);
//  double spacing[3];
//  this->Input->GetSpacing(spacing);

//  for (int i=0; i < 3; i++) bounds[2*i+1] = 2*bounds[2*i+1]+1;

  // just to be sure
  for (int i=0; i < 3; i++)
    {
//    bounds[2*i] -= 10;
    bounds[2*i+1] = bounds[2*i+1]+1; //10;
    }

// TODO: this may be removed?
  this->renderer->ResetCameraClippingRange(bounds); // stil one off??
  camera->Delete(); camera = NULL;

//  this->GetRenderer()->ResetCamera();
  //this->GetRenderer()->GetActiveCamera()->GetDirectionOfProjection(dop);
  //cout<<"dop = "<<dop[0]<<", "<<dop[1]<<", "<<dop[2]<<"."<<endl;
//  this->Modified();

	this->tracerWidget->setMatrix(m, normal);
	this->tracerWidget->ProjectToPlaneOff();
	this->tracerWidget->SetProjectionNormal(axis);
	this->tracerWidget->SetProjectionPosition(0); // why?
}

void RoiDrawDialog::close()
{
    this->hide();
}

void RoiDrawDialog::apply()
{
    // Get the path from the tracer widget:
    vtkPolyData* drawnRoi = vtkPolyData::New();
    this->tracerWidget->GetPath(drawnRoi);
    Q_ASSERT(drawnRoi);

    // Update the depth of the points in the path to match the current slice:
    vtkPoints* points = drawnRoi->GetPoints();
    Q_ASSERT(points);

	vtkIdList * roiList = vtkIdList::New();
	for (vtkIdType ptId = 0; ptId < points->GetNumberOfPoints(); ++ptId)
	{
		double p[3];
		points->GetPoint(ptId, p);
		qDebug() << p[0] << " " << p[1] << " " << p[2];
		roiList->InsertNextId(ptId);
	}
	vtkCellArray * lines = vtkCellArray::New();
	lines->InsertNextCell(roiList);
	drawnRoi->SetLines(lines);
	lines->Delete();
	roiList->Delete();

    Q_ASSERT(this->selectedData != -1);
    data::DataSet* sdata = this->sliceDataSets.at(this->selectedData);
    vtkImageSliceActor* actor = this->getSliceActorFromData(sdata);
    int axis = actor->GetSliceOrientation();
 
    float depth = actor->GetSliceLocation();

    vtkIdType numPoints = points->GetNumberOfPoints();
    double point[3];
    for (vtkIdType i=0; i < numPoints; i++)
	{
	points->GetPoint(i, point);
	point[axis] = depth;
	points->SetPoint(i, point);
	} // for i

	this->plugin->updateROI(drawnRoi, actor->GetUserMatrix());
    drawnRoi->Delete(); drawnRoi = NULL;
}


void RoiDrawDialog::turnPickingOff()
{
	// Turn off picking for the dialog render window. Doing so may
	// prevent crashes when switching between windows.

	this->vtkWidget->GetRenderWindow()->IsPickingOff();
}

bool RoiDrawDialog::event(QEvent *e)
{
	// Show the window when it receives focus
	if (e->type() == QEvent::WindowActivate)
	{
		this->setWindowState(Qt::WindowActive);

		// Additionally, we manually turn off picking, and manually force
		// the dialog render window to become the active one. This is all part of 
		// a hotfix solution for a bug that causes errors when switching windows
		// in Windows. It is probably due to the fact that we use the same texture in 
		// two different windows, which is not supported according to VTK. A more
		// stable, permanent solution should be found in the near future.

		this->turnPickingOff();
		this->makeCurrent();
	}
	// Minimize the window when it looses focus
	if (e->type() == QEvent::WindowDeactivate)
	{
		this->setWindowState(Qt::WindowMinimized);
	}

	// Use the default event handler
	return QDialog::event(e);
}

void RoiDrawDialog::makeCurrent()
{
	// Render the window and make it the current one. Probably not really needed,
	// but this is all part of the hotfix described above, so it is not meant 
	// to be a permanent solution.

	if (this->isActiveWindow())
	{
		this->vtkWidget->GetRenderWindow()->Render();
		this->vtkWidget->GetRenderWindow()->MakeCurrent();
	}
}


} // namespace bmia
