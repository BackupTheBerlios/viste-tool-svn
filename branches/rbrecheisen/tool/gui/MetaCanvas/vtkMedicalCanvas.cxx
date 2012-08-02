/*
 * vtkMedicalCanvas.cxx
 *
 * 2010-09-16	Tim Peeters
 * - First version
 *
 * 2011-02-28	Evert van Aart
 * - Added support for setting the background color with or without gradient.
 *
 */

#include <vtkObjectFactory.h>

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkInteractorStyleImage.h>

#include "vtkMedicalCanvas.h"
#include "vtkSubCanvas.h"

namespace bmia {

vtkStandardNewMacro(vtkMedicalCanvas);

vtkMedicalCanvas::vtkMedicalCanvas()
{
    this->SubCanvas3D = vtkSubCanvas::New();
	this->SubCanvas3D->subCanvasName = std::string("3D View");
    // set the interactor style for the subcanvas.
    vtkInteractorStyleTrackballCamera* istyle = vtkInteractorStyleTrackballCamera::New();
    this->SubCanvas3D->SetInteractorStyle(istyle);
    istyle->Delete(); istyle = NULL;

    // set the background of the subcanvas
	this->SubCanvas3D->GetRenderer()->SetBackground(0.0, 0.0, 0.0);
    this->SubCanvas3D->GetRenderer()->SetBackground2(0.25, 0.25, 0.25);
    this->SubCanvas3D->GetRenderer()->GradientBackgroundOn();

    // add the subcanvas to the metacanvas and set
    // its viewport to the full metacanvas.
    this->AddSubCanvas(this->SubCanvas3D);
	this->SubCanvas3D->SetViewport(0.2, 0.0, 1.0, 1.0);

    for (int axis=0; axis < 3; axis++)
	{
	this->SubCanvas2D[axis] = vtkSubCanvas::New();
	vtkInteractorStyleImage* is = vtkInteractorStyleImage::New();
	this->SubCanvas2D[axis]->SetInteractorStyle(is);
	is->Delete(); is = NULL;
	this->SubCanvas2D[axis]->GetRenderer()->SetBackground(0.0, 0.0, 0.0);
	this->SubCanvas2D[axis]->GetRenderer()->SetBackground2(0.2, 0.2, 0.2);
	this->SubCanvas2D[axis]->GetRenderer()->GradientBackgroundOn();
	this->AddSubCanvas(this->SubCanvas2D[axis]);
	//this->SubCanvas2D[axis]->SetViewport(1.0/3.0*(float)axis+0.001, 0.0, 1.0/3.0*(float)(axis+1)-0.001, 0.3);
	this->SubCanvas2D[axis]->SetViewport(0.0, 1.0/3.0*(float)axis, 0.2, 1.0/3.0*(float)(axis+1));
	} // for axis

	this->SubCanvas2D[0]->subCanvasName = std::string("2D View - YZ Plane");
	this->SubCanvas2D[1]->subCanvasName = std::string("2D View - XZ Plane");
	this->SubCanvas2D[2]->subCanvasName = std::string("2D View - XY Plane");
}

vtkMedicalCanvas::~vtkMedicalCanvas()
{
    this->SubCanvas3D->Delete(); this->SubCanvas3D = NULL;
}

vtkRenderer* vtkMedicalCanvas::GetRenderer3D()
{
    return this->SubCanvas3D->GetRenderer();
}

vtkSubCanvas* vtkMedicalCanvas::GetSubCanvas3D()
{
    return this->SubCanvas3D;
}

vtkSubCanvas* vtkMedicalCanvas::GetSubCanvas2D(int i)
{
    if ((0 <= i) && (i < 3)) return this->SubCanvas2D[i];
    else return NULL;
}


void vtkMedicalCanvas::setBackgroundColor(double r, double g, double b)
{
	// Set color of 3D view
	this->SubCanvas3D->GetRenderer()->SetBackground(r, g, b);
	this->SubCanvas3D->GetRenderer()->GradientBackgroundOff();

	// Set color of 2D views
	for (int axis = 0; axis < 3; axis++)
	{
		this->SubCanvas2D[axis]->GetRenderer()->SetBackground(r, g, b);
		this->SubCanvas2D[axis]->GetRenderer()->GradientBackgroundOff();
	}

	this->GetRenderWindow()->Render();
}

void vtkMedicalCanvas::setGradientBackground(	double r1, double g1, double b1, 
												double r2, double g2, double b2)
{
	// Set color of 3D view
	this->SubCanvas3D->GetRenderer()->SetBackground(r1, g1, b1);
	this->SubCanvas3D->GetRenderer()->SetBackground2(r2, g2, b2);
	this->SubCanvas3D->GetRenderer()->GradientBackgroundOn();

	// Set color of 2D views
	for (int axis = 0; axis < 3; axis++)
	{
		this->SubCanvas2D[axis]->GetRenderer()->SetBackground(r1, g1, b1);
		this->SubCanvas2D[axis]->GetRenderer()->SetBackground2(r2, g2, b2);
		this->SubCanvas2D[axis]->GetRenderer()->GradientBackgroundOn();
	}

	this->GetRenderWindow()->Render();
}

} // namespace bmia
