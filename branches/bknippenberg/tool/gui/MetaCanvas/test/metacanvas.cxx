/**
 * metacanvas.cxx
 * by Tim Peeters
 *
 * 2005-07-15	Tim Peeters
 * - First version
 */

#include "vtkMetaCanvas.h"

#include "vtkConeSubCanvas.h"
#include "vtkCylinderSubCanvas.h"

using namespace bmia;

int main(int argc, char **argv)
{
  vtkMetaCanvas* mc = vtkMetaCanvas::New();

  vtkConeSubCanvas* sc0 = vtkConeSubCanvas::New();
  vtkCylinderSubCanvas* sc1 = vtkCylinderSubCanvas::New();
  sc1->SetViewport(0.0, 0.0, 0.5, 0.5);
  sc0->SetViewport(0.2, 0.2, 1.0, 1.0);

  mc->AddSubCanvas(sc0);
  mc->AddSubCanvas(sc1);
  sc1->Delete(); sc0->Delete();

  mc->SetSize(640,400);
  mc->Start();
  mc->Delete();
}
