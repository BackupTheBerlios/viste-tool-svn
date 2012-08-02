/**
 * fibermapper.cxx
 * by Tim Peeters
 *
 * 2005-06-20	Tim Peeters
 * - First version. Reads fibers from VTK file and visualizes using standard
 *   vtkPolyDataMapper.
 *
 * 2005-06-22	Tim Peeters
 * - Use my own vtkOpenGLFiberMapper instead of vtkPolyDataMapper.
 *
 * 2005-07-04	Tim Peeters
 * - Read data from file specified on command-line.
 *
 * 2005-12-07	Tim Peeters
 * - Added command-line options:
 *   --help
 *   --no-shading
 *
 * 2006-01-30	Tim Peeters
 * - Call InitializeExtensions() instead of using glew.
 *
 * 2006-02-22	Tim Peeters
 * - Remove InitializeExtensions again. This stuff is now dealt with in the
 *   mapper itself.
 */

#include "vtkFiberMapper.h"

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataReader.h>
#include <vtkActor.h>
#include <vtkProperty.h>

#include "vtkAnisotropicLightingSP.h"
//#include "vtkAnisoLiShadowMapSP.h"
#include <assert.h>
//#include "ExtensionInitialize.h"

int main(int argc, char **argv) {
  cout << "============================================================"<<endl;

  cout<<argc<<" arguments."<<endl;
  bool printhelp = false;
  for (int i=0; i < argc; i++)
    {
    if (strcmp(argv[i], "--help") == 0) printhelp = true;
    cout<<"-- argument "<<i<<" == "<<argv[i]<<endl;
    }

  if (argc < 2) printhelp = true;

  if (printhelp) {
    cout<<"Usage: "<<argv[0]<<" INPUT_FILE [--no-shading] [--no-shadows]"<<endl;
    exit(0);
  }

  bool shading = true;
  bool shadows = true;
  for (int i=2; i < argc; i++)
    {
    if (strcmp(argv[i], "--no-shading") == 0) shading = false;
    if (strcmp(argv[i], "--no-shadows") == 0) shadows = false;

    }

  const char* filename = argv[1];

  vtkRenderer* r = vtkRenderer::New();
  //r->DebugOn();
  vtkRenderWindow* rw = vtkRenderWindow::New();
 // rw->AddRenderer(r);
  vtkRenderWindowInteractor* rwi = vtkRenderWindowInteractor::New();
  rwi->SetRenderWindow(rw);
  rwi->SetInteractorStyle(vtkInteractorStyleTrackballCamera::New());

  vtkPolyDataReader* reader = vtkPolyDataReader::New();
  reader->SetFileName(filename);
  //bmia::vtkFiberMapper* mapper = bmia::vtkFiberMapper::New();
  vtkPolyDataMapper* mapper;
//  if (shading)
//    {
    bmia::vtkFiberMapper* fibermapper = bmia::vtkFiberMapper::New();
    fibermapper->ShadowingOn();
    fibermapper->LightingOn();

    if (!shadows)
      {
      fibermapper->ShadowingOff();
      }

    if (!shading)
      {
      fibermapper->LightingOff();
      }

    mapper = fibermapper;
//    }
//  else // no shading
//    {
//    mapper = vtkPolyDataMapper::New();

    // disable scalar visibility. Otherwise all fibers are blue (?) 
//    mapper->ScalarVisibilityOff();
//    }

//  mapper->GetShaderProgram()->SetDiffuseContribution(0.5);
//  mapper->GetShaderProgram()->SetSpecularContribution(0.5);
//  mapper->GetShaderProgram()->SetSpecularPower(20);
  //mapper->DebugOn();
  cout<<"Reading data..."<<endl;
  assert(reader->GetOutput());
  reader->GetOutput()->Update();
  cout<<"Done!"<<endl;
  mapper->SetInput(reader->GetOutput());
  reader->Delete(); reader = NULL;
  //mapper->LightingOff();

  vtkActor* actor = vtkActor::New();
  actor->SetMapper(mapper);
  mapper->Delete(); mapper = NULL;

  r->AddActor(actor);
  actor->GetProperty()->SetColor(1.0, 0.9, 0.2);
  actor->Delete(); actor = NULL;

  r->SetBackground(0, 0, 0);
  rw->SetSize(800, 600);

  rwi->Initialize();
  //glewInit();
  //InitializeExtensions(rw);
  rw->AddRenderer(r);

  rwi->Start();

  r->Delete(); r = NULL;
  rw->Delete(); rw = NULL;
  rwi->Delete(); rwi = NULL;
}
