/**
 * sliceactor.cxx
 * by Tim Peeters
 *
 * 2006-05-02	Tim Peeters
 * - First version.
 */

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkImageData.h>
#include "vtkImageSliceActor.h"
#include "vtkImageOrthogonalSlicesActor.h"
//#include "vtkDTIComponentReader.h"

#include <vtkVolume16Reader.h>

//using namespace bmia;

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

  // Create the reader for the data
//  vtkDTIReader2* reader = vtkDTIReader2::New();
//  reader->SetFileName(filename);

/*
  bmia::vtkDTIComponentReader* reader = bmia::vtkDTIComponentReader::New();
  reader->SetDataScalarTypeToFloat();
  reader->SetFileName(filename);
*/

  vtkVolume16Reader *reader = vtkVolume16Reader::New();
    reader->SetDataDimensions (64,64);
    reader->SetImageRange (1,93);
    reader->SetDataByteOrderToLittleEndian();
    reader->SetFilePrefix (argv[1]);
    reader->SetDataSpacing (3.2, 3.2, 1.5);

  int extent[6];
//  reader->GetOutput()->Update();
  reader->GetOutput()->GetExtent(extent);
  //reader->Delete(); reader = NULL;
  cout<<"Extent == "<<extent[0]<<", "<<extent[1]<<", "
  	  <<extent[2]<<", "<<extent[3]<<", "<<extent[4]
  	  <<", "<<extent[5]<<".\n";


  bmia::vtkImageSliceActor* sliceActor = bmia::vtkImageSliceActor::New();
  sliceActor->SetInput(reader->GetOutput());
  bmia::vtkImageOrthogonalSlicesActor* slicesActor = bmia::vtkImageOrthogonalSlicesActor::New();
  slicesActor->SetInput(reader->GetOutput());
//  slicesActor->SetInput(reader->GetOutput());
//  r->AddActor(sliceActor);
  r->AddActor(slicesActor);
  sliceActor->SetSliceOrientationToYZ();

  rw->SetSize(400,400);

  rwi->Initialize();
  slicesActor->UpdateInput();
  slicesActor->CenterSlices();

  rwi->Start();

  sliceActor->Delete();
  slicesActor->Delete();
  r->Delete();
  rw->Delete();
  rwi->Delete();

  return 0;
}

