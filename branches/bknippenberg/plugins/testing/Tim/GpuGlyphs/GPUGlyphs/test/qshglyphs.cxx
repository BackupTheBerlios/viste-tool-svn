/**
 * shglyphs2.cxx
 * by Tim Peeters
 *
 * 2008-10-03	Tim Peeters
 * - First version
 *
 * 2008-11-20	Tim Peeters
 * - Add support for vtkCHGlyphMapper. The user
 *   can now choose between the two different mappers
 *   using --cyl
 */

#include <QtGui/QApplication>
#include "SHGlyphsGUI.h"
#include "CHGlyphsGUI.h"
#include <QtGui/QStyleFactory>
#include "vtkSHReader.h"
#include "vtkSHGlyphMapper.h"
#include "vtkCHGlyphMapper.h"
#include <vtkPlaneWidget.h>
#include <vtkImageData.h>
#include <vtkRenderer.h>

int main(int argc, char *argv[])
{
  if (argc < 2) {
    cout<<"Usage: "<<argv[0]<<" INPUT_FILE"<<endl;
    exit(0);
  } // if

  cout<<argc<<" arguments."<<endl;
  bool printhelp = false;
  for (int i=0; i < argc; i++)
    {
    if (strcmp(argv[i], "--help") == 0) printhelp = true;
    cout<<"-- argument "<<i<<" == "<<argv[i]<<endl;
    }

  if (argc < 2) printhelp = true;

  if (printhelp) {
    cout<<"Usage: "<<argv[0]<<" INPUT_FILE [--cyl]"<<endl;
    exit(0);
  }

  bool cyl = false;
  for (int i=2; i < argc; i++)
    {
    if (strcmp(argv[i], "--cyl") == 0) cyl = true;
    }

  const char* filename = argv[1];

  bmia::vtkSHReader* reader = bmia::vtkSHReader::New();
  reader->SetFileName(filename);
  cout<<"Reading data..."<<endl;
  reader->Update();
  cout<<"Finished reading data!"<<endl;

//  QApplication app(argc, argv);

  if (!cyl)
    {
    QApplication app(argc, argv);
    bmia::SHGlyphsGUI mainWin;
    mainWin.GetMapper()->SetInputConnection(reader->GetOutputPort());

    mainWin.PlaneWidget->SetInput(reader->GetOutput());
    mainWin.PlaneWidget->On();
    mainWin.PlaneWidget->NormalToZAxisOn();
    mainWin.PlaneWidget->PlaceWidget();

    mainWin.Renderer->ResetCamera();

    mainWin.show();
    app.setStyle(QStyleFactory::create("Macintosh"));
    return app.exec();
    }
  else // cyl
    {
    QApplication app(argc, argv);
    bmia::CHGlyphsGUI mainWin;
    mainWin.GetMapper()->SetInputConnection(reader->GetOutputPort());

    mainWin.PlaneWidget->SetInput(reader->GetOutput());
    mainWin.PlaneWidget->On();
    mainWin.PlaneWidget->NormalToZAxisOn();
    mainWin.PlaneWidget->PlaceWidget();

    mainWin.Renderer->ResetCamera();

    mainWin.show();
    app.setStyle(QStyleFactory::create("Macintosh"));
    return app.exec();
    }
}
