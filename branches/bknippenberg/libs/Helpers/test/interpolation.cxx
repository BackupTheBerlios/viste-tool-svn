/**
 * interpolation.cxx
 * by Tim Peeters
 *
 * 2005-01-26	Tim Peeters	First version
 * 2005-01-31	Tim Peeters	Added test for interpolating tensors
 */

#include "vtkImageDataInterpolator.h"
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsReader.h>

bmia::vtkImageDataInterpolator* interpolator;

void testInterpolation(double xmin, double ymin, double zmin,
                       double xmax, double ymax, double zmax, int steps)
{
  double x = xmin; double y = ymin; double z = zmin;
  double xinc = (xmax-xmin)/(double)steps;
  double yinc = (ymax-ymin)/(double)steps;
  double zinc = (zmax-zmin)/(double)steps;

  cout<<"-- Going from ("<<xmin<<", "<<ymin<<", "<<zmin<<") to (";
  cout<<xmax<<", "<<ymax<<", "<<zmax<<"), step ("<<xinc<<", "<<yinc<<", "<<zinc<<")."<<endl;

  
  //while ( (x <= xmax+0.01) && (y <= ymax+0.01) && (z<=zmax+0.01) )
  for (int i = -1; i < steps; i++)
    {
    cout<<"Value at ("<<x<<", "<<y<<", "<<z<<") is "<<interpolator->GetInterpolatedScalar1At(x, y, z)<<endl;
    x += xinc; y += yinc; z += zinc;
    }
}

void tensorInterpol(double xmin, double ymin, double zmin,
		double xmax, double ymax, double zmax, int steps)
{
  double x = xmin; double y = ymin; double z = zmin;
  double xinc = (xmax-xmin)/(double)steps;
  double yinc = (ymax-ymin)/(double)steps;
  double zinc = (zmax-zmin)/(double)steps;

  cout<<"-- Interpolating tensors from ("<<xmin<<", "<<ymin<<", "<<zmin<<") to (";
  cout<<xmax<<", "<<ymax<<", "<<zmax<<"), step ("<<xinc<<", "<<yinc<<", "<<zinc<<")."<<endl;

  double* tensor = NULL;
  for (int i=-1; i < steps; i++)
    {
    cout<<"Tensor at ("<<x<<", "<<y<<", "<<z<<") is:"<<endl;

    tensor = interpolator->GetInterpolatedTensorAt(x, y, z);
    if (tensor == NULL) cout << "   NULL"<<endl;
    else for (int j=0; j < 3; j++)
      {
      cout<<"   ( "<<tensor[3*j]<<", "<<tensor[3*j+1]<<", "<<tensor[3*j+2]<<" )"<<endl;
      } // for
    x += xinc; y += yinc; z += zinc;
  } // for
}

int main(int argc, char **argv) {
  // Create the reader for the data
  vtkStructuredPointsReader* reader = vtkStructuredPointsReader::New();
  reader->SetFileName("simplevolume.vtk");

  //vtkStructuredPoints* data = reader->GetOutput();
  //data->Update();
  //data->Print(cout);
  //data = NULL;

  cout<<"Creating interpolator..."<<endl;
  interpolator = bmia::vtkImageDataInterpolator::New();
  interpolator->SetInput(reader->GetOutput());
  reader->Delete(); reader = NULL;

  cout<<"========== Nearest-neighbour interpolation ==========="<<endl;
  interpolator->SetInterpolationTypeToNearest();
  testInterpolation(0, 0, 0, 4, 0, 0, 20);
  testInterpolation(0, 0, 0, 4, 4, 4, 10);
  cout<<"================ Linear interpolation ================"<<endl;
  interpolator->SetInterpolationTypeToLinear();
  testInterpolation(0, 0, 0, 4, 0, 0, 20);
  testInterpolation(0, 0, 0, 4, 4, 4, 10);

  cout<<"Interpolating tensors..."<<endl; 
  reader = vtkStructuredPointsReader::New();
  reader->SetFileName("tensors.vtk");
  interpolator->SetInput(reader->GetOutput());
  reader->Delete(); reader = NULL;

  cout<<"========== Nearest-neighbour interpolation ==========="<<endl;
  interpolator->SetInterpolationTypeToNearest();
  tensorInterpol(0, 0, 0, 3, 0, 0, 6);
  cout<<"================ Linear interpolation ================"<<endl;
  interpolator->SetInterpolationTypeToLinear();
  tensorInterpol(0, 0, 0, 3, 0, 0, 6);
  tensorInterpol(0, 0, 0, 2, 1, 0, 4);

  cout<<"Deleting interpolator..."<<endl;
  interpolator->Delete();
  interpolator = NULL;
  cout<<"Done."<<endl;
}
