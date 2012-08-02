/**
 * vtkEigensystemToGPU.cxx
 * by Tim Peeters
 *
 * 2008-02-06	Tim Peeters
 * - First version.
 */

#include "vtkEigensystemToGPU.h"
#include "vtkTensorMath.h"

#include <vtkImageData.h>
#include <vtkObjectFactory.h>

#include <vtkPointData.h>
//#include <vtkDoubleArray.h>
#include <vtkFloatArray.h>
#include "AnisotropyMeasures.h"
#include <assert.h>
#include <math.h>

namespace bmia {

vtkStandardNewMacro(vtkEigensystemToGPU);

void vtkEigensystemToGPU::SimpleExecute(vtkImageData* input, vtkImageData* output)
{
  vtkDebugMacro(<<"Computing eigen systems from tensors...");

  this->SetProgressText("Computing eigenvalues and eigendirections from tensors");
  this->UpdateProgress(0.0);

  // Let's start with a bunch of checks. These are not strictly necessary, but may
  // prove handy when something goes wrong.
  if (!input)
    {
    vtkErrorMacro(<<"No input!");
    return;
    }

  if (!output)
    {
    vtkErrorMacro(<<"No output!");
    return;
    }

  vtkPointData* inPD = input->GetPointData();
  if (!inPD)
    {
    vtkErrorMacro(<<"No input pointdata!");
    return;
    }

  vtkDataArray* inTensors = inPD->GetTensors();
  if (!inTensors)
    {
    vtkWarningMacro(<<"Input data has no tensors!");
    return;
    }

  vtkPointData* outPD = output->GetPointData();
  if (!outPD)
    {
    vtkErrorMacro(<<"No output pointdata!");
    return;
    }

  int numPts = input->GetNumberOfPoints();
  if ( numPts != inTensors->GetNumberOfTuples() )
    {
    vtkErrorMacro(<<"Number of tuples mismatch between poindata and tensors!");
    return;
    }

  if (numPts < 1)
    {
    vtkWarningMacro(<<"No data to extract!");
    return;
    }

  // finished the checks. Now do the actual work.

  int i; // counter
//  vtkDoubleArray* Arrays[6];
//  for (i=0; i < 6; i++) Arrays[i] = vtkDoubleArray::New();
  vtkFloatArray* Arrays[2]; //[6];
  for (i=0; i < 2; i++) Arrays[i] = vtkFloatArray::New();

  Arrays[0]->SetName("Eigenvector 1"); // for eigenvector 1 and normalized eigenvalue 1
  Arrays[1]->SetName("Eigenvector 2"); // for eigenvector 2 and normalized eigenvalue 2
//  Arrays[2]->SetName("Eigenvector 3");
//  Arrays[3]->SetName("Eigenvalue 1");
//  Arrays[4]->SetName("Eigenvalue 2");
//  Arrays[5]->SetName("Eigenvalue 3");

  for (i=0; i < 2; i++) Arrays[i]->SetNumberOfComponents(4);
//  for (i=3; i < 6; i++) Arrays[i]->SetNumberOfComponents(1);
  for (i=0; i < 2; i++) Arrays[i]->SetNumberOfTuples(numPts);

  vtkIdType ptId;
  double tensor[9]; 		// the current tensor
  double* val = new double[3];	// eigenvalues
  double* vec9 = new double[9];	// eigenvectors

  double val3[3];
  double vec3x3[3][3];

  for (ptId = 0; ptId < numPts; ptId++)
    {
    inTensors->GetTuple(ptId, tensor);

    // this IsNullTensor part is not required for correctness,
    // but it will save computation time on sparse datasets.
    if (vtkTensorMath::IsNullTensor(tensor))
      {
      for (i=0; i < 2; i++) Arrays[i]->SetTuple4(ptId, 0.0, 0.0, 0.0, 0.0);
//      for (i=3; i < 6; i++) Arrays[i]->SetTuple1(ptId, 0.0);
      } // if (IsNullTensor(tensor)
//    else if (!vtkTensorMath::EigenSystem(tensor, vec9, val))
//    else if (!vtkTensorMath::EigenSystem(tensor, vec3x3, val3))
      else if (!vtkTensorMath::EigenSystemSorted(tensor, vec9, val))
      {
//      vtkErrorMacro(<<"Could not compute eigen system for tensor (("
      vtkDebugMacro(<<"Could not compute eigen system for tensor (("
		      <<tensor[0]<<", "<<tensor[1]<<", "<<tensor[2]<<"), ("
		      <<tensor[2]<<", "<<tensor[3]<<", "<<tensor[4]<<"), ("
		      <<tensor[5]<<", "<<tensor[6]<<", "<<tensor[7]<<"))");
      } // no valid eigensystem. There may be negative eigenvalues.
    else
      { // eigensystem ok. proceed..
      //double evalsum = val3[0]+val3[1]+val3[2];
      double evalsum = val[0]+val[1]+val[2];
      for (i=0; i < 3; i++)
        {
//	double aniso;
//	if (i == 0) aniso = AnisotropyMeasures::LinearAnisotropy(val3);
//	else aniso = AnisotropyMeasures::PlanarAnisotropy(val3);

	double eval;
//	eval = val3[i] / evalsum;
	eval = val[i] / evalsum;
//if (!(val3[i] > 0.0))
if (!(val[i] > 0.0))
  {
  // some times val3 = {nan, nan, nan} :s
  cout<<"ERROR!! val = "<<val[0]<<", "<<val[1]<<", "<<val[2]<<endl;
  }
//	assert(val3[i] > 0.0);
//	assert(eval > 0.0);
/*
	float veclengthsq = vec3x3[i][0]*vec3x3[i][0] + vec3x3[i][1]*vec3x3[i][1] + vec3x3[i][2]*vec3x3[i][2];
	if ((veclengthsq > 1.01) || (veclengthsq < 0.99))
          {
	  cout<<"AAAAAH ------------------------- EIGENVECTORS DON'T HAVE UNIT-LENGTH! " << veclengthsq<<endl;
	  eval = 0.0;
	  }
*/
	

//if (ptId<50)
//cout<<"GPU ptId = "<<ptId<<", eigenvector "<<i<< " = ("<<vec3x3[i][0]<<", "<<vec3x3[i][1]<<", "<<vec3x3[i][2]<<")."<<endl;
//if (i != 2) Arrays[i]->SetTuple4(ptId, (vec3x3[i][0]+1.0)/2.0, (vec3x3[i][1]+1.0)/2.0, (vec3x3[i][2]+1.0)/2.0, eval); //aniso);
if (i != 2) Arrays[i]->SetTuple4(ptId, (vec9[3*i]+1.0)/2.0, (vec9[3*i+1]+1.0)/2.0, (vec9[3*i+2]+1.0)/2.0, eval); //aniso);
        //Arrays[i]->SetTuple4(ptId, (vec3x3[0][i]+1.0)/2.0, (vec3x3[1][i]+1.0)/2.0, (vec3x3[2][i]+1.0)/2.0, eval); //aniso);
        } // for
//    assert((val3[0]+val3[1])/evalsum <= 1.0);

    if (ptId % 50000 == 0) this->UpdateProgress(((float)ptId) / ((float)numPts));

      } // else
    } // for ptId
  delete[] val; delete[] vec9; val = NULL; vec9 = NULL;

  // copy extent, dimensions, origin, spacing, etc(?) to output.
  output->CopyStructure(input);

  // add the arrays to the output and select active scalars and vectors.
  for (i=0; i < 2; i++) outPD->AddArray(Arrays[i]);
//  outPD->SetActiveScalars(Arrays[3]->GetName());
//  outPD->SetActiveVectors(Arrays[0]->GetName());
  outPD->SetActiveScalars(Arrays[0]->GetName());

  for (i=0; i < 2; i++)
    {
    Arrays[i]->Delete();
    Arrays[i] = NULL;
    }

  this->UpdateProgress(1.0);
  vtkDebugMacro(<<"Eigensystem computation finished.");

} // SimpleExecute()

} // namespace bmia
