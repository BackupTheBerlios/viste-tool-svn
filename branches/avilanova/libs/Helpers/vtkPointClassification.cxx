/**
 * vtkPointClassification.cxx
 * by Tim Peeters
 *
 * 2009-03-24	Tim Peeters
 * - First version.
 */

#include "vtkPointClassification.h"
#include <vtkObjectFactory.h>
#include <vtkInformationVector.h>
#include <vtkInformation.h>
#include <vtkImageData.h>
#include <vtkPointSet.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkUnstructuredGrid.h>
#include <assert.h>

namespace bmia {

vtkStandardNewMacro(vtkPointClassification);
vtkCxxSetObjectMacro(vtkPointClassification, InputScalarImage, vtkImageData);

vtkPointClassification::vtkPointClassification()
{
  this->SetNumberOfInputPorts(1); // I need two but now I handle the other one myself.
  this->SetNumberOfOutputPorts(3); // its not used now. I use this->Input{Lower, Middle, Higher}
  // TODO: set type of input 2 to vtkImageData
  // probably in FillOutputPortInformation. See if FillInputPortInformation needs to be set also.
  
  this->LowerThreshold = 0.1;
  this->UpperThreshold = 0.75;

  this->InputScalarImage = NULL;
}

vtkPointClassification::~vtkPointClassification()
{
  if (this->InputScalarImage) this->InputScalarImage->UnRegister(this);
}

int vtkPointClassification::RequestData(
  //vtkInformation *vtkNotUsed(request),
  vtkInformation* request,
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
//  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
//  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the input and ouptut
//  vtkPointSet *input = vtkPointSet::SafeDownCast(
//    inInfo->Get(vtkDataObject::DATA_OBJECT()));
//  vtkPointSet *output = vtkPointSet::SafeDownCast(
//    outInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPointSet* input = vtkPointSet::SafeDownCast(this->GetInput());
  if (!input) return 1;

  //assert(Middle) == output;

 // assert(this->OutputLower);
  //assert(this->OutputMiddle);
  //this->OutputMiddle = output;
  //assert(this->OutputUpper);

  vtkPoints* inPts = input->GetPoints();
  if (!inPts)
    {
    vtkErrorMacro(<<"No input!");
    return 1;
    } // if

  vtkPoints* lowerPts = vtkPoints::New();
  vtkPoints* middlePts = vtkPoints::New();
  vtkPoints* upperPts = vtkPoints::New();

  int numPts = inPts->GetNumberOfPoints();

  if (!this->InputScalarImage)
    {
    vtkErrorMacro(<<"No input scalar volume!");
    return 1;
    }

  vtkPointData* inPD = this->InputScalarImage->GetPointData();
  if (!inPD)
    {
    vtkErrorMacro(<<"No input point data!");
    return 1;
    }

  vtkDataArray* scalars = inPD->GetScalars();
  if (!scalars)
    {
    vtkErrorMacro("No input scalars!");
    return 1;
    }

  double point[3];
  vtkIdType pointId;
  double scalar;

  for (vtkIdType i=0; i < numPts; i++)
    {
    // get the input scalar for the current point
    inPts->GetPoint(i, point);
    pointId = this->InputScalarImage->FindPoint(point);
    scalar = scalars->GetTuple1(pointId);

    // classify point and add point to one of the output vtkPoints:
    if (scalar < this->LowerThreshold) lowerPts->InsertNextPoint(point);
    else if (scalar > this->UpperThreshold) upperPts->InsertNextPoint(point);
    else middlePts->InsertNextPoint(point);
    }  // for i
  
cout<<"******** Classified points"<<endl;
cout<<"Lower: "<<lowerPts->GetNumberOfPoints()<<" points"<<endl;
cout<<"Middle: "<<middlePts->GetNumberOfPoints()<<" points"<<endl;
cout<<"Upper: "<<upperPts->GetNumberOfPoints()<<" points"<<endl;

 // this->OutputLower->SetPoints(lowerPts);
//  this->OutputMiddle->SetPoints(middlePts);
//  this->OutputUpper->SetPoints(upperPts);
  this->GetOutput(0)->SetPoints(lowerPts);
  this->GetOutput(1)->SetPoints(middlePts);
  this->GetOutput(2)->SetPoints(upperPts);

  return 1;
}

void vtkPointClassification::OptimizeThresholds(vtkPointSet* pointset)
{
  cout<<"vtkPointClassification::OptimizeThresholds("<<pointset<<");"<<endl;
  if (!pointset) return;
  vtkPoints* points = pointset->GetPoints();
  cout<<"points = "<<points<<endl;
  if (!points) return;
  int n = points->GetNumberOfPoints();
  cout<<"n = "<<n<<endl;
  double* values = new double[n];
  int num_values = this->GetValuesInPoints(points, values, n);
  cout<<"num_values = "<<num_values <<endl;
  if (num_values <= 0)
    {
    delete[] values; values = NULL;
    return;
    }

  double min = 1.0; double max = 0.0;
  for (int i=0; i < num_values; i++)
    {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
    } // for i

  cout<<"min = "<<min<<", max = "<<max<<endl;
  this->SetLowerThreshold(min);
  this->SetUpperThreshold(max);
}

void vtkPointClassification::OptimizeThresholds(vtkPointSet* pos, vtkPointSet* neg)
{
  cout<<"vtkPointClassification::OptimizeThresholds("<<pos<<", "<<neg<<");"<<endl;
  if (!pos) return;
  if (!neg) return;
  vtkPoints* posPoints = pos->GetPoints();
  vtkPoints* negPoints = neg->GetPoints();
  cout<<"posPoints = "<<posPoints<<", negPoints = "<<negPoints<<endl;
  if (!posPoints) return;
  if (!negPoints) return;
  int posN = posPoints->GetNumberOfPoints();
  int negN = negPoints->GetNumberOfPoints();
  cout<<"posN = "<<posN<<", negN = "<<negN<<endl;
  double* posValues = new double[posN];
  double* negValues = new double[negN];
  int num_pos_values = this->GetValuesInPoints(posPoints, posValues, posN);
  int num_neg_values = this->GetValuesInPoints(negPoints, negValues, negN);
  int i;

  double pmin = 1.0; double pmax = 0.0; double pavg;
  for (i=0; i < num_pos_values; i++)
    {
    if (posValues[i] > pmax) pmax = posValues[i];
    if (posValues[i] < pmin) pmin = posValues[i];
    pavg += posValues[i];
    } // for i	    
  pavg /= (double)num_pos_values;
  cout<<"pmax = "<<pmax<<", pmin = "<<pmin<<", pavg = "<<pavg<<endl;

  double nmin = 1.0; double nmax = 0.0; double navg;
  for (i=0; i < num_neg_values; i++)
    {
    if (negValues[i] > nmax) nmax = negValues[i];
    if (negValues[i] < nmin) nmin = negValues[i];
    navg += negValues[i];
    } // for i
  navg /= (double)num_neg_values;
  cout<<"nmax = "<<nmax<<", nmin = "<<nmin<<", navg = "<<navg<<endl;

  this->SetLowerThreshold(pmin);

  if (pmax > nmin)
    { // find the median
    //this->SetUpperThreshold((pavg+navg)/2.0);
    // Bubble sort is not the most efficient sorting algorithm, but
    // this is not called very often and the lists are short.
    this->BubbleSort(posValues, num_pos_values);
    this->BubbleSort(negValues, num_neg_values);
    double p_median; double n_median;
    if (num_pos_values%2 == 0)
      {
      p_median = 0.5*(posValues[num_pos_values/2 - 1] + posValues[num_pos_values/2]);
      } // if
    else
      {
      p_median = posValues[(num_pos_values-1)/2];
      }
    if (num_neg_values%2 == 0)
      {
      n_median = 0.5*(negValues[num_neg_values/2 - 1] + negValues[num_neg_values/2]);
      }
    else
      {
      n_median = negValues[(num_neg_values-1)/2];
      }
    this->SetUpperThreshold((p_median+n_median)/2.0);
    } // if (pmax > nmin)
  else this->SetUpperThreshold(pmax);
}

int vtkPointClassification::GetValuesInPoints(vtkPoints* points, double* values, int num_points)
{
  if (!this->InputScalarImage) return -1;
  assert(points);
  assert(values);
  
  vtkPointData* pd = this->InputScalarImage->GetPointData();
  if (!pd) return -2;

  vtkDataArray* scalars = pd->GetScalars();
  if (!scalars) return -3;  
  
  double point[3];
  double scalar;

  vtkIdType pointId;
  int n = 0;
  for (vtkIdType i=0; i < num_points; i++)
    {
    points->GetPoint(i, point);
    pointId = this->InputScalarImage->FindPoint(point);
    if (pointId != -1)
      {
      scalar = scalars->GetTuple1(pointId);
      if (scalar != 0.0)
	{
	values[n] = scalar;
	n++;
	} // if scalar
      } // if pointId
    } // for i
  return n;
}

void vtkPointClassification::BubbleSort(double* list, int len)
{
  cout<<"unsorted list:";
  for (int j=0; j < len; j++) cout<<" "<<list[j];
  cout<<endl;

  int i; double tmp; bool swapped = true;
  while (swapped)
    {
    swapped = false;
    for (i = 0; i < len-2; i++)
      {
      if (list[i] > list[i+1])
        { // swap the values.
        tmp = list[i+1];
	list[i+1] = list[i];
	list[i] = tmp;
	swapped = true;
        } // if
      } // for i
    } // while

  cout<<"sorted list:";
  for (int j=0; j < len; j++) cout<<" "<<list[j];
  cout<<endl;

} // BubbleSort

} // namespace bmia
