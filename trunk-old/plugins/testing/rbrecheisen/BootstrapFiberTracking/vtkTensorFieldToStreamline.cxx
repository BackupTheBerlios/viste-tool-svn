#define _CRT_SECURE_NO_WARNINGS
/**
 * vtkTensorFieldToStreamline.cxx
 * by Anna Vilanova
 *
 * 2005-01-24	Anna Vilanova
 * - First version for the DTITool2 based on the class vtkStreamline of the DTITool
 *
 * 2005-05-30	Tim Peeters
 * - Add vtkErrorMacro in Execute() if no stop anisotropy index image was set.
 * - Add initialization of IntegrationStepLength in constructor.
 * - Add initialization of MaximumPropagationDistance in constructor.
 *
 * 2006-10-01	Tim Peeters
 * - Make streamlinePointArray.Direction an int instead of float.
 *   (it was always 1.0 or -1.0, and cast to int later which gave compiler warnings)
 *
 * 2007-04-01	Tim Peeters
 * - Add progress updates in Execute().
 */
#include "vtkDataObject.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkFloatArray.h"
#include "vtkPolyLine.h"
#include "vtkUnstructuredGrid.h"
#include "vtkTensorFieldToStreamline.h"
#include "vtkImageData.h"
#include "vtkPolygon.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkCellArray.h"
//#include "FluVtkProgressMeter.h"

#define TOLERANCE_DENOMINATOR 1000

#define VTK_START_FROM_POSITION 0
#define VTK_START_FROM_LOCATION 1

namespace bmia {

streamlinePoint::streamlinePoint()
{
  this->V[0] = this->V0;
  this->V[1] = this->V1;
  this->V[2] = this->V2;
}

streamlinePoint& streamlinePoint::operator=(const streamlinePoint& hp)
{
  int i, j;

  for (i=0; i<3; i++)
    {
    this->x[i] = hp.x[i];
    this->P[i] = hp.P[i];
    this->W[i] = hp.W[i];
    for (j=0; j<3; j++)
      {
      this->V[j][i] = hp.V[j][i];
      }
    }
  this->CellId = hp.CellId;
  this->SubId = hp.SubId;
  this->stopAI = hp.stopAI;
  this->D = hp.D;
  this->dotProduct=hp.dotProduct;
  return *this;
}

void streamlinePointArray::initialize()
{
	this->MaxId = -1;
	this->Array = NULL; 
	this->Size = 0;
	this->Extend = 500;	
}
streamlinePointArray::streamlinePointArray()
{
 	initialize();//extend with this amoiunt if more streamlinePoints are required
 }

streamlinePoint **streamlinePointArray::Resize(vtkIdType sz)
{//resizign streamlinePointarray
  streamlinePoint **newArray;
  vtkIdType newSize;

  if (sz >= this->Size)
    {
    newSize = this->Size +
      this->Extend*(((sz-this->Size)/this->Extend)+1);
    }
  else
    {
    newSize = sz;
    }

  newArray = (streamlinePoint **)malloc(sizeof(streamlinePoint*)*newSize);//number of streamlinePoint in an array
  streamlinePoint *points= new streamlinePoint[newSize];
  int i=0;
  for (i=0;i<sz;i++)
  {
	  points[i]= *(this->Array[i]);
	  newArray[i]=&(points[i]);
  }

  for (;i<newSize;i++)
  {
	  newArray[i]=&(points[i]);
  }

  this->Size = newSize;
  if (this->Array != NULL)
  {
	delete [] this->Array[0];
	free(this->Array);
  }
  this->Array = newArray;

  return this->Array;
}

/*
//------------------------------------------------------------------------------
vtkTensorFieldToStreamline* vtkTensorFieldToStreamline::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkTensorFieldToStreamline");
  if(ret)
    {
    return (vtkTensorFieldToStreamline*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTensorFieldToStreamline;
}
*/

vtkStandardNewMacro(vtkTensorFieldToStreamline);


//all this stuff is userdefined....
vtkTensorFieldToStreamline::vtkTensorFieldToStreamline()
{

  this->Streamers = NULL;
  this->cellScalarsStopAI = NULL;

  // default: do not add ROI ID's as scalars to output
  this->copySeedPointScalars = false;

  this->IntegrationStepLength = 1.0;
  this->MaximumPropagationDistance = 100.0;

}

vtkTensorFieldToStreamline::~vtkTensorFieldToStreamline()
{
//	cout<<"~vtkTensorFieldToStreamline() started."<<endl;
  if ( this->Streamers )
    {
    delete [] this->Streamers; this->Streamers = NULL;
    }
//  cout<<"~vtkTensorFieldToStreamline() ended."<<endl;
}


void vtkTensorFieldToStreamline::SetSource(vtkDataSet *source)
{
  this->vtkProcessObject::SetNthInput(2, source);
}

vtkDataSet *vtkTensorFieldToStreamline::GetSource()
{
  if (this->NumberOfInputs < 3)
    {
    return NULL;
    }
  return (vtkDataSet *)(this->Inputs[2]);
}

void vtkTensorFieldToStreamline::SetStopAnisotropyIndexImage(vtkImageData *stopAIImage)
{
  this->vtkProcessObject::SetNthInput(1, stopAIImage);
}

vtkImageData *vtkTensorFieldToStreamline::GetStopAnisotropyIndexImage()
{
	if (this->NumberOfInputs < 2)
    {
		return NULL;
    }
  return (vtkImageData *)(this->Inputs[1]);
}

/*ToDo Add progressMeter
void vtkTensorFieldToStreamline::AddProgressMeter(char *text)
{
	vtkProgressCallback *progressCallback = vtkProgressCallback::New();
		
	// add the callback as an observer to the filter
	this->AddObserver(vtkCommand::StartEvent,progressCallback);
	this->AddObserver(vtkCommand::ProgressEvent,progressCallback);
	this->AddObserver(vtkCommand::EndEvent,progressCallback);
	this->SetProgressText(text);

	progressCallback->Delete();
}*/

// Make sure coordinate systems are consistent, it returns the dotproduct between the two vectors
float vtkTensorFieldToStreamline::FixVectors(float **prev, float **current)
{
  float p0[3], p1[3], p2[3];	//previous vector
  float v0[3], v1[3], v2[3];	//current vector
  float temp[3];
  float dotProduct=0;
  int i;

  for (i=0; i<3; i++)
    {
    v0[i] = current[i][IV];
    v1[i] = current[i][IX];
    v2[i] = current[i][IY];
    }

  if ( prev == NULL ) //make sure coord system is right handed
    {
    vtkMath::Cross(v0,v1,temp);
    if ( vtkMath::Dot(v2,temp) < 0.0 )
      {
      for (i=0; i<3; i++)
        {
        current[i][IY] *= -1.0;
        }
      }
    }

  else //make sure vectors consistent from one point to the next
    {
    for (i=0; i<3; i++)
      {
      p0[i] = prev[i][IV];
      p1[i] = prev[i][IX];
      p2[i] = prev[i][IY];
      }
    if ( vtkMath::Dot(p0,v0) < 0.0 )
      {
      for (i=0; i<3; i++)
        {
        current[i][IV] *= -1.0;
		v0[i]*=-1.0;				//change first eigenvector
        }
      }
    if ( vtkMath::Dot(p1,v1) < 0.0 )
      {
      for (i=0; i<3; i++)
        {
        current[i][IX] *= -1.0;
        }
      }
    if ( vtkMath::Dot(p2,v2) < 0.0 )
      {
      for (i=0; i<3; i++)
        {
        current[i][IY] *= -1.0;
        }
      }

//this is the dotproduct between this and the previous vector (main ev)
	dotProduct = vtkMath::Dot(p0,v0);
    }
	return (dotProduct);
}

void vtkTensorFieldToStreamline::initializeStreamLine(float* seedPoint)
{
	// We assume that the streamers 0 and 1 contain the streamline in + and negative direction.
	// WARNING: We do not check that seedPoint is inside the dataset we assume it is.

	streamlinePoint *sPtr;
	streamlinePoint *sPtrb;

	double w[8];
	vtkCell *cell;
	double xNext[3];

	float *interpolatedTensor[3];
	float m0[3], m1[3], m2[3];
	interpolatedTensor[0] = m0; interpolatedTensor[1] = m1; interpolatedTensor[2] = m2;


	this->Streamers[0].Reset();
	this->Streamers[1].Reset();

	//  insert first point for each streamer
	this->Streamers[0].insertNextStreamlinePoint();
	sPtr = this->Streamers[0].getStreamlinePoint(0);

	sPtr->SubId = 0;

	sPtr->x[0]=seedPoint[0];//start in the voxel itself we assume it is inside the volume no check is done
	sPtr->x[1]=seedPoint[1];
	sPtr->x[2]=seedPoint[2];
	//ToDo: The tolerance parameter should be calculated in advance it is a bit of a nonsense to make the division each time.
	sPtr->CellId = tensorImageData->FindCell(sPtr->x, NULL, -1, tensorImageData->GetLength()/TOLERANCE_DENOMINATOR,sPtr->SubId, sPtr->P, w);
	sPtr->D = 0.0;

	if (sPtr->CellId != -1) // The seed point is inside the volume
	{

		cell = tensorImageData->GetCell(sPtr->CellId);

		//w is weigtfactor for each point within the cell, P=parametric coordinates within cell
		//subId=cell subID
		cell->EvaluateLocation(sPtr->SubId, sPtr->P, xNext, w);
		inTensors->GetTuples(cell->PointIds, cellTensors);

		// interpolate to get the tensor and anisotropy indices of the position
		interpolateTensor(interpolatedTensor,w);
		stopAIArray->GetTuples(cell->PointIds, cellScalarsStopAI);
		interpolateScalars(sPtr,w);

		//compute eigenvalues W and eigenvectors (both sorted) on current position
		vtkMath::Jacobi(interpolatedTensor, sPtr->W, sPtr->V);
		//fix the sign of the vectors by comparing to previous ones
		FixVectors(NULL, sPtr->V);

		//the second point is the same as the first, exept direction
		//insert first point for backward direction. It should be the same than in direction 0
		this->Streamers[1].insertNextStreamlinePoint();
		sPtrb = this->Streamers[1].getStreamlinePoint(0);
		*sPtrb = *sPtr;

		this->Streamers[0].Direction = 1;
		this->Streamers[1].Direction = -1;
	}
	else
	{
		this->Streamers[0].Reset();
	}
}

void vtkTensorFieldToStreamline::calculateStreamLine(int ptId)
{

	streamlinePoint *sPtr;
	streamlinePoint *sNext;
	vtkCell *cell;
	double	xNext[3];
	int dir = 1;
	float testDotProduct;
	double closestPoint[3];
	double dist2;
	float d;
	double w[8];

	float *interpolatedTensor[3];
	float m0[3], m1[3], m2[3];
	interpolatedTensor[0] = m0; interpolatedTensor[1] = m1; interpolatedTensor[2] = m2;
	dir = this->Streamers[ptId].Direction;
	if (this->Streamers[ptId].GetNumberOfPoints() >0) // If there is a seed point
	{
		//get starting step
		sPtr = this->Streamers[ptId].getStreamlinePoint(0);
		testDotProduct=1;  //set testDotProduct to 1 again to not let it stop the first time

		cell = tensorImageData->GetCell(sPtr->CellId);
		cell->EvaluateLocation(sPtr->SubId, sPtr->P, xNext, w);
		step = this->IntegrationStepLength * sqrt((double)cell->GetLength2());

		inTensors->GetTuples(cell->PointIds, cellTensors);
		stopAIArray->GetTuples(cell->PointIds, cellScalarsStopAI);

		while (continueTracking(sPtr,testDotProduct)) 
		{

			RungeKutta2nd (sPtr,xNext,dir,testDotProduct, cell,w);

			sNext = this->Streamers[ptId].insertNextStreamlinePoint();		//stop berekend punt in de hyperarray
			//if count is bigger than 500 (default max number of points in array)
			//it resizes, after this sPtr is empty
			sPtr=this->Streamers[ptId].getStreamlinePoint(this->Streamers[ptId].MaxId-1);

			if ( cell->EvaluatePosition(xNext, closestPoint, sNext->SubId, sNext->P, dist2, w) )
			{ //integration still in cell

				sNext->x[0] = closestPoint[0];sNext->x[1] = closestPoint[1];sNext->x[2] = closestPoint[2];
				sNext->CellId = sPtr->CellId;
				sNext->SubId = sPtr->SubId;
			}
			else
			{ //integration has passed out of cell
				sNext->CellId = tensorImageData->FindCell(xNext, cell, sPtr->CellId, tensorImageData->GetLength()/TOLERANCE_DENOMINATOR,sNext->SubId, sNext->P, w);
				if ( sNext->CellId >= 0 ) //make sure not out of dataset
				{
					sNext->x[0] = xNext[0];sNext->x[1] = xNext[1];sNext->x[2] = xNext[2];
					cell = tensorImageData->GetCell(sNext->CellId);
					inTensors->GetTuples(cell->PointIds, cellTensors);
					stopAIArray->GetTuples(cell->PointIds, cellScalarsStopAI);
				}
			}

			if ( sNext->CellId >= 0 )
			{
				cell->EvaluateLocation(sNext->SubId, sNext->P, xNext, w);

				interpolateTensor(interpolatedTensor,w);

				vtkMath::Jacobi(interpolatedTensor, sNext->W, sNext->V);
				FixVectors(sPtr->V, sNext->V);

				interpolateScalars(sNext,w);

				d = sqrt((double)vtkMath::Distance2BetweenPoints(sPtr->x,sNext->x));
				sNext->D = sPtr->D + d;
			}
			sPtr = sNext;
		}
		if (sPtr->CellId < 0)
		{
			this->Streamers[ptId].MaxId--;
		}
	}

}

void vtkTensorFieldToStreamline::initializeBuildingFibers()
{

  vtkDataSet *source;
  source = this->GetSource();

  vtkPoints *newPts;
  vtkFloatArray *newVectors;
 
  vtkPointData *outPD;
  vtkPolyData *output = this->GetOutput();
  
  vtkCellArray *newFiberLine;	//array for the lines

  // for the ROI ID's 
  vtkDataArray *sourceScalars;
  vtkDataArray *scalars;


  //
  // Initialize
  //
  
  outPD = output->GetPointData();	

  //
  // Allocate and update
  //

   //Vectors are added to fibers of all shapes
	newVectors = vtkFloatArray::New();
	newVectors->SetNumberOfComponents(3);
	newVectors->Allocate(7500);

    outPD->SetVectors(newVectors);
    newVectors->Delete();

    newPts  = vtkPoints::New();
	newPts ->Allocate(2500);
	output->SetPoints(newPts);
	newPts->Delete();


   	newFiberLine = vtkCellArray::New();
	newFiberLine->Allocate(newFiberLine->EstimateSize(2,VTK_CELL_SIZE));
	output->SetLines(newFiberLine);
	newFiberLine->Delete();
  
	if (copySeedPointScalars) {
		sourceScalars = source->GetPointData()->GetScalars();
	
		if (sourceScalars != NULL) // if scalaras where generated this are also transfered
		{
			int scalarsNumberOfComponents = sourceScalars->GetNumberOfComponents();
			int scalarsDataType = sourceScalars->GetDataType();
			scalars = vtkDataArray::CreateDataArray(scalarsDataType);
			scalars->Allocate(75000);
			scalars->SetNumberOfComponents(scalarsNumberOfComponents);
			output->GetPointData()->SetScalars(scalars);
			scalars->Delete();
			seedPointScalar = new double[scalarsNumberOfComponents];
		}
	}

}

// Calculate the length of the current streamline 
float vtkTensorFieldToStreamline::CalculateLengthCurrentStreamline()
{
  float length = 0;
  streamlinePoint *sPtr1,*sPtr2;

  // Get the length of the first half
  int numIntPts=this->Streamers[0].GetNumberOfPoints();	
  if (numIntPts>1)
  {
    sPtr1 = this->Streamers[0].getStreamlinePoint(numIntPts-1);
    for (int i=numIntPts-2; i>=0; i--)
    {
		  sPtr2=this->Streamers[0].getStreamlinePoint(i);
      length += sqrt(vtkMath::Distance2BetweenPoints(sPtr1->x,sPtr2->x));
      sPtr1=sPtr2;
    }
  }

  // Get the length of the second half
  numIntPts=this->Streamers[1].GetNumberOfPoints();	
  if (numIntPts>1)
  {
    sPtr1 = this->Streamers[1].getStreamlinePoint(0);
    for (int i=1; i<numIntPts; i++)
    {
      sPtr2=this->Streamers[1].getStreamlinePoint(i);
      length += sqrt(vtkMath::Distance2BetweenPoints(sPtr1->x,sPtr2->x));
      sPtr1=sPtr2;
    }
  }

  return length;

}
void vtkTensorFieldToStreamline::Execute()
{
  this->SetProgressText("Initializing Fiber Tracking");
  this->UpdateProgress(0.01);
  vtkPolyData *output = this->GetOutput();

  // Get input data: Inputs[1]= seed points Inputs[2]= the vtkImageData with the Anisotropy Index used in the stop criteria
  tensorImageData = (vtkImageData *) (this->GetInput());			//input data (tensordata and scalar data)

  vtkImageData * stopAIImage;
  stopAIImage = GetStopAnisotropyIndexImage();

  // TODO: don't use a stop AI image if none was set.
  if (!stopAIImage)
    {
    vtkErrorMacro("No stop anisotropy index image specified! Hmm.. let's crash! On the next try, use SetStopAnisotropyIndexImage(vtkImageData*).");
    }

  this->stopAIArray=stopAIImage->GetPointData()->GetScalars();
	
  vtkDataSet *source;
  source = this->GetSource();
 
  vtkDebugMacro(<<"Generating Streamline(s)");
  if ( ! (inTensors=tensorImageData->GetPointData()->GetTensors()) )
    {
    vtkErrorMacro(<<"No tensor data defined!");
    return;
    }

  cellTensors = vtkDataArray::CreateDataArray(inTensors->GetDataType());
  cellScalarsStopAI = vtkDataArray::CreateDataArray(stopAIArray->GetDataType());

  if (inTensors)
    {
    int numComp = inTensors->GetNumberOfComponents();
    cellTensors->SetNumberOfComponents(numComp);
    cellTensors->SetNumberOfTuples(VTK_CELL_SIZE);
    }

  if (stopAIArray)
    {
    int numComp=stopAIArray->GetNumberOfComponents();
    cellScalarsStopAI->SetNumberOfComponents(numComp);
    cellScalarsStopAI->SetNumberOfTuples(VTK_CELL_SIZE);
    }

  int NumberOfStreamers=2;
  this->Streamers = new streamlinePointArray[NumberOfStreamers];	//create new extendedstreamlinePointArray's

  this->initializeBuildingFibers();
	
  /* ToDo they should be changed to double */
  float seedPoint[3];

  vtkDataArray *scalars;

  if(!source)
    {
    vtkErrorMacro(<<"No input source defined!");
    }
  else
    {
    // Propagate the id's of the ROI's?
    if (copySeedPointScalars)
      {
      scalars = source->GetPointData()->GetScalars();
      } // if

    this->SetProgressText("Tracking Fibers...");
    int progressSteps = source->GetNumberOfPoints() / 25;
    if (progressSteps < 2) progressSteps = 1;
    for (int ptId =0; ptId < source->GetNumberOfPoints();ptId++)
      {
      double auxPoint[3];
      source->GetPoint(ptId,auxPoint);
      seedPoint[0]=auxPoint[0];seedPoint[1]=auxPoint[1];seedPoint[2]=auxPoint[2];

      if (copySeedPointScalars && scalars) 
        {
        scalars->GetTuple(ptId,seedPointScalar); 
	    } // if

      // update progress
      if (0==(ptId%progressSteps))
        {
	    this->UpdateProgress((float)ptId/(float)source->GetNumberOfPoints());
        } // if

      this->initializeStreamLine(seedPoint);

      // For each part of the streamline, integrate in appropriate direction.
      this->calculateStreamLine(0);
      this->calculateStreamLine(1);

      int numPts0 = (this->Streamers[0].GetNumberOfPoints());
      int numPts1 = (this->Streamers[1].GetNumberOfPoints());
			
      float fiberLength = CalculateLengthCurrentStreamline();
			
      if((fiberLength> this->MinimumFiberSize)&& ((numPts0+numPts1)> 2) && extraBuildFiberCondition())
        {
        this->BuildOutput();
        }

      this->Streamers[0].Reset();
      this->Streamers[1].Reset();
      } // for
    } // else (source)

  //clean the mess
  delete [] this->Streamers;
  this->Streamers = NULL;
  cellTensors->Delete();
  //cellScalarsStopAI->Delete();
  output->Squeeze();

  this->UpdateProgress(1.0);
}

void vtkTensorFieldToStreamline::BuildOutput()
{
  streamlinePoint *sPtr;
  vtkPoints *newPts;
  vtkDataArray *newVectors;
  vtkIdType i;
  int j, id;
  vtkPointData *outPD;
  float v[3];
  
  vtkPolyData *output = this->GetOutput();
  
  vtkIdList *idFiberPoints;
  vtkCellArray *newFiberLine;	//array for the lines

  vtkDataArray *scalars;

  idFiberPoints = vtkIdList::New();
  idFiberPoints->Allocate(2500);

  //
  // Initialize
  //
  outPD = output->GetPointData();

  // get the scalars
  scalars = outPD->GetScalars();

  //Vectors are added to fibers of all shapes
  newVectors = outPD->GetVectors();
  newPts = output->GetPoints();
  newFiberLine = output->GetLines();

  //
  // Loop over all hyperstreamlines generating points
  //
  // First we deal with streamer 0 (we do it from end to middle)
  int numIntPts=this->Streamers[0].GetNumberOfPoints();	
    
//  float f = (this->regionID);

  for ( i=numIntPts-1; i >= 0; i-- )
  {
		sPtr=this->Streamers[0].getStreamlinePoint(i);
		id = newPts->InsertNextPoint(sPtr->x);
		idFiberPoints->InsertNextId(id);
		for (j=0; j<3; j++) //compute point in center of tube
		{
			v[j] = sPtr->V[j][IV];	//major ev
		}
		newVectors->InsertTuple(id,v);
    if (copySeedPointScalars) {
		  scalars->InsertTuple(id,seedPointScalar);
    }
   }
	


    numIntPts=this->Streamers[1].GetNumberOfPoints();	
    
     // 0 is the same than in the previous.
    for ( i=1; i < numIntPts ; i++)
    {
		sPtr=this->Streamers[1].getStreamlinePoint(i);
		id = newPts->InsertNextPoint(sPtr->x);
		idFiberPoints->InsertNextId(id);
		for (j=0; j<3; j++) //compute point in center of tube
		{
			v[j] = sPtr->V[j][IV];	//major ev
		}
		newVectors->InsertTuple(id,v);
    if (copySeedPointScalars) {
		  scalars->InsertTuple(id,seedPointScalar);
    }
    }

	if ( idFiberPoints->GetNumberOfIds() > 1 )
	{
		newFiberLine->InsertNextCell(idFiberPoints);
		idFiberPoints->Reset();
	}
   
    idFiberPoints->Delete();
}


void vtkTensorFieldToStreamline::saveStreamline(char *baseName)
{
	char filenameFBS[255];

	strcpy(filenameFBS,baseName);
	strcat(filenameFBS,"fbs");

	vtkPolyDataWriter *writer=vtkPolyDataWriter::New();
	writer->SetFileName(filenameFBS);
	writer->SetInput(this->GetOutput());
	writer->Write();

	writer->Delete();

}
void vtkTensorFieldToStreamline::loadStreamline(char *baseName)
{
	char filenameFBS[255];
	
	strcpy(filenameFBS,baseName);
	strcat(filenameFBS,"fbs");

	vtkPolyDataReader *Reader=vtkPolyDataReader::New();
	Reader->SetFileName(filenameFBS);
	Reader->Update();
	vtkPolyData *output = Reader->GetOutput();
	vtkPolyData *outputStreamline = this->GetOutput();
	outputStreamline->DeepCopy(output);
}

void vtkTensorFieldToStreamline::SetCopySeedPointScalars(bool flag)
{
  this->copySeedPointScalars = flag;
}
} // namespace bmia

