/**
 * vtkTensorFieldToStreamline.h
 * by Anna Vilanova
 *
 * 2005-01-24	Anna Vilanova
 * - First version for the DTITool2 based on the class vtkStreamline of the DTITool
 *   Marked with ToDo are things that still need to happen
 *				
 * 2006-10-01	Tim Peeters
 * - Make streamlinePointArray.Direction an int instead of float.
 *   (it was always 1.0 or -1.0, and cast to int later which gave compiler warnings)
 */
#ifndef bmia_vtkTensorFieldToStreamline_h
#define bmia_vtkTensorFieldToStreamline_h

#include "vtkDataArray.h"
#include "vtkDataSetToPolyDataFilter.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMath.h"
#include "vtkPolyDataWriter.h"
#include "vtkPolyDataReader.h"
#include "vtkCell.h"
#include "vtkImageData.h"

#define IV 0
#define	IX 1
#define IY 2

/*ToDo another solution has to be found for the inline */
//#define INLINE __forceinline
#define INLINE inline


#define VTK_INTEGRATE_FORWARD 0
#define VTK_INTEGRATE_BACKWARD 1
#define VTK_INTEGRATE_BOTH_DIRECTIONS 2

namespace bmia {

/** Class that implements a point used for the vtkTensorFieldToStreamline for tracking of the fiber.
	@see vtkTensorFieldToStreamline
*/
class streamlinePoint { //a point of a fiber with its attributes....
public:
    streamlinePoint(); // method sets up storage
    /** Assignation operation */
	streamlinePoint &operator=(const streamlinePoint& hp); //for resizing

    /** position */
	double   x[3];    // position
    /** No comment */
	vtkIdType     CellId;  // cell
    /** No comment */
	int     SubId; // cell sub id
    /** No comment */
	double   P[3];    // parametric coords in cell
    /** No comment */
	float   W[3];    // eigenvalues (sorted in decreasing value)
    /** No comment */
	float   *V[3];   // pointers to eigenvectors (also sorted)
    /** No comment */
	float   V0[3];   // storage for eigenvectors
    /** No comment */
	float   V1[3];
    /** No comment */
	float   V2[3];
    /** No comment */
	float   stopAI;       // scalar value
    /** No comment */
	float   D;       // distance travelled so far
	/** No comment */
	float	dotProduct; // dotProduct main eigenvector with main eigenvector previous streamlinePoint
};

/** Class that implements a Streamline an array of a streamlinePoint. It is used during the tracking of the vtkTensorFieldToStreamline.
	@see vtkTensorFieldToStreamline
*/
class streamlinePointArray { //;prevent man page generation
public:
/** No comment */
	streamlinePoint **Array;  // pointer to data
	/** No comment */
	vtkIdType MaxId;             // maximum index inserted thus far
	/** No comment */
	vtkIdType Size;              // allocated size of data
	/** No comment */
	vtkIdType Extend;            // grow array by this amount
	/** No comment */
	int Direction;       // integration direction
	/** No comment */
	bool lineInPolygon;
	/** No comment */
	streamlinePointArray();
	/** No comment */
	virtual ~streamlinePointArray()
    	{
		if (this->Array)
		{
			if (this->Array[0])
				delete [] Array[0];
		free(this->Array);
		}
    	};
	/** No comment */
	virtual void initialize();
	/** No comment */
	vtkIdType GetNumberOfPoints() {return this->MaxId + 1;};
	/** No comment */
	streamlinePoint *getStreamlinePoint(vtkIdType i) {return this->Array[i];};
	/** No comment */
	INLINE virtual streamlinePoint *insertNextStreamlinePoint()
	{ 
	if ( ++this->MaxId >= this->Size )
		{
		Resize(this->MaxId);
		}
	return this->Array[this->MaxId];
	}
	/** No comment */
	virtual streamlinePoint **Resize(vtkIdType sz); //reallocates data
	/** No comment */
	void Reset() {this->MaxId = -1;};

};

/** 
  Class that implements the filter that given seed points and the tensor data returns the streamlines that follow the seed points. It is a son of the vtkDataSetToPolyDataFilter.
*/
class vtkTensorFieldToStreamline : public vtkDataSetToPolyDataFilter
{
public:
		/** No comment*/
		vtkTypeMacro(vtkTensorFieldToStreamline,vtkDataSetToPolyDataFilter);

		/** Construct new vtkTensorFieldToStreamline*/
		static vtkTensorFieldToStreamline *New();

		/**
		Set the maximum length of the streamline expressed as absolute
		distance (i.e., arc length) value.
		*/
		vtkSetClampMacro(MaximumPropagationDistance,float,0.0,VTK_LARGE_FLOAT);
		/**
		Get the maximum length of the streamline expressed as absolute
		distance (i.e., arc length) value.
		*/
		vtkGetMacro(MaximumPropagationDistance,float);

		/**
		Set the length of the segments composing the fibers (also called: fiberStep). 
		*/
		vtkSetClampMacro(IntegrationStepLength,float,0.001,1.0);
		/**
		Get the length of the segments composing the fibers (also called: fiberStep). 
		*/
		vtkGetMacro(IntegrationStepLength,float);

		/** Set stop criterium value for the anisotropy index*/
		vtkSetClampMacro(StopAIValue,float,0.0,1.0);
		/** Get stop criterium value for the anisotropy index*/
		vtkGetMacro(StopAIValue,float);

		/** Set stop criterium for maximum allowable curve in degrees/integrationStep; assuming that the voxelsize is given in mm */
		INLINE void SetStopDegrees(float StopDegrees);
		/** Get stop criterium for maximum allowable curve in degrees/integrationStep; assuming that the voxelsize is given in mm */
		vtkGetMacro(StopDegrees,float);

		/** Set minimum length of a fiber to be displayed.*/
		vtkSetClampMacro(MinimumFiberSize,float,0.0,VTK_LARGE_FLOAT);
		/** Get minimum length of a fiber to be displayed.*/
		vtkGetMacro(MinimumFiberSize,float);

		/**
		Set the seed points as a source object as vtkDataSet.
		*/ 
		void SetSource(vtkDataSet *source);
		/**
		Get the seed points as a source object as vtkDataSet.
		*/ 
		vtkDataSet *GetSource();

		/** Set the anisotropy index array */
	     void SetStopAnisotropyIndexImage(vtkImageData *stopAIImage);
		/** Get the anisotropy index array */
		vtkImageData  *GetStopAnisotropyIndexImage();


		/** ToDo Add a progress callback to this filter 
		void AddProgressMeter(char *text);*/ 

		/** Will the scalars of the seed points be added to the output point? Default: false */
		 void SetCopySeedPointScalars(bool flag);

		/** Save the result of the filter streamlines
			@param baseName used for the name of the file to be saved adding extention is ".fbs"*/
		virtual void saveStreamline(char *baseName);
		/** Load the result of the current streamlines 
			@param baseName used for the name of the file to be loaded adding extention is ".fbs"*/
		virtual void loadStreamline(char *baseName);

protected:
		/** Constructor */
		vtkTensorFieldToStreamline();
		/** Destructor */
		~vtkTensorFieldToStreamline();

		/** Executes the filter. Updates the output according to the input*/
		void Execute();

		/** Length of the streamline in absolute distance*/
		float MaximumPropagationDistance;
		/** The length (fraction of cross diameter cell) of integration steps*/
		float IntegrationStepLength;
		/** Integration step in mm*/
		float step;

		/** Stopcriterium for anisotropy index*/
		float StopAIValue;
		/** Stopcriterium for maximum allowable curve in degrees/integrationStep; assuming that the voxelsize is given in mm*/
		float StopDegrees;
		/** Used for the comparision
			convert degrees/voxellength to resulting dotprodukt for used stepsize */
		float StopDotProduct;

		/** Minimum length of a fiber to be displayed. This is in mm, assuming that the voxelsize is in mm*/
		float MinimumFiberSize;

		/** Scalar of the current seedpoint */
		double *seedPointScalar;

		/** This flag indicates if the scalars associated from the seed points are copied to the output */
		bool copySeedPointScalars;

		/** Streamers are the array of points that will contain the integrated points of the current streamline*/
		streamlinePointArray *Streamers;

		/** To copy the scalar field that it is given in the seedpoints **/


		/** Variable that contains the tensor ImageData to avoid the calling of GetInput */
		vtkImageData *tensorImageData;
		/** Variable contains the array of the anisotropy index image. We used to avoid getting the information from the input[2] vtkImageData each time */
		vtkDataArray *stopAIArray;


		/** ToDo find a more elegant way to interpolate using the vtkImageDataInterpolator. I am afraid that as it is implemented now the vtkImageDataInterpolator will be much slower.*/
		/** Linear interpolation of the tensor data. WARNING this should not be part of this but of the tensor data.
		@param interpolatedTensor result of the interpolation
		@param cellTensors cell to be interpolated
		@param w weights of the interpolation*/
		INLINE void interpolateTensor(float* interpolatedTensor[3], double* w);
		
		/** Linear interpolation of the scalars needed for the stop criterium 
			@param sPtr streamline Pointwhere the result should be left
			@param cellScalarsStopAI cell values with the stopAI values
			@param w weights of the interpolation*/
		virtual INLINE void interpolateScalars(streamlinePoint* sPtr, double* w);
		/** Initializes the streamers with the seedpoint for the tracing */

		/** ToDo the next variables might disapear if we use a more structured interpolation , see previous ToDo.*/
		/** Variable used during the calculations. We avoid passing them by parameter */
		vtkDataArray  *inTensors;
		/** Variable used during the calculations. We avoid passing them by parameter */
		vtkDataArray  *cellTensors;
		/** Variable used during the calculations. We avoid passing them by parameter */
		vtkDataArray  *cellScalarsStopAI;

		virtual void initializeStreamLine(float* seedPoint);
		/** Calculates one streamline. Results are left in the streamer. */
		void calculateStreamLine(int ptId);
		/** Initializes the vtkPolyData output for updating with the fibers */
		virtual void initializeBuildingFibers();
		/** Build the polyline (vtkPolyData) given streamlinePointArray*/
		virtual void BuildOutput();

		/** Condition for stoping tracking.
		@param sPtr current streamline point
		@param testDotProduct current angle dot product
		@result true contunue tracing and false stop tracing */
		virtual INLINE bool continueTracking(streamlinePoint *sPtr, float testDotProduct);

		/** Given a Tensor frame sets the sign of the eigenvectors such that the frame is 
			right handed and coherent with the previous. (Tensors eigenvectors have no meaningful sign) 
			@param prev previous frame
			@param current current frame*/
		static float FixVectors(float **prev, float **current);


		/** ToDo the integration scheme should not be inside this class. However it would probably slowdown the class quite a lot */ 
		/** Integration method used. It is RungeKutta 2nd order. 
			@param sPtr current point with tensor information
			@param xNext result of the next possition after integration
			@param dir direction of tracking (1 or -1)
			@param cell cell where the point is in
			@param w  weightd of the point for interpolation in/out*/
		INLINE void RungeKutta2nd (streamlinePoint *sPtr, double *xNext, int dir,float &testDotProduct, vtkCell *cell, double *w);
		
		/** Allows the children of this class to add a conditions to the building of the fiber as output */
		virtual INLINE bool extraBuildFiberCondition();

		
	    /** Get the length of the current streamline */
		float CalculateLengthCurrentStreamline();

		
private:
		/** No comment */
		vtkTensorFieldToStreamline(const vtkTensorFieldToStreamline&);  // Not implemented.
		/** No comment */
		void operator=(const vtkTensorFieldToStreamline&);  // Not implemented.

};


// Implementation of inline functions of the vtkTensorFieldToStreamline

INLINE bool vtkTensorFieldToStreamline::extraBuildFiberCondition()
{
	return true;
}



INLINE void vtkTensorFieldToStreamline::SetStopDegrees(float StopDegrees)
{
	if(this->StopDegrees != StopDegrees)
	{
	this->StopDegrees = StopDegrees;
	this->StopDotProduct=cos(vtkMath::RadiansFromDegrees(StopDegrees)); // cos(vtkMath::DegreesToRadians()*StopDegrees);
	this->Modified();
	}
}


INLINE void vtkTensorFieldToStreamline::interpolateTensor(float* interpolatedTensor[3],/*vtkDataArray* cellTensors,*/ double* w)
{
	// interpolate tensor, compute eigenfunctions

	int k;
	double *tensor;

	tensor=cellTensors->GetTuple9(0);
	interpolatedTensor[0][0] = tensor[0+3*0] * w[0];	
	interpolatedTensor[1][1] = tensor[1+3*1] * w[0];	
	interpolatedTensor[2][2] = tensor[2+3*2] * w[0]; 	
	interpolatedTensor[0][1] = tensor[0+3*1] * w[0];	
	interpolatedTensor[0][2] = tensor[0+3*2] * w[0];	
	interpolatedTensor[1][2] = tensor[1+3*2] * w[0];	
	interpolatedTensor[1][0] = interpolatedTensor[0][1];
	interpolatedTensor[2][0] = interpolatedTensor[0][2];	
	interpolatedTensor[2][1] = interpolatedTensor[1][2];


	for (k=1; k < 8; k++) //weeg alle punten binnen de cell met weegfactor w[k]
	{
		tensor =cellTensors->GetTuple(k);
		interpolatedTensor[0][0] += tensor[0+3*0] * w[k];	
		interpolatedTensor[1][1] += tensor[1+3*1] * w[k];	
		interpolatedTensor[2][2] += tensor[2+3*2] * w[k];	
		interpolatedTensor[0][1] += tensor[0+3*1] * w[k];	
		interpolatedTensor[0][2] += tensor[0+3*2] * w[k];	
		interpolatedTensor[1][2] += tensor[1+3*2] * w[k];	
		interpolatedTensor[1][0] += interpolatedTensor[0][1];
		interpolatedTensor[2][0] += interpolatedTensor[0][2];	
		interpolatedTensor[2][1] += interpolatedTensor[1][2];
	}

};

INLINE void vtkTensorFieldToStreamline::interpolateScalars(streamlinePoint* sPtr, double* w)
{

	sPtr->stopAI=0;
	int i;
	double *auxscalar;
	for ( i=0; i < 8; i++)
	{
		auxscalar=cellScalarsStopAI->GetTuple(i);
		sPtr->stopAI+= w[i]* (*auxscalar);				
		//input scalar data is also interpolated
	}
}
INLINE bool  vtkTensorFieldToStreamline::continueTracking(streamlinePoint *sPtr, float testDotProduct)
{
	bool bAngle = testDotProduct > StopDotProduct;
	return ((sPtr->CellId >= 0) && (sPtr->D < this->MaximumPropagationDistance) && 
			(sPtr->stopAI >StopAIValue) && bAngle);
}	

INLINE void vtkTensorFieldToStreamline::RungeKutta2nd (streamlinePoint *sPtr, double *xNext, int dir,float &testDotProduct, vtkCell *cell, double *w)
{
	float *interpolatedTensor[3];
	float m0[3], m1[3], m2[3];
	float *eigenVectors[3];
	float v0[3], v1[3], v2[3];
	float eigenValues[3];
	//float withoutScaling[3];

	eigenVectors[0] = v0; eigenVectors[1] = v1; eigenVectors[2] = v2;
	interpolatedTensor[0] = m0; interpolatedTensor[1] = m1; interpolatedTensor[2] = m2;

	double closestPoint[3];
	int subId;
	double p[3];
	double dist2;
	

	//compute updated position using this step (Euler integration)
	xNext[0] = sPtr->x[0] + dir * step * sPtr->V[0][IV];
	xNext[1] = sPtr->x[1] + dir * step * sPtr->V[1][IV];
	xNext[2] = sPtr->x[2] + dir * step * sPtr->V[2][IV];


	//compute updated position using updated step
	cell->EvaluatePosition(xNext, closestPoint, subId, p, dist2, w);

	//interpolate tensor
	interpolateTensor(interpolatedTensor,w);

	//compute eigenvalue/vectors form interpolated tensor
	vtkMath::Jacobi(interpolatedTensor, eigenValues, eigenVectors);

	//use this dotprodukt to test for maximum allowable angle between 2 vectors
	testDotProduct=  FixVectors(sPtr->V, eigenVectors);

	//now compute final position
	//compute next position, X is position in world coordinates
	xNext[0] = sPtr->x[0] + dir * (step/2.0) * (sPtr->V[0][IV] + eigenVectors[0][IV]);	
	xNext[1] = sPtr->x[1] + dir * (step/2.0) * (sPtr->V[1][IV] + eigenVectors[1][IV]);	
	xNext[2] = sPtr->x[2] + dir * (step/2.0) * (sPtr->V[2][IV] + eigenVectors[2][IV]);	
		
}

} // namespace bmia
#endif


