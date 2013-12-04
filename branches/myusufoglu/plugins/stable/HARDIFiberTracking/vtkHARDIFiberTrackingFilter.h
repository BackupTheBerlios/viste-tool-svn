/*
 * vtkHARDIFiberTrackingFilter.h
 *
 * 2011-10-14	Anna Vilanova
 * - First version. 
 *
 * 2011-10-31 Bart van Knippenberg
 * - Added user-defined variables 
 * - Added semi-probabilistic functionality
 *
 *
 *  2013-03-15 Mehmet Yusufoglu, Bart Knippenberg
 * -Can process a discrete sphere data which already have Spherical Directions and Triangles arrays. 
 *  The Execute() function calls different CalculateFiber functions for different data. 
 *  ComputeGeometryFromDirections and readDirectionsFile functions can be used in the future. 
 *- HARDIFiberTrackingFilter has a data type parameter(sphericalHarmonics) anymore, 
 * parameter is either 1 or 0 depending on the data type read.
 */



/** ToDo List for "vtkHARDIFiberTrackingFilter"
	Last updated 20-09-2010 by Evert van Aart

	- Older versions supported copying of seed point scalar values. This is not yet
	  supported in the current version. However, if the scalar value is the same
	  for all seed points in a set (which I recall was the case), it might be better
	  to add this value as an attribute to the output dataset.
	- The filter stored the main eigenvector in the output, in the "Vectors" field.
	  This is currently disabled. Old comments suggest that it may have been related to 
	  "tubes" (i.e., different fiber visualisation methods). Once tubes have been added 
	  back in, it may be neccesary to re-enable storing the main eigenvectors.
	- Is it really necessary to have a "TOLERANCE" definition?
	- Class currently contains dialog boxes and a progress bar using Qt functions.
	  In the long run, all these user-noticifactions should be done by the core.
*/


#ifndef bmia_vtkHARDIFiberTrackingFilter_h
#define bmia_vtkHARDIFiberTrackingFilter_h


/** Class Declerations */

class vtkHARDIFiberTrackingFilter;


/** Includes - Main header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkDataSetToPolyDataFilter.h>
#include <vtkMath.h>
#include <vtkCell.h>
#include <vtkImageData.h>
#include <vtkDataObject.h>
#include <vtkObjectFactory.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>

/** Includes - Qt */

#include <QMessageBox>
#include <QProgressDialog>

#include "vtkXMLImageDataReader.h"
/** Includes - Custom Files */

#include "HARDIdeterministicTracker.h"


/** Includes - STL */

#include <vector>
#include <ctime>


namespace bmia {

/** Class declarations */

class HARDIstreamlinePoint;


/** Filter that executes the default fiber tracking process. The filter takes 
    two "vtkImageData" objects as input: One containing the HARDI tensors, and one
    containing one of the Anisotropy Index measure values. It also takes a set of
    seed points as a third input. From these seed points, it calculates fibers
    based on a number of user settings. These fibers are returned in the form of
    a "vtkPolyData" object. 
	
	The actual tracking is done by the "HARDIdeterministicTracker". Classes that want to
	use this tracker should inherit from the "vtkHARDIFiberTrackingFilter", and, if
	needed, should re-implement the "continueTracking" function of this class.
*/

class vtkHARDIFiberTrackingFilter : public vtkDataSetToPolyDataFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkHARDIFiberTrackingFilter, vtkDataSetToPolyDataFilter);

		/** Constructor Call */

		static vtkHARDIFiberTrackingFilter * New();

		/** "Set" macros for user processing variables. */

		vtkSetClampMacro(MaximumPropagationDistance,	float,		0.0,	VTK_LARGE_FLOAT);
		vtkSetClampMacro(MinimumFiberSize,				float,		0.0,	VTK_LARGE_FLOAT);
		vtkSetClampMacro(IntegrationStepLength,			float,		0.001,				1.0);
		vtkSetMacro(MinScalarThreshold,	float);
		vtkSetMacro(MaxScalarThreshold,	float);
		vtkSetMacro(Iterations,			unsigned int);
		vtkSetMacro(CleanMaxima,		bool);
		vtkSetMacro(TesselationOrder,	unsigned int);
		vtkSetMacro(Treshold,			float);
		vtkSetMacro(UseMaximaFile,			bool);
		vtkSetMacro(UseRKIntegration,bool);
		vtkSetMacro(InitialConditionType,int);
		vtkSetMacro(WriteMaximaToFile,			bool);
		vtkSetMacro(loopAngleSelectMaximaCombinationType,int);
		vtkSetMacro(loopAngleSingleCompareOrAverage,int);
			 
		/** "Get" macros for user processing variables. */

		vtkGetMacro(MaximumPropagationDistance,		float);
		vtkGetMacro(IntegrationStepLength,			float);
		vtkGetMacro(MinScalarThreshold,				float);
		vtkGetMacro(MaxScalarThreshold,				float);
		vtkGetMacro(StopDegrees,					float);
		vtkGetMacro(MinimumFiberSize,				float);
		vtkGetMacro(Iterations,						unsigned int);
		vtkGetMacro(CleanMaxima,					bool);
		vtkGetMacro(UseMaximaFile,			bool);
		vtkGetMacro(WriteMaximaToFile,			bool);
		vtkGetMacro(TesselationOrder,				unsigned int);
		vtkGetMacro(Treshold,						float);
		vtkGetMacro(UseRKIntegration,			bool);
	    vtkGetMacro(InitialConditionType,int);
		
         vtkGetMacro(loopAngleSelectMaximaCombinationType,int);
			 vtkGetMacro(loopAngleSingleCompareOrAverage,int);
		/** Sets the maximum angle between two consecutive line segments, and
			computes the threshold for the dot product ("StopDotProduct"). This
			last threshold is used to determine whether or not the "Maximum Angle"
			stopping condition holds. 
			@param StopDegrees	Maximum angle in degrees */

		void SetStopDegrees(float StopDegrees);

		void  readMaximaVectorsFile(vtkImageData * maximaVolume);
 
		 
		/** Set the current set of seed points. 
			@param seedPoint	Seed point set */

		void SetSeedPoints(vtkDataSet * seedPoints);

		/** Returns a pointer to the current set of seed points. */

		vtkDataSet * GetSeedPoints();

		/** Set the AI image. 
			@param AIImage		Anisotropy Index image */

	    void SetAnisotropyIndexImage(vtkImageData * AIImage);

		/** Returns a pointer to the AI image */

		vtkImageData * GetAnisotropyIndexImage();


		void SetMaximaDirectionsVolume(vtkImageData * img);
		 
		/** Returns a pointer to the AI image */

		vtkImageData * GetMaximaDirectionsVolume()
		{
			return maximaVolume;
		}

		/** Return "false" when one of the stopping conditions is met. Children
			of this class can re-implement this function with additional
			stopping criteria, should this be necessary.
			@param currentPoint		Data of current fiber point
			@param testDot			Dot product between current and previous point
			@param currentCellId	Id of the cell containing the current point */

		virtual bool continueTracking(bmia::HARDIstreamlinePoint * currentPoint, double testDot, vtkIdType currentCellId);

		/** function which only checks the angle
		returns true is the angle meets the user-defined settings
		@param testDot			Dot product between current and previous point*/
		virtual bool continueTrackingTESTDOT(double testDot);

		//made variable member of the class, can now be used by HARDIdeterministicTracker
		int shOrder;

		/** Store the name of the current ROI.
			@param rName			ROI Name */

		void setROIName (QString rName)
		{
			roiName = rName;
		};

		/*  Spherical harmonics or discrete sphere ? SH:1 , DS:0 */
		int sphericalHarmonics;
		float StopDotProduct;					/**< Threshold for dot product. */

	protected:

		/** Constructor */

		vtkHARDIFiberTrackingFilter();

		/** Destructor */

		~vtkHARDIFiberTrackingFilter();

		/** User processing variables, set by the user through the GUI. */

		float MaximumPropagationDistance;		/**< Maximum length of the fibers */
		float IntegrationStepLength;			/**< Length in voxels of an integration step */
		float MinScalarThreshold;				/**< Threshold values for scalar value */
		float MaxScalarThreshold;				/**< Threshold values for scalar value */
		float StopDegrees;						/**< Maximum angle between subsequent lines */
		float MinimumFiberSize;					/**< Minimum length of the fibers */
		unsigned int   Iterations;				/**< Number of iterations */
		unsigned int TesselationOrder;			/**< Order of tesselation for the maximum detection of the ODF*/
		bool CleanMaxima;						/**< Enable or disable maxima cleaning */
		bool UseMaximaFile;                       /**<Maximas initially saved as arraymax0 arraymax1 etc ..>*/
		bool UseRKIntegration;                    /**<Runge Kutta Integration  ..>*/
		bool WriteMaximaToFile;
		float Treshold;							/**< Set the ODF treshold for maxima detection */
	    int InitialConditionType;               /**<initial fiber orientation: 0:Interpolation of SH  1:Average of firstmax  2:Average of second Max.> */
	   int loopAngleSelectMaximaCombinationType;               /**<initial fiber orientation: 0:Interpolation of SH  1:Average of firstmax  2:Average of second Max.> */
	int loopAngleSingleCompareOrAverage;               /**<initial fiber orientation: 0:Interpolation of SH  1:Average of firstmax  2:Average of second Max.> */
	
		/** Derived processing variables, computed from the user variables. */

		

		/** Name of the current ROI. Used for the progress bar. */
		
		QString roiName;

		/** Point lists for positive and negative direction of integration. 
			Objects of class "streamlinePoint" (defined in "streamlineTracker.h"
			are added to these lists by the "streamlineTracker" class. */

		std::vector<HARDIstreamlinePoint> streamlinePointListPos;
		std::vector<HARDIstreamlinePoint> streamlinePointListNeg;

		/** Input image data for the DTI tensors and the scalar values of
			the Anisotropy Index images. */

		vtkImageData * HARDIimageData;
		vtkImageData * aiImageData;

		//maximas and directions of each point 
		vtkImageData *maximaVolume;

		/** Point data of the input images. */

		vtkPointData * HARDIPointData;
		vtkPointData * aiPointData;
		
		/** Low-level data arrays of the tensor- and AI-data images. Used
			to fetch tensors (through "GetTuple") and scalar values. */

		vtkDataArray * aiScalars;
		vtkDataArray * HARDIArray;

		/** Main entry point for the execution of the filter */

		virtual void Execute();

		/** Initializes a single fiber. In this version, this comes down to 
			storing the coordinates of the seed point in the point lists.
			@param seedPoint	3D Coordinates of the seed point */

		virtual bool initializeFiber(double * seedPoint);

		/** Allocated memory space for the output fibers. */

		void initializeBuildingFibers();

		/** Copies the points stored in the point lists to the "vtkPolyData" output. */

		virtual void BuildOutput();

	
		/** reads the directions, these directions will be used together with the radius data */
		void readDirectionsFile( vtkPoints *points, std::string filename);

		/** Directions are read from the directions file then triangulation and spherical coordinate angle calculations are done.
		 Fills trianglesarray and anglesarray. Trianglesarray has the result of triangulation now.
	     If Discrete Sphere data comes with its own directions and triangles, no need to use this function.
		 @param unitVectors Initial Directions or Points
		 @param anglesArray2 Calculated angles of the directions 
		 @param trianglesArray Calculated triangles array, ie. result of triangulation.
		  */
		void  computeGeometryFromDirections(double **unitVectors, std::vector<double*> &anglesArray2 ,vtkIntArray * trianglesArray) ;

	

}; // class vtkHARDIFiberTrackingFilter
 

} // namespace bmia


#endif // bmia_vtkHARDIFiberTrackingFilter_h


