/*
 * vtkFiberTrackingFilter.h
 *
 * 2010-09-13	Evert van Aart
 * - First version. 
 *
 * 2010-09-15	Evert van Aart
 * - Fixed errors in the code that computes and checks the dot product.
 * - Removed "vtkMath::DegreesToRadians()", replaced by static value.
 * 
 * 2010-09-17	Evert van Aart
 * - Added message boxes for error scenarios.
 * - Added support for the new DTI tensor storage system.
 * - Fiber Tracking is now done in "streamlineTracker". The motivation
 *   for this is that other classes (like "vtkFiberTrackingWVSFilter")
 *   can reuse this class, with custom stopping criteria if needed.
 * - Added basic support for displaying filter progress.
 * - Replaced QLists by std::lists.
 *
 * 2010-09-20	Evert van Aart
 * - Added a progress bar.
 *
 * 2010-09-30	Evert van Aart
 * - Fixed a bug in the "fixVectors" function.
 * - Aligning consecutive line segments now works correctly for second-
 *   order Runge-Kutte solver.
 *
 * 2010-11-10	Evert van Aart
 * - Fixed a bug that caused infinite loops in Whole Volume Seeding.
 *
 * 2011-02-09	Evert van Aart
 * - Added support for maximum scalar threshold values.
 *
 */



/** ToDo List for "vtkFiberTrackingFilter"
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


#ifndef bmia_vtkFiberTrackingFilter_h
#define bmia_vtkFiberTrackingFilter_h


/** Class Declerations */

class vtkFiberTrackingFilter;


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

/** Includes - Custom Files */

#include "streamlineTracker.h"

/** Includes - STL */

#include <vector>


namespace bmia {

/** Class declarations */

class streamlinePoint;


/** Filter that executes the default fiber tracking process. The filter takes 
    two "vtkImageData" objects as input: One containing the DTI tensors, and one
    containing one of the Anisotropy Index measure values. It also takes a set of
    seed points as a third input. From these seed points, it calculates fibers
    based on a number of user settings. These fibers are returned in the form of
    a "vtkPolyData" object. This class is loosely based on the old "vtkTensorField-
    ToStreamline" class. 
	
	The actual tracking is done by the "streamlineTracker". Classes that want to
	use this tracker should inherit from the "vtkFiberTrackingFilter", and, if
	needed, should re-implement the "continueTracking" function of this class.
*/

class vtkFiberTrackingFilter : public vtkDataSetToPolyDataFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkFiberTrackingFilter, vtkDataSetToPolyDataFilter);

		/** Constructor Call */

		static vtkFiberTrackingFilter * New();

		/** "Set" macros for user processing variables. */

		vtkSetClampMacro(MaximumPropagationDistance,	float,		0.0,	VTK_LARGE_FLOAT);
		vtkSetClampMacro(MinimumFiberSize,				float,		0.0,	VTK_LARGE_FLOAT);
		vtkSetClampMacro(IntegrationStepLength,			float,		0.001,				1.0);
		vtkSetMacro(MinScalarThreshold,	float);
		vtkSetMacro(MaxScalarThreshold,	float);

		/** "Get" macros for user processing variables. */

		vtkGetMacro(MaximumPropagationDistance,		float);
		vtkGetMacro(IntegrationStepLength,			float);
		vtkGetMacro(MinScalarThreshold,				float);
		vtkGetMacro(MaxScalarThreshold,				float);
		vtkGetMacro(StopDegrees,					float);
		vtkGetMacro(MinimumFiberSize,				float);

		/** Sets the maximum angle between two consecutive line segments, and
			computes the threshold for the dot product ("StopDotProduct"). This
			last threshold is used to determine whether or not the "Maximum Angle"
			stopping condition holds. 
			@param StopDegrees	Maximum angle in degrees */

		void SetStopDegrees(float StopDegrees);

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

		/** Return "false" when one of the stopping conditions is met. Children
			of this class can re-implement this function with additional
			stopping criteria, should this be necessary.
			@param currentPoint		Data of current fiber point
			@param testDot			Dot product between current and previous point
			@param currentCellId	Id of the cell containing the current point */

		virtual bool continueTracking(bmia::streamlinePoint * currentPoint, double testDot, vtkIdType currentCellId);

		/** Store the name of the current ROI.
			@param rName			ROI Name */

		void setROIName (QString rName)
		{
			roiName = rName;
		};


	protected:

		/** Constructor */

		vtkFiberTrackingFilter();

		/** Destructor */

		~vtkFiberTrackingFilter();

		/** User processing variables, set by the user through the GUI. */

		float MaximumPropagationDistance;		/**< Maximum length of the fibers */
		float IntegrationStepLength;			/**< Length in voxels of an integration step */
		float MinScalarThreshold;				/**< Threshold values for scalar value */
		float MaxScalarThreshold;				/**< Threshold values for scalar value */
		float StopDegrees;						/**< Maximum angle between subsequent lines */
		float MinimumFiberSize;					/**< Minimum length of the fibers */

		/** Derived processing variables, computed from the user variables. */

		float StopDotProduct;					/**< Threshold for dot product. */

		/** Name of the current ROI. Used for the progress bar. */
		
		QString roiName;

		/** Point lists for positive and negative direction of integration. 
			Objects of class "streamlinePoint" (defined in "streamlineTracker.h"
			are added to these lists by the "streamlineTracker" class. */

		std::vector<streamlinePoint> streamlinePointListPos;
		std::vector<streamlinePoint> streamlinePointListNeg;

		/** Input image data for the DTI tensors and the scalar values of
			the Anisotropy Index images. */

		vtkImageData * dtiImageData;
		vtkImageData * aiImageData;

		/** Point data of the input images. */

		vtkPointData * dtiPointData;
		vtkPointData * aiPointData;
		
		/** Low-level data arrays of the tensor- and AI-data images. Used
			to fetch tensors (through "GetTuple") and scalar values. */

		vtkDataArray * aiScalars;
		vtkDataArray * dtiTensors;

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

}; // class vtkFiberTrackingFilter
 

} // namespace bmia


#endif // bmia_vtkFiberTrackingFilter_h


