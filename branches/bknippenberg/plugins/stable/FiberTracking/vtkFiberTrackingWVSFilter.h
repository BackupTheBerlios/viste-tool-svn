/*
 * vtkFiberTrackingWVSFilter.h
 *
 * 2010-09-20	Evert van Aart
 * - First version. 
 * 
 * 2011-04-26	Evert van Aart
 * - Improved progress reporting.
 * - Slight speed improvements.
 *
 * 2011-06-06	Evert van Aart
 * - Changed the criterion for adding computed fibers to the output; in previous
 *   versions, fibers of length less than the minimum fiber length criterion could
 *   still be added to the output, which was deemed undesirable behavior.
 *
 */


/** ToDo List for "vtkFiberTrackingWVSFilter"
	Last updated 26-04-2011 by Evert van Aart
	
	- Initialization of extra seed points was implemented in the old version,
	  but was never actually used. Check if it can be deleted.
	- In the old version, the variable "distance" computed by the "exactGood-
	  Distance" function used in "continueTracking" was saved in the "closedistance"
	  variable of the fiber point structure ("sPtr"). However, this value was never
	  actually used; it was originally saved in the "Scalars" field of the output,
	  but the corresponding lines had been commented out. Check if we need this
	  distance value in the output.
	- Is it really necessary to have a "TOLERANCE" definition?
*/


#ifndef bmia_vtkFiberTrackingWVSFilter_h
#define bmia_vtkFiberTrackingWVSFilter_h

/** Includes - Main header */

#include "DTITool.h"

/** Includes - VTK */

#include "vtkDataArray.h"
#include "vtkDataSetToPolyDataFilter.h"
#include "vtkUnstructuredGrid.h"
#include "vtkMath.h"
#include "vtkCell.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkPolyLine.h"
#include "vtkImageData.h"
#include "vtkPolygon.h"
#include "vtkCellData.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"

/** Includes - STL */

#include <list>
#include <vector>

/** Includes - Custom Files */

#include "vtkFiberTrackingFilter.h"
#include "CDistanceVolume.h"


namespace bmia {


/** Whole Volume Seeding filter for Fiber Tracking. It is loosely based on the 
	"TensorFieldToStreamlineAll" class of the old DTITool, with some improvements
	in the code and clearer names. It inherits from "vtkFiberTrackingFilter", and
	like this class, it uses the "streamlineTracker" class to perform the actual
	fiber tracking. The differences are in the way it generates seed points, and
	in the stopping condition for fiber tracking (see the "continueTracking"
	function), which is more elaborate than that of its parent class. */

class vtkFiberTrackingWVSFilter : public vtkFiberTrackingFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtkFiberTrackingWVSFilter, vtkDataSetToPolyDataFilter);
  
		/** Constructor Call */

		static vtkFiberTrackingWVSFilter * New();

		/** "Get" and "Set" macros */

		vtkSetMacro(SeedDistance, float);
		vtkGetMacro(SeedDistance, float);
		vtkGetMacro(MinDistancePercentage, float);
		vtkSetClampMacro(MinDistancePercentage, float, 0.0, 1.0);
	
		/** Simple struct for 3D coordinates */

		typedef struct
		{ 
			double x;
			double y;
			double z; 	 
		} Point3d;

		/** Define a list of "Point3d" elements */

		typedef std::list<Point3d> VSeedPoints;

		/** Set the list of extra seed points.
			@param extraSeedPoint	List containing seed points */

		void SetExtraSeedPoints (VSeedPoints * extraSeedPoints)
		{
			// Remove existing list
			if (extraSeedPoints)
			{
				extraSeedPoints->clear();
				delete extraSeedPoints;
			}

			// Store new list
			this->extraSeedPoints = extraSeedPoints;
		};

		/** Stopping condition for the tracker of class "streamlineTracker". 
			Re-implemented from the parent "vtkFiberTrackingFilter" class.
			@param currentPoint		Data of current fiber point
			@param testDot			Dot product between current and previous point
			@param currentCellId	Id of the cell containing the current point */

		virtual bool continueTracking(bmia::streamlinePoint * currentPoint, double testDot, vtkIdType currentCellId);

	protected:

		/** Helper class stores a set of 3D coordinates, along with the AI
			scalar value at that point. The redefined "less than" operator
			is needed in order to sort the list. */

		class initialPoint
		{
			public:
				double X[3];	// 3D coordinates
				double AI;		// Anisotropy Index value
		
				// Define "less than" operator
				bool operator < (const initialPoint point)
				{
					return (point.AI < this->AI);
				};	
		};
	
		/** Define a queue of 3D coordinates as a "queueOfPoints". */

		typedef std::list<Point3d> queueOfPoints;

		/** Constructor */

		vtkFiberTrackingWVSFilter();

		/** Destructor */

		~vtkFiberTrackingWVSFilter();
	
		/** List containing additional seed points. */

		VSeedPoints * extraSeedPoints;

		/** Distance volume to keep track of fiber distances. */

		CDistanceVolume distanceVolume;

		/** Process variables */

		float SeedDistance;				// Distance between seed points
		float MinDistancePercentage;	// User-defined percentage of the seed distance
		float minimumDistance;			// Squared product of seed distance and percentage
		float seedDistanceSquared;		// Squared seed distance

		/** False when a fiber is too close to an existing fiber. */

		bool bIsNotTooClose;

		/** Main entry point for the execution of the filter */

		void Execute();

		/** Create a seed point at every grid point of the input image, store tem
			all in the seed pint list, and sort this list in descending order of
			seed point Anistropy Index values.
			@param seedPointList	Output list for seed points */

		void createInitialPoints(std::list<initialPoint> * seedPointList);

		/** Add additional seed points to the list. Additional seed points must be
			defined using the "SetExtraSeedPoints" function. Extra seed points are
			given the AI value "1.0" to ensure maximum priority after sorting. 
			@param seedPointList	Output list for seed points */

		void initializeExtraSeedPoints(std::list<initialPoint> * seedPointList);		

		/** Add all points in the newly computed fiber to the distance volume */

		void addFiberToDistance();

		/** Use the points of the newly computed fiber to create new seed points. 
			@param pointQueue			Output queue for new points
			@param streamlinePointList	List of fiber points */

		void generateNewSeedPoints(queueOfPoints * pointQueue, 
									std::vector<streamlinePoint> * streamlinePointList);

		/** Add a new seed point to the queue.
			@param point			Coordinates of the point
			@param pointQueue		Queue containing new fiber points */

		void addNewSeedPoint(double * point, queueOfPoints * pointQueue);

}; // class vtkFiberTrackingWVSFilter


} // namespace bmia


#endif // bmia_vtkFiberTrackingWVSFilter_h



