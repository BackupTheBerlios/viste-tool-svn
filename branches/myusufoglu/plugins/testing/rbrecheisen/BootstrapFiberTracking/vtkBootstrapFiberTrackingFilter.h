#ifndef __vtkBootstrapFiberTrackingFilter_h
#define __vtkBootstrapFiberTrackingFilter_h

// Includes VTK
#include <vtkObject.h>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPolyData.h>

// Includes C++
#include <vector>
#include <string>

class vtkBootstrapFiberTrackingFilter : public vtkObject
{
public:

	/** Instantiates new object */
	static vtkBootstrapFiberTrackingFilter * New();

	/** Type revision macro */
	vtkTypeRevisionMacro( vtkBootstrapFiberTrackingFilter, vtkObject );

	/** Sets maximum distance a fiber is allowed to propagate
		@param distance The maximum propagation distance */
	void SetMaximumPropagationDistance( float distance );

	/** Sets integration step length
		@param stepLength The step length to use for integration */
	void SetIntegrationStepLength( float stepLength );

	/** Sets step length to use for simplifying the streamlines after
		they have been tracked. This will significantly reduce the
		number of vertices
		@param stepLength The step length */
	void SetSimplificationStepLength( float stepLength );

	/** Sets minimum length of a fiber. This is done to prevent large
		quantities of very short fibers
		@param fiberSize The minimum size of the fiber */
	void SetMinimumFiberSize( float fiberSize );

	/** Sets the anistropy threshold to be used for terminating
		the fiber tracking process
		@param aiValue The threshold value */
	void SetStopAIValue( float aiValue );

	/** Sets the curvature fibers are allowed to have. Curvature
		beyond this value will result in fiber termination
		@param degrees The number of degrees */
	void SetStopDegrees( float degrees );

	/** Sets the seed points to be used. Each seed point will
		result in N streamlines, depending on how many bootstrap
		tensor volumes are defined
		@param seedPoints The dataset containing the seed points */
	void SetSeedPoints( vtkDataSet * seedPoints );

	/** Sets the anisotropy volume to be used. This volume contains
		for each voxel the anisotropy value (based on the measure
		selected by the user)
		@param image The image data containing the value */
	void SetAnisotropyIndexImage( vtkImageData * anisotropy );

	/** Sets the number of iterations the bootstrapping procedure
		should take. In practice, there may be 100 tensor volumes
		for bootstrapping, however the user may wish to process only
		10 volumes to save time or to experiment
		@param numberOfIterations The number of iterations to use */
	void SetNumberOfBootstrapIterations( int numberOfIterations );

	/** Sets the filenames for each bootstrap tensor volume
		@param fileNames The vector of string containing the names */
	void SetFileNames( std::vector< std::string > & fileNames );

	/** Updates the processing pipeline of the internal fiber tracking
		object. This will cause the streamlines to be tracked */
	void Update();

	/** Returns the collection of streamlines tracked by this tracker.
		This collection contains all streamlines tracked across all
		bootstrap tensor volumes. If necessary, also automatically
		calls \see Update()
		@return The bootstrap streamlines */
	vtkPolyData * GetOutput();

	/** Returns a list of fiber ID's. This list provides a mapping
		between a seed point and fibers that originate from that
		a seed point because these fibers all have the same ID in
		the list
		@return The list of fiber ID's */
	vtkIntArray * GetFiberIds();

protected:

	/** Constructor */
	vtkBootstrapFiberTrackingFilter();

	/** Destructor */
	virtual ~vtkBootstrapFiberTrackingFilter();

private:

	/** Updates the global list of fiber ID's. For each set of fibers,
		an index is added to the list of ID's, starting at 0 and
		increasing until N-1, if N is the number of fibers in the set.
		The final list will contain fiber ID's where all fibers
		originating from the same seed point have the same ID. It looks
		a bit like this:

		[0,...,N-1,0,...,N-1,0,...]

		@param fibers The current fibers */
	void UpdateFiberIds( vtkPolyData * fibers );

	/** Builds the output for this tracker by loading the streamlines
		written to disk and collecting them in a single set */
	void BuildOutput();

private:

	float MaximumPropagationDistance;		// Maximum propagation distance for each fiber
	float IntegrationStepLength;			// Integration step size
	float SimplificationStepLength;			// Step size for simplification of each fiber
	float MinimumFiberSize;					// Minimum fiber length
		float StopAIValue;					// Anisotropy threshold
	float StopDegrees;						// Curvature threshold

	vtkImageData	* Anisotropy;			// Volume with anistropy values
	vtkDataSet		* SeedPoints;			// Collection of seed points
	vtkPolyData		* Output;				// Fiber collection produced by this tracker
	vtkIntArray		* FiberIds;				// List of fiber ID's for all streamlines

	int NumberOfBootstrapIterations;		// Number of iterations

	std::vector< std::string > FileNames;	// Bootstrap file names to process
};

#endif
