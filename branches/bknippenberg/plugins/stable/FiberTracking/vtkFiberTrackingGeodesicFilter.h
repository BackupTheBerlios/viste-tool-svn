/*
 * vtkFiberTrackingGeodesicFilter.h
 *
 * 2011-05-31	Evert van Aart
 * - First Version.
 *
 * 2011-06-08	Evert van Aart
 * - Improved progress reporting: Progress now also depends on the number of additional
 *   angles, not just on the number of seed points.
 *
 */


#ifndef bmia_FiberTrackingPlugin_vtkFiberTrackingGeodesicFilter_h
#define bmia_FiberTrackingPlugin_vtkFiberTrackingGeodesicFilter_h


/** Includes - Custom Files */

#include "vtkFiberTrackingFilter.h"
#include "geodesicPreProcessor.h"
#include "TensorMath/vtkTensorMath.h"
#include "HARDI/sphereTesselator.h"

/** Includes - Qt */

#include <QList>


namespace bmia {


/** Class Declarations */

class geodesicFiberTracker;


/** Subclass of the main fiber tracking filter used for geodesic fiber-tracking.
	Extends on the base class with a number of additional options, most of which
	are passed either to the tracker ("geodesicFiberTracker"), or to the class
	used to preprocess the tensor data ("geodesicPreProcessor"). Additionally,
	this filter can create multiple directions per seed point, using one of 
	several supported patterns. The in- and outputs of this filter are exactly
	the same as those of the base class.
*/

class vtkFiberTrackingGeodesicFilter : public vtkFiberTrackingFilter
{
	public:
		
		/** VTK Macro */

		vtkTypeMacro(vtkFiberTrackingGeodesicFilter, vtkFiberTrackingFilter);

		/** Constructor Call */

		static vtkFiberTrackingGeodesicFilter * New();

		/** Returns false if one of the stopping criteria is met. In this filter,
			most stopping criteria are optional; if all optional stopping criteria 
			(length, angle, and scalar value) are turned off, this function only
			returns false if "currentCellId" is -1 (i.e., fiber has left the volume).
			@param currentPoint		Current fiber point.
			@param testDot			Dot product, used to test the angle. 
			@param currentCellId	ID of the current cell, -1 if out of volume. */

		virtual bool continueTracking(bmia::streamlinePoint * currentPoint, double testDot, vtkIdType currentCellId);

		/** Enumeration for the pattern of additional shooting angles. */

		enum AdditionalAnglesPattern
		{
			AAP_Cone = 0,		/**< Cone around the main eigenvector. */
			AAP_SimpleSphere,	/**< Simple sphere, using spherical coordinates. */
			AAP_Icosahedron		/**< Tessellated icosahedron. */
		};

		/** ODE solver used to compute the fibers. */

		enum ODESolver
		{
			OS_Euler = 0,		/**< Euler's method. */
			OS_RK2_Heun,		/**< Second-order Runge-Kutta (Heun's Method). */
			OS_RK2_MidPoint,	/**< Second-order Runge-Kutta (Midpoint Method). */
			OS_RK4				/**< Fourth-order Runge-Kutta. */
		};

		/** Profile for the performance of this filter. */

		enum PerformanceProfile
		{
			PERF_NoProcomputation = 0,		/**< No precomputation. */
			PERF_PreProcessAll,				/**< Precompute preprocessed tensors. */
			PERF_PreProcessAndInvertAll		/**< Precompute PP'd and inverted tensors. */
		};

		/** Set the options related to the additional angles (i.e., multiple starting
			directions per seed point).
			@param rUseAA		Whether or not to use additional angles. 
			@param rPattern		Desired pattern for the additional angles. */

		void setAdditionalAnglesOptions(bool rUseAA, AdditionalAnglesPattern rPattern)
		{
			useAdditionAngles = rUseAA;
			aaPattern = rPattern;
		}

		/** Set the options for additional angles for the 'Cone around MEV' pattern.
			@param rNOA			Number of additional angles.
			@param rWidth		Desired width of the cone. */

		void setAAConeOptions(int rNOA, double rWidth)
		{
			aaConeNumberOfAngles = rNOA;
			aaConeWidth = rWidth;
		}

		/** Set the options for additional angles for the 'Simple sphere' pattern.
			@param NOAP			Number of angles for phi (angle with positive X-axis).
			@param NOAT			Number of angles for theta (angle with XY-plane). */

		void setAASphereOptions(int NOAP, int NOAT)
		{
			aaSphereNumberOfAnglesP = NOAP;
			aaSphereNumberOfAnglesT = NOAT;
		}

		/** Set the tessellation order for the 'Icosahedron' angles pattern.
			@param rOrder		Desired tessellation order. */

		void setAAIcoTessOrder(int rOrder)
		{
			aaIcoTessOrder = rOrder;
		}

		/** Set the options related to tensor preprocessing. 
			@param rEnable				Whether or not to use preprocessing.
			@param rSharpeningMethod	Desired method for sharpening the tensors.
			@param rGain				Tensor gain.
			@param rThreshold			Scalar threshold for sharpening the tensors.
			@param rExponent			Exponent used to sharpen the tensors. */

		void setPreProcessingOptions(bool rEnable, int rSharpeningMethod, int rGain, double rThreshold, int rExponent)
		{
			ppEnable		= rEnable;
			ppSharpenMethod	= (geodesicPreProcessor::SharpeningMethod) rSharpeningMethod;
			ppGain			= rGain;
			ppThreshold		= rThreshold;
			ppExponent		= rExponent;
		}

		/** Set the performance profile, which determines how much of the required
			tensor data should be precomputed.
			@param rPP			Desired performance profile. */

		void setPerformanceProfile(PerformanceProfile rPP)
		{
			performanceProfile = rPP;
		}

		/** Turn the 'Maximum fiber length' stopping criterion on or off.
			@param rStop		Use optional stopping criterion. */

		void setUseStopLength(bool rStop)	{	useStopLength = rStop;		}

		/** Turn the 'Maximum fiber angle' stopping criterion on or off.
			@param rStop		Use optional stopping criterion. */

		void setUseStopAngle( bool rStop)	{	useStopAngle  = rStop;		}

		/** Turn the 'Scalar threshold' stopping criterion on or off.
			@param rStop		Use optional stopping criterion. */

		void setUseStopScalar(bool rStop)	{	useStopScalar = rStop;		}

		/** Set the ODE solver used to compute the fibers.
			@param rSolver		Desired ODE solver. */

		void setODESolver(ODESolver rSolver)	{	myODESolver = rSolver;	}

	protected:

		/** Constructor */

		vtkFiberTrackingGeodesicFilter();

		/** Destructor */

		~vtkFiberTrackingGeodesicFilter();

		/** Initialize a fiber. Like the base class, this will add a point with
			the input coordinates to both fiber lists (for positive and negative
			fibers). In addition, this filter takes one direction from the "dirList"
			list. This direction is used as "dX" for the first point in the positive
			list, and this direction multiplied by -1 is used as "dX" for the first
			point in the negative list. 
			@param seedPoint	Coordinates of the current seed point. */

		virtual bool initializeFiber(double * seedPoint);

		/** Execute the filter. */

		virtual void Execute();

		/** Create the initial directions for the current seed point. All directions
			are added to the "dirList" list. If "useAdditionalAngles" is false, only
			the direction of the main eigenvector is added; otherwise, additional
			directions are generated based on the select additional angles pattern.
			@param p			Coordinates of the current seed point. */

		void generateFiberDirections(double * p);

		/** Whether or not to use additional shooting angles. */

		bool useAdditionAngles;

		/** Pattern for the additional angles. */

		AdditionalAnglesPattern aaPattern;

		/** Number of angles for the cone pattern. */

		int aaConeNumberOfAngles;

		/** Width of the cone. Should be between 0 and 1. */

		double aaConeWidth;

		/** Number of additional angles for the 'Simple sphere' pattern for phi.
			This pattern is created by varying phi (angle with the positive X-axis)
			from 0 to 2*pi in "aaSphereNumberOfAnglesP" steps, and theta (angle
			with the XY-plane) from 0 to pi/2 in "aaSphereNumberOfAnglesT" steps. */

		int aaSphereNumberOfAnglesP;

		/** Number of additional angles for the 'Simple sphere' pattern for theta.
			This pattern is created by varying phi (angle with the positive X-axis)
			from 0 to 2*pi in "aaSphereNumberOfAnglesP" steps, and theta (angle
			with the XY-plane) from 0 to pi/2 in "aaSphereNumberOfAnglesT" steps. */

		int aaSphereNumberOfAnglesT;

		/** Tessellation order for the icosahedron pattern. Should between 1 (12
			additional angles per seed point) and 6 (10242 angles). */

		int aaIcoTessOrder;

		/** Whether or not to use the fiber length stopping criterion. */

		bool useStopLength;

		/** Whether or not to use the fiber angle stopping criterion. */

		bool useStopAngle;

		/** Whether or not to use the scalar threshold stopping criterion. */

		bool useStopScalar;

		/** ODE solver used to construct the fibers. */

		ODESolver myODESolver;

		/** List of initial fiber directions for the current seed point. */

		QList<double *> dirList;

		/** Whether or not we should preprocess the tensors. */

		bool ppEnable;

		/** Method used to sharpen the tensors. */

		geodesicPreProcessor::SharpeningMethod ppSharpenMethod;

		/** Constant gain factor applied to all tensor elements. */

		int ppGain;

		/** Threshold for tensor sharpening. Tensors will only be sharpened if the
			scalar value at the seed point location is less than this threshold. */

		double ppThreshold;

		/** Exponent used to sharpen the tensors. */

		int ppExponent;

		/** Performance profile. Determines whether or not the preprocessed tensors
			and/or their inverse should be precomputed for the entire image. 
			Precomputing this data will potentially speed up the algorithm, since
			we will avoid redundant computations, but it also requires more memory,
			so it is best avoided for large images. */

		PerformanceProfile performanceProfile;

}; // class vtkFiberTrackingGeodesicFilter


} // namespace bmia


#endif // bmia_FiberTrackingPlugin_vtkFiberTrackingGeodesicFilter_h