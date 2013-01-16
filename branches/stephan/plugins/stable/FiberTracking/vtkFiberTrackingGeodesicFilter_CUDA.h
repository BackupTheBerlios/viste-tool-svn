/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
 * 2011-07-06	Evert van Aart
 * - First version for the CUDA-enabled version. Changed the way the fibers are 
 *   computed and added to the output.
 *
 */


#ifndef bmia_FiberTrackingPlugin_vtkFiberTrackingGeodesicFilter_h
#define bmia_FiberTrackingPlugin_vtkFiberTrackingGeodesicFilter_h


/** Includes - Custom Files */

#include "geodesicFiberTracker_CUDA.h"
#include "vtkFiberTrackingFilter.h"
#include "geodesicPreProcessor.h"
#include "TensorMath/vtkTensorMath.h"
#include "HARDI/sphereTesselator.h"

/** Includes - Qt */

#include <QList>


namespace bmia {


/** Class Declarations */

class geodesicFiberTracker;



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

		/** Information about a single seed point. Describes which fiber this seed
			point belongs to, and which direction it has. The direction is either
			positive or negative (1 or -1). Each fiber ID has two associated seed
			points, one for each direction. When writing the output, fibers with 
			the same ID are merged. */

		struct inputFiberInfo
		{
			int id;					/**< Fiber index. */
			int dir;				/**< Direction of the fiber. */
		};

		/** Information about output fibers. Contains a list of all point indices
			(in the correct order) of the points that make up the fibers. The indices
			refer to the points of the output. After calling the tracking kernel,
			all valid fiber points are added to the output point set, and their
			indices within this point set are stored in the list. When both directions
			of a fiber have terminated ("finishedDirs" is two), the ID list is added
			to the lines array of the output. */

		struct outputFiberInfo
		{
			QList<int> * idList;	/**< Point IDs of the fiber points. */
			int finishedDirs;		/**< Number of finished directions for this fiber (0, 1, or 2). */
		};

		/** Complete information for a seed point. Includes its position and initial
			direction, as well as its fiber index, and its direction. When a seed point
			is selected for computation on the GPU, its position and direction are
			uploaded to the GPU, while its ID and direction are added to an array
			of "inputFiberInfo" structs, to keep track of which seed points are being
			processed at which positions. */

		struct geodesicSeedPoint
		{
			float px;				/**< Seed point position (X). */
			float py;				/**< Seed point position (Y). */
			float pz;				/**< Seed point position (Z). */
			float dx;				/**< Seed point direction (X). */
			float dy;				/**< Seed point direction (Y). */
			float dz;				/**< Seed point direction (Z). */
			int dir;				/**< Fiber direction (1 or -1). */
			int id;					/**< Fiber index. */
		};

		/** Constructor */

		vtkFiberTrackingGeodesicFilter();

		/** Destructor */

		~vtkFiberTrackingGeodesicFilter();

		/** Execute the filter. */

		virtual void Execute();

		/** Create the initial directions for the current seed point. All directions
			are added to the "dirList" list. If "useAdditionalAngles" is false, only
			the direction of the main eigenvector is added; otherwise, additional
			directions are generated based on the select additional angles pattern.
			@param p			Coordinates of the current seed point. */

		void generateFiberDirections(double * p);

		/** Add all fiber points computed by the tracking kernel to the output.
			The IDs of the points in the output point set are then added to the 
			ID list in the fiber's "outputFiberInfo" struct. The list of IDs is
			not yet added to the output; this is because the fiber may still
			grow in size, and the "vtkPolyData" class does not really support
			dynamic resizing of fibers. Only when both directions of the fiber
			(positive and negative) have terminated, do we add the list of point
			IDs to the output polydata object. 
			@param info			Array of input fiber information (ID and direction).
			@param fiberPoints	Computed fiber points. */

		void addFibersToOutput(inputFiberInfo * info, float * fiberPoints);

		/** Add a single fiber to the output. This function is called when both
			directions of a fiber (positive and negative) have terminated. At
			that point, the ID list for this fiber will contain all point IDs
			of the fiber points (in order), which have already been added to the
			output point set. This function copies this list of IDs to VTK format,
			and adds it to the output. In addition, it checks if the length of the
			fiber exceeds the minimum fiber length. 
			@param idList		List of point IDs of the fiber points. */

		void addSingleFiberToOutput(QList<int> * idList);

		/** Checks if there are still fibers left to process. First checks if the
			list of seed points is non-empty; if it is empty, we check the array
			of active seed points, to see if there are still valid seed points.
			@param info			Array of fiber input information (ID and direction). */

		bool fibersLeft(inputFiberInfo * info);

		/** Fill up the array of active seed points (i.e., those seed points that
			will be sent to the GPU). Replaces invalid seed points (i.e., points
			with an ID of -1 in their input information struct) with a new point
			from the main seed list. This point is then removed from the list.
			@param info			Array of fiber input information (ID and direction). 
			@param seeds		Array of active seed points. */

		void fillSeedPointArray(inputFiberInfo * info, GC_fiberPoint * seeds);

		/** Preprocess the tensors. First copies the input DTI tensors to arrays
			that can be loaded onto the GPU (i.e., convert doubles to floats), and
			then calls the "GC_GPU_PreProcessing" function, which takes care of
			the actual preprocessing. Returns true on success, false otherwise. */

		bool preProcessTensors(GC_imageInfo grid);

		/** Track a load of fibers. This function is mainly used to setup the 
			input arguments for the "GC_GPU_Tracking", where the actual work is done.
			@param grid			Information about the DTI image (size, spacing).
			@param seeds		Array of active seed points. 
			@param outFibers	Array for the computed fiber points. */

		bool trackFibers(GC_imageInfo grid, GC_fiberPoint * seeds, float * outFibers);

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

		/** Whether or not we should preprocess the tensors. */

		bool ppEnable;

		/** Constant gain factor applied to all tensor elements. */

		int ppGain;

		/** Threshold for tensor sharpening. Tensors will only be sharpened if the
			scalar value at the seed point location is less than this threshold. */

		double ppThreshold;

		/** Exponent used to sharpen the tensors. */

		int ppExponent;
	
		/** Sharpening method used to sharpen the tensors. */

		geodesicPreProcessor::SharpeningMethod ppSharpenMethod;

		/** Maximum number of steps per fiber per tracking kernel call. Default is 1024. */

		int maxNumberOfSteps;

		/** Number of fibers computed per 'load' (i.e., per call to the tracking
			kernel. Default is 4096. */

		int numberOfFibersPerLoad;

		/** Maps input fiber IDs to output information structs. Fiber IDs are 
			assigned sequentially when generating the seed points, but they may
			not be sent to the output in the same order. When a fiber is added 
			to the output, we first check if its input fiber ID has been mapped
			to an output information struct (because of previous tracking kernel
			calls, and/or because a fiber with the same fiber ID but a different
			direction (e.g., positive instead of negative) has already been added).
			If this is the case, we continue building the existing output fiber;
			otherwise, we create a new output struct, and add it to the map. */

		QMap<int, outputFiberInfo> fiberIdMap;

		/** Array containing the input DTI tensors, cast to floats. Tensor elements
			are grouped into "float4" structs, since we need 4-tuples in order to
			use textures. This array contains the first four tensor elements. */

		float4 * inTensorsA;

		/** Array containing the input DTI tensors, cast to floats. Tensor elements
			are grouped into "float4" structs, since we need 4-tuples in order to
			use textures. This array contains the last two tensor elements. 
			Optionally, it can also contain scalar values in the W-component,
			which can be used by the preprocessing kernel, to check against the
			sharpening threshold. This scalar value will be overwritten by the
			first call to the derivatives kernel, at which point we no longer
			need it. */

		float4 * inTensorsB;

		/** Array containing the final points and segments of fiber parts that 
			have been postprocessed by the angle kernel. In the next call to this
			kernel, these values can be used to check the angle between the last 
			segment of the previous part, and the first segment of the new part. */

		GC_fiberPoint * anglePrevPoints;

		/** Array used to contain fiber lengths. Used by the distance kernel to 
			store the current length of a fiber; during the next call to this kernel,
			the values in this array are used as the initial distance values. */

		float * distanceArray;

		/** Array containing scalars (cast to floats) from the input scalar image.
			Used by the scalar kernel to check against the scalar thresholds. */

		float * scalarArray;

		/** List of all remaining seed points. Often, the number of seed points
			will be larger than the maximum load size (i.e., the maximum number of
			fiber processed at once), so we store seed point that cannot yet be
			processed in this list. When active seed points become invalid, because
			their fiber has terminated, they are replaced by new seed points from
			this list. */

		QList<geodesicSeedPoint> seedPointList;

		/** Global counter for the seed points. Used to assign sequential indices
			to the seed when generating them. */

		int seedPointCounter;

}; // class vtkFiberTrackingGeodesicFilter


} // namespace bmia


#endif // bmia_FiberTrackingPlugin_vtkFiberTrackingGeodesicFilter_h