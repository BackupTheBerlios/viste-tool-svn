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
 * vtk2DRoiToSeedFilter.h
 *
 * 2010-10-29	Evert van Aart
 * - First Version.
 *
 * 2010-12-15	Evert van Aart
 * - Added support for voxel seeding.
 *
 * 2011-03-16	Evert van Aart
 * - Removed the need to compute the normal for primary planes, making the seeding
 *   more robust for elongated ROIs.
 * - Increased stability for voxel seeding when a ROI is touching the edges of
 *   an image. 
 *
 */

 
#ifndef bmia_vtkRoiToSeedFilter_h
#define bmia_vtkRoiToSeedFilter_h


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - VTK */

#include <vtkPolyData.h>
#include <vtkPolygon.h>
#include <vtkMath.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSetToUnstructuredGridFilter.h>
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkDataObject.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>

/** Includes - Custom Files */

#include "SeedingPlugin.h"


namespace bmia {


/** This class generates a grid of seed points inside a two-dimensional
	Region-of-Interest (ROI). The ROI is defined as a closed polygon, located
	on some plane in 3D space. The input "vtkPolyData" may contain more than
	one ROI (stored as seperate lines), although one ROI per input is default.
	The plane containing the ROI can be of any orientation (i.e., oblique
	planes are fully supported). The filter inherits from the "vtkDataSetTo-
	UnstructuredGridFilter" class; its input is a "vtkPolyData" object containing
	the ROI(s), its output is a "vtkUnstructuredGrid" object containing the
	seed points. 
*/

class vtk2DRoiToSeedFilter : public vtkDataSetToUnstructuredGridFilter
{
	public:

		/** VTK Macro */

		vtkTypeMacro(vtk2DRoiToSeedFilter, vtkDataSetToUnstructuredGridFilter);

		/** Constructor Call */

		static vtk2DRoiToSeedFilter * New();

		/** Constructor */
	
		vtk2DRoiToSeedFilter();

		/** Destructor */
	
		~vtk2DRoiToSeedFilter();

		/** Set the distance between seed points, and pre-computed its square. 
			@param rDistance	New seed point distance. */
		
		void setSeedDistance(double rDistance)
		{
			if (rDistance != 0.0)
			{
				seedDistance  = rDistance;
				seedDistance2 = rDistance * rDistance;
			}
			else
			{
				seedDistance  = 1.0;
				seedDistance2 = 1.0;
			}
		}

		/** Set the image data object used for voxel seeding. 
			@param rVoxels	Voxel seeding image. */

		void setSeedVoxels(vtkImageData * rVoxels)
		{
			voxels = rVoxels;
		}

		/** Set the seeding method. Either "RST_Distance" for distance seeding
			(seed points are placed on a regular grid with spacing equal to 
			"seedDistance"), or "RST_Voxel" (seed points are placed on the voxels
			contained in the "voxels" image data. 
			@param rSM		Required seeding method. */
		
		void setSeedMethod(SeedingPlugin::roiSeedingType rSM)
		{
			seedMethod = rSM;
		}

	protected:

		/** Main point of entry of the filter. */

		void Execute();

	private:

		/** Compute the normal of the plane containing the ROI, using the
			points of the ROI.
			@param normal	Output normal. */
		
		bool getPlaneNormal(double * normal);

		/** Given inputs "V" and "W", which are two coefficients of the 
			plane equation of the plane containing the ROI, compute "incA" 
			and "incB", which together define a step of length "seedDistance"
			on the plane. Refer to the ".cxx" file for detailed comments.
			@param V		First plane coefficient.
			@param W		Second plane coefficient.
			@param incA		Output increment.
			@param incB		Output increment. */

		void computeIncrement(double V, double W, double * incA, double * incB);

		/** If the ROI is located on a plane parallel to one of the three 
			primary planes, we can use this function to generate seed points. */

		void generateSeedsOnPrimaryPlane();

		/** If the ROI is in an oblique plane, seed points are generated using this
			function. Using "computeIncrement", we ensure that all seed points are
			located a distance "seedDistance" from each other, regardless of the
			orientation of the plane. Refer to the ".cxx" file for detailed comments. */

		void generateSeedsOnObliquePlane();

		/** Get the bounds (minima and maxima in all three dimensions) of the
			current polygon. 
			@param numnerOfPoints	Number of points in the polygon.
			@param pointList		List of pointIds. */

		void getLineBounds(vtkIdType numberOfPoints, vtkIdType * pointList);

		/** Input polydata containing the ROIs. */

		vtkPolyData * input;

		/** Single ROI extracted from the input. */

		vtkPolygon * inputPolygon;

		/** Output seed points. */
	
		vtkUnstructuredGrid * output;

		/** Distance between seed points. */
	
		double seedDistance2;

		/** Squared seed point distance (pre-computed). */
	
		double seedDistance;

		/** Bounds of the current ROI. */
	
		double bounds[6];

		/** Enumeration of the three dimensions. Used in the "generate-
			SeedsOnObliquePlane" function, which uses the concept of 
			a primary, secondary and tertiary dimenion. Enumeration allows
			for more descriptive, less confusing labeling of dimensions. */
	
		enum dims
		{	
			DIM_X = 0,
			DIM_Y,
			DIM_Z
		};

		/** Seeding method (seed distance or voxel seeding). */

		SeedingPlugin::roiSeedingType seedMethod;

		/** Image data used for voxel seeding. */

		vtkImageData * voxels;

}; // class vtk2DRoiToSeedFilter


} // namespace bmia


#endif // bmia_vtkRoiToSeedFilter_h