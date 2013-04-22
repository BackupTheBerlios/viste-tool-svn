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
 * CDistanceVolume.h
 *
 * 2010-09-21	Evert van Aart
 * - First version. 
 *
 * 2010-10-04	Evert van Aart
 * - Fixed an error that prevented the class from working correctly.
 *
 * 2010-11-10	Evert van Aart
 * - Fixed behaviour near borders. In older versions, a global 1D "index" was
 *   computed from the 3D element coordinates ("dsepCoord"), and we only checked
 *   if "index" was between zero and the total number of indices. If one of the
 *   3D coordinates exceeded the distance volume dimension, the 1D index could 
 *   still be within global range, and the functions would select a distance
 *   element in a completely different part of the volume.
 *
 * 2011-04-21	Evert van Aart
 * - "exactGoodDistance" now immediately returns true if the distance threshold is zero.
 *
 */


#ifndef bmia_CDistanceVolume_h
#define bmia_CDistanceVolume_h


/** Includes - STL */

#include <math.h>
#include <stdlib.h>


namespace bmia {


/**	Used by the "vtkFiberTrackingWVSFilter" class to keep track of the positions of 
	existing fibers. The main goal of the distance volume is to determine whether or
	not a new fiber point is within a certain minimum distance of existing fiber 
	points. This process is accelerated by the use of distance elements, which can be
	seen as small sub-volumes of the image. Given a point located in a certain element,
	we check either only the existing points in that element (fast, but possibly 
	inaccurate), or we check the existing points in the element plus those in the 26
	elements surrounding it. The number of elements in each direction is directly set
	by the user through the GUI, and influences both the average speed of the query 
	(smaller elements are generally faster), and the total memory requirements of
	this class (larger elements lead to less memory used). 
	
	This class was based directly on the "CDistanceVolume" class developed by Anna
	Vilanova for DTITool2, with some minor changes in naming, comments, and code. */

class CDistanceVolume
{
	public:

		/** Simple struct for 3D coordinates */
		typedef struct
		{ 
			float x;
			float y;
			float z; 	 
		} Point3f;

		/** List item containing a set of 3D coordinates, and a pointer
			to the next item in the list. Allows for low-memory storage of
			all points. */

		typedef struct pointListItem
		{
			Point3f p;
		
			pointListItem * Next;
		} pointListItem;

		/** Struct containing a list of points in a single sub-volume of
			the image. The size of the subvolume is determined by the
			"elementDistance" variable. */

		struct DistanceElement
		{
			pointListItem * listHead;

			int numberOfItems;
		};

		/** Constructor */
		CDistanceVolume();

		/** Destructor */
		~CDistanceVolume();

		/** Distance between elements. Equal to the "seedDistance" value
			used in "vtkFiberTrackingWVSFilter". */

		float elementDistance;

		/** Distance volume used to keep track of the fibers. */

		DistanceElement * distanceVolume;

		/** Dimensions of the distance volume. */

		int dsepDim[3];

		/** Size of the distance volume. */
		int dsepSize;

		/** Size of XY slice of the volume. */

		int XYdsep;

		/** Frees the space and empties the distance volume */

		/** Number of cells that contain at least one point */

		int filledCellsCount;

		void clearVolume();
	
		/** Initializes the distance volume grid, given a distance between elements 
			and the original size of the elements. 
			@param rElementDistance	Distance between distance volume elements.
			@param volumeSize		Size (in mm) of the DTI image. */

		void initializeVolume(float rElementDistance, double volumeSize[3]);

		/** Check whether the point is a good approximate distance. This is done 
			simply by checking whether or not the distance element closest to the 
			point is empty. 
			@param point			3D point coordinates. */
		
		bool goodDistance(double * point);

		/** Calculate the minimum distance of the point to the closest exsting fiber. 
			@param index			Index of distance element.
			@param minimumDistance2	Squared minimum distance.
			@param point			3D point coordinates. */

		float calculateMinDist(int index, float * point, float minimumDistance2);

		/** Calculate the minimum distance of the point to the closest exsting fiber. 
			Also returns the coordinates of the existing fiber point closest to "point".
			@param index			Index of distance element.
			@param minimumDistance2	Squared minimum distance.
			@param point			3D point coordinates. 
			@param closestPoint		Coordinates of closest fiber point. */

		float calculateMinDist(int index, float * point, float minimumDistance2, double * closestPoint);

		/** Check whether the point is a good distance from existing points. It is more 
			accurate than "goodDistance", since it not only looks at the points in the
			distance element containing "point", but also at the point of the 26 elements
			surrounding this central element. 
			@param point			3D point coordinates.
			@param minimumDistance2 Squared minimum distance. */

		bool exactGoodDistance(double * point, float minimumDistance2);

		/** Like previous function, but also return the coordinates of the existing fiber
			point closest to the coordinates of "point". 
			@param point			3D point coordinates.
			@param minimumDistance2 Squared minimum distance. 
			@param closestPoint		Point closest to "point" */

		bool exactGoodDistance(double * point, float minimumDistance2, double * closestPoint);

		/** Like previous functions, but also return the distance to the closest point.
			@param point			3D point coordinates.
			@param minimumDistance2 Squared minimum distance. 
			@param closestPoint		Point closest to "point"
			@param distance			Distance between "point" and "closestPoint" */
			
		bool exactGoodDistance(double * point, float minimumDistance2, double * closestPoint, double * distance);

		/** Add a single point to distance volume.
			@param point			Coordinates of the point. */

		void addPointToDistance(double * point);

		/** Returns the percentage of the cells that contain at least one point. */

		float getPercentageFilled();	

}; // class CDistanceVolume


} // namespace bmia


#endif  // bmia_CDistanceVolume_h
