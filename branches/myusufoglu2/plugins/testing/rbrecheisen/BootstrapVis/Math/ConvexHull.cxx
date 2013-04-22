#include "ConvexHull.h"

#include <limits>
#include "SeparatingAxis.h"

namespace math
{	
	ConvexHull::ConvexHull(OrientedBoundingBox boundingBox, const Vector3& viewDir, const Vector3& upDir)		
	{
		mCamAxes[2] = viewDir.normalizedCopy();
		mCamAxes[1] = upDir.normalizedCopy();
		mCamAxes[0] = mCamAxes[1].crossProduct(mCamAxes[2]).normalizedCopy();

		mCenter = boundingBox.getCenter();
		mCenter = Vector3(
			mCenter.scalarProjectOn(mCamAxes[0]),
			mCenter.scalarProjectOn(mCamAxes[1]),
			
			// Flatten the depth component so that we
			// work exclusively in a view aligned plane.
			0
		);

		std::vector<Vector3> worldCorners = boundingBox.getCornerPoints();
		std::vector<Vector3> viewCorners;
		viewCorners.reserve(worldCorners.size());

		std::vector<Vector3>::const_iterator itWorld = worldCorners.begin();
		for(; itWorld != worldCorners.end(); ++ itWorld)
		{
			Vector3 viewCorner(
				(*itWorld).scalarProjectOn(mCamAxes[0]),
				(*itWorld).scalarProjectOn(mCamAxes[1]),
				
				// Flatten the depth component so that we
				// work exclusively in a view aligned plane.
				0 
			);

			viewCorners.push_back(viewCorner);
		}


		// Gets the point with smallest X as the starting point of our convex hull.
		Vector3 root = *(std::min_element(viewCorners.begin(), viewCorners.end(), Vector3Comparer()));
		
		// Sorts according to ascending angle (descending dotproduct) with the Y axis.
		// Root always sorts first with the given comparer.
		std::sort(viewCorners.begin(), viewCorners.end(), Vector3HullComparer(Vector3::UNIT_Y, root));
				
		// push the root and the first point onto the list of vertices
		mVertices.push_back(viewCorners[0]);
		mVertices.push_back(viewCorners[1]);

		std::vector<Vector3>::const_iterator itCorner = viewCorners.begin() + 2;
		for(; itCorner != viewCorners.end(); ++itCorner)
		{
			Vector3 current = *itCorner;

			while(mVertices.size() > 1)
			{
				Vector3 next2last = mVertices[mVertices.size() - 2];
				Vector3 last = mVertices.back();
						
				Vector3 cross = (last - next2last).crossProduct(current - next2last);
				if (cross.z < 0) // pointing out of plane: left turn
				{
					break;
				}
				else // pointing into plane: right turn, or colinear
				{
					mVertices.pop_back();
				}
			}

			mVertices.push_back(current);
		}

		std::vector<Vector3>::const_iterator itHull = mVertices.begin();
		for(; itHull != mVertices.end(); ++itHull)
		{
			Vector3 start = *itHull;
			Vector3 end =
				(itHull+1 != mVertices.end())
					? *(itHull+1) : *(mVertices.begin());

			Vector3 edge = (end - start).normalizedCopy();

			mEdgeNormals.push_back(Vector3::UNIT_Z.crossProduct(edge).normalizedCopy());
		}		
	}

	ConvexHull::~ConvexHull(void)
	{
	}

	ConvexHull::VectorSet ConvexHull::getSeparatingAxesWith(ConvexHull other)
	{
		VectorSet axes;

		axes.insert(this->mEdgeNormals.begin(), this->mEdgeNormals.end());
		axes.insert(other.mEdgeNormals.begin(), other.mEdgeNormals.end());

		return axes;		
	}

	double ConvexHull::findPenetrationAlong(ConvexHull other, const Vector3& projDir)
	{
		VectorSet axes = this->getSeparatingAxesWith(other);
		std::vector<Vector3> cornersA = this->mVertices;
		std::vector<Vector3> cornersB = other.mVertices;

		double minPenetration = std::numeric_limits<double>::max();

		VectorSet::iterator itAxis = axes.begin();
		for(; itAxis != axes.end(); ++itAxis)
		{
			SeparatingAxis sepAxis(*itAxis);
			
			double penetration =
				sepAxis.test(
					cornersA.begin(), cornersA.end(),
					cornersB.begin(), cornersB.end(),
					this->getCenter(), other.getCenter()
				);

			if (penetration == 0) 
			{
				// No intersection found
				return 0;
			}
			else
			{
				// Discard a degenerate case. We are guaranteed to have
				// atleast one non-degenerate case anyway.
				if ((*itAxis).dotProduct(projDir) == 0)
				{					
					continue;
				}

				// Map the penetration back onto the direction passed in as vector 'dir'.
				penetration *= std::abs((*itAxis).squaredLength() / (*itAxis).dotProduct(projDir));

				// We only neeed to keep the minimum penetration along vector 'dir'.
				// Shifting by that minimum penetration will already turn the axes
				// from which this minimum penetration originated into a seperating
				// axes, i.e. will stop the two boxes from intersecting, which is all
				// we need.
				minPenetration = std::min<double>(minPenetration, penetration);
			}
		}
		
		return minPenetration;
	}
}