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

#include "OrientedBoundingBox.h"

#include <limits>
#include <cmath>

#include <vtkActor.h>
#include <vtkMapper.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>

#include "SeparatingAxis.h"

namespace ICMath
{
	OrientedBoundingBox::OrientedBoundingBox()
		: mCenter( Vector3::ZERO), mMaxExtents(Vector3::ZERO), mMinExtents(Vector3::ZERO)
	{
		mAxis[0] = Vector3::ZERO;
		mAxis[1] = Vector3::ZERO;
		mAxis[2] = Vector3::ZERO;
	}

	OrientedBoundingBox::OrientedBoundingBox(Vector3 center, Vector3 axes[3], Vector3 minExtents, Vector3 maxExtents)
		: mCenter(center), mMinExtents(minExtents), mMaxExtents(maxExtents)
	{
		mAxis[0] = axes[0].normalizedCopy();
		mAxis[1] = axes[1].normalizedCopy();
		mAxis[2] = axes[2].normalizedCopy();
	}

	OrientedBoundingBox::OrientedBoundingBox(vtkPoints* points)	
		: mCenter( Vector3::ZERO), mMaxExtents(Vector3::ZERO), mMinExtents(Vector3::ZERO)
	{
		mAxis[0] = Vector3::ZERO;
		mAxis[1] = Vector3::ZERO;
		mAxis[2] = Vector3::ZERO;	

		this->buildFromPoints(points);
	}	
	
	OrientedBoundingBox::OrientedBoundingBox(vtkActor* actor)
		: mCenter( Vector3::ZERO), mMaxExtents(Vector3::ZERO), mMinExtents(Vector3::ZERO)
	{
		mAxis[0] = Vector3::ZERO;
		mAxis[1] = Vector3::ZERO;
		mAxis[2] = Vector3::ZERO;		

		vtkMapper* mapper = actor->GetMapper();
		vtkDataSet* dataSet = mapper->GetInput();		

		if (!dataSet->IsA("vtkPolyData")) return;		
		
		this->buildFromPoints(	
			static_cast<vtkPolyData*>(dataSet)->GetPoints()
		);
	}	

	OrientedBoundingBox::~OrientedBoundingBox(void)
	{
	}

	void OrientedBoundingBox::buildFromPoints(vtkPoints* points)
	{
		vtkIdType nrPoints = points->GetNumberOfPoints();

		// Compute the center point of the OBB
		mCenter = Vector3::ZERO;
		for(vtkIdType id = 0; id < nrPoints; ++id)
		{
			Vector3 point(points->GetPoint(id));
			mCenter = mCenter + point;
		}
		mCenter = (1.0 / nrPoints) * mCenter;
	
		// Compute the covariance matrix of the OBB
		Matrix3 covariance = Matrix3::ZERO;
		for(vtkIdType id = 0; id < nrPoints; ++id)
		{
			Vector3 point(points->GetPoint(id));
			point = point - mCenter;

			Matrix3 matrix;
			Matrix3::TensorProduct(point, point, matrix);

			covariance = covariance + matrix;		
		}
		covariance = (1.0 / nrPoints) * covariance;		


		// Retrieve the axes of the OBB by using the eigen vectors
		// of the covariance matrix
		double eigenValues[3]; // Solver needs these even if we don't
		covariance.EigenSolveSymmetric(eigenValues, mAxis);
	

		// Compute the extents of each bounding box axis
		for(vtkIdType id = 0; id < nrPoints; ++id)
		{
			Vector3 point(points->GetPoint(id));
			point = point - mCenter;

			for(int i = 0; i < 3; i++)
			{
				double extent = mAxis[i].dotProduct(point);
				mMaxExtents[i] = std::max<double>(mMaxExtents[i], extent);
				mMinExtents[i] = std::min<double>(mMinExtents[i], extent);
			}
		}

		// Keep track of the center as an offset with respect to the data set's
		// origin. This allows us to reposition the box acccording to a vtkActor's
		// position.
		mPositionOffset = mCenter;
	}

	void OrientedBoundingBox::setPosition(Vector3 pos)
	{
		mCenter = pos + mPositionOffset;
	}

	std::vector<Vector3> OrientedBoundingBox::getCornerPoints(void) const
	{
		std::vector<Vector3> points;

		points.push_back(mCenter + mAxis[0] * mMaxExtents[0] + mAxis[1] * mMaxExtents[1] + mAxis[2] * mMaxExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMinExtents[0] + mAxis[1] * mMaxExtents[1] + mAxis[2] * mMaxExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMaxExtents[0] + mAxis[1] * mMinExtents[1] + mAxis[2] * mMaxExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMinExtents[0] + mAxis[1] * mMinExtents[1] + mAxis[2] * mMaxExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMaxExtents[0] + mAxis[1] * mMaxExtents[1] + mAxis[2] * mMinExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMinExtents[0] + mAxis[1] * mMaxExtents[1] + mAxis[2] * mMinExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMaxExtents[0] + mAxis[1] * mMinExtents[1] + mAxis[2] * mMinExtents[2]);
		points.push_back(mCenter + mAxis[0] * mMinExtents[0] + mAxis[1] * mMinExtents[1] + mAxis[2] * mMinExtents[2]);

		return points;
	}

	OrientedBoundingBox::VectorSet OrientedBoundingBox::getSeparatingAxesWith(OrientedBoundingBox other)
	{
		VectorSet axes;

		// Seperating axes consist of the unique(!) edge ribs of both bounding boxes.
		// This corresponds to the axes that form both boxes' local coordinate system.
		axes.insert(this->mAxis[0]);
		axes.insert(this->mAxis[1]);
		axes.insert(this->mAxis[2]);

		axes.insert(other.mAxis[0]);
		axes.insert(other.mAxis[1]);
		axes.insert(other.mAxis[2]);

		// In addition the crossproduct of each rib from the first box with each rib
		// of the second box needs to be considered.
		for(int i = 0; i < 3; ++i)
			for(int j = 0; j < 3; ++j)
			{
				Vector3 axis = this->mAxis[i].crossProduct(other.mAxis[j]);				

				if (!axis.isZeroLength())
					axes.insert(axis.normalizedCopy());
			}

		return axes;
	}

	double OrientedBoundingBox::findPenetrationAlong(OrientedBoundingBox other, const Vector3& dir)
	{
		VectorSet axes = this->getSeparatingAxesWith(other);
		std::vector<Vector3> cornersA = this->getCornerPoints();
		std::vector<Vector3> cornersB = other.getCornerPoints();
				
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
				if ((*itAxis).dotProduct(dir) == 0)
				{					
					continue;
				}

				// Map the penetration back onto the direction passed in as vector 'dir'.
				penetration *= std::abs((*itAxis).squaredLength() / (*itAxis).dotProduct(dir));

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

	Vector3 OrientedBoundingBox::findMinimumPenetration(OrientedBoundingBox other)
	{
		VectorSet axes = this->getSeparatingAxesWith(other);
		std::vector<Vector3> cornersA = this->getCornerPoints();
		std::vector<Vector3> cornersB = other.getCornerPoints();
				
		double minPenetration = std::numeric_limits<double>::max();
		Vector3 minPenetrationVector = Vector3::ZERO;

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

			if (penetration < minPenetration)
			{
				minPenetration = penetration;
				minPenetrationVector = (*itAxis).normalizedCopy() * minPenetration;
			}
		}

		return minPenetrationVector;
	}
}