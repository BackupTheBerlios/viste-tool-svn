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

#ifndef math_OrientedBoundingBox_h
#define math_OrientedBoundingBox_h

#include <set>
#include <vector>
#include <algorithm>

#include "Vector3.h"
#include "Matrix3.h"

class vtkActor;
class vtkPoints;

namespace ICMath
{

	class OrientedBoundingBox
	{
	public:		
		OrientedBoundingBox();
		OrientedBoundingBox(vtkActor* actor);
		OrientedBoundingBox(vtkPoints* points);
		OrientedBoundingBox(Vector3 center, Vector3 axes[3], Vector3 minExtents, Vector3 maxExtents);
		virtual ~OrientedBoundingBox(void);

		double findPenetrationAlong(OrientedBoundingBox other, const Vector3& dir);	
		Vector3 findMinimumPenetration(OrientedBoundingBox other);
		
		inline Vector3 getCenter(void) const { return mCenter; }
		std::vector<Vector3> getCornerPoints(void) const;
		void setPosition(Vector3 position);

	protected:
		class Vector3Comparer: std::binary_function<Vector3, Vector3, bool>
		{
		public:
			bool operator() (const Vector3& lhs, const Vector3& rhs) const
			{
				if (lhs.x == rhs.x)
				{
					if (lhs.y == rhs.y)
					{
						return (lhs.z < rhs.z);
					}

					return (lhs.y < rhs.y);
				}
					
				return (lhs.x < rhs.x);
			}
		};

		class Vector3Projector: std::unary_function<Vector3, double>
		{
		public:
			Vector3Projector(const Vector3& target): mTarget(target) {}

			double operator() (const Vector3& source) const
			{
				double length = mTarget.dotProduct(source) / mTarget.squaredLength();
				
				return length;
			}

		private:
			Vector3 mTarget;
		};

		typedef std::set<Vector3, Vector3Comparer> VectorSet;				
		VectorSet getSeparatingAxesWith(OrientedBoundingBox other);

	private:
		Vector3 mAxis[3];
		Vector3 mCenter;
		Vector3 mMaxExtents;	
		Vector3 mMinExtents;

		Vector3 mPositionOffset;

		void buildFromPoints(vtkPoints* points);
	};
	
}

#endif