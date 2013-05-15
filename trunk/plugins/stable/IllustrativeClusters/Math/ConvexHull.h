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

#ifndef math_ConvexHull_h
#define math_ConvexHull_h

#include "OrientedBoundingBox.h"

namespace ICMath
{
	class ConvexHull
	{
	public:
		ConvexHull(OrientedBoundingBox boundingBox, const Vector3& viewDir, const Vector3& upDir);
		virtual ~ConvexHull(void);

		inline Vector3 getCenter(void) const { return mCenter; }
		double findPenetrationAlong(ConvexHull other, const Vector3& projDir);		
		
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

		class Vector3HullComparer: std::binary_function<Vector3, Vector3, bool>
		{
		public:
			Vector3HullComparer(const Vector3& refAxis, const Vector3& root)
				:mRefAxis(refAxis), mRoot(root) {}

			bool operator() (Vector3 lhs, Vector3 rhs) const
			{
				// A node does not come earlier than itself
				if (lhs == rhs) return false;

				// The root always comes first
				if (mRoot == lhs) return true;
				if (mRoot == rhs) return false;

				double lhsDp = (lhs - mRoot).normalizedCopy().dotProduct(mRefAxis);
				double rhsDp = (rhs - mRoot).normalizedCopy().dotProduct(mRefAxis);

				return (lhsDp == rhsDp)
					? (lhs.distance(mRoot) < rhs.distance(mRoot))
					: (lhsDp > rhsDp);
			}

		private:
			Vector3 mRefAxis;
			Vector3 mRoot;
		};

		typedef std::set<Vector3, Vector3Comparer> VectorSet;
		VectorSet getSeparatingAxesWith(ConvexHull other);
		
	private:
		std::vector<Vector3> mVertices;
		std::vector<Vector3> mEdgeNormals;
		
		Vector3 mCenter;	
		Vector3 mCamAxes[3];			
	};
	
}

#endif