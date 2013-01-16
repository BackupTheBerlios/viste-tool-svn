#ifndef math_ConvexHull_h
#define math_ConvexHull_h

#include "OrientedBoundingBox.h"

namespace math
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