#ifndef math_OrientedBoundingBox_h
#define math_OrientedBoundingBox_h

#include <set>
#include <vector>
#include <algorithm>

#include "Vector3.h"
#include "Matrix3.h"

class vtkActor;
class vtkPoints;

namespace math
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