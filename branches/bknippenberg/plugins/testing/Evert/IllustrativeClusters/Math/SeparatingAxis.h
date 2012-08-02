#ifndef math_SeparatingAxis_h
#define math_SeparatingAxis_h

#include <vector>
#include <functional>
#include <algorithm>
#include "Vector3.h"

namespace ICMath
{
	class SeparatingAxis
	{
	public:
		SeparatingAxis(const Vector3& axis);
		virtual ~SeparatingAxis(void);

		double test(
			std::vector<Vector3>::iterator first1,
			std::vector<Vector3>::iterator last1,
			std::vector<Vector3>::iterator first2,
			std::vector<Vector3>::iterator last2,
			const Vector3& center1, const Vector3& center2
		);

	protected:
		class ScalarProjection: std::unary_function<Vector3, double>
		{
		public:
			ScalarProjection(const Vector3& target): mTarget(target) {}

			double operator() (const Vector3& source) const
			{
				return source.scalarProjectOn(mTarget);
			}

		private:
			Vector3 mTarget;
		};

	private:
		Vector3 mAxis;
	};
}

#endif