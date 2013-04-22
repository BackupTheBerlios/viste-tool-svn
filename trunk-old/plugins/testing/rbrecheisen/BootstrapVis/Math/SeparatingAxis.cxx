#include "SeparatingAxis.h"

namespace math
{
	SeparatingAxis::SeparatingAxis(const Vector3& axis)
		:mAxis(axis)
	{
	}

	SeparatingAxis::~SeparatingAxis(void)
	{
	}

	
	double SeparatingAxis::test(
		std::vector<Vector3>::iterator first1,
		std::vector<Vector3>::iterator last1,
		std::vector<Vector3>::iterator first2,
		std::vector<Vector3>::iterator last2,
		const Vector3& center1, const Vector3& center2
	)
	{
		ScalarProjection projector(mAxis);
		
		// Project points onto axis. Only scalar values are needed
		std::vector<double> proj1, proj2;
		std::transform(first1, last1, std::inserter(proj1, proj1.begin()), projector);
		std::transform(first2, last2, std::inserter(proj2, proj2.begin()), projector);

		// Compute intervals of projected points.
		double min1 = *(std::min_element(proj1.begin(), proj1.end()));
		double max1 = *(std::max_element(proj1.begin(), proj1.end()));
		
		double min2 = *(std::min_element(proj2.begin(), proj2.end()));
		double max2 = *(std::max_element(proj2.begin(), proj2.end()));
		
		// If no intersection, then no penetration
		if ((max1 <= min2) || (min1 >= max2))
		{			
			return 0;
		}

		// Penetration along the seperating axis according to topology of both
		// projected centers, rather than according to which direction would
		// lead to least penetration.
		return (projector(center1) - projector(center2) < 0)
			? (max1 - min2) : (max2 - min1);
	}
}