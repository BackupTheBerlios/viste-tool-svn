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

#include "SeparatingAxis.h"

namespace ICMath
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