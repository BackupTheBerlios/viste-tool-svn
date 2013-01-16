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