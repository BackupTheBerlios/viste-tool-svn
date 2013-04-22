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

/*
 * MapEnumerator.h
 *
 *  Created on: Feb 19, 2009
 *      Author: ron
 */

#ifndef opengl_MapEnumerator_h
#define opengl_MapEnumerator_h

namespace opengl
{

template<class T>
class MapEnumerator
{
public:
	MapEnumerator(typename T::iterator begin, typename T::iterator end) :
		mCurrent(begin), mEnd(end), mFirst(true)
	{
	}

	MapEnumerator(T& container) :
		mCurrent(container.begin()), mEnd(container.end()), mFirst(true)
	{
	}

	MapEnumerator(const MapEnumerator<T> &rhs)
	{
		mCurrent = rhs.mCurrent;
		mEnd = rhs.mEnd;
	}

	bool moveNext()
	{
		if (!mFirst)
		{
			mCurrent++;
		}
		mFirst = false;

		return (mCurrent != mEnd);
	}

	typename T::key_type getCurrentKey() const
	{
		return mCurrent->first;
	}

	typename T::mapped_type getCurrentValue() const
	{
		return mCurrent->second;
	}

	MapEnumerator<T>& operator =(MapEnumerator<T> &rhs)
	{
		mCurrent = rhs.mCurrent;
		mEnd = rhs.mEnd;
		return *this;
	}

private:
	// private, since only the parametrized versions should be used.
	MapEnumerator()
	{
	}

	typename T::iterator mCurrent;
	typename T::iterator mEnd;
	bool mFirst;
};

}

#endif /* MAPENUMERATOR_H_ */
