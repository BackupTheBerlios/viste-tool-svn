/*
 * VectorEnumerator.h
 *
 *  Created on: Feb 19, 2009
 *      Author: ron
 */

#ifndef opengl_VectorEnumerator_h
#define opengl_VectorEnumerator_h

namespace opengl
{

template<class T>
class VectorEnumerator
{
public:
	VectorEnumerator(typename T::iterator begin, typename T::iterator end) :
		mCurrent(begin), mEnd(end), mFirst(true)
	{
	}

	VectorEnumerator(T& container) :
		mCurrent(container.begin()), mEnd(container.end()), mFirst(true)
	{
	}

	VectorEnumerator(const VectorEnumerator<T> &rhs)
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

	typename T::value_type getCurrent() const
	{
		return *mCurrent;
	}

	VectorEnumerator<T>& operator =(VectorEnumerator<T> &rhs)
	{
		mCurrent = rhs.mCurrent;
		mEnd = rhs.mEnd;
		return *this;
	}

private:
	// private, since only the parametrized versions should be used.
	VectorEnumerator()
	{
	}

	typename T::iterator mCurrent;
	typename T::iterator mEnd;
	bool mFirst;
};

}

#endif /* VECTORENUMERATOR_H_ */
