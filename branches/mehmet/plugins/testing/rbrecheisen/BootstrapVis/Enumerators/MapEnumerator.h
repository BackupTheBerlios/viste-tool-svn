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
