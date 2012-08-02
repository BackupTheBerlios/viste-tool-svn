/*
 * TransformFeedback.cpp
 *
 * 2009-02-27	Ron Otten
 * - First version
 */


#include "TransformFeedback.h"

namespace opengl
{

TransformFeedback::TransformFeedback() :
	mIsActive(false)
{
	// TODO Auto-generated constructor stub

}

TransformFeedback::~TransformFeedback()
{
	QueryMap::iterator it = mQueries.begin();
	QueryMap::iterator itEnd = mQueries.end();

	for (; it != itEnd; ++it)
	{
		delete it->second;
	}
	mQueries.clear();
}

TransformFeedbackQuery* TransformFeedback::createQuery(
		TransformFeedbackQuery::QueryType type)
{
	QueryMap::const_iterator it = mQueries.find(type);
	if (it != mQueries.end())
	{
		// Query type already exists
		// TODO: Throw exception instead?
		return it->second;
	}

	TransformFeedbackQuery* query = new TransformFeedbackQuery(type);
	mQueries.insert(std::make_pair(type, query));

	return query;
}

TransformFeedbackQuery* TransformFeedback::getQuery(
		TransformFeedbackQuery::QueryType type)
{
	QueryMap::const_iterator it = mQueries.find(type);
	if (it == mQueries.end())
	{
		// Query type does not exist.
		// TODO: Throw exception instead?
		return NULL;
	}

	return it->second;
}

void TransformFeedback::destroyQuery(TransformFeedbackQuery::QueryType type)
{
	QueryMap::iterator it = mQueries.find(type);
	if (it == mQueries.end())
	{
		// Query type does not exist.
		// TODO: Throw exception instead?
		return;
	}

	delete it->second;
	mQueries.erase(it);
}

void TransformFeedback::destroyQuery(TransformFeedbackQuery* query)
{
	this->destroyQuery(query->getType());
}

void TransformFeedback::start(GeometryType type, bool discardRasterizer)
{
	if (mIsActive)
		return;

	mIsActive = true;
	glBeginTransformFeedbackNV(static_cast<GLenum> (type));
	if (discardRasterizer)
		glEnable(GL_RASTERIZER_DISCARD_NV);

	QueryMap::const_iterator it = mQueries.begin();
	QueryMap::const_iterator itEnd = mQueries.end();

	for (; it != itEnd; ++it)
	{
		it->second->start();
	}

}

void TransformFeedback::end()
{
	if (!mIsActive)
		return;

	mIsActive = false;
	QueryMap::const_iterator it = mQueries.begin();
	QueryMap::const_iterator itEnd = mQueries.end();

	for (; it != itEnd; ++it)
	{
		it->second->end();
	}

	glDisable(GL_RASTERIZER_DISCARD_NV);
	glEndTransformFeedbackNV();
}

}
