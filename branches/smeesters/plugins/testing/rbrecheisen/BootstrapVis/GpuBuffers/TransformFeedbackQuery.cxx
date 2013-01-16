/*
 * GpuTransformFeedbackQuery.cpp
 *
 *  Created on: Feb 26, 2009
 *      Author: ron
 */

#include "TransformFeedbackQuery.h"

namespace opengl
{

TransformFeedbackQuery::TransformFeedbackQuery(QueryType type) :
	mQueryHandle(0), mQueryType(type), mResult(0), mIsActive(false)
{
	glGenQueries(1, &mQueryHandle);
}

TransformFeedbackQuery::~TransformFeedbackQuery()
{
	if (glIsQuery(mQueryHandle) == GL_TRUE)
	{
		glDeleteQueries(1, &mQueryHandle);
	}
}

void TransformFeedbackQuery::start()
{
	if (mIsActive)
		return;

	mIsActive = true;
	glBeginQuery(static_cast<GLenum>(mQueryType), mQueryHandle);
}

void TransformFeedbackQuery::end()
{
	if (!mIsActive)
		return;

	mIsActive = false;
	glEndQuery(static_cast<GLenum>(mQueryType));
	glGetQueryObjectuiv(mQueryHandle, GL_QUERY_RESULT, &mResult);
}

}
