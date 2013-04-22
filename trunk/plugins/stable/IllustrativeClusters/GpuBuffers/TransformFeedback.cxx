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
