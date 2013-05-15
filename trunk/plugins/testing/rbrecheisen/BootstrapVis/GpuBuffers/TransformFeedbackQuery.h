/*
 * GpuTransformFeedbackQuery.h
 *
 *  Created on: Feb 26, 2009
 *      Author: ron
 */

#ifndef opengl_TransformFeedbackQuery_h
#define opengl_TransformFeedbackQuery_h

#include <GL/glew.h>
#include <GL/gl.h>

namespace opengl
{

class TransformFeedbackQuery
{
public:
	enum QueryType
	{
		QU_PRIMITIVES_WRITTEN = GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV
	};

	TransformFeedbackQuery(QueryType type);
	virtual ~TransformFeedbackQuery();

	inline unsigned int getResult() const { return mResult; }
	inline QueryType getType() const { return mQueryType; }

protected:
	friend class TransformFeedback;

	void start();
	void end();


private:
	GLuint mQueryHandle;
	QueryType mQueryType;
	GLuint mResult;
	bool mIsActive;
};

}

#endif /* GPUTRANSFORMFEEDBACKQUERY_H_ */
