/*
 * TransformFeedback.h
 *
 * 2009-02-27	Ron Otten
 * - First version
 */

#ifndef opengl_TransformFeedback_h
#define opengl_TransformFeedback_h

#include <map>
#include <GL/glew.h>
#include <GL/gl.h>

#include "TransformFeedbackQuery.h"

namespace opengl
{

/** Represents a transform feedback operation and allows configuration and
 *  result retrieval of transform feedback related GL queries.
 */
class TransformFeedback
{
public:
	/** Enumerates the types of geometry for which a transform feedback
	 *  operation can take place.
	 */
	enum GeometryType
	{
		GEO_POINTS = GL_POINTS,
		GEO_LINES = GL_LINES,
		GEO_TRIANGLES = GL_TRIANGLES
	};

	/** Creates a new transform feedback operation. */
	TransformFeedback();

	/** Destroys the transform feedback operation. */
	virtual ~TransformFeedback();

	/** Starts the transform feedback operation for the specified
	 *  geometry type and any member queries.
	 *	@param type
	 *		The geometry type to run the transform feedback for.
	 *	@param discardRasterizer
	 *		Whether to disable the rasterizer during transform feedback.
	 */
	void start(GeometryType type, bool discardRasterizer = true);

	/** Ends the transform feedback operation and any member queries. */
	void end();

	/** Creates a new transform feedback query of the specified type,
	 *	@param type
	 *		The type of transform feedback query to create.
	 */
	TransformFeedbackQuery* createQuery(TransformFeedbackQuery::QueryType type);

	/** Retrieves the existing transform feedback query of the specfied type.
	 *  @param type
	 *  	The query type to retrieve.
	 */
	TransformFeedbackQuery* getQuery(TransformFeedbackQuery::QueryType type);

	/** Destroys the existing transform feedback query of the specfied type.
	 *  @param type
	 *  	The query type to destory.
	 */
	void destroyQuery(TransformFeedbackQuery::QueryType type);

	/** Destroys the specified transform feedback query.
	 *  @param query
	 *  	The query to destory.
	 */
	void destroyQuery(TransformFeedbackQuery* query);

protected:
	/** Represents a mapping from types of transform feedback query to actual
	 * 	TransformFeedbackQuery class instances
	 */
	typedef std::map<TransformFeedbackQuery::QueryType, TransformFeedbackQuery*> QueryMap;

private:
	QueryMap mQueries;
	bool mIsActive;
};

}

#endif /* TRANSFORMFEEDBACK_H_ */
