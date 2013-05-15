/*
 * GpuProgram.cxx
 *
 * 2009-02-18	Ron Otten
 * - First version
 *
 * 2009-04-08	Ron Otten
 * - Added setUniform overload for uniform vec2
 */

#ifndef opengl_GpuProgram_h
#define opengl_GpuProgram_h

#include <map>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

#include "GpuShader.h"
#include "../Enumerators/VectorEnumerator.h"

namespace opengl
{

/** A high level representation of an OpenGL program object.
 *  @remarks
 *  	Currently requires that GLEW (GL Extension Wrangler) has been
 *  	succesfully initialized prior to usage of this class.
 */
class GpuProgram
{
public:
	/** Represents a list of GLSL variables declared as 'varying'. */
	typedef std::vector<std::string> VaryingsList;

	/** Allows enumeration over a VaryingsList without public
	 *  exposure of the list's iterators.
	 */
	typedef VectorEnumerator<std::vector<std::string> > VaryingsEnumerator;

	/** Enumerates the possible types of input geometry for the GPU program.
	 *  @remarks
	 *  	The type of input geometry must be specified when using
	 *  	geometry shaders with a GPU program.
	 */
	enum InputGeometry
	{
		INGEO_POINTS                   = GL_POINTS,
		INGEO_LINES                    = GL_LINES,
		INGEO_LINES_WITH_ADJACENCY     = GL_LINES_ADJACENCY_EXT,
		INGEO_TRIANGLES                = GL_TRIANGLES,
		INGEO_TRIANGLES_WITH_ADJACENCY = GL_TRIANGLES_ADJACENCY_EXT
	};

	/** Enumerates the possible types of output geometry from the GPU program.
	 * 	@remarks
	 * 		The type of output geometry must be specified when using
	 * 		geometry shaders with a GPU program.
	 */
	enum OutputGeometry
	{
		OUTGEO_POINTS                  = GL_POINTS,
		OUTGEO_LINE_STRIP              = GL_LINE_STRIP,
		OUTGEO_TRIANGLE_STRIP          = GL_TRIANGLE_STRIP
	};

	/** Creates a new GPU program.
	 */
	GpuProgram();

	/** Destroys the GPU program.
	 */
	virtual ~GpuProgram();

	/** Binds the GPU program for use */
	void bind();

	/** Unbinds the GPU program if it is the currently bound program. */
	void unbind();

	/** Creates a new GPU shader and attaches it to the GPU program.
	 *  @param name
	 *  	The name by which the shader can be identified.
	 *  @param filePath
	 *  	The path to the file holding the shader's code.
	 *  @param type
	 *  	The shader's type: vertex, fragment or geometry.
	 *  @return
	 *  	The newly created shader.
	 */
	GpuShader* createShader(const std::string& name,
			const std::string& filePath, GpuShader::GpuShaderType type);

	/** Retrieves a previously created GPU shader.
	 * 	@param name
	 * 		The name of the shader to retrieve.
	 * 	@return
	 * 		The requested shader.
	 */
	GpuShader* getShader(const std::string& name);

	/** Detaches a previously created GPU shader from the GPU program and destroys it.
	 * 	@param name
	 * 		The name of the shader to destroy.
	 */
	void destroyShader(const std::string& name);

	/** Detaches a previously created GPU shader from the GPU program and destroys it.
	 * 	@param shader
	 * 		The shader to destroy.
	 */
	void destroyShader(GpuShader* shader);

	/** Links the GPU program
	 * 	@param inputType
	 * 		The type of input geometry to use for geometry shaders in this program. Defaults to points.
	 * 	@param outputType
	 * 		The type of output geometry built by geometry shaders in this program. Defaults to points.
	 * 	@param nrOutputVertices
	 * 		The maximum number of vertices a geometry shader in this program will output for each input
	 * 		primitive. Defaults to 1.
	 * 	@return
	 * 		Whether linking was successful.
	 */
	bool link(InputGeometry inputType = INGEO_POINTS,
			OutputGeometry outputType = OUTGEO_POINTS,
			unsigned int nrOutputVertices = 1);

	/** Builds the GPU program: compiles all attached shaders and then links the program.
	 * 	@param inputType
	 * 		The type of input geometry to use for geometry shaders in this program. Defaults to points.
	 * 	@param outputType
	 * 		The type of output geometry built by geometry shaders in this program. Defaults to points.
	 * 	@param nrOutputVertices
	 * 		The maximum number of vertices a geometry shader in this program will output for each input
	 * 		primitive. Defaults to 1.
	 *  @return
	 *  	Whether building was successful.
	 */
	bool build(InputGeometry inputType = INGEO_POINTS,
			OutputGeometry outputType = OUTGEO_POINTS,
			unsigned int nrOutputVertices = 1);

	/** Get an enumerator over the collection of GPU program variable names of the 'varyings' class.
	 *  @return
	 *  	The enumerator over the collection.
	 */
	inline VaryingsEnumerator getVaryings()
	{
		return VaryingsEnumerator(mVaryings);
	}

	void setUniform(const std::string& name, bool value);
	void setUniform(const std::string& name, bool v1, bool v2, bool v3);

	void setUniform(const std::string& name, int value);
	void setUniform(const std::string& name, int v1, int v2, int v3);

	void setUniform(const std::string& name, float value);
	void setUniform(const std::string& name, float v1, float v2);
	void setUniform(const std::string& name, float v1, float v2, float v3);
	void setUniform(const std::string& name, float v1, float v2, float v3, float v4);

	/** Adds a new 'varying'-class variable name to the GPU program's collection of 'varying' variable names.
	 *	@remarks
	 *  	!! Default 'varying' names such as 'gl_Position' and 'gl_FrontColor' must be included when wanting
	 *   	!! to grab their values from the output of the GPU program with the transform feedback extension.
	 *
	 *  	The program has to be relinked for the change to take effect.
	 *
	 *  @param varying
	 *  	The variable name to add.
	 */
	void addVarying(const std::string& varying);

	/** Removes an existing 'varying'-class variable name from the GPU program's collection of 'varying' variable names.
	 * 	@remarks
	 * 		 The program has to be relinked for the change to take effect.
	 *
	 *  @param varying
	 *  	The variable name to remove.
	 */
	void removeVarying(const std::string& varying);

	/** Gets the log of the GPU program's most recent build.
	 *  @return
	 *  	The GPU program's last build log.
	 */
	inline std::string getLastBuildLog() const
	{
		return mBuildLog;
	}

protected:
	/** Represents a mapping from names to GpuShader class instances. */
	typedef std::map<std::string, GpuShader*> GpuShaderMap;

private:
	GpuShaderMap mShaders;
	VaryingsList mVaryings;

	GLuint mProgramHandle;
	std::string mBuildLog;

	/** Sets all varying variables specified for the GPU program to active.
	 *	@return
	 *		Whether the operation was successful.
	 */
	bool setActiveVaryings();

	/** Sets all varying variables specified for the GPU program to be fed back
	 *  during a transform feedback operation.
	 *  @return
	 *  	Whether the operation was successful.
	 */
	bool setFeedbackVaryings();
};

}

#endif /* GPUPROGRAM_H_ */
