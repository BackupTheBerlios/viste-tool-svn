/*
 * vtkIllustrativeFiberBundleMapper.cxx
 *
 * 2009-03-26	Ron Otten
 * - First Version.
 *
 * 2011-03-28	Evert van Aart
 * - Ported to DTITool3.
 * - Changed initialization to avoid unneccesary re-initialization.
 * - Removed a million VTK error messages if/when the 3D subcanvas is hidden.
 *
 */


#include "vtkIllustrativeFiberBundleMapper.h"

#include <string>
#include <sstream>
#include <cmath> // Required for std::abs() which has floating point overloads for C99's integer-only abs() function
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkDataArray.h>
#include <vtkCommand.h>
#include <vtkTimerLog.h>
#include <vtkCamera.h>

// Include shader texts
#include "BuildHaloFins.geom.h"
#include "BuildHaloLines.geom.h"
#include "DilateSilhouetteContours.frag.h"
#include "DilateSilhouetteContours.vert.h"
#include "InkHaloFins.frag.h"
#include "InkHaloLines.frag.h"
#include "Passthrough.vert.h"

namespace bmia
{

vtkCxxRevisionMacro(vtkIllustrativeFiberBundleMapper, "$Revision: 1.0 $")
vtkStandardNewMacro(vtkIllustrativeFiberBundleMapper)

void vtkIllustrativeFiberBundleMapper::PrintSelf(ostream& os, vtkIndent indent)
{
	this->Superclass::PrintSelf(os, indent);
}

vtkIllustrativeFiberBundleMapper::vtkIllustrativeFiberBundleMapper() :
	vtkPolyDataMapper(),
	mFinWidth(0.6f), mFinRecision(0.1f),
	mUseStroking(true), mMinStrokeWidth(0.1f), mMaxStrokeWidth(0.6f),
	mLightLines(true), mMinLuminosity(0), mMaxLuminosity(1.0f),
	mSilhouette(true), mOutlineWidth(1), mFillDilation(5), mInnerOutlineDepthThreshold(10.0f), 
	mFinProgram(NULL), mSilhouetteProgram(NULL), mLineProgram(NULL), mFrameBuffer(NULL),
	GEOMETRY_OUTPUT_VERTICES(14)
{
	mLineColor[0] = 0.0f;
	mLineColor[1] = 0.0f;
	mLineColor[2] = 0.0f;

	mFillColor[0] = 1.0f;
	mFillColor[1] = 1.0f;
	mFillColor[2] = 1.0f;

	mLighting[0] = 0.0f;
	mLighting[1] = 0.7f;
	mLighting[2] = 0.2f;

	mShinyness = 3;

	this->initialized = false;
}

vtkIllustrativeFiberBundleMapper::~vtkIllustrativeFiberBundleMapper()
{
}

bool vtkIllustrativeFiberBundleMapper::InitializeGraphicsResources(vtkViewport* viewport)
{
	int* windowSize = viewport->GetSize();

	if (mFinProgram == NULL)
	{
		// Make sure GLEW has initialized so we can test for presence of
		// the required GL extensions and use them more easily.
		glewInit();

		mFinProgram = new opengl::GpuProgram();	

		mFinProgram
			->createShader("Vertex passthrough", "", opengl::GpuShader::GST_VERTEX)
			->setSourceCode(Passthrough_VertexShaderCode);
				
		mFinProgram
			->createShader("Build fins", "", opengl::GpuShader::GST_GEOMETRY)
			->setSourceCode(BuildHaloFins_GeometryShaderCode);

		mFinProgram
			->createShader("Ink fins", "", opengl::GpuShader::GST_FRAGMENT)
			->setSourceCode(InkHaloFins_FragmentShaderCode);

		if (!mFinProgram->build(opengl::GpuProgram::INGEO_LINES_WITH_ADJACENCY,
			opengl::GpuProgram::OUTGEO_TRIANGLE_STRIP, GEOMETRY_OUTPUT_VERTICES))
		{
			// Jumping through hoops because the VTK error reporting macro is a PoS.

			std::stringstream errorMsg(std::stringstream::in
					| std::stringstream::out);

			errorMsg << "Could not build the fin rendering GPU program."
					<< std::endl << "Build log:"
					<< mFinProgram->getLastBuildLog();

			vtkErrorMacro( << errorMsg.str().c_str());			

			return false;
		}		
	}

	if (mLineProgram == NULL)
	{
		mLineProgram = new opengl::GpuProgram();		

		mLineProgram
			->createShader("Vertex passthrough", "", opengl::GpuShader::GST_VERTEX)
			->setSourceCode(Passthrough_VertexShaderCode);

		mLineProgram
			->createShader("Build fins", "", opengl::GpuShader::GST_GEOMETRY)
			->setSourceCode(BuildHaloLines_GeometryShaderCode);

		mLineProgram
			->createShader("Ink fins", "", opengl::GpuShader::GST_FRAGMENT)
			->setSourceCode(InkHaloLines_FragmentShaderCode);

		if (!mLineProgram->build(opengl::GpuProgram::INGEO_LINES_WITH_ADJACENCY,
			opengl::GpuProgram::OUTGEO_LINE_STRIP, GEOMETRY_OUTPUT_VERTICES))
		{
			// Jumping through hoops because the VTK error reporting macro is a PoS.

			std::stringstream errorMsg(std::stringstream::in
					| std::stringstream::out);

			errorMsg << "Could not build the line rendering GPU program."
					<< std::endl << "Build log:"
					<< mLineProgram->getLastBuildLog();

			vtkErrorMacro( << errorMsg.str().c_str());			

			return false;
		}		
	}

	if (mSilhouetteProgram == NULL)
	{
		mSilhouetteProgram = new opengl::GpuProgram();

		mSilhouetteProgram
			->createShader("Silhouette quadrilateral", "", opengl::GpuShader::GST_VERTEX)
			->setSourceCode(DilateSilhouetteContours_VertexShaderCode);
		
		mSilhouetteProgram
			->createShader("Dilate silhouette and contours", "", opengl::GpuShader::GST_FRAGMENT)
			->setSourceCode(DilateSilhouetteContours_FragmentShaderCode);
		

		mSilhouetteProgram->addVarying("minScreenExtent");
		mSilhouetteProgram->addVarying("maxScreenExtent");
		if (!mSilhouetteProgram->build())
		{
			// Jumping through hoops because the VTK error reporting macro is a PoS.

			std::stringstream errorMsg(std::stringstream::in
					| std::stringstream::out);

			errorMsg << "Could not build the silhouette composition GPU program."
					<< std::endl << "Build log:"
					<< mSilhouetteProgram->getLastBuildLog();

			vtkErrorMacro( << errorMsg.str().c_str());			

			return false;
		}

		mSilhouetteProgram->setUniform("colorSampler", 0);
		mSilhouetteProgram->setUniform("depthSampler", 1);
	}

	this->initialized = true;

	return true;
}


bool vtkIllustrativeFiberBundleMapper::createFrameBuffer(int w, int h)
{
	if (mFrameBuffer != NULL)
	{
		// If the window has changed size, then the frame buffer must be reinitialized accordingly.
		if ((mFrameBuffer->getWidth() != w) || (mFrameBuffer->getHeight() != h))
		{
			delete mFrameBuffer;
			mFrameBuffer = NULL;
		}
	}

	if (mFrameBuffer == NULL)
	{	
		opengl::FrameBufferDeclaration declaration(w, h);
		declaration.createBinding(
			opengl::FrameBufferBinding::BND_COLOR_ATTACHMENT,
			opengl::FrameBufferBinding::ELM_UNSIGNED_BYTE, 0
			);
		declaration.createBinding(
			opengl::FrameBufferBinding::BND_DEPTH_ATTACHMENT,
			opengl::FrameBufferBinding::ELM_UNSIGNED_BYTE, 1
			);

		mFrameBuffer = new opengl::FrameBuffer();

		if (!mFrameBuffer->declare(declaration))
		{
			return false;
		}		
	}

	return true;
}

void vtkIllustrativeFiberBundleMapper::ReleaseGraphicsResources(vtkWindow *window)
{	
	if (mFinProgram != NULL)
	{
		delete mFinProgram;
		mFinProgram = NULL;
	}

	if (mLineProgram != NULL)
	{
		delete mLineProgram;
		mLineProgram = NULL;
	}

	if (mSilhouetteProgram != NULL)
	{
		delete mSilhouetteProgram;
		mSilhouetteProgram = NULL;
	}

	if (mFrameBuffer != NULL)
	{
		delete mFrameBuffer;
		mFrameBuffer = NULL;
	}

	this->initialized = false;
}

void vtkIllustrativeFiberBundleMapper::RenderPiece(vtkRenderer *renderer, vtkActor *actor)
{
	vtkPolyData *input = this->GetInput();

	//
	// make sure that we've been properly initialized
	//
	if (renderer->GetRenderWindow()->CheckAbortStatus())
	{
		return;
	}

	if (input == NULL)
	{
		vtkErrorMacro(<< "No input!");
		return;
	}
	else
	{
		this->InvokeEvent(vtkCommand::StartEvent, NULL);
		if (!this->Static)
		{
			input->Update();
		}
		this->InvokeEvent(vtkCommand::EndEvent, NULL);

		vtkIdType numPts = input->GetNumberOfPoints();
		if (numPts == 0)
		{
			vtkDebugMacro(<< "No points!");
			return;
		}
	}

	// make sure our window is current
	renderer->GetRenderWindow()->MakeCurrent();

	vtkCellArray* lines = input->GetLines();
	vtkPoints* points = input->GetPoints();

	vtkIdType *connections = 0;
	vtkIdType nrConnections = 0;

	GLint maxPossibleVertices;
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &maxPossibleVertices);

	// -3 so there is room for two connecting vertices used by GL_LINE_STRIP_ADJACENCY_EXT and
	// the re-use of the last vertex from the preceding iteration's line segment.
	int maxConnectionsPerIteration = (maxPossibleVertices / GEOMETRY_OUTPUT_VERTICES) - 3;
	
	double vertex3f[3];

	for(lines->InitTraversal(); lines->GetNextCell(nrConnections, connections);)
	{
		
		int iterations = ceil(nrConnections / static_cast<float>(maxConnectionsPerIteration));

		for (int iteration = 0; iteration < iterations; ++iteration)
		{
			vtkIdType startNr = iteration * maxConnectionsPerIteration;
			vtkIdType endNr = (iteration + 1) * maxConnectionsPerIteration;

			glBegin(GL_LINE_STRIP_ADJACENCY_EXT);

			for (vtkIdType connectionNr = startNr - 2; connectionNr < (endNr + 1); ++connectionNr)
			{
				if (connectionNr > nrConnections || connectionNr < -1) continue;

				if (connectionNr == -1)
				{
					points->GetPoint(connections[0], vertex3f);
				}
				else if (connectionNr == nrConnections)
				{
					points->GetPoint(connections[nrConnections - 1], vertex3f);
				}				
				else
				{
					points->GetPoint(connections[connectionNr], vertex3f);
				}

				glVertex3f(vertex3f[0], vertex3f[1], vertex3f[2]);
			}

			glEnd();
		}
	}
}

void vtkIllustrativeFiberBundleMapper::Render(vtkRenderer* renderer, vtkActor* actor)
{
	if (this->initialized == false)
	{
		if (!(this->InitializeGraphicsResources(renderer)))
		{
			this->ReleaseGraphicsResources(renderer->GetVTKWindow());
			return;
		}
	}

	int * windowSize = renderer->GetSize();
	if (!(this->createFrameBuffer(windowSize[0], windowSize[1])))
	{
		return;
	}

	this->Timer->StartTimer();	

	double farplane;
	double nearplane;	
	renderer->GetActiveCamera()->GetClippingRange(nearplane, farplane);

	glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT);
	{
		glDisable(GL_LIGHTING);

		mFrameBuffer->bind();

		glPushAttrib(GL_VIEWPORT_BIT | GL_SCISSOR_BIT);
		{
			glViewport(0, 0, mFrameBuffer->getWidth(), mFrameBuffer->getHeight());
			glScissor(0, 0, mFrameBuffer->getWidth(), mFrameBuffer->getHeight());
			glDepthRange(0,1);

			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			{
				mFinProgram->bind();
				mFinProgram->setUniform("finWidth", std::abs(mFinWidth));
				mFinProgram->setUniform("finRecision", mFinRecision);
				
				mFinProgram->setUniform("inkLines", mUseStroking);
				mFinProgram->setUniform("minStrokeWidth", mMinStrokeWidth);
				mFinProgram->setUniform("maxStrokeWidth", mMaxStrokeWidth);
				
				mFinProgram->setUniform("finColor", mFillColor[0], mFillColor[1], mFillColor[2]);
				mFinProgram->setUniform("lineColor", mLineColor[0], mLineColor[1], mLineColor[2]);
				
				if (mLightLines)
				{
					mFinProgram->setUniform("phongParams", mLighting[0], mLighting[1], mLighting[2], static_cast<float>(mShinyness));
					mFinProgram->setUniform("minLuminosity", mMinLuminosity);
					mFinProgram->setUniform("maxLuminosity", mMaxLuminosity);
				}
				else
				{
					mFinProgram->setUniform("phongParams", 0.0f, 0.0f, 0.0f, 1.0f);
					mFinProgram->setUniform("minLuminosity", 0.0f);
					mFinProgram->setUniform("maxLuminosity", 0.0f);
				}

				this->Superclass::Render(renderer, actor);

				mFinProgram->unbind();

				if (!mUseStroking)
				{
					mLineProgram->bind();
					mLineProgram->setUniform("finColor", mFillColor[0], mFillColor[1], mFillColor[2]);
					mLineProgram->setUniform("lineColor", mLineColor[0], mLineColor[1], mLineColor[2]);
					
					if (mLightLines)
					{
						mLineProgram->setUniform("phongParams", mLighting[0], mLighting[1], mLighting[2], static_cast<float>(mShinyness));
						mLineProgram->setUniform("minLuminosity", mMinLuminosity);
						mLineProgram->setUniform("maxLuminosity", mMaxLuminosity);
					}
					else
					{
						mLineProgram->setUniform("phongParams", 0.0f, 0.0f, 0.0f, 1.0f);
						mLineProgram->setUniform("minLuminosity", 0.0f);
						mLineProgram->setUniform("maxLuminosity", 0.0f);
					}

					glDepthFunc(GL_LEQUAL);

					this->Superclass::Render(renderer, actor);

					mLineProgram->unbind();
				}
			}
		}
		glPopAttrib();

		mFrameBuffer->unbind();

		mFrameBuffer->getBoundTexture(opengl::FrameBufferBinding::BND_COLOR_ATTACHMENT)->bind();
		mFrameBuffer->getBoundTexture(opengl::FrameBufferBinding::BND_DEPTH_ATTACHMENT)->bind();
		{
			mSilhouetteProgram->bind();
			mSilhouetteProgram->setUniform("pixelRatio", static_cast<float>(1.0 / mFrameBuffer->getWidth()), static_cast<float>(1.0 / mFrameBuffer->getHeight()));
			mSilhouetteProgram->setUniform("depthTreshold", mInnerOutlineDepthThreshold);
			mSilhouetteProgram->setUniform("nearPlane", static_cast<float>(nearplane));
			mSilhouetteProgram->setUniform("farPlane", static_cast<float>(farplane));

			mSilhouetteProgram->setUniform("lineColor", mLineColor[0], mLineColor[1], mLineColor[2]);
			mSilhouetteProgram->setUniform("fillColor", mFillColor[0], mFillColor[1], mFillColor[2]);

			if (mSilhouette)
			{
				mSilhouetteProgram->setUniform("fillDilation", static_cast<int>(mFillDilation));			
				mSilhouetteProgram->setUniform("outlineWidth", static_cast<int>(mOutlineWidth));
			}
			else
			{
				mSilhouetteProgram->setUniform("fillDilation", static_cast<int>(0));			
				mSilhouetteProgram->setUniform("outlineWidth", static_cast<int>(0));
			}
			
			if (renderer->GetActiveCamera()->GetParallelProjection())
				mSilhouetteProgram->setUniform("orthographic", true);
			else
				mSilhouetteProgram->setUniform("orthographic", false);

			double* bounds = this->GetInput()->GetBounds();
			mSilhouetteProgram->setUniform("minExtent", static_cast<float>(bounds[0]), static_cast<float>(bounds[2]), static_cast<float>(bounds[4]));
			mSilhouetteProgram->setUniform("maxExtent", static_cast<float>(bounds[1]), static_cast<float>(bounds[3]), static_cast<float>(bounds[5]));

			glDepthFunc(GL_LEQUAL);
			
			glBegin(GL_QUADS);
			glTexCoord2d(0, 0);
			glVertex3f(-1, -1, 0);

			glTexCoord2d(1, 0);
			glVertex3f( 1, -1, 0);

			glTexCoord2d(1, 1);
			glVertex3f( 1,  1, 0);

			glTexCoord2d(0, 1);
			glVertex3f(-1,  1, 0);
			glEnd();
			
			mSilhouetteProgram->unbind();
		}	

		mFrameBuffer->getBoundTexture(opengl::FrameBufferBinding::BND_COLOR_ATTACHMENT)->unbind();
		mFrameBuffer->getBoundTexture(opengl::FrameBufferBinding::BND_DEPTH_ATTACHMENT)->unbind();

	}
	glPopAttrib();	

	this->Timer->StopTimer();
	this->TimeToDraw = this->Timer->GetElapsedTime();
}

}
