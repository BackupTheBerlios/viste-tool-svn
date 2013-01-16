/**
 * vtkFiberConfidenceMapper.cxx
 * by Ralph Brecheisen
 *
 * 08-01-2010	Ralph Brecheisen
 * - First version
 */
#include "vtkFiberConfidenceMapper.h"

#include "vtkObjectFactory.h"
#include "vtkTimerLog.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkCamera.h"
#include "vtkProperty.h"
#include "vtkCellArray.h"

#include <cassert>
#include <sstream>
#include <algorithm>

#include "DensityMap.vert.h"
#include "DensityMap.frag.h"
#include "SilhouetteMap.vert.h"
#include "SilhouetteMap.frag.h"
#include "Erosion.vert.h"
#include "Erosion.frag.h"
#include "ToScreen.vert.h"
#include "ToScreen.frag.h"
#include "DensitySmoothing.vert.h"
#include "DensitySmoothing.frag.h"

bool compareLevels( std::pair<std::string,float> A, std::pair<std::string,float> B )
{
    return (A.second < B.second);
}

bool compareColors( std::pair<float,vtkColor4> A, std::pair<float,vtkColor4> B )
{
    return (A.first < B.first);
}

namespace bmia {

    vtkStandardNewMacro( vtkFiberConfidenceMapper );

	////////////////////////////////////////////////////////////////////////
    vtkFiberConfidenceMapper::vtkFiberConfidenceMapper() : vtkPolyDataMapper()
	{
		this->Table = NULL;
        this->TableOriginal = NULL;
		this->TableBundle = NULL;
		this->Interval[0] = 0.0f;
		this->Interval[1] = 1.0f;
		this->RenderList = 0;
		this->RebuildList = true;
		this->SilhouetteEnabled = true;
		this->SilhouetteStreamlinesEnabled = false;

		//this->FiberMapper = NULL;

        this->RenderLists = NULL;
        this->Renderer = NULL;
        this->Actor = NULL;

		this->DepthThreshold = 10.0f;
		this->FillDilation = 5;
		this->OutlineWidth = 1;
        this->MaximumFiberDensity = 20;
        this->ErosionEnabled = false;
        this->DensityColoringEnabled = false;
		this->DensityWeightingEnabled = false;
        this->DensitySmoothingEnabled = false;
		this->OverwriteEnabled = true;

		this->FillColor[0] = 1.0f;
		this->FillColor[1] = 1.0f;
		this->FillColor[2] = 1.0f;

		this->OutlineColor[0] = 0.0f;
		this->OutlineColor[1] = 0.0f;
		this->OutlineColor[2] = 0.0f;

        this->SilhouetteProgram = NULL;
        this->DensityProgram = NULL;
        this->ToScreenProgram = NULL;
        this->ErosionProgram = NULL;
        this->DensitySmoothingProgram = NULL;

        this->FrameBuffer = 0;
        this->ColorBuffer = 0;
        this->DepthBuffer = 0;
        this->DensityBuffer = 0;
        this->DensityDepthBuffer = 0;
        this->DensitySmoothingBuffer = 0;
        this->DensitySmoothingDepthBuffer = 0;
        this->ErosionBuffer[0] = 0;
        this->ErosionBuffer[1] = 0;
        this->ErosionDepthBuffer[0] = 0;
        this->ErosionDepthBuffer[1] = 0;
        this->SilhouetteBuffer[0] = 0;
        this->SilhouetteBuffer[1] = 0;
        this->SilhouetteDepthBuffer[0] = 0;
        this->SilhouetteDepthBuffer[2] = 0;

		this->ConfidenceLevels = NULL;
		this->FillColors = NULL;
		this->LineColors = NULL;

        this->DensityColor[0] = 1.0f;
        this->DensityColor[1] = 0.0f;
        this->DensityColor[2] = 0.0f;
        this->DensityColor[3] = 1.0f;

		this->PreviousNumberOfConfidenceLevels = 0;
		this->DistanceMode = 0; // Relative to median fiber
		this->PercentageMode = 1; // Relative to fiber per seedpoint
	}

	////////////////////////////////////////////////////////////////////////
    vtkFiberConfidenceMapper::~vtkFiberConfidenceMapper()
	{
		if( this->Table )
			this->Table->UnRegister( this );
        if( this->TableOriginal )
            this->TableOriginal->UnRegister( this );

		this->ReleaseGraphicsResources( NULL );
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::Render( vtkRenderer * _renderer, vtkActor * _actor )
	{
        this->Renderer = _renderer;
        this->Actor = _actor;

		if( ! this->InitializeGraphicsResources( _renderer ) )
		{
            std::cout << "vtkFiberConfidenceMapper::Render() cannot initialize graphics resources" << std::endl;
			this->ReleaseGraphicsResources( _renderer->GetVTKWindow() );
			return;
		}

		this->vtkPolyDataMapper::Render( _renderer, _actor );
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::RenderPiece( vtkRenderer * _renderer, vtkActor * _actor )
	{
		double depthNear, depthFar;
        _renderer->GetRenderWindow()->MakeCurrent();
        _renderer->GetActiveCamera()->GetClippingRange( depthNear, depthFar );

        int * viewportSize = _renderer->GetSize();

        if( this->RebuildList )
            this->RebuildDisplayList();

        GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT,GL_NONE};

        glPushAttrib( GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT );
        {
            glDisable( GL_LIGHTING );
            glEnable( GL_BLEND );

            int write = 0;
            int read  = 1;

            for( unsigned int i = 0; i < this->ConfidenceLevels->size(); i++ )
            {
                //------------------------------------------------------------------------------------------
                // Render N % streamlines to offscreen buffer

                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->FrameBuffer );
                {
                    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->ColorBuffer, 0 );
                    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->DepthBuffer, 0 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    glDrawBuffers( 2, buffers );

                    glPushAttrib( GL_VIEWPORT_BIT | GL_SCISSOR_BIT );
                    {
                        glViewport( 0, 0, viewportSize[0], viewportSize[1] );
                        glScissor ( 0, 0, viewportSize[0], viewportSize[1] );
                        glDepthRange( 0, 1 );

                        glDepthFunc( GL_LESS );

                        glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
                        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

                        glCallList( this->RenderLists[i] );
                    }
                    glPopAttrib();

                }
                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

                //------------------------------------------------------------------------------------------
                // Render N % streamlines to density buffer

                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->FrameBuffer );
				{
                    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->DensityBuffer, 0 );
                    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->DensityDepthBuffer, 0 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    glDrawBuffers( 2, buffers );

                    glPushAttrib( GL_VIEWPORT_BIT | GL_SCISSOR_BIT );
                    {
                        glViewport( 0, 0, viewportSize[0], viewportSize[1] );
                        glScissor ( 0, 0, viewportSize[0], viewportSize[1] );
                        glDepthRange( 0, 1 );

                        glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
                        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

                        // Disable depth test because we want *all* fragments to be processed
                        // by the density shader program
                        glDisable( GL_DEPTH_TEST );
                        {
                            // Enable additive blending so each fragment shader's output is
                            // added to the color buffer
                            glBlendFunc( GL_ONE, GL_ONE );

                            this->DensityProgram->bind();
                            {
                                glCallList( this->RenderLists[i] );
                            }
                            this->DensityProgram->unbind();

                            // Reset blending function otherwise VTK will get transparency problems
                            // else in the rendering pipeline
                            glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
                        }
                        glEnable( GL_DEPTH_TEST );
                    }
                    glPopAttrib();
				}
                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

                //------------------------------------------------------------------------------------------
                // Smooth the density buffer

                if( this->IsDensitySmoothingEnabled() )
                {
                    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->FrameBuffer );
                    {
                        glBindTexture( GL_TEXTURE_2D, 0 );

                        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->DensitySmoothingBuffer, 0 );
                        glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->DensitySmoothingDepthBuffer, 0 );
                        glBindTexture( GL_TEXTURE_2D, 0 );
                        glDrawBuffers( 2, buffers );

                        glActiveTexture( GL_TEXTURE0 );
                        glBindTexture( GL_TEXTURE_2D, this->DensityBuffer );
                        glActiveTexture( GL_TEXTURE1 );
                        glBindTexture( GL_TEXTURE_2D, this->DensityDepthBuffer );

                        this->DensitySmoothingProgram->bind();
                        {
                            this->DensitySmoothingProgram->setUniform( "densityBuffer", 0 );
                            this->DensitySmoothingProgram->setUniform( "densityDepthBuffer", 1 );
                            this->DensitySmoothingProgram->setUniform( "pixelRatio", (1.0f / viewportSize[0]), (1.0f / viewportSize[1]) );
                            this->DensitySmoothingProgram->setUniform( "kernelRadius", this->GetSmoothingKernelSize() );
                            this->DensitySmoothingProgram->setUniform( "maxDensity", (float) this->GetMaximumFiberDensity() );

                            glPushAttrib( GL_VIEWPORT_BIT | GL_SCISSOR_BIT );
                            {
                                glViewport( 0, 0, viewportSize[0], viewportSize[1] );
                                glScissor ( 0, 0, viewportSize[0], viewportSize[1] );
                                glDepthRange( 0, 1 );

                                glDepthFunc( GL_LEQUAL );

                                glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
                                glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

                                glBegin(GL_QUADS);
                                {
                                    glTexCoord2f( 0, 0 ); glVertex3f( -1, -1, 0 );
                                    glTexCoord2f( 1, 0 ); glVertex3f(  1, -1, 0 );
                                    glTexCoord2f( 1, 1 ); glVertex3f(  1,  1, 0 );
                                    glTexCoord2f( 0, 1 ); glVertex3f( -1,  1, 0 );
                                }
                                glEnd();
                            }
                            glPopAttrib();
                        }
                        this->DensitySmoothingProgram->unbind();

                        glActiveTexture( GL_TEXTURE1 );
                        glBindTexture( GL_TEXTURE_2D, 0 );
                        glActiveTexture( GL_TEXTURE0 );
                        glBindTexture( GL_TEXTURE_2D, 0 );
                    }
                    glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
                }

                //------------------------------------------------------------------------------------------
                // Render silhouette

                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->FrameBuffer );
                {
                    glBindTexture( GL_TEXTURE_2D, 0 );

                    // Only use first silhouette buffer, we don't need to do ping-pong
                    // now that we have an additional erosion step
                    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->SilhouetteBuffer[write], 0 );
                    glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->SilhouetteDepthBuffer[write], 0 );
                    glDrawBuffers( 2, buffers );

                    glActiveTexture( GL_TEXTURE0 );
                    glBindTexture( GL_TEXTURE_2D, this->ColorBuffer );
                    glActiveTexture( GL_TEXTURE1 );
                    glBindTexture( GL_TEXTURE_2D, this->DepthBuffer );

                    glActiveTexture( GL_TEXTURE2 );
                    if( this->IsDensitySmoothingEnabled() )
                        glBindTexture( GL_TEXTURE_2D, this->DensitySmoothingBuffer );
                    else
                        glBindTexture( GL_TEXTURE_2D, this->DensityBuffer );

					if( this->IsErosionEnabled() )
					{
						glActiveTexture( GL_TEXTURE3 );
						glBindTexture( GL_TEXTURE_2D, this->ErosionBuffer[read] );
						glActiveTexture( GL_TEXTURE4 );
						glBindTexture( GL_TEXTURE_2D, this->ErosionDepthBuffer[read] );
					}
					else
					{
						glActiveTexture( GL_TEXTURE3 );
						glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[read] );
						glActiveTexture( GL_TEXTURE4 );
						glBindTexture( GL_TEXTURE_2D, this->SilhouetteDepthBuffer[read] );
					}

                    double * bounds = this->GetInput()->GetBounds();

                    this->SilhouetteProgram->bind();
                    {
                        std::pair<float,vtkColor4> fillPair = this->FillColors->at( i );
                        float fillColor[4];
                        fillColor[0] = fillPair.second.r / 255.0f;
                        fillColor[1] = fillPair.second.g / 255.0f;
                        fillColor[2] = fillPair.second.b / 255.0f;
                        fillColor[3] = fillPair.second.a / 255.0f;

                        std::pair<float,vtkColor4> linePair = this->LineColors->at( i );
                        float lineColor[4];
                        lineColor[0] = linePair.second.r / 255.0f;
                        lineColor[1] = linePair.second.g / 255.0f;
                        lineColor[2] = linePair.second.b / 255.0f;
                        lineColor[3] = 1.0f;

                        const float * densityColor = this->GetDensityColor();

                        bool densityEnabled =
								this->IsDensityColoringEnabled() && (! this->IsErosionEnabled());

                        bool densityWeightingEnabled =
								this->IsDensityWeightingEnabled() && (! this->IsErosionEnabled());

                        this->SilhouetteProgram->setUniform( "colorSampler", 0 );
                        this->SilhouetteProgram->setUniform( "depthSampler", 1 );
                        this->SilhouetteProgram->setUniform( "densitySampler", 2 );
                        this->SilhouetteProgram->setUniform( "previousSampler", 3 );
                        this->SilhouetteProgram->setUniform( "previousDepthSampler", 4 );
                        this->SilhouetteProgram->setUniform( "pixelRatio", (1.0f / viewportSize[0]), (1.0f / viewportSize[1]) );
                        this->SilhouetteProgram->setUniform( "depthTreshold", this->GetDepthThreshold() );
                        this->SilhouetteProgram->setUniform( "nearPlane", (float) depthNear );
                        this->SilhouetteProgram->setUniform( "farPlane", (float) depthFar );
                        this->SilhouetteProgram->setUniform( "fillColor", fillColor[0], fillColor[1], fillColor[2], fillColor[3] );
                        this->SilhouetteProgram->setUniform( "lineColor", lineColor[0], lineColor[1], lineColor[2], lineColor[3] );
                        this->SilhouetteProgram->setUniform( "fillDilation", this->GetFillDilation() );
                        this->SilhouetteProgram->setUniform( "outlineWidth", this->GetOutlineWidth() );
                        this->SilhouetteProgram->setUniform( "minExtent", (float) bounds[0], (float) bounds[2], (float) bounds[4] );
                        this->SilhouetteProgram->setUniform( "maxExtent", (float) bounds[1], (float) bounds[3], (float) bounds[5] );
                        this->SilhouetteProgram->setUniform( "orthographic", _renderer->GetActiveCamera()->GetParallelProjection() ? true : false );
                        this->SilhouetteProgram->setUniform( "densityColor", densityColor[0], densityColor[1], densityColor[2], densityColor[3] );
                        this->SilhouetteProgram->setUniform( "densityColoring", densityEnabled );
                        this->SilhouetteProgram->setUniform( "densityWeighting", densityWeightingEnabled );
                        this->SilhouetteProgram->setUniform( "maxDensity", (float) this->GetMaximumFiberDensity() );
                        this->SilhouetteProgram->setUniform( "firstPass", (i == 0) );
						this->SilhouetteProgram->setUniform( "overwriteEnabled", this->IsOverwriteEnabled() );

                        glPushAttrib( GL_VIEWPORT_BIT | GL_SCISSOR_BIT );
                        {
                            glViewport( 0, 0, viewportSize[0], viewportSize[1] );
                            glScissor ( 0, 0, viewportSize[0], viewportSize[1] );
                            glDepthRange( 0, 1 );

                            glDepthFunc( GL_LEQUAL );

                            glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
                            glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

                            glBegin(GL_QUADS);
                            {
                                glTexCoord2f( 0, 0 ); glVertex3f( -1, -1, 0 );
                                glTexCoord2f( 1, 0 ); glVertex3f(  1, -1, 0 );
                                glTexCoord2f( 1, 1 ); glVertex3f(  1,  1, 0 );
                                glTexCoord2f( 0, 1 ); glVertex3f( -1,  1, 0 );
                            }
                            glEnd();
                        }
                        glPopAttrib();
                    }
                    this->SilhouetteProgram->unbind();

                    glActiveTexture( GL_TEXTURE4 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    glActiveTexture( GL_TEXTURE3 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    glActiveTexture( GL_TEXTURE2 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    glActiveTexture( GL_TEXTURE1 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    glActiveTexture( GL_TEXTURE0 );
                    glBindTexture( GL_TEXTURE_2D, 0 );
                }
                glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );

                //------------------------------------------------------------------------------------------
                // Apply erosion to silhouettes

				if( this->IsErosionEnabled() )
				{
					glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, this->FrameBuffer );
					{
						glBindTexture( GL_TEXTURE_2D, 0 );

						glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, this->ErosionBuffer[write], 0 );
						glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, this->ErosionDepthBuffer[write], 0 );
						glDrawBuffers( 2, buffers );

						glActiveTexture( GL_TEXTURE0 );
						glBindTexture( GL_TEXTURE_2D, this->ErosionBuffer[read] );
						glActiveTexture( GL_TEXTURE1 );
						glBindTexture( GL_TEXTURE_2D, this->ErosionDepthBuffer[read] );

						glActiveTexture( GL_TEXTURE2 );
                        if( this->IsDensitySmoothingEnabled() )
                            glBindTexture( GL_TEXTURE_2D, this->DensitySmoothingBuffer );
                        else
                            glBindTexture( GL_TEXTURE_2D, this->DensityBuffer );

						glActiveTexture( GL_TEXTURE3 );
						glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[write] );
						glActiveTexture( GL_TEXTURE4 );
						glBindTexture( GL_TEXTURE_2D, this->SilhouetteDepthBuffer[write] );

						double * bounds = this->GetInput()->GetBounds();

						this->ErosionProgram->bind();
						{
							std::pair<float,vtkColor4> fillPair = this->FillColors->at( i );
							float fillOpacity = fillPair.second.a / 255.0f;

							std::pair<float,vtkColor4> linePair = this->LineColors->at( i );
							float lineColor[4];
							lineColor[0] = linePair.second.r / 255.0f;
							lineColor[1] = linePair.second.g / 255.0f;
							lineColor[2] = linePair.second.b / 255.0f;
							lineColor[3] = 1.0f;

							const float * densityColor = this->GetDensityColor();

							this->ErosionProgram->setUniform( "erosionSampler", 0 );
							this->ErosionProgram->setUniform( "erosionDepthSampler", 1 );
							this->ErosionProgram->setUniform( "densitySampler", 2 );
							this->ErosionProgram->setUniform( "silhouetteSampler", 3 );
							this->ErosionProgram->setUniform( "silhouetteDepthSampler", 4 );
							this->ErosionProgram->setUniform( "depthTreshold", this->GetDepthThreshold() );
							this->ErosionProgram->setUniform( "pixelRatio", (1.0f / viewportSize[0]), (1.0f / viewportSize[1]) );
							this->ErosionProgram->setUniform( "fillErosion", this->GetFillDilation() );
							this->ErosionProgram->setUniform( "lineWidth", this->GetOutlineWidth() );
							this->ErosionProgram->setUniform( "minExtent", (float) bounds[0], (float) bounds[2], (float) bounds[4] );
							this->ErosionProgram->setUniform( "maxExtent", (float) bounds[1], (float) bounds[3], (float) bounds[5] );
							this->ErosionProgram->setUniform( "fillOpacity", fillOpacity );
							this->ErosionProgram->setUniform( "lineColor", lineColor[0], lineColor[1], lineColor[2], lineColor[3] );
							this->ErosionProgram->setUniform( "firstPass", (i == 0) );
							this->ErosionProgram->setUniform( "densityColor", densityColor[0], densityColor[1], densityColor[2], densityColor[3] );
							this->ErosionProgram->setUniform( "densityColoring", this->IsDensityColoringEnabled() );
							this->ErosionProgram->setUniform( "maxDensity", (float) this->GetMaximumFiberDensity() );

							glPushAttrib( GL_VIEWPORT_BIT | GL_SCISSOR_BIT );
							{
								glViewport( 0, 0, viewportSize[0], viewportSize[1] );
								glScissor ( 0, 0, viewportSize[0], viewportSize[1] );
								glDepthRange( 0, 1 );

								glDepthFunc( GL_LEQUAL );

								glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
								glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

								glBegin(GL_QUADS);
								{
									glTexCoord2f( 0, 0 ); glVertex3f( -1, -1, 0 );
									glTexCoord2f( 1, 0 ); glVertex3f(  1, -1, 0 );
									glTexCoord2f( 1, 1 ); glVertex3f(  1,  1, 0 );
									glTexCoord2f( 0, 1 ); glVertex3f( -1,  1, 0 );
								}
								glEnd();
							}
							glPopAttrib();
						}
						this->ErosionProgram->unbind();

						glActiveTexture( GL_TEXTURE4 );
						glBindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE3 );
						glBindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE2 );
						glBindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						glBindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
						glBindTexture( GL_TEXTURE_2D, 0 );
					}
					glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
				}

                // Swap read/write indices
                int tmp = read;
                read = write;
                write = tmp;
            }

            ///////////////////////////////////////////////////////////////////////////////////////
            // Render streamline color buffer to screen

            this->ToScreenProgram->bind();
            {
				if( this->IsErosionEnabled() )
				{
					glActiveTexture( GL_TEXTURE0 );
					glBindTexture( GL_TEXTURE_2D, this->ErosionBuffer[read] );
					glActiveTexture( GL_TEXTURE1 );
					glBindTexture( GL_TEXTURE_2D, this->ErosionDepthBuffer[read] );
				}
				else
				{
					glActiveTexture( GL_TEXTURE0 );
					glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[read] );
					glActiveTexture( GL_TEXTURE1 );
					glBindTexture( GL_TEXTURE_2D, this->SilhouetteDepthBuffer[read] );
				}

				this->ToScreenProgram->setUniform( "colorSampler", 0 );
                this->ToScreenProgram->setUniform( "depthSampler", 1 );

                glDepthFunc( GL_LEQUAL );

                glBegin(GL_QUADS);
                {
                    glTexCoord2f( 0, 0 ); glVertex3f( -1, -1, 0 );
                    glTexCoord2f( 1, 0 ); glVertex3f(  1, -1, 0 );
                    glTexCoord2f( 1, 1 ); glVertex3f(  1,  1, 0 );
                    glTexCoord2f( 0, 1 ); glVertex3f( -1,  1, 0 );
                }
                glEnd();

                glActiveTexture( GL_TEXTURE1 );
                glBindTexture( GL_TEXTURE_2D, 0 );
                glActiveTexture( GL_TEXTURE0 );
                glBindTexture( GL_TEXTURE_2D, 0 );
            }
            this->ToScreenProgram->unbind();
        }
        glPopAttrib();
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::RebuildDisplayList()
    {
        if( this->Table )
        {
            if( this->Table->IsNormalized() == false )
                this->Table->Normalize();
        }

        if( this->TableOriginal )
        {
            if( this->TableOriginal->IsNormalized() == false )
                this->TableOriginal->Normalize();
        }

		if( this->TableBundle )
		{
			if( this->TableBundle->IsNormalized() == false )
				this->TableBundle->Normalize();
		}

		//-----------------------------------------------------------------------------------
        // Render illuminated streamlines using Tim's mapper

        if( this->RenderList != 0 )
            glDeleteLists( this->RenderList, 1 );
        this->RenderList = glGenLists( 1 );

		glNewList( this->RenderList, GL_COMPILE );
		{
            this->RenderStreamlines( 1.0f );
		}
		glEndList();

        //-----------------------------------------------------------------------------------
        // Render confidence intervals

		if( this->ConfidenceLevels->size() == 0 )
		{
			this->RebuildList = false;
			return;
		}

		if( this->RenderLists != NULL )
		{
			for( unsigned int i = 0; i < this->PreviousNumberOfConfidenceLevels; i++ )
				if( this->RenderLists[i] != 0 )
					glDeleteLists( this->RenderLists[i], 1 );
			delete [] this->RenderLists;
		}

		this->RenderLists = new unsigned int[this->ConfidenceLevels->size()];
		for( unsigned int i = 0; i < this->ConfidenceLevels->size(); i++ )
        {
            this->RenderLists[i] = glGenLists( 1 );
        }

        // Sort the confidence levels and colors from small to large so that the
        // smallest is rendered first.
        std::sort( this->ConfidenceLevels->begin(),
                   this->ConfidenceLevels->end(), compareLevels );
        std::sort( this->FillColors->begin(),
                   this->FillColors->end(), compareColors );
        std::sort( this->LineColors->begin(),
                   this->LineColors->end(), compareColors );

        int k = 0;
        std::vector<std::pair<std::string,float> >::iterator i = this->ConfidenceLevels->begin();
        for( ; i != this->ConfidenceLevels->end(); i++ )
        {
            glNewList( this->RenderLists[k], GL_COMPILE );
            {
                float threshold = (*i).second;
                this->RenderStreamlines( threshold );
            }
            glEndList();
            k++;
        }

        this->RebuildList = false;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::RenderIlluminatedStreamlines( const float _opacity )
    {
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::RenderStreamlines( const float _threshold )
	{
        vtkCellArray * lines = this->GetInput()->GetLines();
        vtkPoints * points   = this->GetInput()->GetPoints();
        int nrLines          = this->GetInput()->GetNumberOfLines();

        // Select distance table depending on distance mode
        vtkDistanceTable * table = NULL;
        if( this->DistanceMode == 0 )
            table = this->Table;
		else if( this->DistanceMode == 1 )
            table = this->TableOriginal;
		else if( this->DistanceMode == 2 )
			table = this->TableBundle;
		else {}

        int cellIdx = 0;
        int nrElements = table ? table->GetNumberOfElements() : nrLines;

        for( int i = 0; i < nrElements; i++ )
        {
            int nrPtIds = 0;
            int * ptIds = NULL;

            if( table )
                lines->GetCell( table->GetCellIndex( i ), nrPtIds, ptIds );
            else
                lines->GetCell( cellIdx, nrPtIds, ptIds );
            cellIdx += (nrPtIds + 1);

            // Skip this iteration if the distance falls outside the required confidence
            // interval. This should only render the streamlines that are needed
            double distance = 0.0;

            if( table )
            {
				distance = i / (double) nrElements;
                if( distance > _threshold )
                    break;
            }

            glBegin( GL_LINES );
            {
                double previous[3];

                double current[3];
                points->GetPoint( ptIds[0], current );

                for( int j = 1; j < nrPtIds; j++ )
                {
                    previous[0] = current[0];
                    previous[1] = current[1];
                    previous[2] = current[2];

                    points->GetPoint( ptIds[j], current );

					// TODO: apply different colors for different confidence levels
                    glColor3f( this->FillColor[0], this->FillColor[1], this->FillColor[2] );
                    glVertex3dv( previous );
                    glVertex3dv( current );
                }
            }
            glEnd();
        }
	}

	////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::InitializeGraphicsResources( vtkViewport * _viewport )
	{
		int * windowSize = _viewport->GetSize();

		if( this->SilhouetteProgram == NULL )
		{
			glewInit();
			
			this->SilhouetteProgram = new opengl::GpuProgram;			

			this->SilhouetteProgram
				->createShader( "SilhouetteVertexShader", "", opengl::GpuShader::GST_VERTEX )
                ->setSourceCode( SilhouetteMap_VertexShaderCode );	

			this->SilhouetteProgram
				->createShader( "SilhouetteFragmentShader", "", opengl::GpuShader::GST_FRAGMENT )
                ->setSourceCode( SilhouetteMap_FragmentShaderCode );

            this->SilhouetteProgram->addVarying( "minScreenExtent" );
            this->SilhouetteProgram->addVarying( "maxScreenExtent" );

            if( ! this->SilhouetteProgram->build() )
			{
				std::stringstream errMsg( 
					std::stringstream::in | std::stringstream::out );
				errMsg << "Could not build silhouette program (" << this->SilhouetteProgram->getLastBuildLog() << std::endl;
				
				vtkErrorMacro( << errMsg.str().c_str() );
				return false;
			}
		}

        if( this->ErosionProgram == NULL )
        {
            this->ErosionProgram = new opengl::GpuProgram;

            this->ErosionProgram
                ->createShader( "ErosionVertexShader", "", opengl::GpuShader::GST_VERTEX )
                ->setSourceCode( Erosion_VertexShaderCode );

            this->ErosionProgram
                ->createShader( "ErosionFragmentShader", "", opengl::GpuShader::GST_FRAGMENT )
                ->setSourceCode( Erosion_FragmentShaderCode );

            if( ! this->ErosionProgram->build() )
            {
                std::stringstream errMsg(
                    std::stringstream::in | std::stringstream::out );
                errMsg << "Could not build erosion program (" << this->ErosionProgram->getLastBuildLog() << std::endl;

                vtkErrorMacro( << errMsg.str().c_str() );
                return false;
            }
        }

        if( this->DensityProgram == NULL )
        {
            this->DensityProgram = new opengl::GpuProgram;

            this->DensityProgram
                ->createShader( "DensityMapVertexShader", "", opengl::GpuShader::GST_VERTEX )
                ->setSourceCode( DensityMap_VertexShaderCode );

            this->DensityProgram
                ->createShader( "DensityMapFragmentShader", "", opengl::GpuShader::GST_FRAGMENT )
                ->setSourceCode( DensityMap_FragmentShaderCode );

            if( ! this->DensityProgram->build() )
            {
                std::stringstream errMsg(
                    std::stringstream::in | std::stringstream::out );
                errMsg << "Could not build silhouette program (" << this->DensityProgram->getLastBuildLog() << std::endl;

                vtkErrorMacro( << errMsg.str().c_str() );
                return false;
            }
        }

        if( this->DensitySmoothingProgram == NULL )
        {
            this->DensitySmoothingProgram = new opengl::GpuProgram;

            this->DensitySmoothingProgram
                    ->createShader( "DensitySmoothingVertexShader", "", opengl::GpuShader::GST_VERTEX )
                    ->setSourceCode( DensitySmoothing_VertexShaderCode );

            this->DensitySmoothingProgram
                    ->createShader( "DensitySmoothingFragmentShader", "", opengl::GpuShader::GST_FRAGMENT )
                    ->setSourceCode( DensitySmoothing_FragmentShaderCode );

            if( ! this->DensitySmoothingProgram->build() )
            {
                std::stringstream errMsg(
                    std::stringstream::in | std::stringstream::out );
                errMsg << "Could not build density smoothing program (" << this->DensitySmoothingProgram->getLastBuildLog() << std::endl;

                vtkErrorMacro( << errMsg.str().c_str() );
                return false;
            }
        }

        if( this->ToScreenProgram == NULL )
        {
            this->ToScreenProgram = new opengl::GpuProgram;

            this->ToScreenProgram
                ->createShader( "ToScreenVertexShader", "", opengl::GpuShader::GST_VERTEX )
                ->setSourceCode( ToScreen_VertexShaderCode );

            this->ToScreenProgram
                ->createShader( "ToScreenFragmentShader", "", opengl::GpuShader::GST_FRAGMENT )
                ->setSourceCode( ToScreen_FragmentShaderCode );

            if( ! this->ToScreenProgram->build() )
            {
                std::stringstream errMsg(
                    std::stringstream::in | std::stringstream::out );
                errMsg << "Could not build to-screen program (" << this->ToScreenProgram->getLastBuildLog() << std::endl;

                vtkErrorMacro( << errMsg.str().c_str() );
                return false;
            }
        }

        if( this->FrameBuffer > 0 )
        {
            glDeleteFramebuffersEXT( 1, & this->FrameBuffer );
            glDeleteTextures( 1, & this->ColorBuffer );
            glDeleteTextures( 1, & this->DepthBuffer );
            glDeleteTextures( 1, & this->DensityBuffer );
            glDeleteTextures( 1, & this->DensityDepthBuffer );
            glDeleteTextures( 1, & this->DensitySmoothingBuffer );
            glDeleteTextures( 1, & this->DensitySmoothingDepthBuffer );
            glDeleteTextures( 2, this->SilhouetteBuffer );
            glDeleteTextures( 2, this->SilhouetteDepthBuffer );
            glDeleteTextures( 2, this->ErosionBuffer );
            glDeleteTextures( 2, this->ErosionDepthBuffer );

            this->FrameBuffer = 0;
            this->ColorBuffer = 0;
            this->DepthBuffer = 0;
            this->DensityBuffer = 0;
            this->DensityDepthBuffer = 0;
            this->DensitySmoothingBuffer = 0;
            this->DensitySmoothingDepthBuffer = 0;
            this->ErosionBuffer[0] = 0;
            this->ErosionBuffer[1] = 0;
            this->ErosionDepthBuffer[0] = 0;
            this->ErosionDepthBuffer[1] = 0;
            this->SilhouetteBuffer[0] = 0;
            this->SilhouetteBuffer[1] = 0;
            this->SilhouetteDepthBuffer[0] = 0;
            this->SilhouetteDepthBuffer[1] = 0;
        }

        if( this->FrameBuffer == 0 )
        {
            glActiveTexture( GL_TEXTURE0 );

            int * windowSize = _viewport->GetSize();

            glGenTextures( 1, & this->ColorBuffer );
            glBindTexture( GL_TEXTURE_2D, this->ColorBuffer );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ); // Is nearest-neighbor sufficient?
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glGenTextures( 1, & this->DepthBuffer );
            glBindTexture( GL_TEXTURE_2D, this->DepthBuffer );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

            glGenTextures( 1, & this->DensityBuffer );
            glBindTexture( GL_TEXTURE_2D, this->DensityBuffer );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glGenTextures( 1, & this->DensityDepthBuffer );
            glBindTexture( GL_TEXTURE_2D, this->DensityDepthBuffer );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

            glGenTextures( 1, & this->DensitySmoothingBuffer );
            glBindTexture( GL_TEXTURE_2D, this->DensitySmoothingBuffer );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glGenTextures( 1, & this->DensitySmoothingDepthBuffer );
            glBindTexture( GL_TEXTURE_2D, this->DensitySmoothingDepthBuffer );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

            glGenTextures( 2, this->ErosionBuffer );

            glBindTexture( GL_TEXTURE_2D, this->ErosionBuffer[0] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glBindTexture( GL_TEXTURE_2D, this->ErosionBuffer[1] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glGenTextures( 2, this->ErosionDepthBuffer );

            glBindTexture( GL_TEXTURE_2D, this->ErosionDepthBuffer[0] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

            glBindTexture( GL_TEXTURE_2D, this->ErosionDepthBuffer[1] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

			glGenTextures( 2, this->SilhouetteBuffer );

            glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[0] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[1] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize[0], windowSize[1], 0, GL_RGBA, GL_FLOAT, 0 );

            glGenTextures( 2, this->SilhouetteDepthBuffer );

            glBindTexture( GL_TEXTURE_2D, this->SilhouetteDepthBuffer[0] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

            glBindTexture( GL_TEXTURE_2D, this->SilhouetteDepthBuffer[1] );
            glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, windowSize[0], windowSize[1], 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

            glGenFramebuffersEXT( 1, & this->FrameBuffer );
            glBindFramebufferEXT( GL_FRAMEBUFFER_EXT, 0 );
        }

		return true;
	}
	
	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::ReleaseGraphicsResources( vtkWindow * _window )
	{
		if( this->SilhouetteProgram != NULL )
			delete this->SilhouetteProgram;
		
		this->SilhouetteProgram = NULL;

        if( this->FrameBuffer > 0 )
        {
            glDeleteFramebuffersEXT( 1, & this->FrameBuffer );
            glDeleteTextures( 1, & this->ColorBuffer );
            glDeleteTextures( 1, & this->DepthBuffer );
            glDeleteTextures( 1, & this->DensityBuffer );
            glDeleteTextures( 1, & this->DensityDepthBuffer );
            glDeleteTextures( 1, & this->DensitySmoothingBuffer );
            glDeleteTextures( 1, & this->DensitySmoothingDepthBuffer );
            glDeleteTextures( 2, this->SilhouetteBuffer );
            glDeleteTextures( 2, this->SilhouetteDepthBuffer );
            glDeleteTextures( 2, this->ErosionBuffer );
            glDeleteTextures( 2, this->ErosionDepthBuffer );

			this->FrameBuffer = 0;
			this->ColorBuffer = 0;
			this->DepthBuffer = 0;
			this->DensityBuffer = 0;
			this->DensityDepthBuffer = 0;
            this->DensitySmoothingBuffer = 0;
            this->DensitySmoothingDepthBuffer = 0;
            this->ErosionBuffer[0] = 0;
            this->ErosionBuffer[1] = 0;
            this->ErosionDepthBuffer[0] = 0;
            this->ErosionDepthBuffer[1] = 0;
            this->SilhouetteBuffer[0] = 0;
            this->SilhouetteBuffer[1] = 0;
            this->SilhouetteDepthBuffer[0] = 0;
            this->SilhouetteDepthBuffer[1] = 0;
		}
	}
	
    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDistanceTable( vtkDistanceTable * _table )
	{
		if( this->Table )
			this->Table->UnRegister( this );
		this->Table = _table;
		if( this->Table )
			this->Table->Register( this );
		
		this->RebuildList = true;
	}

	////////////////////////////////////////////////////////////////////////
    vtkDistanceTable * vtkFiberConfidenceMapper::GetDistanceTable()
	{
		return this->Table;
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDistanceTableOriginalFibers( vtkDistanceTable * _table )
    {
        if( this->TableOriginal )
            this->TableOriginal->UnRegister( this );
        this->TableOriginal = _table;
        if( this->TableOriginal )
            this->TableOriginal->Register( this );

        this->RebuildList = true;
    }

    ////////////////////////////////////////////////////////////////////////
    vtkDistanceTable * vtkFiberConfidenceMapper::GetDistanceTableOriginalFibers()
    {
        return this->TableOriginal;
    }

	////////////////////////////////////////////////////////////////////////
	void vtkFiberConfidenceMapper::SetDistanceTableBundle( vtkDistanceTable * _table )
	{
		if( this->TableBundle )
			this->TableBundle->UnRegister( this );
		this->TableBundle = _table;
		if( this->TableBundle )
			this->TableBundle->Register( this );

		this->RebuildList = true;
	}

	////////////////////////////////////////////////////////////////////////
	vtkDistanceTable * vtkFiberConfidenceMapper::GetDistanceTableBundle()
	{
		return this->TableBundle;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetInterval( float _minimum, float _maximum )
	{
		assert( _minimum <= _maximum );
		assert( _minimum >= 0.0f && _minimum <= 1.0f );
		assert( _maximum >= 0.0f && _maximum <= 1.0f );

		this->Interval[0] = _minimum;
		this->Interval[1] = _maximum;
		
		this->RebuildList = true;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::GetInterval( float & _minimum, float & _maximum )
	{
		_minimum = this->Interval[0];
		_maximum = this->Interval[1];
	}

	////////////////////////////////////////////////////////////////////////
    float * vtkFiberConfidenceMapper::GetInterval()
	{
		return this->Interval;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetSilhouetteEnabled( bool _enabled )
	{
		if( this->SilhouetteEnabled != _enabled )
			this->RebuildList = true;
			
		this->SilhouetteEnabled = _enabled;
	}

	////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsSilhouetteEnabled()
	{
		return this->SilhouetteEnabled;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetSilhouetteStreamlinesEnabled( bool _enabled )
	{
		if( this->SilhouetteStreamlinesEnabled != _enabled )
			this->RebuildList = true;
			
		this->SilhouetteStreamlinesEnabled = _enabled;
	}

	////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsSilhouetteStreamlinesEnabled()
	{
		return this->SilhouetteStreamlinesEnabled;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetFillColor( const float _red, const float _green, const float _blue )
	{
		assert( _red >= 0.0f && _red <= 1.0f );
		assert( _green >= 0.0f && _green <= 1.0f );
		assert( _blue >= 0.0f && _blue <= 1.0f );

		this->FillColor[0] = _red;
		this->FillColor[1] = _green;
		this->FillColor[2] = _blue;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetFillColor( const float _color[3] )
	{
		this->SetFillColor( _color[0], _color[1], _color[2] );
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::GetFillColor( float & _red, float & _green, float & _blue ) const
	{
		_red   = this->FillColor[0];
		_green = this->FillColor[1];
		_blue  = this->FillColor[2];
	}

	////////////////////////////////////////////////////////////////////////
    const float * vtkFiberConfidenceMapper::GetFillColor() const
	{
		return this->FillColor;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetFillOpacity( const float _opacity )
	{
		assert( _opacity >= 0.0f && _opacity <= 1.0f );
		this->FillOpacity = _opacity;
	}

	////////////////////////////////////////////////////////////////////////
    const float vtkFiberConfidenceMapper::GetFillOpacity() const
	{
		return this->FillOpacity;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetOutlineColor( const float _red, const float _green, const float _blue )
	{
		assert( _red >= 0.0f && _red <= 1.0f );
		assert( _green >= 0.0f && _green <= 1.0f );
		assert( _blue >= 0.0f && _blue <= 1.0f );

		this->OutlineColor[0] = _red;
		this->OutlineColor[1] = _green;
		this->OutlineColor[2] = _blue;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetOutlineColor( const float _color[3] )
	{
		this->SetOutlineColor( _color[0], _color[1], _color[2] );
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::GetOutlineColor( float & _red, float & _green, float & _blue ) const
	{
		_red   = this->OutlineColor[0];
		_green = this->OutlineColor[1];
		_blue  = this->OutlineColor[2];
	}

	////////////////////////////////////////////////////////////////////////
    const float * vtkFiberConfidenceMapper::GetOutlineColor() const
	{
		return this->OutlineColor;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDepthThreshold( const float _threshold )
	{
		assert( _threshold >= 0.0f );
		this->DepthThreshold = _threshold;
	}

	////////////////////////////////////////////////////////////////////////
    float vtkFiberConfidenceMapper::GetDepthThreshold() const
	{
		return this->DepthThreshold;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetFillDilation( const int _size )
	{
		assert( _size >= 0 );
		this->FillDilation = _size;
	}

	////////////////////////////////////////////////////////////////////////
    int vtkFiberConfidenceMapper::GetFillDilation() const
	{
		return this->FillDilation;
	}

	////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetOutlineWidth( const int _width )
	{
		assert( _width >= 0 );
		this->OutlineWidth = _width;
	}

	////////////////////////////////////////////////////////////////////////
    int vtkFiberConfidenceMapper::GetOutlineWidth() const
	{
		return this->OutlineWidth;
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetMaximumFiberDensity( const int _density )
    {
        assert( _density >= 0 );
        this->MaximumFiberDensity = _density;
    }

    ////////////////////////////////////////////////////////////////////////
    const int vtkFiberConfidenceMapper::GetMaximumFiberDensity() const
    {
        return this->MaximumFiberDensity;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetSmoothingKernelSize( const int _size )
    {
        assert( _size > 0 );
        this->SmoothingKernelSize = _size;
    }

    ////////////////////////////////////////////////////////////////////////
    const int vtkFiberConfidenceMapper::GetSmoothingKernelSize() const
    {
        return this->SmoothingKernelSize;
    }

    ////////////////////////////////////////////////////////////////////////
	void vtkFiberConfidenceMapper::SetConfidenceLevels( std::vector<std::pair<std::string,float> > * _levels )
	{
//        if( this->ConfidenceLevels == NULL )
//            this->ConfidenceLevels = new std::vector<std::pair<std::string,float> >;
//
//		if( this->ConfidenceLevels )
//        {
//			this->PreviousNumberOfConfidenceLevels = this->ConfidenceLevels->size();
//            this->ConfidenceLevels->clear();
//        }
//
//        // Copy the list so we can resort it without conflicts
//        for( unsigned int i = 0; i < _levels->size(); i++ )
//        {
//            std::pair<std::string,float> newPair = _levels->at( i );
//            this->ConfidenceLevels->push_back( newPair );
//        }

		this->ConfidenceLevels = _levels;
		this->RebuildList = true;
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetFillColors( std::vector<std::pair<float,vtkColor4> > * _colors )
	{
//        if( this->FillColors == NULL )
//            this->FillColors = new std::vector<std::pair<float,vtkColor4> >;
//
//        if( this->FillColors )
//            this->FillColors->clear();
//
//        for( unsigned int i = 0; i < _colors->size(); i++ )
//        {
//            std::pair<float,vtkColor4> newPair = _colors->at( i );
//            this->FillColors->push_back( newPair );
//        }

		this->FillColors = _colors;
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetLineColors( std::vector<std::pair<float,vtkColor4> > * _colors )
	{
//        if( this->LineColors == NULL )
//            this->LineColors = new std::vector<std::pair<float,vtkColor4> >;
//
//        if( this->LineColors )
//            this->LineColors->clear();
//
//        for( unsigned int i = 0; i < _colors->size(); i++ )
//        {
//            std::pair<float,vtkColor4> newPair = _colors->at( i );
//            this->LineColors->push_back( newPair );
//        }

		this->LineColors = _colors;
    }

    ////////////////////////////////////////////////////////////////////////
	void vtkFiberConfidenceMapper::SetDistanceModeRelativeToMedianFiber()
	{
		this->DistanceMode = 0;
		this->RebuildList = true;
	}

    ////////////////////////////////////////////////////////////////////////
	void vtkFiberConfidenceMapper::SetDistanceModeRelativeToFiberOriginalDataset()
	{
		this->DistanceMode = 1;
		this->RebuildList = true;
	}

    ////////////////////////////////////////////////////////////////////////
	void vtkFiberConfidenceMapper::SetDistanceModeRelativeToBundleCenter()
	{
		this->DistanceMode = 2;
		this->RebuildList = true;
	}

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDensityColor( const float _red, const float _green, const float _blue, const float _alpha )
    {
        this->DensityColor[0] = _red;
        this->DensityColor[1] = _green;
        this->DensityColor[2] = _blue;
        this->DensityColor[3] = _alpha;
    }

    ////////////////////////////////////////////////////////////////////////
    const float * vtkFiberConfidenceMapper::GetDensityColor() const
    {
        return this->DensityColor;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDensityColoringEnabled( bool _enabled )
    {
        this->DensityColoringEnabled = _enabled;
    }

    ////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsDensityColoringEnabled()
    {
        return this->DensityColoringEnabled;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDensitySmoothingEnabled( bool _enabled )
    {
        this->DensitySmoothingEnabled = _enabled;
    }

    ////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsDensitySmoothingEnabled()
    {
        return this->DensitySmoothingEnabled;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetDensityWeightingEnabled( bool _enabled )
    {
        this->DensityWeightingEnabled = _enabled;
    }

    ////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsDensityWeightingEnabled()
    {
        return this->DensityWeightingEnabled;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetOverwriteEnabled( bool _enabled )
    {
        this->OverwriteEnabled = _enabled;
    }

    ////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsOverwriteEnabled()
    {
        return this->OverwriteEnabled;
    }

    ////////////////////////////////////////////////////////////////////////
    void vtkFiberConfidenceMapper::SetErosionEnabled( bool _enabled )
    {
        this->ErosionEnabled = _enabled;
    }

    ////////////////////////////////////////////////////////////////////////
    bool vtkFiberConfidenceMapper::IsErosionEnabled()
    {
        return this->ErosionEnabled;
    }
} // namespace bmia
