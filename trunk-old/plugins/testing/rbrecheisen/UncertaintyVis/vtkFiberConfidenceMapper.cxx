#include <vtkFiberConfidenceMapper.h>
#include <vtkFiberConfidenceMapperShaders.h>
#include <vtkConfidenceTable.h>
#include <vtkConfidenceInterval.h>
#include <vtkConfidenceIntervalProperties.h>

#include <vtkOpenGLExtensionManager.h>
#include <vtkTimerLog.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkViewport.h>
#include <vtkCamera.h>
#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkMath.h>
#include <vtkgl.h>

#include <vector>
#include <string>

vtkCxxRevisionMacro( vtkFiberConfidenceMapper, "$Revision: 1.0 $");
vtkStandardNewMacro( vtkFiberConfidenceMapper );

///////////////////////////////////////////////////////////////////////////
vtkFiberConfidenceMapper::vtkFiberConfidenceMapper()
{
	this->ShadersInitialized = false;
	this->TexturesInitialized = false;
	this->ExtensionsInitialized = false;
	this->Orthographic = false;
	this->NumberOfDisplayLists = 0;
	this->ScreenSize[0] = 0;
	this->ScreenSize[1] = 0;
	this->ROI[0] = 0;
	this->ROI[1] = 0;
	this->ROI[2] = 0;
	this->ROI[3] = 0;
	this->ROIEnabled = false;
	this->DepthThreshold = 5.0;
	this->DepthNear = 0.0;
	this->DepthFar = 1.0;
	this->DisplayLists = 0;
	this->VertexShader = 0;
	this->BlurringFragShader = 0;
	this->BlurringProgram = 0;
	this->OutputFragShader = 0;
	this->OutputProgram = 0;
	this->FragShader = 0;
	this->Program = 0;
	this->FBO = 0;
	this->StreamlineBuffer[0] = 0;
	this->StreamlineBuffer[1] = 0;
	this->SilhouetteBuffer[0] = 0;
	this->SilhouetteBuffer[1] = 0;
	this->BlurringBuffer[0] = 0;
	this->BlurringBuffer[1] = 0;
	this->Table = 0;
	this->Interval = 0;
    this->Ids = 0;
    this->Scores = 0;

	this->Mode = RENDERMODE_SOLID;
}

///////////////////////////////////////////////////////////////////////////
vtkFiberConfidenceMapper::~vtkFiberConfidenceMapper()
{
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::Render( vtkRenderer * renderer, vtkActor * actor )
{
	if( this->GetInput() == 0 || this->GetInterval() == 0 )
	{
		return;
	}

    renderer->GetRenderWindow()->MakeCurrent();

	if( this->InitializeGraphicsResources( renderer ) == false )
	{
		std::cout << "vtkFiberConfidenceMapper::Render() could not initialize mapper" << std::endl;
		this->ReleaseGraphicsResources( renderer->GetVTKWindow() );
		return;
	}

	if( this->DisplayLists == 0 || this->GetInterval()->HasChanged() )
	{
		this->RebuildDisplayLists();

		if( this->GetInterval()->HasChanged() )
		{
			this->GetInterval()->SetChanged( false );
		}
	}

	this->vtkOpenGLPolyDataMapper::Render( renderer, actor );
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::RenderPiece( vtkRenderer * renderer, vtkActor * actor )
{
    renderer->GetActiveCamera()->GetClippingRange( this->DepthNear, this->DepthFar );

	this->Timer->StartTimer();

	this->Orthographic = renderer->GetActiveCamera()->GetParallelProjection() ? true : false;

	int width  = this->ScreenSize[0];
	int height = this->ScreenSize[1];

	// Clipping?

	glPushAttrib( GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT );
	{
		glEnable ( GL_DEPTH_TEST );
		glEnable ( GL_CULL_FACE );
		glDisable( GL_BLEND );
		glDisable( GL_LIGHTING );
		glDisable( GL_ALPHA_TEST );
		glDisable( GL_STENCIL_TEST );

		glStencilFunc( GL_EQUAL, 0, 1 );
		glStencilOp  ( GL_KEEP, GL_KEEP, GL_INCR );
		glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
		glDepthFunc  ( GL_LESS );
		glFrontFace  ( GL_CCW );
		glCullFace   ( GL_BACK );

		glPushAttrib( GL_SCISSOR_BIT | GL_VIEWPORT_BIT );
		{
			glViewport( 0, 0, width, height );
			glScissor ( 0, 0, width, height );

			if( this->IsROIEnabled() )
			{
				glScissor(
					this->ROI[0],
					this->ROI[1],
					this->ROI[2], this->ROI[3] );
			}

			glEnable( GL_BLEND );

			glDepthRange  ( 0, 1 );
			glClearDepth  ( 1 );
			glClearStencil( 0 );
			glClearColor  ( 0, 0, 0, 0 );

			bool first = true;

			bool blurringEnabled = 
				this->GetInterval()->GetProperties()->IsBlurringEnabled();

			int index = blurringEnabled ? 
				0 : this->GetInterval()->GetNumberOfIntervals() - 1;
			int indexStop = blurringEnabled ?
				this->GetInterval()->GetNumberOfIntervals() : -1;
			int delta = blurringEnabled ? 1 : -1;

			for( ; index != indexStop; index += delta )
			//for( int index = 0; index < this->GetInterval()->GetNumberOfIntervals(); ++index )
			//for( int index = this->GetInterval()->GetNumberOfIntervals() - 1; index >= 0; --index )
			{
				if( this->GetInterval()->GetProperties()->IsEnabled( index ) == false )
				{
					continue;
				}

				// Render streamlines

				vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, this->FBO );
				{
					glBindTexture( GL_TEXTURE_2D, 0 );
					vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT,
							GL_TEXTURE_2D, this->StreamlineBuffer[0], 0 );
					vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT,
							GL_TEXTURE_2D, this->StreamlineBuffer[1], 0 );
					vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::STENCIL_ATTACHMENT_EXT,
							GL_TEXTURE_2D, 0, 0 );

					glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
					glCallList( this->DisplayLists[index] );
				}
				vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, 0 );

                // Render silhouettes

                vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, this->FBO );
                {
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT,
                            GL_TEXTURE_2D, this->SilhouetteBuffer[0], 0 );
                    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT,
                            GL_TEXTURE_2D, this->SilhouetteBuffer[1], 0 );

                    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

                    vtkgl::UseProgram( this->Program );
                    {
                        this->ApplyParameters( index, this->Program );

                        glBegin( GL_QUADS );
                        glTexCoord2f( 0, 0 ); glVertex3f( -1.0f, -1.0f, 0.0f );
                        glTexCoord2f( 1, 0 ); glVertex3f(  1.0f, -1.0f, 0.0f );
                        glTexCoord2f( 1, 1 ); glVertex3f(  1.0f,  1.0f, 0.0f );
                        glTexCoord2f( 0, 1 ); glVertex3f( -1.0f,  1.0f, 0.0f );
                        glEnd();

                        glBindTexture( GL_TEXTURE_2D, 0 );
                    }
                    vtkgl::UseProgram( 0 );
                }
                vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, 0 );

                // Do blurring

                vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, this->FBO );
                {
                    glBindTexture( GL_TEXTURE_2D, 0 );
                    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::COLOR_ATTACHMENT0_EXT,
                            GL_TEXTURE_2D, this->BlurringBuffer[0], 0 );
                    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::DEPTH_ATTACHMENT_EXT,
                            GL_TEXTURE_2D, this->BlurringBuffer[1], 0 );
                    vtkgl::FramebufferTexture2DEXT( vtkgl::FRAMEBUFFER_EXT, vtkgl::STENCIL_ATTACHMENT_EXT,
                            GL_TEXTURE_2D, this->BlurringBuffer[1], 0 );

                    // Do a stencil test only when the intervals are rendered in reverse order
                    // (from high to low confidence). In that case, if you reduce the opacity
                    // of one interval is will not mix with the underlying interval, resulting
                    // in unwanted colors. However, the blurring in this case causes ugly edges.
                    // So, when blurring is enable, we render the interval is normal order (from
                    // low to high confidence)

                    if( blurringEnabled )
                    glDepthFunc( GL_ALWAYS );
                    else
                    glEnable( GL_STENCIL_TEST );

                    if( first )
                    {
                        glClear( GL_COLOR_BUFFER_BIT |
                                 GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );
                        first = false;
                    }

                    vtkgl::UseProgram( this->BlurringProgram );
                    {
                        this->ApplyBlurringParameters( index, this->BlurringProgram );

                        glBegin( GL_QUADS );
                        glTexCoord2f( 0, 0 ); glVertex3f( -1.0f, -1.0f, 0.0f );
                        glTexCoord2f( 1, 0 ); glVertex3f(  1.0f, -1.0f, 0.0f );
                        glTexCoord2f( 1, 1 ); glVertex3f(  1.0f,  1.0f, 0.0f );
                        glTexCoord2f( 0, 1 ); glVertex3f( -1.0f,  1.0f, 0.0f );
                        glEnd();

                        glBindTexture( GL_TEXTURE_2D, 0 );
                    }
                    vtkgl::UseProgram( 0 );

                    if( blurringEnabled )
                    glDepthFunc( GL_LESS );
                    else
                    glDisable( GL_STENCIL_TEST );
					
                }
                vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, 0 );
			}

			vtkgl::UseProgram( this->OutputProgram );
			{
				this->ApplyOutputParameters( this->OutputProgram );

				glBegin( GL_QUADS );
				glTexCoord2f( 0, 0 ); glVertex3f( -1.0f, -1.0f, 0.0f );
				glTexCoord2f( 1, 0 ); glVertex3f(  1.0f, -1.0f, 0.0f );
				glTexCoord2f( 1, 1 ); glVertex3f(  1.0f,  1.0f, 0.0f );
				glTexCoord2f( 0, 1 ); glVertex3f( -1.0f,  1.0f, 0.0f );
				glEnd();

				glBindTexture( GL_TEXTURE_2D, 0 );
			}
			vtkgl::UseProgram( 0 );
		}
		glPopAttrib();
	}
	glPopAttrib();

	this->Timer->StopTimer();
	this->TimeToDraw = this->Timer->GetElapsedTime();
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::ApplyParameters( int index, unsigned int progId )
{
	vtkConfidenceIntervalProperties * properties = this->GetInterval()->GetProperties();

	int width  = this->ScreenSize[0];
	int height = this->ScreenSize[1];

	vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, this->StreamlineBuffer[1] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "depthBuffer" ), 0 );

	float pixelRatio[] = { 1.0f / width, 1.0f / height };
	vtkgl::Uniform2f( vtkgl::GetUniformLocation( progId, "pixelRatio" ),
			pixelRatio[0], pixelRatio[1] );

	float * color = properties->GetColor( index );
	float   opacity = properties->GetOpacity( index );
	vtkgl::Uniform4f( vtkgl::GetUniformLocation( progId, "color" ), 
			color[0], 
			color[1], 
			color[2], opacity );

	float * outlineColor = properties->GetOutlineColor( index );
	float outlineOpacity = properties->GetOutlineOpacity( index );
	vtkgl::Uniform4f( vtkgl::GetUniformLocation( progId, "outlineColor" ), 
			outlineColor[0], 
			outlineColor[1], 
			outlineColor[2], outlineOpacity );

	float * outlineThicknessRange = properties->GetOutlineThicknessRange();
	int outlineThickness = static_cast< int >(
			properties->GetOutlineThickness( index ) *
					(outlineThicknessRange[1] - outlineThicknessRange[0]) + outlineThicknessRange[0] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "outlineThickness" ), outlineThickness );

	float * dilationRange = properties->GetDilationRange();
	int dilation = static_cast< int >( 
			properties->GetDilation( index ) *
					(dilationRange[1] - dilationRange[0]) + dilationRange[0] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "dilation" ), dilation );

	float * checkerSizeRange = properties->GetCheckerSizeRange();
	int checkerSize = static_cast< int >(
			properties->GetCheckerSize( index ) *
					(checkerSizeRange[1] - checkerSizeRange[0]) + checkerSizeRange[0] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "checkerSize" ), checkerSize );

	float * holeSizeRange = properties->GetHoleSizeRange();
	int holeSize = static_cast< int >(
			properties->GetHoleSize( index ) *
					(holeSizeRange[1] - holeSizeRange[0]) + holeSizeRange[0] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "holeSize" ), holeSize );

	vtkgl::Uniform1f( vtkgl::GetUniformLocation( progId, "depthThreshold" ), this->DepthThreshold );
	vtkgl::Uniform1f( vtkgl::GetUniformLocation( progId, "depthNear" ), this->DepthNear );
	vtkgl::Uniform1f( vtkgl::GetUniformLocation( progId, "depthFar" ), this->DepthFar );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "ortho" ), this->Orthographic );
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::ApplyBlurringParameters( int index, unsigned int progId )
{
	vtkConfidenceIntervalProperties * properties = this->GetInterval()->GetProperties();

	int width  = this->ScreenSize[0];
	int height = this->ScreenSize[1];

	vtkgl::ActiveTexture( vtkgl::TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[1] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "depthBuffer" ), 1 );

	vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[0] );
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "colorBuffer" ), 0 );

	float pixelRatio[] = { 1.0f / width, 1.0f / height };
	vtkgl::Uniform2f( vtkgl::GetUniformLocation( progId, "pixelRatio" ),
			pixelRatio[0], pixelRatio[1] );

	int blurringRadius = 0;
	if( properties->IsBlurringEnabled() )
	{
		float * blurringRadiusRange = properties->GetBlurringRadiusRange();
		blurringRadius = static_cast< int >( 
				properties->GetBlurringRadius( index ) *
						(blurringRadiusRange[1] - blurringRadiusRange[0]) + blurringRadiusRange[0] );
	}
	vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "blurringRadius" ), blurringRadius );

	float * blurringBrightnessRange = properties->GetBlurringBrightnessRange();
	float blurringBrightness = properties->GetBlurringBrightness( index ) *
					(blurringBrightnessRange[1] - blurringBrightnessRange[0]) + blurringBrightnessRange[0];
	vtkgl::Uniform1f( vtkgl::GetUniformLocation( progId, "blurringBrightness" ), blurringBrightness );
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::ApplyOutputParameters( unsigned int progId )
{
	//vtkgl::ActiveTexture( vtkgl::TEXTURE2 );
	//glBindTexture( GL_TEXTURE_2D, this->DensityBuffer[0] );
	//vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "densityBuffer" ), 2 );

    vtkgl::ActiveTexture( vtkgl::TEXTURE1 );
    glBindTexture( GL_TEXTURE_2D, this->BlurringBuffer[1] );
    vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "depthBuffer" ), 1 );

    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, this->BlurringBuffer[0] );
    vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "colorBuffer" ), 0 );

//    vtkgl::ActiveTexture( vtkgl::TEXTURE1 );
//    glBindTexture( GL_TEXTURE_2D, this->StreamlineBuffer[1] );
//    vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "depthBuffer" ), 1 );

//    vtkgl::ActiveTexture( vtkgl::TEXTURE0 );
//    glBindTexture( GL_TEXTURE_2D, this->StreamlineBuffer[0] );
//    vtkgl::Uniform1i( vtkgl::GetUniformLocation( progId, "colorBuffer" ), 0 );
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::RebuildDisplayLists()
{
	if( this->GetInterval() == 0 )
	{
		return;
	}

	int nrIntervals = this->GetInterval()->GetNumberOfIntervals();
	if( this->DisplayLists )
	{
		for( int i = 0; i < this->NumberOfDisplayLists; ++i )
			glDeleteLists( this->DisplayLists[i], 1 );
		delete [] this->DisplayLists;
		this->DisplayLists = 0;
	}

	// Walk through the list of intervals and create a display
	// list for each one. Then render the streamlines associated
	// with each interval into the corresponding display list

	this->DisplayLists = new unsigned int[nrIntervals];
	for( int i = 0; i < nrIntervals; ++i )
	{
		float * interval = this->GetInterval()->GetInterval( i );
		this->DisplayLists[i] = glGenLists( 1 );
		glNewList( this->DisplayLists[i], GL_COMPILE );
			this->RenderStreamlines( interval[0], interval[1] );
		glEndList();
	}

	this->NumberOfDisplayLists = nrIntervals;
}

void vtkFiberConfidenceMapper::RenderStreamlines( float min, float max )
{
    if(!this->Scores)
        return;
    glDisable(GL_LIGHTING);
    vtkCellArray * cells = this->GetInput()->GetLines();
    vtkPoints * points =this->GetInput()->GetPoints();
    vtkIdType nrPtIds, * ptIds, cellIdx = 0;
    cells->InitTraversal();
    while(cells->GetNextCell(nrPtIds, ptIds))
    {
        float score = this->Scores->at(cellIdx++);
        if(score < min || score > max)
            continue;
        double P[3], Q[3];
        points->GetPoint(ptIds[0], P);
        for(int i = 1; i < nrPtIds; ++i)
        {
            points->GetPoint(ptIds[i], Q);
            glBegin(GL_LINES);
                glVertex3dv(P);
                glVertex3dv(Q);
            glEnd();
            for(int j = 0; j < 3; ++j)
                P[j] = Q[j];
        }
    }
    glEnable(GL_LIGHTING);
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::RenderStreamlines2( float min, float max )
{
    if( this->GetInput() == 0)// || this->GetTable() == 0 )
	{
		return;
	}

	vtkCellArray * lines = this->GetInput()->GetLines();
	vtkPoints * points = this->GetInput()->GetPoints();
    int nrLines = this->Ids->size(); //this->GetTable()->GetNumberOfValues();
	if( nrLines != lines->GetNumberOfCells() )
	{
		std::cout << "vtkConfidenceIntervalMapper::RenderStreamlines() number " \
				"of table entries does not match number of lines in polydata" << std::endl;
		return;
	}

//	if( this->GetTable()->IsNormalized() == false )
//	{
//		this->GetTable()->Normalize();
//	}

	glDisable( GL_LIGHTING );

    for( int i = 0; i < nrLines; ++i )
	{
        vtkIdType nrPtIds, * ptIds;
        //vtkIdType id = this->GetTable()->GetIdAt( i );
        //lines->GetCell( id, nrPtIds, ptIds );
        lines->GetCell( i, nrPtIds, ptIds );

        std::cout << "CELL " << i << ", nrPtIds " << nrPtIds << std::endl;

		// If the confidence is larger than the maximum, continue with
		// the next iteration. If it is smaller than the maximum, the
		// loop should quit.

        //float confidence = this->GetTable()->GetConfidenceAt( i );
        float confidence = this->Scores->at(i);
		if( confidence > max )
			continue;
		if( confidence < min )
			break;

        double previous[3], current[3];
        points->GetPoint(ptIds[0], previous);

        std::cout << ptIds[0] << " ";

        for(int j = 1; j < nrPtIds; ++j)
        {
            points->GetPoint(ptIds[j], current);
            double x = previous[0] - current[0];
            double y = previous[1] - current[1];
            double z = previous[2] - current[2];
            double d = sqrt(x*x + y*y + z*z);

            std::cout << ptIds[j] << " ";

//            std::cout << "J " << j << " ID " << ptIds[j] << " D " << d << " P(" <<
//                         previous[0] << " " << previous[1] << " " << previous[2] << ") C(" <<
//                         current[0] << " " << current[1] << " " << current[2] << ")" << std::endl;

            previous[0] = current[0];
            previous[1] = current[1];
            previous[2] = current[2];
        }

        std::cout << std::endl;

//		double previous[3], current[3];
//        points->GetPoint( ptIds[0], current );
//        std::cout << "POINT " << ptIds[0] << " " << current[0] << " " << current[1] << " " << current[2] << std::endl;
//        for( int j = 1; j < nrPtIds; ++j )
//		{
//			previous[0] = current[0];
//			previous[1] = current[1];
//			previous[2] = current[2];
//            points->GetPoint( ptIds[j], current );
//			glBegin( GL_LINES );
//				glVertex3dv( previous );
//				glVertex3dv( current );
//			glEnd();
//            std::cout << "POINT " << ptIds[j] << " " << current[0] << " " << current[1] << " " << current[2] << std::endl;
//		}
	}

	glEnable( GL_LIGHTING );
}

///////////////////////////////////////////////////////////////////////////
bool vtkFiberConfidenceMapper::InitializeGraphicsResources( vtkViewport * viewport )
{
	int * screenSize = viewport->GetVTKWindow()->GetSize();

	if( this->ScreenSize[0] != screenSize[0] || this->ScreenSize[1] != screenSize[1] )
	{
		this->ScreenSize[0] = screenSize[0];
		this->ScreenSize[1] = screenSize[1];

		this->TexturesInitialized = false;
	}

	if( ! this->ExtensionsInitialized )
	{
		vtkRenderer * renderer = (vtkRenderer *) viewport;
		vtkOpenGLExtensionManager * extensions = vtkOpenGLExtensionManager::New();
		extensions->SetRenderWindow( renderer->GetRenderWindow() );

		if( extensions->ExtensionSupported( "GL_VERSION_2_0" ) == 0 )
		{
			std::cout << "vtkFiberConfidenceMapper::InitializeGraphicsResources() " \
					"OpenGL 2.0 not supported" << std::endl;
			extensions->Delete();
			return false;
		}

		extensions->LoadExtension( "GL_VERSION_1_3" );
		extensions->LoadExtension( "GL_VERSION_2_0" );
		extensions->LoadExtension( "GL_EXT_framebuffer_object" );
		extensions->Delete();

		this->ExtensionsInitialized = true;
	}

	if( ! this->ShadersInitialized )
	{
		if( ! this->InitializeShaders() )
		{
			return false;
		}

		this->ShadersInitialized = true;
	}

	if( ! this->TexturesInitialized )
	{
		if( ! this->InitializeTextures() )
		{
			return false;
		}

		this->TexturesInitialized = true;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////
bool vtkFiberConfidenceMapper::InitializeShaders()
{
	if( this->VertexShader > 0 )
		vtkgl::DeleteShader( this->VertexShader );
	if( this->FragShader > 0 )
		vtkgl::DeleteShader( this->FragShader );
	if( this->Program > 0 )
		vtkgl::DeleteProgram( this->Program );
	if( this->BlurringFragShader > 0 )
		vtkgl::DeleteShader( this->BlurringFragShader );
	if( this->BlurringProgram > 0 )
		vtkgl::DeleteProgram( this->BlurringProgram );
	if( this->OutputFragShader > 0 )
		vtkgl::DeleteShader( this->OutputFragShader );
	if( this->OutputProgram > 0 )
		vtkgl::DeleteProgram( this->OutputProgram );

	int length = 0;

	this->VertexShader = vtkgl::CreateShader( vtkgl::VERTEX_SHADER );
	vtkgl::ShaderSource( this->VertexShader, 1, 
		& vtkFiberConfidenceMapperShaders_VertexShader, 0 );
	vtkgl::CompileShader( this->VertexShader );
	vtkgl::GetShaderiv( this->VertexShader, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetShaderInfoLog( this->VertexShader, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapper::InitializeShaders() " \
				"error compiling vertex shader" << std::endl;
		std::cout << log << std::endl;
		this->PrintShader( vtkFiberConfidenceMapperShaders_VertexShader );

		delete log;
		return false;
	}

	if( this->Mode == RENDERMODE_SOLID )
	{
		this->FragShader = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
		vtkgl::ShaderSource( this->FragShader, 1, 
			& vtkFiberConfidenceMapperShaders_SilhouetteFragShader, 0 );
	}
	else if( this->Mode == RENDERMODE_CHECKER_BOARD )
	{
		this->FragShader = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
		vtkgl::ShaderSource( this->FragShader, 1, 
			& vtkFiberConfidenceMapperShaders_CheckerBoardFragShader, 0 );
	}
	else if( this->Mode == RENDERMODE_HOLES )
	{
		this->FragShader = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
		vtkgl::ShaderSource( this->FragShader, 1, 
			& vtkFiberConfidenceMapperShaders_HolesFragShader, 0 );
	}
	else
	{
		std::cout << "vtkConfidenceIntervalMapper::InitializeShaders() " \
			"unknown render mode" << std::endl;
		this->FragShader = 0;
	}

	vtkgl::CompileShader( this->FragShader );
	vtkgl::GetShaderiv( this->FragShader, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetShaderInfoLog( this->FragShader, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapperShaders::InitializeShaders() " \
				"error compiling fragment shader" << std::endl;
		std::cout << log << std::endl;

		if( this->Mode == RENDERMODE_SOLID )
		{
			this->PrintShader( vtkFiberConfidenceMapperShaders_SilhouetteFragShader );
		}
		else if( this->Mode == RENDERMODE_CHECKER_BOARD )
		{
			this->PrintShader( vtkFiberConfidenceMapperShaders_CheckerBoardFragShader );
		}
		else if( this->Mode == RENDERMODE_HOLES )
		{
			this->PrintShader( vtkFiberConfidenceMapperShaders_HolesFragShader );
		}
		else {}

		delete log;
		return false;
	}

	length = 0;

	this->Program = vtkgl::CreateProgram();
	vtkgl::AttachShader( this->Program, this->VertexShader );
	vtkgl::AttachShader( this->Program, this->FragShader );
	vtkgl::LinkProgram( this->Program );
	vtkgl::GetProgramiv( this->Program, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetProgramInfoLog( this->Program, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapper::InitializeShaders() " \
				"error linking program" << std::endl;
		std::cout << log << std::endl;
		delete log;
		return false;
	}

	length = 0;

	this->BlurringFragShader = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
	vtkgl::ShaderSource( this->BlurringFragShader, 1, 
		& vtkFiberConfidenceMapperShaders_BlurringFragShader, 0 );
	vtkgl::CompileShader( this->BlurringFragShader );
	vtkgl::GetShaderiv( this->BlurringFragShader, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetShaderInfoLog( this->BlurringFragShader, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapperShaders::InitializeShaders() " \
				"error compiling blurring fragment shader" << std::endl;
		std::cout << log << std::endl;
		this->PrintShader( vtkFiberConfidenceMapperShaders_BlurringFragShader );
		delete log;
		return false;
	}

	length = 0;

	this->BlurringProgram = vtkgl::CreateProgram();
	vtkgl::AttachShader( this->BlurringProgram, this->VertexShader );
	vtkgl::AttachShader( this->BlurringProgram, this->BlurringFragShader );
	vtkgl::LinkProgram( this->BlurringProgram );
	vtkgl::GetProgramiv( this->BlurringProgram, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetProgramInfoLog( this->BlurringProgram, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapperShaders::InitializeShaders() " \
				"error linking blurring program" << std::endl;
		std::cout << log << std::endl;

		delete log;
		return false;
	}

	length = 0;

	this->OutputFragShader = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
	vtkgl::ShaderSource( this->OutputFragShader, 1, 
		& vtkFiberConfidenceMapperShaders_OutputFragShader, 0 );
	vtkgl::CompileShader( this->OutputFragShader );
	vtkgl::GetShaderiv( this->OutputFragShader, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetShaderInfoLog( this->OutputFragShader, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapperShaders::InitializeShaders() " \
				"error compiling output fragment shader" << std::endl;
		std::cout << log << std::endl;
		this->PrintShader( vtkFiberConfidenceMapperShaders_OutputFragShader );
		delete log;
		return false;
	}

	length = 0;

	this->OutputProgram = vtkgl::CreateProgram();
	vtkgl::AttachShader( this->OutputProgram, this->VertexShader );
	vtkgl::AttachShader( this->OutputProgram, this->OutputFragShader );
	vtkgl::LinkProgram( this->OutputProgram );
	vtkgl::GetProgramiv( this->OutputProgram, vtkgl::INFO_LOG_LENGTH, & length );

	if( length > 2 )
	{
		char * log = new char[length];
		vtkgl::GetProgramInfoLog( this->OutputProgram, length, NULL, log );
		std::cout << "vtkFiberConfidenceMapperShaders::InitializeShaders() " \
				"error linking output program" << std::endl;
		std::cout << log << std::endl;

		delete log;
		return false;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////
bool vtkFiberConfidenceMapper::InitializeTextures()
{
	int width  = this->ScreenSize[0];
	int height = this->ScreenSize[1];

	vtkgl::ActiveTexture( vtkgl::TEXTURE0 );

	if( this->StreamlineBuffer[0] > 0 )
		glDeleteTextures( 2, this->StreamlineBuffer );
	if( this->SilhouetteBuffer[0] > 0 )
		glDeleteTextures( 2, this->SilhouetteBuffer );
	if( this->BlurringBuffer[0] > 0 )
		glDeleteTextures( 2, this->BlurringBuffer );
	if( this->FBO > 0 )
		vtkgl::DeleteFramebuffersEXT( 1, & this->FBO );

	glGenTextures( 2, this->StreamlineBuffer );

	glBindTexture( GL_TEXTURE_2D, this->StreamlineBuffer[0] );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0 );

	glBindTexture( GL_TEXTURE_2D, this->StreamlineBuffer[1] );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

	glGenTextures( 2, this->SilhouetteBuffer );

	glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[0] );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0 );

	glBindTexture( GL_TEXTURE_2D, this->SilhouetteBuffer[1] );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );

	glGenTextures( 2, this->BlurringBuffer );

	glBindTexture( GL_TEXTURE_2D, this->BlurringBuffer[0] );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, 0 );

	glBindTexture( GL_TEXTURE_2D, this->BlurringBuffer[1] );
	glTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	//glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::DEPTH_COMPONENT32, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0 );
	glTexImage2D( GL_TEXTURE_2D, 0, vtkgl::DEPTH24_STENCIL8_EXT, width, height,
				  0, vtkgl::DEPTH_STENCIL_EXT, vtkgl::UNSIGNED_INT_24_8_EXT, 0 );

	glBindTexture( GL_TEXTURE_2D, 0 );

	vtkgl::GenFramebuffersEXT( 1, & this->FBO );
	vtkgl::BindFramebufferEXT( vtkgl::FRAMEBUFFER_EXT, 0 );
	return true;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::ReleaseGraphicsResources( vtkWindow * window )
{
	if( this->StreamlineBuffer[0] > 0 )
		glDeleteTextures( 2, this->StreamlineBuffer );
	if( this->SilhouetteBuffer[0] > 0 )
		glDeleteTextures( 2, this->SilhouetteBuffer );
	if( this->BlurringBuffer[0] > 0 )
		glDeleteTextures( 2, this->BlurringBuffer );
	if( this->FBO > 0 )
		vtkgl::DeleteFramebuffersEXT( 1, & this->FBO );

	if( this->VertexShader > 0 )
		vtkgl::DeleteShader( this->VertexShader );
	if( this->FragShader > 0 )
		vtkgl::DeleteShader( this->FragShader );
	if( this->Program > 0 )
		vtkgl::DeleteProgram( this->Program );
	if( this->BlurringFragShader > 0 )
		vtkgl::DeleteShader( this->BlurringFragShader );
	if( this->BlurringProgram > 0 )
		vtkgl::DeleteProgram( this->BlurringProgram );
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetTable( vtkConfidenceTable * table )
{
    this->Table = table;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceTable * vtkFiberConfidenceMapper::GetTable()
{
	return this->Table;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetInterval( vtkConfidenceInterval * interval )
{
	this->Interval = interval;
}

///////////////////////////////////////////////////////////////////////////
vtkConfidenceInterval * vtkFiberConfidenceMapper::GetInterval()
{
	return this->Interval;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetRenderModeToSolid()
{
	if( this->Mode == RENDERMODE_SOLID )
	{
		return;
	}

	this->Mode = RENDERMODE_SOLID;
	this->ShadersInitialized = false;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetRenderModeToCheckerBoard()
{
	if( this->Mode == RENDERMODE_CHECKER_BOARD )
	{
		return;
	}

	this->Mode = RENDERMODE_CHECKER_BOARD;
	this->ShadersInitialized = false;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetRenderModeToHoles()
{
	if( this->Mode == RENDERMODE_HOLES )
	{
		return;
	}

	this->Mode = RENDERMODE_HOLES;
	this->ShadersInitialized = false;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetROIEnabled( bool enabled )
{
	this->ROIEnabled = enabled;
}

///////////////////////////////////////////////////////////////////////////
bool vtkFiberConfidenceMapper::IsROIEnabled()
{
	return this->ROIEnabled;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetROI( int x, int y, int width, int height )
{
	this->ROI[0] = x;
	this->ROI[1] = y;
	this->ROI[2] = width;
	this->ROI[3] = height;
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::SetROI( int roi[4] )
{
	for( int i = 0; i < 4; ++i )
		this->ROI[i] = roi[i];
}

///////////////////////////////////////////////////////////////////////////
int vtkFiberConfidenceMapper::GetROIX()
{
	return this->ROI[0];
}

///////////////////////////////////////////////////////////////////////////
int vtkFiberConfidenceMapper::GetROIY()
{
	return this->ROI[1];
}

///////////////////////////////////////////////////////////////////////////
int vtkFiberConfidenceMapper::GetROIWidth()
{
	return this->ROI[2];
}

///////////////////////////////////////////////////////////////////////////
int vtkFiberConfidenceMapper::GetROIHeight()
{
	return this->ROI[3];
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::CheckStatusFBO()
{
	GLenum status = vtkgl::CheckFramebufferStatusEXT( vtkgl::FRAMEBUFFER_EXT );
	std::cout << "vtkFiberConfidenceMapper::CheckStatusFBO() ";

	switch( status )
	{
	case vtkgl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT :
		std::cout << "no buffers attached" << std::endl;
		break;
	case vtkgl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT :
		std::cout << "missing required buffer attachment" << std::endl;
		break;
	case vtkgl::FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT :
		std::cout << "mismatch image and buffer dimensions" << std::endl;
		break;
	case vtkgl::FRAMEBUFFER_INCOMPLETE_FORMATS_EXT :
		std::cout << "color buffers have different types" << std::endl;
		break;
	case vtkgl::FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT :
		std::cout << "attempting to draw to non-attached color buffer" << std::endl;
		break;
	case vtkgl::FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT :
		std::cout << "attempting to read from non-attached color buffer" << std::endl;
		break;
	case vtkgl::FRAMEBUFFER_UNSUPPORTED_EXT :
		std::cout << "format is not supported by graphics card" << std::endl;
		break;
	default :
		std::cout << "framebuffer complete" << std::endl;
		break;
	}
}

///////////////////////////////////////////////////////////////////////////
void vtkFiberConfidenceMapper::PrintShader( const char * text )
{
	std::vector< std::string > tokens;
	std::string str( text );
	std::string delimiter( "\n" );

	size_t p0 = 0;
	size_t p1 = std::string::npos;

	while( p0 != std::string::npos )
	{
		p1 = str.find_first_of( delimiter, p0 );

		if( p1 != p0 )
		{
			std::string token = str.substr( p0, p1 - p0 );
			tokens.push_back( token );
		}

		p0 = str.find_first_not_of( delimiter, p1 );
	}

	std::vector< std::string >::iterator i = tokens.begin();
	int lineNr = 0;
	for( ; i != tokens.end(); ++i )
	{
		std::cout << lineNr << "\t" << (*i) << std::endl;
		lineNr++;
	}
}
