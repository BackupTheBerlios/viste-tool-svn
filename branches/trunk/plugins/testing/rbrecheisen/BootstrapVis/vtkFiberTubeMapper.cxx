/**
 * vtkFiberTubeMapper.cxx
 * by Ralph Brecheisen
 *
 * 31-01-2010	Ralph Brecheisen
 * - First version
 */
#include "vtkFiberTubeMapper.h"

#include "vtkObjectFactory.h"
#include "vtkProperty.h"

#include <string>
#include <sstream>

#include "Default.vert.h"
#include "Lighting.frag.h"

namespace bmia {

	vtkStandardNewMacro( vtkFiberTubeMapper );

	////////////////////////////////////////////////////////////////////////
	vtkFiberTubeMapper::vtkFiberTubeMapper() : vtkOpenGLPolyDataMapper()
	{
		this->Program = NULL;
		this->ShadersEnabled = false;
	}

	////////////////////////////////////////////////////////////////////////
	vtkFiberTubeMapper::~vtkFiberTubeMapper()
	{
	}

	////////////////////////////////////////////////////////////////////////
	void vtkFiberTubeMapper::SetShadersEnabled( bool _enabled )
	{
		this->ShadersEnabled = _enabled;
	}

	////////////////////////////////////////////////////////////////////////
	bool vtkFiberTubeMapper::IsShadersEnabled()
	{
		return this->ShadersEnabled;
	}

	////////////////////////////////////////////////////////////////////////
	bool vtkFiberTubeMapper::InitializeGraphicsResources( vtkViewport * _viewport )
	{
		if( this->Program == NULL )
		{
			glewInit();

			this->Program = new opengl::GpuProgram();
			this->Program
				->createShader( "Default", "", opengl::GpuShader::GST_VERTEX )
				->setSourceCode( Default_VertexShaderCode );
					
			this->Program
				->createShader( "Lighting", "", opengl::GpuShader::GST_FRAGMENT )
				->setSourceCode( Lighting_FragmentShaderCode );

			if( ! this->Program->build() )
			{
				std::stringstream errorMsg( std::stringstream::in
						| std::stringstream::out );

				errorMsg << "Could not build the GPU program" << std::endl << "Build log:"
						<< this->Program->getLastBuildLog();

				vtkErrorMacro( << errorMsg.str().c_str() );
				return false;
			}
		}

		return true;
	}

	////////////////////////////////////////////////////////////////////////
	void vtkFiberTubeMapper::RenderPiece( vtkRenderer * _renderer, vtkActor * _actor )
	{
		if( this->IsShadersEnabled() )
		{
			double red, green, blue;
			_actor->GetProperty()->GetColor( red, green, blue );

			double shininess = _actor->GetProperty()->GetSpecularPower();

			this->Program->bind();
			this->Program->setUniform( "ambient", 0.0f, 0.0f, 0.0f, 1.0f );
			this->Program->setUniform( "diffuse", 
				static_cast<float>(red), static_cast<float>(green), static_cast<float>(blue), 1.0f );
			this->Program->setUniform( "shininess", static_cast<float>(shininess) );

			this->vtkOpenGLPolyDataMapper::RenderPiece( _renderer, _actor );

			this->Program->unbind();
		}
		else
		{
			this->vtkOpenGLPolyDataMapper::RenderPiece( _renderer, _actor );
		}
	}

	////////////////////////////////////////////////////////////////////////
	void vtkFiberTubeMapper::Render( vtkRenderer * _renderer, vtkActor * _actor )
	{
		if( ! this->InitializeGraphicsResources( _renderer ) )
			return;

		this->vtkOpenGLPolyDataMapper::Render( _renderer, _actor );
	}

} // namespace bmia
