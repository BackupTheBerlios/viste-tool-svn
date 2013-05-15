/**
 * vtkFiberTubeMapper.h
 * by Ralph Brecheisen
 *
 * 31-01-2010	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkFiberTubeMapper_h
#define bmia_vtkFiberTubeMapper_h

#include <GL/glew.h>

#include "vtkOpenGLPolyDataMapper.h"
#include "vtkRenderer.h"
#include "vtkActor.h"
#include "vtkWindow.h"
#include "vtkViewport.h"

#include "GpuPrograms/GpuProgram.h"

namespace bmia
{
	class vtkFiberTubeMapper : public vtkOpenGLPolyDataMapper
	{
	public:

		/** Creates new instance of the mapper */
		static vtkFiberTubeMapper * New();

		/** Sets lighting mode to hardware */
		void SetShadersEnabled( bool _enabled );
		bool IsShadersEnabled();

		/** Renders the streamtubes using a fragment shader for high-detail
		    illumination effects */
		virtual void RenderPiece( vtkRenderer * _renderer, vtkActor * _actor );
		virtual void Render( vtkRenderer * _renderer, vtkActor * _actor );

	protected:

		/** Constructor and destructor */
		vtkFiberTubeMapper();
		virtual ~vtkFiberTubeMapper();

		/** Initializes shaders */
		virtual bool InitializeGraphicsResources( vtkViewport * _viewport );

	private:

		/** NOT IMPLEMENTED copy constructor and assignment operator */
		vtkFiberTubeMapper( const vtkFiberTubeMapper & );
		void operator = ( const vtkFiberTubeMapper & );

		opengl::GpuProgram * Program;

		bool ShadersEnabled;
	};

} // namespace bmia

#endif // vtkFiberTubeMapper
