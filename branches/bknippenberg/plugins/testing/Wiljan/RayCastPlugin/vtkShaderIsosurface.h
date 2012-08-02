#ifndef __vtkShaderIsosurface_h
#define __vtkShaderIsosurface_h

#include "vtkShaderBase.h"

class vtkShaderIsosurface : public vtkShaderBase
{
public:

	static vtkShaderIsosurface * New();

protected:

	vtkShaderIsosurface();
	virtual ~vtkShaderIsosurface();

	virtual std::string GetVertexShader();
	virtual std::string GetFragShader();
};

#endif
