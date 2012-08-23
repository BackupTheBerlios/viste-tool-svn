#ifndef __vtkShaderRayDirections_h
#define __vtkShaderRayDirections_h

#include "vtkShaderBase.h"

class vtkShaderRayDirections : public vtkShaderBase
{
public:

	static vtkShaderRayDirections * New();

protected:

	vtkShaderRayDirections();
	virtual ~vtkShaderRayDirections();

	virtual std::string GetVertexShader();
	virtual std::string GetFragShader();
};

#endif
