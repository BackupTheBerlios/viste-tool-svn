#ifndef __vtkShaderMIP_h
#define __vtkShaderMIP_h

#include "vtkShaderBase.h"

class vtkShaderMIP : public vtkShaderBase
{
public:

	static vtkShaderMIP * New();

protected:

	vtkShaderMIP();
	virtual ~vtkShaderMIP();

	virtual std::string GetVertexShader();
	virtual std::string GetFragShader();
};

#endif
