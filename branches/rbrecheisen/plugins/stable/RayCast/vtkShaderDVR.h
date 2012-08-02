#ifndef __vtkShaderDVR_h
#define __vtkShaderDVR_h

#include "vtkShaderBase.h"

class vtkShaderDVR : public vtkShaderBase
{
public:

	static vtkShaderDVR * New();

protected:

	vtkShaderDVR();
	virtual ~vtkShaderDVR();

	virtual std::string GetVertexShader();
	virtual std::string GetFragShader();
};

#endif
