#ifndef __vtkShaderToon_h
#define __vtkShaderToon_h

#include "vtkShaderBase.h"

class vtkShaderToon : public vtkShaderBase
{
public:

        static vtkShaderToon * New();

protected:

        vtkShaderToon();
        virtual ~vtkShaderToon();

	virtual std::string GetVertexShader();
	virtual std::string GetFragShader();
};

#endif
