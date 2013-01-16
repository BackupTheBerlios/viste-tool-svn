/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Copyright 2003 Sandia Corporation.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the
 * U.S. Government. Redistribution and use in source and binary forms, with
 * or without modification, are permitted provided that this Notice and any
 * statement of authorship are reproduced on all copies.
 */

#ifndef __vtkShadowRenderer_h
#define __vtkShadowRenderer_h

//#include "vtksnlRenderingWin32Header.h"
#include <vtkOpenGLRenderer.h>


//class VTK_SNL_RENDERING_EXPORT vtkShadowRenderer : public vtkOpenGLRenderer {
class vtkShadowRenderer : public vtkOpenGLRenderer {

public:
  static vtkShadowRenderer *New();
  vtkTypeMacro(vtkShadowRenderer,vtkOpenGLRenderer);
  void PrintSelf(ostream& os, vtkIndent indent);
  void SetShadows(bool b) {SHADOW_RENDER = b;}
  bool GetShadows() {return SHADOW_RENDER;}
  
protected:
  virtual void DeviceRender(void);
  vtkShadowRenderer();
  ~vtkShadowRenderer();

private:
  vtkShadowRenderer(const vtkShadowRenderer&);  // Not implemented
  void operator=(const vtkShadowRenderer&);     // Not implemented

  void InitShadowStuff();

  // Attributes
  int win_size_x;
  int win_size_y;
  bool SHADOW_RENDER;
 
};

#endif
