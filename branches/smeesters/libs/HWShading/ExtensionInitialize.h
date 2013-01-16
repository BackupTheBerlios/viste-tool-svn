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

/**
 * ExtensionInitialize.h
 * by Tim Peeters
 *
 * 2006-01-30	Tim Peeters
 * - First version
 */

#include <vtkOpenGLExtensionManager.h>
#include <vtkgl.h>
#include <vtkRenderWindow.h>

bool InitializeExtensions(vtkRenderWindow* rw)
{
  cout<<"InitializeExtensions("<<rw<<")"<<endl;
  bool success = false;

  vtkOpenGLExtensionManager *extensions = vtkOpenGLExtensionManager::New();
//  extensions->DebugOn();
  extensions->SetRenderWindow(rw);

//  cout<<"Read OpenGL extensions."<<endl;
//  extensions->ReadOpenGLExtensions();

  cout<<"Updating extension manager.."<<endl;
  extensions->Update();

//  cout<<"Getting extensions string.."<<endl;
  cout<<"Extensions: "<< extensions->GetExtensionsString()<<endl;

  cout<<"Checking for OpenGL 2.0"<<endl;
  extensions->ExtensionSupported("GL_VERSION_2_0");

   if ( !extensions->ExtensionSupported("GL_VERSION_2_0") 
       || !extensions->ExtensionSupported("GL_EXT_framebuffer_object") )
//     || !extensions->ExtensionSupported("GL_ARB_multitexture") ) {
    {
//    vtkErrorMacro("Required extensions not supported!");
    return success;
    }

  extensions->LoadExtension("GL_VERSION_2_0");
  extensions->LoadExtension("GL_EXT_framebuffer_object");
  success = true;

  extensions->Delete(); extensions = NULL;

  return success;
}
