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
