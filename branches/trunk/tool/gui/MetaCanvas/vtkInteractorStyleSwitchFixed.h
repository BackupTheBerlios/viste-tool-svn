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

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkInteractorStyleSwitchFixed.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkInteractorStyleSwitch - class to swap between interactory styles
// .SECTION Description
// The class vtkInteractorStyleSwitch allows handles interactively switching
// between four interactor styles -- joystick actor, joystick camera,
// trackball actor, and trackball camera.  Type 'j' or 't' to select
// joystick or trackball, and type 'c' or 'a' to select camera or actor.
// The default interactor style is joystick camera.
// .SECTION See Also
// vtkInteractorStyleJoystickActor vtkInteractorStyleJoystickCamera
// vtkInteractorStyleTrackballActor vtkInteractorStyleTrackballCamera

#ifndef bmia_vtkInteractorStyleSwitchFixed_h
#define bmia_vtkInteractorStyleSwitchFixed_h

#include <vtkInteractorStyle.h>

#define VTKIS_JOYSTICK  0
#define VTKIS_TRACKBALL 1

#define VTKIS_CAMERA    0
#define VTKIS_ACTOR     1

class vtkInteractorStyleJoystickActor;
class vtkInteractorStyleJoystickCamera;
class vtkInteractorStyleTrackballActor;
class vtkInteractorStyleTrackballCamera;

namespace bmia {

class vtkInteractorStyleSwitchFixed : public vtkInteractorStyle
{
public:
  static vtkInteractorStyleSwitchFixed *New();
  vtkTypeRevisionMacro(vtkInteractorStyleSwitchFixed, vtkInteractorStyle);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // The sub styles need the interactor too.
  void SetInteractor(vtkRenderWindowInteractor *iren);
  
  // Description:
  // We must override this method in order to pass the setting down to
  // the underlying styles
  void SetAutoAdjustCameraClippingRange( int value );
  
  // Description:
  // Set/Get current style
  vtkGetObjectMacro(CurrentStyle, vtkInteractorStyle);
  void SetCurrentStyleToJoystickActor();
  void SetCurrentStyleToJoystickCamera();
  void SetCurrentStyleToTrackballActor();
  void SetCurrentStyleToTrackballCamera();

  // Description:
  // Only care about the char event, which is used to switch between
  // different styles.
  virtual void OnChar();

  // Added by Tim Peeters
  virtual void SetDefaultRenderer(vtkRenderer*);
  virtual void SetCurrentRenderer(vtkRenderer*);
  
protected:
  vtkInteractorStyleSwitchFixed();
  ~vtkInteractorStyleSwitchFixed();
  
  void SetCurrentStyle();
  
  vtkInteractorStyleJoystickActor *JoystickActor;
  vtkInteractorStyleJoystickCamera *JoystickCamera;
  vtkInteractorStyleTrackballActor *TrackballActor;
  vtkInteractorStyleTrackballCamera *TrackballCamera;
  vtkInteractorStyle* CurrentStyle;

  int JoystickOrTrackball;
  int CameraOrActor;

private:
  vtkInteractorStyleSwitchFixed(const vtkInteractorStyleSwitchFixed&);  // Not implemented.
  void operator=(const vtkInteractorStyleSwitchFixed&);  // Not implemented.
};

} //  namespace bmia

#endif
