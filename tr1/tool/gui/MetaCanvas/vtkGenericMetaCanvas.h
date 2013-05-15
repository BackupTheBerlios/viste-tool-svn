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
 * vtkGenericMetaCanvas.h
 *
 * 2005-01-05	Tim Peeters
 * - First version
 *
 * 2005-01-12	Tim Peeters
 * - Moved all selection related functions and variables to subclass
 * - Use vtkAbstractSubCanvas instead of vtkSubCanvas
 * - Added SetInteractor()
 * - Make use of vtkAbstractSubCanvasCollection
 *
 * 2005-07-15	Tim Peeters
 * - Added GetRenderWindowInteractor() and SetRenderWindowInteractor()
 *   as aliases for GetInteractor() and SetInteractor()
 * - Added Start() and SetSize() functions.
 *
 * 2005-11-14	Tim Peeters
 * - Renamed from vtkMetaCanvas to vtkGenericMetaCanvas.
 *
 * 2005-11-15	Tim Peeters
 * - Made SetInteractor() virtual so that it can be overridden in
 *   vtkMetaCanvas.
 *
 * 2006-02-13	Tim Peeters
 * - Added SetRenderWindow(rw).
 *
 * 2011-02-08	Evert van Aart
 * - Added support for maximization of subcanvasses.
 *
 * 2011-02-21	Evert van Aart
 * - Increased support for maximization of subcanvasses.
 *
 */

#ifndef bmia_vtkGenericMetaCanvas_h
#define bmia_vtkGenericMetaCanvas_h

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkCallbackCommand.h>
#include "vtkAbstractSubCanvas.h"
#include "vtkAbstractSubCanvasCollection.h"

namespace bmia {

/**
 * Class for representing a canvas which can have multiple
 * sub canvasses. It does not handle interaction styles. For that,
 * assign a (MetaCanvas)InteractionStyle to RenderWindow of this
 * MetaCanvas.
 */
class vtkGenericMetaCanvas : public vtkObject
{
public:
  static vtkGenericMetaCanvas *New();

  /**
   * Adds a new subcanvas to the vtkGenericMetaCanvas.
   *
   * @param subcanvas The new subcanvas to add to the metacanvas.
   */
  virtual void AddSubCanvas(vtkAbstractSubCanvas* subcanvas);

  /**
   * Removes the specified subcanvas if it is part of this metacanvas
   */
  virtual void RemoveSubCanvas(vtkAbstractSubCanvas* subcanvas);

  /**
   * Gets/Sets the interactor for this metacanvas's render window and
   * and all the subcanvasses.
   */
  virtual void SetInteractor(vtkRenderWindowInteractor* rwi);
  vtkGetObjectMacro(Interactor, vtkRenderWindowInteractor);
  void SetRenderWindowInteractor(vtkRenderWindowInteractor* rwi)
    {
    this->SetInteractor(rwi);
    }
  vtkRenderWindowInteractor* GetRenderWindowInteractor()
    {
    return this->GetInteractor();
    }

  vtkAbstractSubCanvasCollection* GetSubCanvasses()
    {
    return this->SubCanvasses;
    }

  /**
   * Calls the Start() method of this metacanvas's RenderWindowInteractor.
   */
  virtual void Start();

  /**
   * Sets the size of the render window in pixels.
   */
  virtual void SetSize(int h, int v);

  // needed for vtkMetaCanvasInteractorStyleWM to determine where inside a
  // subcanvas the user clicked (size of the renderwindow is needed).
  // AND in order to call Render() on the render window.
  vtkGetObjectMacro(RenderWindow, vtkRenderWindow);

  void SetRenderWindow(vtkRenderWindow* rw);

  /**
   * Reset the cameras for all subcanvasses.
   */
  virtual void ResetAllCameras();

	/** Maximize one of the subcanvasses
		@param ID	Index of the target subcanvas. */

	void MaximizeSubCanvas(int ID);

	/** Restore all subcanvasses to their original size. */

	void RestoreCanvasSizes();

protected:

  vtkGenericMetaCanvas();
  ~vtkGenericMetaCanvas();

  /**
   * Returns the subcanvas at the specified coordinates, or
   * NULL if there is none at those coordinates.
   */
  vtkAbstractSubCanvas* GetSubCanvasAt(int x, int y);

	/** Resize the selected subcanvas. Either maximize it, or, if it already 
		maximized, restore the initial layout of viewports.
		@param x	X-Coordinate of the mouse cursor.
		@param y	Y-Coordinate of the mouse cursor. */

	void ResizeSubCanvasAt(int x, int y);

  /**
   * The current interactor.
   * It is protected so that vtkMetaCanvas can use it in
   * SelectPokedSubCanvas() to find out where the last event
   * took place.
   */
  vtkRenderWindowInteractor* Interactor;

  static void OnEvent(vtkObject* object, unsigned long event,
		      void* clientdata, void* calldata);

  virtual void OnEvent(vtkObject* object, unsigned long event,
		       void* calldata);

private:
  vtkGenericMetaCanvas(const vtkGenericMetaCanvas&);  // Not implemented.
  void operator=(const vtkGenericMetaCanvas&);  // Not implemented.

  vtkRenderWindow* RenderWindow;
  vtkAbstractSubCanvasCollection* SubCanvasses;

  /**
   * Renders the background (black).
   */
  vtkRenderer* BackgroundRenderer;

  vtkCallbackCommand* CallbackCommand;

}; // class vtkGenericMetaCanvas

} // namespace bmia

#endif
