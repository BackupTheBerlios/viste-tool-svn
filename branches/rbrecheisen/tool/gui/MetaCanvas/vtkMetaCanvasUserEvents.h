/**
 * vtkMetaCanvasUserEvents.h
 * by Tim Peeters
 *
 * 2005-01-18	Tim Peeters
 * - First version
 *
 * 2005-11-14	Tim Peeters
 * - changed VTK_ prefixes in variable names to BMIA_
 */

#ifndef bmia_vtkMetaCanvasUserEvents_h
#define bmia_vtkMetaCanvasUserEvents_h

#include <vtkCommand.h>

namespace bmia {
  #define BMIA_USER_EVENT_SUBCANVAS_ADDED	100
  #define BMIA_USER_EVENT_SUBCANVAS_REMOVED	101
  #define BMIA_USER_EVENT_SUBCANVAS_SELECTED	102
	#define BMIA_USER_EVENT_SUBCANVAS_CAMERA_RESET 103
	#define BMIA_USER_EVENT_SUBCANVASSES_RESIZED 104
}

#endif
