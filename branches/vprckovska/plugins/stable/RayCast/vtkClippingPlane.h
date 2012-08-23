#ifndef __vtkClippingPlane_h
#define __vtkClippingPlane_h

#include "vtkPlane.h"
#include <vtkgl.h>
//#include <GL/gl.h>
//#include <GL/glew.h>

class vtkClippingPlane : public vtkPlane
{
public:

    static vtkClippingPlane * New();

    void SetId( int iId );
    void SetEnabled( bool bEnabled );

    void Enable()
    {
        SetEnabled( true );
    };

    void Disable()
    {
        SetEnabled( false );
    };

    bool IsEnabled()
    {
        return m_bEnabled;
    }

    int GetId()
    {
        return m_iId;
    }

    GLenum GetGLId()
    {
        return m_enumClipPlaneId;
    }

    void Update();

protected:
    double origin[3];
    vtkClippingPlane();
    virtual ~vtkClippingPlane();

    int m_iId;
    bool m_bEnabled;
    GLenum m_enumClipPlaneId;
};

#endif
