#include "vtkClippingPlane.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro( vtkClippingPlane )

        ///////////////////////////////////////////////////////////////////
        vtkClippingPlane::vtkClippingPlane() :
        m_iId( -1 ), m_bEnabled( false )
{
}

///////////////////////////////////////////////////////////////////
vtkClippingPlane::~vtkClippingPlane()
{
}

///////////////////////////////////////////////////////////////////
void vtkClippingPlane::SetId( int iId )
{
    m_enumClipPlaneId = (GLenum) (GL_CLIP_PLANE0 + iId);
    m_iId = iId;
    Update();
};

///////////////////////////////////////////////////////////////////
void vtkClippingPlane::SetEnabled( bool bEnabled )
{
    if( bEnabled )
    {
        glEnable( m_enumClipPlaneId );
    }
    else
    {
        glDisable( m_enumClipPlaneId );
    }
    m_bEnabled = bEnabled;
}

///////////////////////////////////////////////////////////////////
void vtkClippingPlane::Update()
{
    double origin[3];
    GetOrigin( origin );

    double planeEquation[4];
    GetNormal( planeEquation );

    planeEquation[3] =
            -(planeEquation[0] * origin[0] +
              planeEquation[1] * origin[1] +
              planeEquation[2] * origin[2]);
    glClipPlane( m_enumClipPlaneId, planeEquation );
}
