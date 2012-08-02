#ifndef __vtkFiberSortFilter_h
#define __vtkFiberSortFilter_h

#include <vtkPolyDataToPolyDataFilter.h>

class vtkFiberSortFilter : public vtkPolyDataToPolyDataFilter
{
public:

    vtkTypeMacro(vtkFiberSortFilter, vtkPolyDataToPolyDataFilter)
    static vtkFiberSortFilter * New();

protected:

    virtual void Execute();

    vtkFiberSortFilter();
    virtual ~vtkFiberSortFilter();
};

#endif
