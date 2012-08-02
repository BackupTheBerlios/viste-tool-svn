/**
 * SHGlyphsGUI.h
 * by Tim Peeters
 *
 * 2008-10-03	Tim Peeters
 * - First version
 */

#ifndef bmia_SHGlyphsGUI_h
#define bmia_SHGlyphsGUI_h

#include <QtGui/QMainWindow>
#include "ui_shglyphsgui.h"

class vtkPlaneWidget;
class vtkRenderer;

namespace bmia {

class vtkSHGlyphMapper;
class vtkWidgetCallback;

class SHGlyphsGUI : public QMainWindow, private Ui::shglyphsgui
{
  Q_OBJECT

public:
  SHGlyphsGUI(QWidget* parent = NULL);
  ~SHGlyphsGUI();

  vtkSHGlyphMapper* GetMapper()
    {
    return this->SHGlyphMapper;
    };

  vtkPlaneWidget* PlaneWidget;
  vtkRenderer* Renderer;

protected:
  vtkSHGlyphMapper* SHGlyphMapper;
  vtkWidgetCallback* Callback;

protected slots:
  void setGlyphScaling(double scale);
  void setA0Scaling(double scale);
  void setLocalScaling(bool local);
  void setStepSize(double step);
  void setNumRefineSteps(int num);
  void setZRotation(int angle);
  void setYRotation(int angle);
  void setSeedPointDistance(double distance);


}; // class SHGlyphsGUI

} // namespace bmia

#endif // bmia_SHGlyphsGUI_h
