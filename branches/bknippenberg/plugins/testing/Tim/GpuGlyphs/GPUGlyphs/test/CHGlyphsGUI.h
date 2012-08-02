/**
 * CHGlyphsGUI.h
 * by Tim Peeters
 *
 * 2008-11-20	Tim Peeters
 * - First version. Based on SHGlyphsGUI.h
 */

#ifndef bmia_CHGlyphsGUI_h
#define bmia_CHGlyphsGUI_h

#include <QtGui/QMainWindow>
#include "ui_shglyphsgui.h"

class vtkPlaneWidget;
class vtkRenderer;

namespace bmia {

class vtkCHGlyphMapper;
class vtkWidgetCallback;

class CHGlyphsGUI : public QMainWindow, private Ui::shglyphsgui
{
  Q_OBJECT

public:
  CHGlyphsGUI(QWidget* parent = NULL);
  ~CHGlyphsGUI();

  vtkCHGlyphMapper* GetMapper()
    {
    return this->CHGlyphMapper;
    };

  vtkPlaneWidget* PlaneWidget;
  vtkRenderer* Renderer;

protected:
  vtkCHGlyphMapper* CHGlyphMapper;
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

}; // class CHGlyphsGUI

} // namespace bmia

#endif // bmia_CHGlyphsGUI_h
