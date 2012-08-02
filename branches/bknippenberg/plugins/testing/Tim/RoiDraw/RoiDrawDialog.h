/*
 * RoiDrawDialog.h
 *
 * 2010-11-16	Tim Peeters
 * - First version
 *
 * 2010-12-14	Evert van Aart
 * - Hotfix solution for the bug in Windows, which causes errors when
 *   switching windows. This is an improvised solution, a better 
 *   solution should be made in the near future.
 *
 * 2011-01-27	Evert van Aart
 * - Added support for transformed planes.
 *
 */

#ifndef bmia_RoiDrawDialog_h
#define bmia_RoiDrawDialog_h

#include <QDialog>
#include "ui_roidraw.h"

class vtkImageTracerWidget2;
class vtkRenderer;

namespace bmia {

namespace data {
    class DataSet;
}

class vtkImageSliceActor;
class RoiDrawPlugin;

/**
 * Qt dialog for drawing a region of interest
 */
class RoiDrawDialog : public QDialog, private Ui::RoiDraw
{
  Q_OBJECT

public:
  RoiDrawDialog(RoiDrawPlugin* pplugin, QWidget* parent = NULL);
  ~RoiDrawDialog();

  // TODO: add planes (data)
  void addSliceData(data::DataSet* ds);

  /**
   * Called if data manager notifies the plugin that a data set
   * was changed.
   */
  void sliceDataChanged(data::DataSet* ds);

  /** Next three functions are part of a hotfix solution for a bug in Windows,
	which causes errors when switching between windows. See the code for "event"
	for details. */

  void turnPickingOff();
  bool event(QEvent * e);
  void makeCurrent();

protected slots:
  /**
   * Called if the user selected a different slice data.
   * Select a slice data. Use -1 as an argument to select no data.
   */
  void selectSliceData(int index);

  void apply();
  void close();

private:
  vtkImageSliceActor* getSliceActorFromData(data::DataSet* ds);
  vtkImageData* sliceInput;

  QList<data::DataSet*> sliceDataSets;
  int selectedData;

  vtkRenderer* renderer;
  void ResetCamera(vtkImageSliceActor* actor);
  vtkImageTracerWidget2* tracerWidget;

  RoiDrawPlugin* plugin;

}; // RoiDrawDialog

} // namespace bmia
#endif // bmia_RoiDrawDialog_h
