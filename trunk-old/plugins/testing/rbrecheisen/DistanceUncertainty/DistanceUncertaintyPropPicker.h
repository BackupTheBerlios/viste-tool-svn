#ifndef bmia_DistanceUncertainty_DistanceUncertaintyPropPicker_h
#define bmia_DistanceUncertainty_DistanceUncertaintyPropPicker_h

#include <DistanceUncertaintyPlugin.h>

#include <vtkActor.h>
#include <vtkObject.h>
#include <vtkCommand.h>
#include <vtkPropPicker.h>
#include <vtkCellPicker.h>

namespace bmia
{
	class DistanceUncertaintyPropPicker : public vtkCommand
	{
	public:

		DistanceUncertaintyPropPicker();
		virtual ~DistanceUncertaintyPropPicker();

		void SetPlugin( DistanceUncertaintyPlugin * plugin );
		DistanceUncertaintyPlugin * GetPlugin();

		void SetEnabled( bool enabled );
		bool IsEnabled();

		vtkPropPicker * GetPicker();

		void Execute( vtkObject * caller, unsigned long eventId, void * callData );

	private:

		DistanceUncertaintyPlugin * Plugin;

		vtkCellPicker * CellPicker;
		vtkPropPicker * PropPicker;
		vtkActor * PickActor;

		bool Enabled;
	};
}

#endif
