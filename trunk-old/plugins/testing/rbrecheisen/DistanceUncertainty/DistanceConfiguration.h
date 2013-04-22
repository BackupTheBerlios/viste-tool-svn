#ifndef bmia_DistanceConfiguration_h
#define bmia_DistanceConfiguration_h

#include <vtkImageData.h>
#include <vtkActor.h>

#include <QString>
#include <QPointF>
#include <QList>

namespace bmia
{
	class DistanceConfiguration
	{
	public:

		DistanceConfiguration( QString name );
		virtual ~DistanceConfiguration();

		QString getName();

		void setThresholdRange( double min, double max );
		double getThresholdMin();
		double getThresholdMax();

		void setSelectedThreshold( double threshold );
		double getSelectedThreshold();

		void setSelectedDistance( double distance );
		double getSelectedDistance();

		void setSelectedPointMarkerPosition( double position[3] );
		double getSelectedPointMarkerPositionX();
		double getSelectedPointMarkerPositionY();
		double getSelectedPointMarkerPositionZ();

		void setMinimumDistanceRange( double min, double max );
		double getMinimumDistanceMin();
		double getMinimumDistanceMax();

		void setMaximumDistanceRange( double min, double max );
		double getMaximumDistanceMin();
		double getMaximumDistanceMax();

		void setTumorCentroid( double centroid[3] );
		double getTumorCentroidX();
		double getTumorCentroidY();
		double getTumorCentroidZ();

		void setFiberCentroid( double centroid[3] );
		double getFiberCentroidX();
		double getFiberCentroidY();
		double getFiberCentroidZ();

		void setTumorMinMaxRadius( double minRadius, double maxRadius );
		double getTumorMinRadius();
		double getTumorMaxRadius();

		void setRiskRadius( double radius );
		double getRiskRadius();

		void setRiskRadiusEnabled( bool enabled );
		bool isRiskRadiusEnabled();

        void setRiskRadiusUncertaintyEnabled( bool enabled );
        bool isRiskRadiusUncertaintyEnabled();

		void setProjectionEnabled( bool enabled );
		bool isProjectionEnabled();

		void setContoursEnabled( bool enabled );
		bool isContoursEnabled();

		void setNumberOfContours( int numberOfContours );
		int getNumberOfContours();

		void setSelectedColorScaleName( QString name );
		QString getSelectedColorScaleName();

		void setDimensions( int dimensions[3] );
		int getDimensionX();
		int getDimensionY();
		int getDimensionZ();

		void setSpacing( double spacing[3] );
		double getSpacingX();
		double getSpacingY();
		double getSpacingZ();

		void setTumorData( vtkImageData * tumorData );
		vtkImageData * getTumorData();

		void setFiberData( vtkImageData * fiberData );
		vtkImageData * getFiberData();

		void setTumorActor( vtkActor * tumorActor );
		vtkActor * getTumorActor();

		void setSelectedVoronoiData( vtkImageData * voronoiData );
		vtkImageData * getSelectedVoronoiData();

		void setTumorVoxelPositions( int ** positions, int numberOfPositions );
		int ** getTumorVoxelPositions();
		int getNumberOfTumorVoxelPositions();

		void setFiberVoxelPositions( int ** positions, int numberOfPositions );
		int ** getFiberVoxelPositions();
		int getNumberOfFiberVoxelPositions();

		QList< double > & getMinimumDistances();
		QList< double > & getMaximumDistances();
		QList< double * > & getMinimumDistanceStartPoints();
		QList< double * > & getMinimumDistanceEndPoints();

		QList< QPointF > & getThresholdDistancePoints();

		void applyTexture( vtkTexture * texture );
		void updateTextureLookupTable( vtkTexture * texture );

		void updateTexture();

		bool isValid();

		void printConfig();

	private:

		QString _name;

		double _minimumDistanceRange[2];
		double _maximumDistanceRange[2];
		double _tumorCentroid[3];
		double _fiberCentroid[3];
		double _tumorMinMaxRadius[2];
		double _thresholdRange[2];
		double _riskRadius;
		double _spacing[3];

		double _selectedThreshold;
		double _selectedDistance;
		double _selectedPointMarkerPosition[3];

		bool _projectionEnabled;
		bool _riskRadiusEnabled;
        bool _riskRadiusUncertaintyEnabled;
		bool _contoursEnabled;

		vtkImageData * _selectedVoronoiData;
		vtkImageData * _fiberData;
		vtkImageData * _tumorData;
		vtkActor * _tumorActor;

		int ** _tumorVoxelPositions;
		int ** _fiberVoxelPositions;

		int _numberOfTumorVoxelPositions;
		int _numberOfFiberVoxelPositions;
		int _numberOfContours;
		int _dimensions[3];

		QList< QPointF > _thresholdDistancePoints;
		QList< double * > _minimumDistanceStartPoints;
		QList< double * > _minimumDistanceEndPoints;
		QList< double > _minimumDistances;
		QList< double > _maximumDistances;

		QString _selectedColorScaleName;
	};
}

#endif
