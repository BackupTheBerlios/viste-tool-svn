#ifndef bmia_DistanceUncertainty_DistanceUncertaintyPlugin_h
#define bmia_DistanceUncertainty_DistanceUncertaintyPlugin_h

#include <DTITool.h>
#include <DistanceConfiguration.h>

#include <QDistanceWidget.h>

#include <vtkDistanceArrowWidget.h>
#include <vtkPointMarkerWidget.h>
#include <vtkInteractorStyleCellPicker.h>

#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorObserver.h>
#include <vtkImageData.h>
#include <vtkTexture.h>
#include <vtkPolyData.h>
#include <vtkActor.h>

#include <QtGui>

#include <RayCastPlugin.h>
#include <RayCastVolumeMapper.h>

namespace bmia
{
	class DistanceUncertaintyPlugin :	public plugin::AdvancedPlugin,
										public data::Consumer,
										public plugin::GUI
	{
		Q_OBJECT

		Q_INTERFACES( bmia::plugin::Plugin )
		Q_INTERFACES( bmia::plugin::AdvancedPlugin )
		Q_INTERFACES( bmia::data::Consumer )
		Q_INTERFACES( bmia::plugin::GUI )

	public:

		DistanceUncertaintyPlugin();
		virtual ~DistanceUncertaintyPlugin();

		QWidget * getGUI();

		void dataSetAdded  ( data::DataSet * dataset );
        void dataSetRemoved( data::DataSet * dataset ) {}
        void dataSetChanged( data::DataSet * dataset ) {}

		vtkRenderer * getRenderer3D();
		vtkRenderWindowInteractor * getInteractor();
		vtkInteractorObserver * getInteractorStyle();

		void render();

	public slots:

		void computeStarted();
		void mapPointSelected( int );
		void graphPointSelected( QPointF & );
		void graphRangeSelected( QPointF &, QPointF & );
		void riskRadiusEnabled( bool );
        void riskRadiusUncertaintyEnabled( bool );
		void riskRadiusChanged( double );
		void contoursEnabled( bool, int );
		void projectionEnabled( bool );
		void configChanged( const QString );
		void automaticViewPointsEnabled( bool );
		void updateIsoValueEnabled( bool );
        void colorLookupChanged( QString );

        void computeSingleDTStarted();

	private:

		// Sets the camera viewpoint to look straight at given point
		void updateViewPoint( double P[3] );

		// For a given binary object, returns the boundary voxels. This can
		// speed up the computation of the centroid and also the distance lookups
		template< class T >
		void getBoundaryVoxelPositions( T * voxels, int **& voxelPositions, int & nrPositions,
			int nx, int ny, int nz );

		// Computes the centroid of a set of voxel positions by taking
		// the average position. The centroid is scaled to mm's
		void computeCentroid( int ** positions, int nrPositions, double centroid[3],
			double sx, double sy, double sz );

		// Computes minimum and maximum radius of binary object defined by the
		// given set of positions and centroid. This allows a speedup of the tumor
		// raycasting by jumping to the minimum radius at the first step
		void computeMinMaxRadius( int ** positions, int nrPositions, double centroid[3],
			double minMaxRadius[2], double sx, double sy, double sz );

		// Computes distance map for the given tumor
		template< class T >
		void computeDistanceMap( T * voxels, float * transform, int * voronoi, double *& map, int *& idxMap, int rows, int columns,
			double centroid[3], double minRadius, double threshold, int nx, int ny, int nz,
            double sx, double sy, double sz );

		// Computes a sinusoidally projected distance map for the given tumor
		template< class T >
		void computeSinusoidalDistanceMap( T * voxels, float * transform, int * voronoi, double *& map, int *& idxMap, int rows, int columns,
			double centroid[3], double minRadius, double threshold, int nx, int ny, int nz,
            double sx, double sy, double sz, double deltaTheta, double deltaPhi );

		// Computes minimal distance at the tumor surface position corresponding to
		// the given spherical coordinates
		template< class T >
		void computeDistanceFromSphericalCoordinates( T * voxels, float * transform, int * voronoi, double centroid[3],
			double minRadius, double threshold, int nx, int ny, int nz, double sx, double sy, double sz,
			double theta, double phi, double & distance, int & voxelIndex );

		// Computes position of the voxel that has the minimal distance in the
		// tumor. We need to be able to unfold the tumor distance map such that
		// the minimal distance is centered in the map
		void computeMinMaxDistancePosition( int ** voxelPositions, int nrPositions, float * transformVoxels,
			int * voronoiVoxels, double & minDist, double minDistPos1[3], double minDistPos2[3], double & maxDist,
			int nx, int ny, int nz, double sx, double sy, double sz );

		// Return trilinearly interpolated value at given position
		template< class T >
		double getInterpolatedValue( T * voxels, int nx, int ny, int nz, double x, double y, double z );

		// Returns nearest voxel position for given position
		int getNearestVoxelIndex( int nx, int ny, int nz, double x, double y, double z );

		// Computes unit vector for given angle pair. The base vector
		// is assumed to be (0,0,1)
		void sphericalToCartesian( double R, double theta, double phi, double V[3] );

		// Computes angle offset between given vector and (0,0,1)
		void cartesianToSpherical( double V[3], double & R, double & theta, double & phi );

		// Normalizes given vector
		void normalizeVector( double V[3] );

		// Multiplies two quaternions P and Q to output R
		void multiplyQuaternions( double P[4], double Q[4], double R[4] );

		// Compute complex conjugate of given quaternion
		void quaternionConjugate( double P[4], double Q[4] );

		// Removes distance line widgets from the scene
		void clearArrowWidgets();

		// Add distance arrow widget to scene
		void addArrowWidget( vtkDistanceArrowWidget * widget );
		void addArrowWidget( double P[3], double Q[3], double distance = -1.0 );

		// Removes point marker widgets from the scene
		void clearPointMarkerWidgets();

		// Add point marker widget to scene
		void addPointMarkerWidget( vtkPointMarkerWidget * widget );
		void addPointMarkerWidget( double P[3] );

		// Builds vtkTexture object from given map
		vtkTexture * buildTexture( double * data, int rows, int columns, double threshold = 0.0 );

        // Builds vtkTexture object from given maps
        vtkTexture * buildTexture( double * dataA, double * dataB, int rows, int columns, double threshold = 0.0 );

		// Finds actor owning the given polydata
		vtkActor * findActor( vtkPolyData * polyData );

		// Gets raycast volume mapper from plugin manager
		RayCastVolumeMapper * findVolumeMapper();

		// Returns distance configuration for the given tumor and fiber set
		DistanceConfiguration * findDistanceConfiguration( QString name );
		DistanceConfiguration * findDistanceConfiguration( vtkImageData * tumorData, vtkImageData * fiberData );

		// Highlight 3D tumor pick position in tumor map
		void showPickPosition( double position[3] );

        // Returns 2D map position for given 3D coordinate, depending on whether
        // sinusoidal projection is enabled or not.
        QPoint getMapPosition(double position[3], double centroid[3], int rows, int columns, bool projection);

	private:

		class CellPickerEventHandler : public vtkCommand
		{
		public:
			CellPickerEventHandler( DistanceUncertaintyPlugin * plugin, vtkInteractorStyleCellPicker * picker );
			void Execute( vtkObject * caller, unsigned long eventId, void * callData );
			vtkInteractorStyleCellPicker * Picker;
			DistanceUncertaintyPlugin * Plugin;
		};

		QDistanceWidget	 * _distanceWidget;

		QList< DistanceConfiguration * > _configurations;
		QList< vtkImageData * >	 _datasets;
		QList< vtkPolyData * >  _polyDatasets;
		QList< vtkDistanceArrowWidget * > _arrowWidgets;
		QList< vtkPointMarkerWidget * > _pointMarkerWidgets;

		QMessageBox _progressDialog;

		DistanceConfiguration * _currentConfig;

		vtkInteractorStyleCellPicker * _pickerStyle;
		vtkInteractorStyleTrackballCamera * _defaultStyle;

		CellPickerEventHandler * _pickerStyleEventHandler;
	};
}

#endif
