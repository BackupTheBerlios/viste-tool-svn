/**
 * vtkFiberConfidenceMapper.h
 * by Ralph Brecheisen
 *
 * 20-01-2010	Ralph Brecheisen
 * - First version
 */
#ifndef bmia_vtkFiberConfidenceMapper_h
#define bmia_vtkFiberConfidenceMapper_h

#include "vtkDistanceTable.h"
#include "vtkColor4.h"

#include "vtkPolyDataMapper.h"
#include "vtkRenderer.h"
#include "vtkActor.h"
#include "vtkWindow.h"
#include "vtkViewport.h"

#include "GpuBuffers/VertexBuffer.h"
#include "GpuBuffers/FrameBuffer.h"
#include "GpuPrograms/GpuProgram.h"

#include <vector>

class QColor;

namespace bmia
{
    class vtkFiberConfidenceMapper : public vtkPolyDataMapper
	{
	public:

		/** Creates new instance of the mapper */
        static vtkFiberConfidenceMapper * New();

		/** Renders the fibers as a confidence area */
		virtual void Render( vtkRenderer * _renderer, vtkActor * _actor );
		virtual void RenderPiece( vtkRenderer * _renderer, vtkActor * _actor );
        virtual void RenderStreamlines( const float _threshold );
        virtual void RenderIlluminatedStreamlines( const float _opacity );
		
		/** Initializes graphics resources such as shader programs and offscreen 
		    render buffers */
		bool InitializeGraphicsResources( vtkViewport * _viewport );
		
		/** Releases graphics resources */
		void ReleaseGraphicsResources( vtkWindow * _window );

		/** Sets/gets the distance table used to determine which fibers belong
		    to the confidence boundary */
		void SetDistanceTable( vtkDistanceTable * _table );
		vtkDistanceTable * GetDistanceTable();

        void SetDistanceTableOriginalFibers( vtkDistanceTable * _table );
        vtkDistanceTable * GetDistanceTableOriginalFibers();

		void SetDistanceTableBundle( vtkDistanceTable * _table );
		vtkDistanceTable * GetDistanceTableBundle();

		/** Sets/gets confidence interval between [0,1] */
		void SetInterval( float _minimum, float _maximum );
		void GetInterval( float & _minimum, float & _maximum );
		float * GetInterval();

		void SetConfidenceLevels( std::vector<std::pair<std::string,float> > * _levels );
		std::vector< std::pair< std::string, float > > * GetConfidenceLevels() { return this->ConfidenceLevels; }
        void SetFillColors( std::vector<std::pair<float,vtkColor4> > * _colors );
		std::vector<std::pair<float,vtkColor4> > * GetFillColors() { return this->FillColors; }
        void SetLineColors( std::vector<std::pair<float,vtkColor4> > * _colors );
		std::vector<std::pair<float,vtkColor4> > * GetLineColors() { return this->LineColors; }

		void SetDistanceModeRelativeToMedianFiber(); // 0
		void SetDistanceModeRelativeToFiberOriginalDataset(); // 1
		void SetDistanceModeRelativeToBundleCenter(); // 2

        /** Sets/gets maximum fiber density per pixel. This is the density that
            is assigned an opacity of 1 */
        void SetMaximumFiberDensity( const int _density );
        const int GetMaximumFiberDensity() const;

		/** Enables/disables silhouette representation. If disabled, the mapper
		    shows normal streamlines */
		void SetSilhouetteEnabled( bool _enabled );
		bool IsSilhouetteEnabled();

		void SetSilhouetteStreamlinesEnabled( bool _enabled );
		bool IsSilhouetteStreamlinesEnabled();

		/** Sets/gets the silhouette fill color */
		void SetFillColor( const float _red, const float _green, const float _blue );
		void SetFillColor( const float _color[3] );
		void GetFillColor( float & _red, float & _green, float & _blue ) const;
		const float * GetFillColor() const;

		/** Sets/gets the fill opacity */
		void SetFillOpacity( const float _opacity );
		const float GetFillOpacity() const;

		/** Sets/gets the silhouette outline color */
		void SetOutlineColor( const float _red, const float _green, const float _blue );
		void SetOutlineColor( const float _color[3] );
		void GetOutlineColor( float & _red, float & _green, float & _blue ) const;
		const float * GetOutlineColor() const;

        /** Sets/gets the color of the density mapping */
        void SetDensityColor( const float _red, const float _green, const float _blue, const float _alpha );
        const float * GetDensityColor() const;

        /** Enables/disables density coloring */
        void SetDensityColoringEnabled( bool _enabled );
        bool IsDensityColoringEnabled();

        /** Enables/disables erosion after dilation */
        void SetErosionEnabled( bool _enabled );
        bool IsErosionEnabled();

		/** Sets/gets the inner depth threshold. This threshold controls.... */
		void SetDepthThreshold( const float _threshold );
		float GetDepthThreshold() const;

		/** Sets/gets the width of the dilation kernel */
		void SetFillDilation( const int _size );
		int GetFillDilation() const;

		/** Sets/gets the width of the silhouette outline */
		void SetOutlineWidth( const int _width );
		int GetOutlineWidth() const;

        /** Enables/disables density weighting for dilation */
		void SetDensityWeightingEnabled( bool _enabled );
		bool IsDensityWeightingEnabled();

        /** Sets the kernel size for smoothing the density map */
        void SetSmoothingKernelSize( const int _size );
        const int GetSmoothingKernelSize() const;

        /** Enables/disables smoothing of density map */
        void SetDensitySmoothingEnabled( bool _enabled );
        bool IsDensitySmoothingEnabled();

        void SetOverwriteEnabled( bool _enabled );
        bool IsOverwriteEnabled();

	protected:

		/** Constructor and destructor */
        vtkFiberConfidenceMapper();
        virtual ~vtkFiberConfidenceMapper();

	private:

		/** NOT IMPLEMENTED copy constructor and assignment operator */
        vtkFiberConfidenceMapper( const vtkFiberConfidenceMapper & );
        void operator = ( const vtkFiberConfidenceMapper & );

        /** Rebuilds display list */
        void RebuildDisplayList();

    private:

		vtkDistanceTable * Table;
        vtkDistanceTable * TableOriginal;
		vtkDistanceTable * TableBundle;

		//vtkFiberMapper * FiberMapper;
        vtkRenderer * Renderer;
        vtkActor * Actor;

		float DepthThreshold;
		float Interval[2];
		float OutlineColor[3];
		float FillColor[3];
		float FillOpacity;
        float DensityColor[4];

        int FillDilation;
		int OutlineWidth;
        int MaximumFiberDensity;
        int SmoothingKernelSize;
		int DistanceMode;
		int PercentageMode;

        bool RebuildList;
		bool SilhouetteEnabled;
		bool SilhouetteStreamlinesEnabled;
        bool DensityColoringEnabled;
        bool ErosionEnabled;
		bool DensityWeightingEnabled;
        bool DensitySmoothingEnabled;
		bool OverwriteEnabled;
		
        opengl::GpuProgram * SilhouetteProgram;
		opengl::GpuProgram * ErosionProgram;
        opengl::GpuProgram * DensityProgram;
        opengl::GpuProgram * DensitySmoothingProgram;
        opengl::GpuProgram * ToScreenProgram;

        unsigned int RenderList;
        unsigned int * RenderLists;

        unsigned int FrameBuffer;
        unsigned int ColorBuffer;
        unsigned int DepthBuffer;
        unsigned int DensityBuffer;
        unsigned int DensitySmoothingBuffer;
        unsigned int DensitySmoothingDepthBuffer;
        unsigned int ErosionBuffer[2];
        unsigned int ErosionDepthBuffer[2];
        unsigned int DensityDepthBuffer;
        unsigned int SilhouetteBuffer[2];
        unsigned int SilhouetteDepthBuffer[2];

		unsigned int PreviousNumberOfConfidenceLevels;
		std::vector<std::pair<std::string,float> > * ConfidenceLevels;
        std::vector<std::pair<float,vtkColor4> > * FillColors;
        std::vector<std::pair<float,vtkColor4> > * LineColors;
	};

} // namespace bmia

#endif // bmia_vtkFiberConfidenceBoundaryMapper_h
