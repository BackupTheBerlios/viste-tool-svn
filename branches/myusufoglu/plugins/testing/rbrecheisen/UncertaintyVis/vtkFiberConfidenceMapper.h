#ifndef __vtkFiberConfidenceMapper_h
#define __vtkFiberConfidenceMapper_h

#include <vtkOpenGLPolyDataMapper.h>

#include <vector>

class vtkRenderer;
class vtkActor;
class vtkWindow;
class vtkViewport;

class vtkConfidenceTable;
class vtkConfidenceInterval;

class vtkFiberConfidenceMapper : public vtkOpenGLPolyDataMapper
{
public:

	static vtkFiberConfidenceMapper * New();
    vtkTypeRevisionMacro( vtkFiberConfidenceMapper, vtkOpenGLPolyDataMapper )

	void Render( vtkRenderer * renderer, vtkActor * actor );
	void RenderPiece( vtkRenderer * renderer, vtkActor * actor );

	void SetTable( vtkConfidenceTable * table );
	vtkConfidenceTable * GetTable();

    void SetTable2(std::vector<int> *& ids, std::vector<float> *& scores)
    {
        this->Ids = ids;
        this->Scores = scores;
    }

	void SetInterval( vtkConfidenceInterval * interval );
	vtkConfidenceInterval * GetInterval();

	void SetROIEnabled( bool enabled );
	bool IsROIEnabled();

	void SetROI( int x, int y, int width, int height );
	void SetROI( int roi[4] );
	int  GetROIX();
	int  GetROIY();
	int  GetROIWidth();
	int  GetROIHeight();

	void SetRenderModeToSolid();
	void SetRenderModeToCheckerBoard();
	void SetRenderModeToHoles();

protected:

	bool InitializeGraphicsResources( vtkViewport * viewport );
	bool InitializeTextures();
	bool InitializeShaders();

	void ReleaseGraphicsResources( vtkWindow * window );
	void RenderStreamlines( float min, float max );
    void RenderStreamlines2( float min, float max );
	void RebuildDisplayLists();

	void CheckStatusFBO();
	void PrintShader( const char * text );

	void ApplyParameters( int index, unsigned int progId );
	void ApplyBlurringParameters( int index, unsigned int progId );
	void ApplyOutputParameters( unsigned int progId );

	vtkFiberConfidenceMapper();
	virtual ~vtkFiberConfidenceMapper();

private:

	enum RenderMode
	{
		RENDERMODE_SOLID,
		RENDERMODE_CHECKER_BOARD,
		RENDERMODE_HOLES
	};

	bool ShadersInitialized;
	bool TexturesInitialized;
	bool ExtensionsInitialized;
	bool Orthographic;
	bool ROIEnabled;

	int ROI[4];
	int ScreenSize[2];
	int NumberOfDisplayLists;
	int Mode;

	double DepthThreshold;
	double DepthNear;
	double DepthFar;

	unsigned int * DisplayLists;
	unsigned int   SilhouetteBuffer[2];
	unsigned int   StreamlineBuffer[2];
	unsigned int   BlurringBuffer[2];
	unsigned int   VertexShader;
	unsigned int   BlurringFragShader;
	unsigned int   BlurringProgram;
	unsigned int   OutputFragShader;
	unsigned int   OutputProgram;
	unsigned int   FragShader;
	unsigned int   Program;
	unsigned int   FBO;

	vtkConfidenceTable * Table;
	vtkConfidenceInterval * Interval;

    std::vector<int> * Ids;
    std::vector<float> * Scores;

	vtkFiberConfidenceMapper( const vtkFiberConfidenceMapper & );
	void operator = ( const vtkFiberConfidenceMapper & );
};

#endif
