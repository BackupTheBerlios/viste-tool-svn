/*
 * vtkIllustrativeFiberBundleMapper.h
 *
 * 2009-03-26	Ron Otten
 * - First Version.
 *
 * 2011-03-28	Evert van Aart
 * - Ported to DTITool3.
 * - Changed initialization to avoid unneccesary re-initialization.
 * - Removed a million VTK error messages if/when the 3D subcanvas is hidden.
 *
 */


#ifndef bmia_vtkIllustrativeFiberBundleMapper_h
#define bmia_vtkIllustrativeFiberBundleMapper_h

#include <vtkObjectFactory.h>
#include <vtkPolyDataMapper.h>
#include <vtkWindow.h>
#include <vtkViewport.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include "GpuBuffers/VertexBuffer.h"
#include "GpuBuffers/FrameBuffer.h"
#include "GpuPrograms/GpuProgram.h"

namespace bmia
{

class vtkIllustrativeFiberBundleMapper: public vtkPolyDataMapper
{
public:
	static vtkIllustrativeFiberBundleMapper* New();
	vtkTypeRevisionMacro(vtkIllustrativeFiberBundleMapper, vtkPolyDataMapper)
	void PrintSelf(ostream& os, vtkIndent indent);

	virtual void ReleaseGraphicsResources(vtkWindow *window);
	virtual bool InitializeGraphicsResources(vtkViewport* viewport);

	virtual void RenderPiece(vtkRenderer *renderer, vtkActor *actor);
	virtual void Render(vtkRenderer* renderer, vtkActor* actor);

	inline float GetFinWidth() const { return mFinWidth; }
	inline void SetFinWidth(float width) { mFinWidth = std::max<float>(0.001f, width); }

	inline float GetFinRecision() const { return mFinRecision; }
	inline void SetFinRecision(float recision) { mFinRecision = std::max<float>(0.0f, recision); }	

	inline const float* GetFillColor() const { return mFillColor; }
	inline void SetFillColor(float red, float green, float blue)
	{
		mFillColor[0] = std::min<float>(1.0f, std::max<float>(0.0f, red));
		mFillColor[1] = std::min<float>(1.0f, std::max<float>(0.0f, green));
		mFillColor[2] = std::min<float>(1.0f, std::max<float>(0.0f, blue));
	}

	inline const float* GetLineColor() const { return mLineColor; }
	inline void SetLineColor(float red, float green, float blue)
	{
		mLineColor[0] = std::min<float>(1.0f, std::max<float>(0.0f, red));
		mLineColor[1] = std::min<float>(1.0f, std::max<float>(0.0f, green));
		mLineColor[2] = std::min<float>(1.0f, std::max<float>(0.0f, blue));
	}

	inline unsigned int GetOutlineWidth() const { return mOutlineWidth; }
	inline void SetOutlineWidth(unsigned int width) { mOutlineWidth = width; }

	inline unsigned int GetFillDilation() const { return mFillDilation; }
	inline void SetFillDilation(unsigned int dilation) { mFillDilation = dilation; }

	inline float GetInnerOutlineDepthThreshold() const { return mInnerOutlineDepthThreshold; }
	inline void SetInnerOutlineDepthThreshold(float threshold)
	{
		mInnerOutlineDepthThreshold = threshold;
	}

	inline float GetMinimumStrokeWidth() const { return mMinStrokeWidth; }
	inline void SetMinimumStrokeWidth(float width)
	{
		mMinStrokeWidth = std::min<float>(std::max<float>(0, width), 1);
	}

	inline float GetMaximumStrokeWidth() const { return mMaxStrokeWidth; }
	inline void SetMaximumStrokeWidth(float width)
	{
		mMaxStrokeWidth = std::min<float>(std::max<float>(0, width), 1);
	}

	inline float GetMinimumLuminosity() const { return mMinLuminosity; }
	inline void SetMinimumLuminosity(float luminosity) 
	{
		mMinLuminosity = std::min<float>(std::max<float>(0, luminosity), 1);
	}

	inline float GetMaximumLuminosity() const { return mMaxLuminosity; }
	inline void SetMaximumLuminosity(float luminosity)
	{
		mMaxLuminosity = std::min<float>(std::max<float>(0, luminosity), 1);
	}

	inline const float* GetLighting() const { return mLighting; }
	inline void SetLighting(float ambient, float diffuse, float specular)
	{
		mLighting[0] = std::min<float>(1.0f, std::max<float>(0.0f, ambient));
		mLighting[1] = std::min<float>(1.0f, std::max<float>(0.0f, diffuse));;
		mLighting[2] = std::min<float>(1.0f, std::max<float>(0.0f, specular));;
	}

	inline unsigned int GetShinyness() const { return mShinyness; }
	inline void SetShinyness(unsigned int shinyness)
	{
		mShinyness = std::max<unsigned int>(1, shinyness);
	}

	inline bool IsUsingStroking() const { return mUseStroking; }
	inline void UseStroking(bool use) { mUseStroking = use; }

	inline bool HasLitLines() const { return mLightLines; }
	inline void ApplyLightingToLines(bool apply) { mLightLines = apply; }

	inline bool HasSilhouette() const { return mSilhouette; }
	inline void CreateSilhouette(bool silhouette) { mSilhouette = silhouette; }

protected:
	vtkIllustrativeFiberBundleMapper();
	virtual ~vtkIllustrativeFiberBundleMapper();	

private:
	vtkIllustrativeFiberBundleMapper(const vtkIllustrativeFiberBundleMapper&); // Not implemented.
	void operator=(const vtkIllustrativeFiberBundleMapper&);      // Not implemented.

	const int GEOMETRY_OUTPUT_VERTICES;

	float mFinWidth;
	float mFinRecision;

	bool mUseStroking;
	float mMinStrokeWidth;
	float mMaxStrokeWidth;

	bool mLightLines;
	float mMinLuminosity;
	float mMaxLuminosity;

	float mInnerOutlineDepthThreshold;

	bool mSilhouette;
	unsigned int mOutlineWidth;
	unsigned int mFillDilation;

	float mFillColor[3];
	float mLineColor[3];
	float mLighting[3];
	unsigned int mShinyness;

	opengl::GpuProgram* mFinProgram;
	opengl::GpuProgram* mLineProgram;
	opengl::GpuProgram* mSilhouetteProgram;
	opengl::FrameBuffer* mFrameBuffer;

	bool initialized;

	bool createFrameBuffer(int w, int h);
};

}

#endif // bmia_vtkIllustrativeFiberBundleMapper_h
