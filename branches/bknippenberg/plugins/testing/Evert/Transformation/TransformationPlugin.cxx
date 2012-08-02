/*
 * TransformationPlugin.cxx
 *
 * 2011-04-27	Evert van Aart
 * - Version 1.0.0.
 * - First version
 *
 */


/** Includes */

#include "TransformationPlugin.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

TransformationPlugin::TransformationPlugin() : Plugin("Transformation")
{
	// Create the GUI of the widget
	this->widget = new QWidget();
	this->ui = new Ui::TransformationForm();
	this->ui->setupUi(this->widget);

	connect(this->ui->flipXButton, SIGNAL(clicked()), this, SLOT(flipImageX()));
	connect(this->ui->flipYButton, SIGNAL(clicked()), this, SLOT(flipImageY()));
	connect(this->ui->flipZButton, SIGNAL(clicked()), this, SLOT(flipImageZ()));
}


//---------------------------------[ init ]--------------------------------\\

void TransformationPlugin::init()
{

}


//------------------------------[ Destructor ]-----------------------------\\

TransformationPlugin::~TransformationPlugin()
{
	delete this->widget;
}


//--------------------------------[ getGUI ]-------------------------------\\

QWidget * TransformationPlugin::getGUI()
{
	// Return the GUI widget
	return this->widget;
}


//-----------------------------[ dataSetAdded ]----------------------------\\

void TransformationPlugin::dataSetAdded(data::DataSet * ds)
{
	if (ds->getKind() == "DTI" || ds->getKind() == "scalar volume")
	{
		if (!(ds->getVtkImageData()))
			return;
		
		QString prefix = "Unknown Image Type - ";

		if (ds->getKind() == "DTI")
			prefix = "DTI - ";
		else if (ds->getKind() == "scalar volume")
			prefix = "Scalars - ";

		this->ui->imageCombo->addItem(prefix + ds->getName());
		this->imageList.append(ds);
	}
}


//----------------------------[ dataSetChanged ]---------------------------\\

void TransformationPlugin::dataSetChanged(data::DataSet * ds)
{

}


//----------------------------[ dataSetRemoved ]---------------------------\\

void TransformationPlugin::dataSetRemoved(data::DataSet * ds)
{

}


void TransformationPlugin::flipImage(int axis)
{
	if (this->ui->imageCombo->currentIndex() < 0 || this->ui->imageCombo->currentIndex() >= this->imageList.size())
		return;

	data::DataSet * ds = this->imageList[this->ui->imageCombo->currentIndex()];

	vtkImageData * image = ds->getVtkImageData();

	if (!image)
		return;

	if (image->GetActualMemorySize() == 0)
	{
		image->Update();
		this->core()->data()->dataSetChanged(ds);
	}

	vtkPointData * pointData = image->GetPointData();

	if (!pointData)
		return;

	QProgressDialog progress;
	progress.setWindowTitle("Transformation");
	progress.setMinimumDuration(500);
	progress.setRange(0, 100);
	progress.setValue(0);

	int numberOfArrays = pointData->GetNumberOfArrays();

	int progressStepSize = image->GetNumberOfPoints() / 25;
	progressStepSize += (progressStepSize == 0) ? 1 : 0;

	for (int arrayIndex = 0; arrayIndex < numberOfArrays; ++arrayIndex)
	{
		vtkDataArray * currentArray = pointData->GetArray(arrayIndex);

		if (!currentArray)
			continue;

		if (currentArray->GetNumberOfTuples() != image->GetNumberOfPoints())
			continue;

		progress.setLabelText("Flipping scalar array #" + QString::number(arrayIndex));

		ImageType arrayType = IT_Scalars;

		QString arrayName = QString(currentArray->GetName());
		if (ds->getKind() == "DTI" && arrayName == "Tensors" && currentArray->GetNumberOfComponents() == 6)
			arrayType = IT_DTI;

		int dims[3];
		image->GetDimensions(dims);

		int dimLimits[3] = {dims[0], dims[1], dims[2]};
		dimLimits[axis] = dimLimits[axis] / 2;
		if ((dims[axis] % 2) != 0)
			dimLimits[axis]++;

		int ijk[3];

		int voxelCount = 0;

		for (ijk[0] = 0; ijk[0] < dimLimits[0]; ++(ijk[0]))					{
		for (ijk[1] = 0; ijk[1] < dimLimits[1]; ++(ijk[1]))					{
		for (ijk[2] = 0; ijk[2] < dimLimits[2]; ++(ijk[2]), ++voxelCount)	{

			if ((voxelCount % progressStepSize) == 0)
			{
				progress.setValue((int) (((float) voxelCount / (float) image->GetNumberOfPoints()) * 200.0f));
			}

			vtkIdType currentPoint = image->ComputePointId(ijk);

			int ijkNew[3] = {ijk[0], ijk[1], ijk[2]};

			ijkNew[axis] = dims[axis] - ijk[axis] - 1;

			vtkIdType targetPoint = image->ComputePointId(ijkNew);

			if (currentPoint == -1 || targetPoint == -1)
				continue;

			if (arrayType == IT_Scalars)
			{
				for (int componentIndex = 0; componentIndex < currentArray->GetNumberOfComponents(); ++componentIndex)
				{
					double tempComponent = currentArray->GetComponent(currentPoint, componentIndex);
					currentArray->SetComponent(currentPoint, componentIndex, currentArray->GetComponent(targetPoint, componentIndex));
					currentArray->SetComponent(targetPoint, componentIndex, tempComponent);
				}
			}

			else if (arrayType == IT_DTI)
			{
				double originalTensorA[6];
				double originalTensorB[6];

				currentArray->GetTuple(currentPoint, originalTensorA);
				currentArray->GetTuple(targetPoint, originalTensorB);
				
				double flippedTensorA[6];
				double flippedTensorB[6];

				vtkTensorMath::Flip(originalTensorA, flippedTensorA, axis);
				vtkTensorMath::Flip(originalTensorB, flippedTensorB, axis);

				currentArray->SetTuple(targetPoint,  flippedTensorA);
				currentArray->SetTuple(currentPoint, flippedTensorB);
			}
		
		} } }
	}

	progress.setValue(100);

	image->Modified();
	this->core()->data()->dataSetChanged(ds);
}


} // namespace


Q_EXPORT_PLUGIN2(libTransformationPlugin, bmia::TransformationPlugin)
