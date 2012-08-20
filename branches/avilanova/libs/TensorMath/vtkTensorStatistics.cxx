/**
 * vtkTensorStatistics.cxx
 *
 * 2007-07-29	Paulo Rodrigues
 * - First version (taken from the other classes).
 *
 * 2010-12-17	Evert van Aart
 * - First version for the DTITool3. I currently only need the function
 *   "LogEuclidean", so I will leave the rest commented out for now.
 *
 */


/** Includes */

#include "vtkTensorStatistics.h"


using namespace std;


namespace bmia {


const char * vtkTensorStatistics::longNames[] =
{
	"Log Euclidean",
	"Geometric",
	"Vemuri"
};

vtkTensorStatistics::vtkTensorStatistics()
{
//	this->tensorSource = NULL;
//	memset(this->meanTensor, 0, sizeof(this->meanTensor));
//	memset(this->meanEigenValues, 0, sizeof(this->meanEigenValues));
//	memset(this->meanEigenVectors, 0, sizeof(this->meanEigenVectors));
//	memset(this->sumLogEuclidean, 0, sizeof(this->sumLogEuclidean));
//	memset(this->A, 0, sizeof(this->A));
//	memset(this->B, 0, sizeof(this->B));

//	this->nPointsInMean = 0;
}

vtkTensorStatistics::~vtkTensorStatistics()
{

}

/*
void vtkTensorStatistics::setTensorSource(vtkImageData* source)
{
	this->tensorSource = source;
}

void vtkTensorStatistics::Reset()
{
	memset(this->meanTensor, 0, sizeof(this->meanTensor));
	memset(this->meanEigenValues, 0, sizeof(this->meanEigenValues));
	memset(this->meanEigenVectors, 0, sizeof(this->meanEigenVectors));
	memset(this->sumLogEuclidean, 0, sizeof(this->sumLogEuclidean));
	memset(this->A, 0, sizeof(this->A));
	memset(this->B, 0, sizeof(this->B));
	this->nPointsInMean = 0;
}

void vtkTensorStatistics::Mean(int mean, double *m1, double *m2, double *outTensor)
{
	//	cout << "Mean:" << this->meanToUse << endl;
	switch(mean)
	{
		case i_LogEuclidean:
			LogEuclidean(m1,m2, outTensor);
			break;
		case i_Geometric:
			Geometric(m1,m2, outTensor);
			break;
		case i_Vemuri:
			Vemuri(m1,m2, outTensor);
			break;
		default:
			cout << "ERROR: unkown mean!" << mean << endl;
	}

	if(vtkTensorMath::IsNullTensor(outTensor))
	{
		cerr << "ERROR: null mean tensor! Try another mean!" << endl;
		printmatrix(outTensor);
		_printtensortomathematica(outTensor);
	}
}
*/

void vtkTensorStatistics::LogEuclidean(double * m1, double * m2, double * outTensor)
{
	double logM1[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double logM2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double sum[9]   = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double prod[9]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	
	// Compute the logarithm tensors
	vtkTensorMath::Log(m1, logM1);
	vtkTensorMath::Log(m2, logM2);

	// Compute the sum of the two logarithm tensors
	vtkTensorMath::Sum(logM1, logM2, sum, 6);

	// Multiply resulting tensor by 0.5
	vtkTensorMath::ProductWithScalar(sum, 0.5, prod, 6);

	// Take exponent of the resulting tensor
	vtkTensorMath::Exp(prod, outTensor);
}

/*
void vtkTensorStatistics::Geometric(double *m1, double *m2, double *outTensor)
{
	double m1inv[9]= {0,0,0,0,0,0,0,0,0};
	double prod[9] = {0,0,0,0,0,0,0,0,0};
	double msqr[9] = {0,0,0,0,0,0,0,0,0};
		
	int res = vtkTensorMath::Inverse(m1, m1inv);
	if(res==0)
	{
		cerr << "GeometricMean: Could not compute inverse matrix!" << endl;
		return;
	}
	
	vtkTensorMath::Product(m1inv, m2, prod);
	vtkTensorMath::SquareRoot(prod, msqr);
	vtkTensorMath::Product(m1, msqr, outTensor);
}

void vtkTensorStatistics::Vemuri(double *m1, double *m2, double *outTensor)
{
	double A[9] 		= {0,0,0,0,0,0,0,0,0};
	double B[9] 		= {0,0,0,0,0,0,0,0,0};
	double invA[9] 		= {0,0,0,0,0,0,0,0,0};
	double invB[9] 		= {0,0,0,0,0,0,0,0,0};

	vtkTensorMath::Sum(m1,m2,A);

	int res = vtkTensorMath::Inverse(m1,invA);
	if(res==0)
	{
		cerr << "VemuriMean: Could not compute inverse matrix 1!" << endl;
		return;
	}
	res = vtkTensorMath::Inverse(m2,invB);
	if(res==0)
	{
		cerr << "VemuriMean: Could not compute inverse matrix 2!" << endl;
		return;
	}
	vtkTensorMath::Sum(invA,invB,B);

	vtkTensorStatistics::Vemuri_AB(A,B,outTensor);

}

void vtkTensorStatistics::Vemuri_AB(double *m1, double *m2, double *outTensor)
{
	double temp1[9]		= {0,0,0,0,0,0,0,0,0};
	double temp2[9]		= {0,0,0,0,0,0,0,0,0};
	double sqB[9] 		= {0,0,0,0,0,0,0,0,0};
	double invB[9]		= {0,0,0,0,0,0,0,0,0};
	double sqinvB[9] 	= {0,0,0,0,0,0,0,0,0};

	vtkTensorMath::SquareRoot(m2, sqB);
	int res = vtkTensorMath::Inverse(m2, invB);
	if(res==0)
	{
		cerr << "CalculateMeanTensor_Vemuri: Could not compute inverse matrix 2!" << endl;
	}
	vtkTensorMath::SquareRoot(invB, sqinvB);
		
	// calculate the mean
	vtkTensorMath::Product(sqB, m1, temp1);
	vtkTensorMath::Product(temp1, sqB, temp2);
	vtkTensorMath::SquareRoot(temp2, temp1);	//middle squareroot
	vtkTensorMath::Product(sqinvB, temp1, temp2);
	vtkTensorMath::Product(temp2, sqinvB, outTensor);
}

void vtkTensorStatistics::Mean(int mean, std::list<vtkIdType>* lst)
{
	//	cout << "Mean:" << this->meanToUse << endl;
	// reset stuff
	this->nPointsInMean = 0;
	memset(this->meanEigenVectors, 0, sizeof(this->meanEigenVectors));
	memset(this->meanEigenValues, 0, sizeof(this->meanEigenValues));
	memset(this->meanTensor, 0, sizeof(this->meanTensor));

	switch(mean)
	{
		case i_LogEuclidean:
			LogEuclidean(lst);
			break;
		case i_Geometric:
			Geometric(lst);
			break;
		case i_Vemuri:
			Vemuri(lst);
			break;
		default:
			cout << "ERROR: unkown mean!" << mean << endl;
	}

	if(vtkTensorMath::IsNullTensor(this->meanTensor))
	{
		cerr << "ERROR: null mean tensor! Try another mean!" << endl;
		printmatrix(this->meanTensor);
		_printtensortomathematica(this->meanTensor);
	}

	_printtensortomathematica(this->meanTensor);

	// update eigensystem of mean tensor
	vtkTensorMath::EigenSystem(this->meanTensor, this->meanEigenVectors, this->meanEigenValues);
}

void vtkTensorStatistics::Geometric(std::list<vtkIdType>* lst)
{
	//cout << "CalculateMeanTensor_Geometric" << endl;
	assert(lst);
	assert(this->tensorSource);

	std::list<vtkIdType>::iterator iter;
	double *tensorA = new double[9];
	double tempMean[9] = {0,0,0,0,0,0,0,0,0};
	double outTensor[9] = {0,0,0,0,0,0,0,0,0};
	
	if(lst->size() > 0)
	{
		// reset the mean
		iter = lst->begin();
		this->tensorSource->GetPointData()->GetTensors()->GetTuple(*iter, tensorA);
		memcpy(outTensor, tensorA, sizeof(outTensor));
		
		iter++;
		int n = 0;
		for(; iter != lst->end(); iter++)
		{
			n++;
			this->tensorSource->GetPointData()->GetTensors()->GetTuple(*iter, tensorA);
			vtkTensorStatistics::Geometric(outTensor, tensorA, tempMean);
			memcpy(outTensor, tempMean, sizeof(outTensor));
		}

		//cout << "number of seed points=" << n << endl;
		memcpy(this->meanTensor, outTensor, sizeof(this->meanTensor));
	}
	delete[]tensorA;
}

void vtkTensorStatistics::LogEuclidean(std::list<vtkIdType>* lst)
{
	//cout << "CalculateMeanTensor_LogEuc" << endl;
	assert(lst);
	assert(this->tensorSource);

	std::list<vtkIdType>::iterator iter;
	double *tensorA = new double[9];
	double logm1[9] = {0,0,0,0,0,0,0,0,0};
	double prod[9]	= {0,0,0,0,0,0,0,0,0};
	double outTensor[9]	= {0,0,0,0,0,0,0,0,0};
	int n = 0;
	this->nPointsInMean = 0;

	if(lst->size() > 0)
	{
		// reset the mean
		memset(this->sumLogEuclidean,0,sizeof(this->sumLogEuclidean));

		// log[mat_]:= Transpose[Eigensystem[mat][[2]]].DiagonalMatrix[Log[Eigensystem[mat][[1]]]].
		//             .Inverse[Transpose[Eigensystem[mat][[2]]]]	
		for(iter = lst->begin(); iter != lst->end(); iter++)
		{
			this->nPointsInMean++;
			this->tensorSource->GetPointData()->GetTensors()->GetTuple(*iter, tensorA);

			vtkTensorMath::Log(tensorA, logm1);
			vtkTensorMath::Sum(logm1,this->sumLogEuclidean, this->sumLogEuclidean);
		}
		vtkTensorMath::ProductWithScalar(this->sumLogEuclidean, 1.0/(double)this->nPointsInMean, prod);		
		vtkTensorMath::Exp(prod, outTensor);

		//calculate eigen of the mean
		memcpy(this->meanTensor, outTensor, sizeof(this->meanTensor));
	}
	delete[]tensorA;
}

void vtkTensorStatistics::Vemuri(std::list<vtkIdType>* lst)
{
	//cout << "CalculateMeanTensor_Vemuri" << endl;
	assert(lst);
	assert(this->tensorSource);

	std::list<vtkIdType>::iterator iter;
	double *tensorA  	= new double[9];
	double inverse[9] 	= {0,0,0,0,0,0,0,0,0};
	double temp1[9]		= {0,0,0,0,0,0,0,0,0};
	double temp2[9]		= {0,0,0,0,0,0,0,0,0};
	double sqB[9] 		= {0,0,0,0,0,0,0,0,0};
	double invB[9]		= {0,0,0,0,0,0,0,0,0};
	double sqinvB[9] 	= {0,0,0,0,0,0,0,0,0};
	double outTensor[9] = {0,0,0,0,0,0,0,0,0};
	this->nPointsInMean = 0;
	int res = 0;

	// A and B
	if(lst->size() > 0)
	{
		for(iter = lst->begin(); iter != lst->end(); iter++)
		{
			this->nPointsInMean++;
			this->tensorSource->GetPointData()->GetTensors()->GetTuple(*iter, tensorA);

			vtkTensorMath::Sum(tensorA, this->A, temp1);
			memcpy(this->A,temp1, sizeof(this->A));
			res = vtkTensorMath::Inverse(tensorA, inverse);
			if(res==0)
			{
				cerr << "CalculateMeanTensor_Vemuri: Could not compute inverse matrix 1!" << endl;
				_printtensortomathematica(tensorA);
			}

			vtkTensorMath::Sum(inverse, this->B, temp2);
			memcpy(this->B,temp2, sizeof(this->B));
		}

		vtkTensorStatistics::Vemuri_AB(this->A, this->B, outTensor);

		//calculate eigen of the mean
		memcpy(this->meanTensor, outTensor, sizeof(this->meanTensor));
	}
	delete[]tensorA;
}


void vtkTensorStatistics::UpdateMean(int mean, double* m2)
{
	assert(m2);

	switch(mean)
	{
	case i_LogEuclidean:
		Update_LogEuclidean(m2);
		break;
	case i_Geometric:
		Update_Geometric(m2);
		break;
	case i_Vemuri:
		Update_Vemuri(m2);
		break;
	default:
		cout << "ERROR: unkown mean!" << endl;
	}
}

void vtkTensorStatistics::Update_LogEuclidean(double *m2)
{
	double logm2[9] = {0,0,0,0,0,0,0,0,0};
	double prod[9]  = {0,0,0,0,0,0,0,0,0};
	
	vtkTensorMath::Log(m2, logm2);
	vtkTensorMath::Sum(this->sumLogEuclidean, logm2, this->sumLogEuclidean);
	this->nPointsInMean++;		// keep track on how many tensors we add to the mean
	vtkTensorMath::ProductWithScalar(this->sumLogEuclidean, 1.0 / this->nPointsInMean, prod);
	vtkTensorMath::Exp(prod, this->meanTensor);
}

void vtkTensorStatistics::Update_Geometric(double *m2)
{
	double temp[9] = {0,0,0,0,0,0,0,0,0};
	this->nPointsInMean++;
	vtkTensorStatistics::Geometric(this->meanTensor, m2, temp);
	memcpy(this->meanTensor, temp, sizeof(this->meanTensor));
}

void vtkTensorStatistics::Update_Vemuri(double *m2)
{
	double *tensorA  	= new double[9];
	double inverse[9] 	= {0,0,0,0,0,0,0,0,0};
	double temp1[9]		= {0,0,0,0,0,0,0,0,0};
	double temp2[9]		= {0,0,0,0,0,0,0,0,0};
	double sqB[9] 		= {0,0,0,0,0,0,0,0,0};
	double invB[9]		= {0,0,0,0,0,0,0,0,0};
	double sqinvB[9] 	= {0,0,0,0,0,0,0,0,0};
	double outTensor[9] = {0,0,0,0,0,0,0,0,0};
	this->nPointsInMean = 0;
	int res = 0;

	this->nPointsInMean++;

	vtkTensorMath::Sum(m2, this->A, temp1);
	memcpy(this->A,temp1, sizeof(this->A));
	res = vtkTensorMath::Inverse(m2, inverse);
	if(res==0)
		{
			cerr << "CalculateMeanTensor_Vemuri: Could not compute inverse matrix 1!" << endl;
			_printtensortomathematica(m2);
		}

	vtkTensorMath::Sum(inverse, this->B, temp2);
	memcpy(this->B,temp2, sizeof(this->B));
	
	vtkTensorMath::SquareRoot(this->B, sqB);
	res = vtkTensorMath::Inverse(this->B, invB);
	if(res==0)
	{
		cerr << "CalculateMeanTensor_Vemuri: Could not compute inverse matrix 2!" << endl;
		_printtensortomathematica(this->B);
	}
	vtkTensorMath::SquareRoot(invB, sqinvB);
	
	// calculate the mean
	vtkTensorMath::Product(sqB, this->A, temp1);
	vtkTensorMath::Product(temp1, sqB, temp2);
	vtkTensorMath::SquareRoot(temp2, temp1);	//middle squareroot
	vtkTensorMath::Product(sqinvB, temp1, temp2);
	vtkTensorMath::Product(temp2, sqinvB, outTensor);

	//calculate eigen of the mean
	memcpy(this->meanTensor, outTensor, sizeof(this->meanTensor));

	delete[]tensorA;
}

void vtkTensorStatistics::Covariance(std::list<vtkIdType>* lst, double* outTensor)
{
	assert(lst);
	assert(this->tensorSource);

	std::list<vtkIdType>::iterator iter;
	double m1[9]      = {0,0,0,0,0,0,0,0,0};
	double logmean[9] = {0,0,0,0,0,0,0,0,0};
	double logm1[9]   = {0,0,0,0,0,0,0,0,0};
	double dif[9]     = {0,0,0,0,0,0,0,0,0};
	double vec[6]	  = {0,0,0,0,0,0};
	double covar[36]; memset(covar,0,sizeof(covar));
	int n = 0;

	vtkTensorMath::Log(this->meanTensor, logmean);

	if(lst->size() > 0)
	{
		for(iter = lst->begin(); iter != lst->end(); iter++)
		{
			n++;
			this->tensorSource->GetPointData()->GetTensors()->GetTuple(*iter, m1);

			vtkTensorMath::Log(m1, logm1);
			vtkTensorMath::Subtraction(logm1, logmean, dif);

			vtkTensorMath::Vec(dif, vec);
			for(int i=0;i<6;i++)
			for(int j=0;j<6;j++)
				covar[6*i + j] = vec[i]*vec[j];

			for(int k=0; k<36;k++)
				outTensor[k] += covar[k];
		}

		for(int k=0; k<36;k++)
			outTensor[k] *= (1.0/static_cast<double>(n));
	}
}

double vtkTensorStatistics::UnbiasMeanSquareDistance(double *covar)
{
	double res = 0.0;
	for(int i=0;i<6;i++)
		res+=covar[7*i];
	return res;
}

*/

} // namespace bmia