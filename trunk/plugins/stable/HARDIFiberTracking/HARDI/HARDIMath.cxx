/*
 * HARDIMath.h
 *
 * 2007-12-11	Vesna Prckovska
 * - First version
 *
 * 2010-12-07	Evert van Aart
 * - First version for DTITool3. 
 * - A lot of functions have been disabled. They will be re-enabled when
 *   they're needed by other parts of the tool.
 *
 */


#include "HARDIMath.h"
#include <vtkMath.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <strstream>

#include <assert.h> // XXX: by Tim. see below.

static const double rel_error= 1E-12;



namespace bmia {

	/*
	bool HARDIMath::Equals(double a, double b, double precision) {
		double abs_a, abs_b;

		abs_a = fabs(a);
		abs_b = fabs(b);

		precision *= (abs_a > abs_b) ? abs_a : abs_b;
		//cout << "Precision for test " << a << ", " << b << " Precision: " << precision << " Difference: " << fabs(a-b) << endl;

		return (fabs(a - b) < precision);
	}

	double HARDIMath::erf(double x)

	{
		static const double two_sqrtpi=  1.128379167095512574;        // 2/sqrt(pi)
		if (fabs(x) > 2.2) {
			return 1.0 - erfc(x);        //use continued fraction when fabs(x) > 2.2
		}
		double sum= x, term= x, xsqr= x*x;
		int j= 1;
		do {
			term*= xsqr/j;
			sum-= term/(2*j+1);
			++j;
			term*= xsqr/j;
			sum+= term/(2*j+1);
			++j;
		} while (fabs(term/sum) > rel_error);   // CORRECTED LINE
		return two_sqrtpi*sum;
	}

	double HARDIMath::erfc(double x)
	{
		static const double one_sqrtpi=  0.564189583547756287;        // 1/sqrt(pi)
		if (fabs(x) < 2.2) {
			return 1.0 - erf(x);        //use series when fabs(x) < 2.2
		}
		//if (signbit(x)) {               //continued fraction only valid for x>0
		//	return 2.0 - erfc(-x);
		//}w
		double a=1, b=x;                //last two convergent numerators
		double c=x, d=x*x+0.5;          //last two convergent denominators
		double q1, q2= b/d;             //last two convergents (a/c and b/d)
		double n= 1.0, t;
		do {
			t= a*n+b*x;
			a= b;
			b= t;
			t= c*n+d*x;
			c= d;
			d= t;
			n+= 0.5;
			q1= q2;
			q2= b/d;
		} while (fabs(q1-q2)/q2 > rel_error);
		return one_sqrtpi*exp(-x*x)*q2;
	}


	double HARDIMath::Legendre(int l, double x)
	{
		switch ( l ) {

  case 0 : 
	  return 1;
	  break;

  case 2 : 
	  return 0.5*(3*pow(x,2)-1);
	  break;
  case 4:
	  return 0.125*(35*pow(x,4)-30*pow(x,2)+3);
	  break;
  case 6:
	  return 0.0625*(231*pow(x,6)- 315*pow(x,4)+105*pow(x,2)-5);
	  break;
  case 8:
	  return (1.0/128.0)*(6435*pow(x,8)-12012*pow(x,6)+6930*pow(x,4)-1260*pow(x,2)+35);
	  break;


  default : 
	  return 1;

		}

	}
*/
double HARDIMath::AssociatedLegendrePolynomial(int l, int m, double x)
{
	double fact;
	double pll;
	double pmm;
	double pmmp1;
	double somx2;
	int i;
	int ll;

	if ( (m < 0) || (m > l) || (fabs(x) > 1.0))
	{
		assert(0);
	}

	// Compute P_m^m.
	pmm = 1.0;

	if (m > 0) 
	{
		somx2 = sqrt((1.0 - x) * (1.0 + x));

		fact = 1.0;

		for (i = 1; i <= m; i++) 
		{
			pmm  *= -fact * somx2;
			fact += 2.0;
		}
	}

	if (l == m)
	{
		return pmm;
	}
	//Compute P^m_{m+1}
	else 
	{ 
		pmmp1 = x * (2.0 * m + 1.0) * pmm;

		if (l == (m + 1))
		{
			return pmmp1;
		} 
		// Compute P^m_l , l > m+ 1
		else 
		{
			for (ll = m + 2; ll <= l; ll++) 
			{
				pll   = (x * (2.0 * ll - 1) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
				pmm   = pmmp1;
				pmmp1 = pll;
			}

			return pll;
		}
	}
}


std::complex<double> HARDIMath::SHTTable(int l, int m, double theta, double phi)
{
	std::complex<double> Y;
	double val = 0.0;
	double subfac = 1.0;

	switch (l)
	{
		case 0:	
			val = 0.5 * sqrt(1.0 / vtkMath::DoublePi());
			break;
			
		case 2:
			if (m == 0)
			{
				val = 0.25 * sqrt(5.0 / vtkMath::DoublePi()) * (3.0 * pow(cos(theta), 2.0) - 1.0);
			}
			else if((m == 1) || (m == -1))
			{
				val = -0.5 * sqrt(15.0 / (2.0 * vtkMath::DoublePi())) * sin(theta) * cos(theta);
			}
			else if((m == 2) || (m == -2))
			{
				val = 0.25 * sqrt(15.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 2.0);
			}
			else 
			{
				cout << "Error in SHT";
			}

			break;
			
		case 4:
			if(m == 0)
			{
				val = (3.0 / 16.0) * sqrt(1.0 / vtkMath::DoublePi()) * (35.0 * pow(cos(theta), 4.0) - 30.0 * pow(cos(theta), 2.0) + 3.0);
			}
			else if ((m == 1) || (m == -1))
			{
				val = -(3.0 / 8.0) * sqrt(5.0 / vtkMath::DoublePi()) * sin(theta) * (7.0 * pow(cos(theta), 3.0) - 3.0 * cos(theta));
			}
			else if ((m == 2) || (m == -2))
			{
				val = (3.0 / 8.0) * sqrt(5.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 2.0) * (7.0 * pow(cos(theta), 2.0) - 1.0);
			}
			else if ((m == 3) || (m == -3))
			{
				val = -(3.0 / 8.0) * sqrt(35.0 / vtkMath::DoublePi()) * pow(sin(theta), 3.0) * cos(theta);
			}
			else if ((m == 4) || (m == -4))
			{
				val = (3.0 / 16.0) * sqrt(35.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 4.0);
			}
			else 
			{
				cout << "Error in SHT";
			}
		
			break;

		case 6:
			if (m == 0)
			{
				val = (1.0 / 32.0) * sqrt(13.0 / vtkMath::DoublePi()) * (231.0 * pow(cos(theta), 6.0) - 315.0 * pow(cos(theta), 4.0) + 105.0 * pow(cos(theta), 2.0) - 5.0);
			}
			else if ((m == 1) || (m == -1))
			{
				val = -(1.0 / 16.0) * sqrt(273.0 / (2.0 * vtkMath::DoublePi())) * sin(theta) * (33.0 * pow(cos(theta), 5.0) - 30.0 * pow(cos(theta), 3.0) + 5.0 * cos(theta));
			}
			else if ((m == 2) || (m == -2))
			{
				val = (1.0 / 64.0) * sqrt(1365.0 / vtkMath::DoublePi()) * pow(sin(theta), 2.0) * (33.0 * pow(cos(theta), 4.0) - 18.0 * pow(cos(theta), 2.0) + 1.0);
			}
			else if ((m == 3) || (m == -3))
			{
				val = -(1.0 / 32.0) * sqrt(1365.0 / vtkMath::DoublePi()) * pow(sin(theta), 3.0) * (11.0 * pow(cos(theta), 3.0) - 3.0 * cos(theta));
			}
			else if ((m == 4) || (m == -4))
			{
				val = (3.0 / 32.0) * sqrt(91.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 4.0) * (11.0 * pow(cos(theta), 2.0) - 1.0);
			}
			else if ((m == 5) || (m == -5))
			{
				val = -(3.0 / 32.0) * sqrt(1001.0 / vtkMath::DoublePi()) * pow(sin(theta), 5.0) * cos(theta);
			}
			else if ((m == 6) || (m == -6))
			{
				val = (1.0 / 64.0) * sqrt(3003.0 / vtkMath::DoublePi()) * pow(sin(theta), 6.0);	
			}
		    else 
			{
				cout << "Error in SHT";
			}
		
			break;

		case 8:
			if (m == 0)
			{
				val = (1.0 / 256.0) * sqrt(17.0 / vtkMath::DoublePi()) * (6435.0 * pow(cos(theta), 8.0) - 12012.0 * pow(cos(theta), 6.0) + 6930.0 * pow(cos(theta), 4.0) - 1260.0 * pow(cos(theta), 2.0) + 35.0);
			}
			else if ((m == 1) || (m == -1))
			{
				val = -(3.0 / 64.0) * sqrt(17.0 / (2.0 * vtkMath::DoublePi())) * sin(theta) * (715.0 * pow(cos(theta), 7.0) - 1001.0 * pow(cos(theta), 5.0) + 385.0 * pow(cos(theta), 3.0) - 35.0 * cos(theta));
			}
			else if ((m == 2) || (m == -2))
			{
				val = (3.0 / 128.0) * sqrt(595.0 / vtkMath::DoublePi()) * pow(sin(theta), 2.0) * (143.0 * pow(cos(theta), 6.0) - 143.0 * pow(cos(theta), 4.0) + 33.0 * pow(cos(theta), 2.0) - 1);
			}
			else if ((m == 3) || (m == -3))
			{
				val = -(1.0 / 64.0) * sqrt(19635.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 3.0) * (39.0 * pow(cos(theta), 5.0) - 26.0 * pow(cos(theta), 3.0) + 3.0 * cos(theta));	
			}
			else if ((m == 4) || (m == -4))
			{
				val = (3.0 / 128.0) * sqrt(1309.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 4.0) * (65.0 * pow(cos(theta), 4.0) - 26.0 * pow(cos(theta), 2.0) + 1.0);
			}
			else if ((m == 5) || (m == -5))
			{
				val = -(3.0 / 64.0) * sqrt(17017.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 5.0) * (5.0 * pow(cos(theta), 3.0) - cos(theta));
			}
			else if ((m == 6) || (m == -6))
			{
				val = (1.0 / 128.0) * sqrt(7293.0 / vtkMath::DoublePi()) * pow(sin(theta), 6.0) * (15.0 * pow(cos(theta), 2.0) - 1.0);
			}
			else if ((m == 7) || (m == -7))
			{
				val = -(3.0 / 64.0) * sqrt(12155.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 7.0) * cos(theta);
			}
			else if ((m == 8) || (m == -8))
			{
				val = (3.0 / 256.0) * sqrt(12155.0 / (2.0 * vtkMath::DoublePi())) * pow(sin(theta), 8.0);
			}
			else 
			{
				cout << "Error in SHT";
			}
		
			break;

		default:
			if (m >= 0)
			{
				// Calculate (l + m)! / (l - m + 1)! 
				for(int k = l - m + 1; k <= l + m; k++)
				{
					subfac *= k;
				}

				val = sqrt((2.0 * l + 1.0) / (4.0 * vtkMath::DoublePi() * subfac) ) * HARDIMath::AssociatedLegendrePolynomial(l, m, cos(theta));
			}
			else 
			{
				// Calculate (l - m)! / (l + m + 1)! 
				for(int k = l + m + 1; k <= l - m; k++)
				{
					subfac *= k;
				}

				// "m" is odd
				if (((-m) % 2) != 0)
				{
					val = -sqrt((2.0 * l + 1.0) / (4.0 * vtkMath::DoublePi() * subfac) ) * HARDIMath::AssociatedLegendrePolynomial(l, -m, cos(theta));
				}
				// "m" is even
				else
				{
					val =  sqrt((2.0 * l + 1.0) / (4.0 * vtkMath::DoublePi() * subfac) ) * HARDIMath::AssociatedLegendrePolynomial(l, -m, cos(theta));
				}
			}

			Y = std::complex<double>(cos(m * phi) * val, sin(m * phi) * val);
			return Y;

	} // switch [l]

	if (m >= 0)
	{
		Y = std::complex<double>(val * cos(m * phi), val * sin(m * phi));
	}
	else
	{
		if ((m % 2) == 0)
		{
			Y = std::complex<double>(val * cos( m * phi), -val * sin(-m * phi));
		}
		else
		{
			Y = std::complex<double>(-val * cos( m * phi), val * sin(-m * phi));
		}
	}

	return Y;
}


double HARDIMath::RealSHTransform(int l, int m, double theta, double phi)
{
	if (m < 0)
	{
		return (sqrt(2.0) * HARDIMath::SHTTable(l, m, theta, phi).real());
	}
	else if (m == 0)
	{
		return (HARDIMath::SHTTable(l, m, theta, phi).real());
	}
	else
	{
		return (sqrt(2.0) * HARDIMath::SHTTable(l, m, theta, phi).imag());
	}
}
/*
		MatrixDouble HARDIMath::LaplaceBeltrami(int l, double lambda)
	{
		int R=(l+1)*(l+2)/2;
		int lOrder;
		
		MatrixDouble L(R,R);

		for (int i=0; i<R;i++)
		{
				if (i==0) { lOrder=0; }
				else if ((i>0)&&(i<=5)) {lOrder = 2;}
				else if ((i>5)&&(i<=14)){lOrder = 4;}
				else if ((i>14)&&(i<=27)){lOrder = 6;}
				else if ((i>27)&&(i<=44)) {lOrder = 8;}

				L(i,i) = lambda*lOrder*lOrder*(lOrder+1)*(lOrder+1);
		}
				
		return L;
	}

		MatrixDouble HARDIMath::ScaleSpace(int l, double t)
	{
		double R=(l+1)*(l+2)/2;
		int lOrder;
		
		MatrixDouble S(R,R);

		for (int i=0; i<R;i++)
		{
				if (i==0) { lOrder=0; }
				else if ((i>0)&&(i<=5)) {lOrder = 2;}
				else if ((i>5)&&(i<=14)){lOrder = 4;}
				else if ((i>14)&&(i<=27)){lOrder = 6;}
				else if ((i>27)&&(i<=44)) {lOrder = 8;}

				S(i,i) = exp(t*lOrder*(lOrder+1));
		}
		
		return S;
	}
	MatrixDouble HARDIMath::Tikhonov(int l, double tikhonov)
	{
		double R=(l+1)*(l+2)/2;
		int lOrder;
		
		MatrixDouble T(R,R);

		for (int i=0; i<R;i++)
		{
				if (i==0) { lOrder=0; }
				else if ((i>0)&&(i<=5)) {lOrder = 2;}
				else if ((i>5)&&(i<=14)){lOrder = 4;}
				else if ((i>14)&&(i<=27)){lOrder = 6;}
				else if ((i>27)&&(i<=44)) {lOrder = 8;}

				T(i,i) = tikhonov*lOrder*(lOrder+1);
		}
		
		return T;
	}
*/

} // namespace bmia

