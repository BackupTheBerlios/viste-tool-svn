varying vec4 vSHCoefficients1;
varying vec4 vSHCoefficients2;
varying vec4 vSHCoefficients3;
varying vec3 vSHCoefficients4;
uniform float StepSize;
uniform int NumRefineSteps;

float cot(float x);
float csc(float x);
float sec(float x);

float infinity();

vec2 SHTTable(int l, int m, float theta, float phi)
{
  vec2 Y; // Y represents an imaginary number. Y[0] is the real part. Y[1] is the imaginary part.

  float temp = 0.0;
  float Pi = 3.14159265;
  if (l == 0)
    {
    temp = (1.0/2.0)*sqrt(1.0/Pi);
    } // if (l == 0)
  else if (l == 2)
    {
    if (m == 0)
      {
      temp = (1.0/4.0)*sqrt(5.0/Pi)*(3.0*pow(cos(theta),2.0) - 1.0);
      }
    else if((m == 1)||(m == -1))
      {
      temp = -1.0*(1.0/2.0)*sqrt(15.0/(2.0*Pi))*sin(theta)*cos(theta);
      } 
    else if((m == 2)||(m == -2))
      {
      temp = (1.0/4.0)*sqrt(15.0/(2.0*Pi))*pow(sin(theta),2.0);
      }
    } // if (l == 2)
  else //if (l == 4)
    {
    if (m == 0)
      {
      // apparently pow(x, 4.0) is undefined if x < 0.
      temp = (3.0/16.0)* sqrt(1.0/Pi)*(35.0*pow(abs(cos(theta)),4.0)-30.0*pow(abs(cos(theta)),2.0) + 3.0);
      }
    else if ((m == 1)||(m == -1))
      {
      temp = -1.0*(3.0/8.0)*sqrt(5.0/Pi)*sin(theta)*(7.0*pow(cos(theta),3.0) - 3.0*cos(theta));
      }
    else if ((m == 2)||(m == -2))
      {
      temp = (3.0/8.0)*sqrt(5.0/(2.0*Pi))*pow(sin(theta),2.0)*(7.0*pow(cos(theta),2.0) - 1.0);
      }
    else if ((m == 3)||(m == -3))
      {
      temp = -1.0*(3.0/8.0)*sqrt(35.0/Pi)*pow(sin(theta),3.0)*cos(theta);
      }
    else if ((m == 4)||(m == -4))
      {
      temp = (3.0/16.0)*sqrt(35.0/(2.0*Pi))*pow(abs(sin(theta)),4.0);
      }
    } // if (l == 4)


    // XXX: The stuff below can be put in RealSHTransform?
    // because some things are computed that are never used.
    // or leave it. This is more clear/general and maybe the compiler
    // takes it out anyway.
    if (m >= 0)
      {
      Y[0] = temp*cos(mod(float(m)*phi, 2.0*Pi));
      Y[1] = temp*sin(mod(float(m)*phi, 2.0*Pi));
      } // if (m >= 0)
    else // if (m < 0)
      {
      if ((m%2) == 0)
        { // m is even
        Y[0] = temp*cos(mod(float(m)*phi, 2.0*Pi));
	Y[1] = temp*sin(mod(-1.0*float(m)*phi, 2.0*Pi));
        } // if ((m%2) == 0)
      else
        { // m is odd
        Y[0] = -1.0*temp*cos(mod(float(m)*phi, 2.0*Pi));
        Y[1] = -1.0*temp*sin(mod(-1.0*float(m)*phi, 2.0*Pi));
        } // else
      } // if (m < 0)

    return Y;
}

float RealSHTransform(int l, int m, float theta, float phi)
{
  float result;

  if (m < 0)
    {
    result = sqrt(2.0) * SHTTable(l,m,theta,phi)[0];
    } // if (m < 0)
  else if (m == 0)
    {
    result = SHTTable(l,0,theta,phi)[0];
    } // if (m == 0)
  else //if (m > 0)
    {
    result = sqrt(2.0) * SHTTable(l,m,theta,phi)[1];
    } // else

  return result;

//  if (l==0) result = 0.5; //SHTTable(0, 0, theta, phi)[0];
//  else result = 0.0;
}

vec3 DSHTable(int l, int m, float theta, float phi)
{
  float Pi = 3.14159265;

  vec3 result;
  if (l == 0)
    {
    result = vec3(0,0,0);
    } // if (l == 0)
  else if (l == 2)
    {
    if (m == -2)
      {
      result.x = -(sqrt(15./Pi)*cos(phi)*sin(theta)*(-1. + cos(2.*phi)*pow(sin(theta),2.0)))/2.;
      result.y = -(sqrt(15./Pi)*sin(theta)*(1. + cos(2.*phi)*pow(sin(theta),2.0))*sin(phi))/2.;
      result.z = -(sqrt(15./Pi)*cos(theta)*cos(2.*phi)*pow(sin(theta),2.0))/2.;
      } // if (m == -2)
    else if (m == -1)
      {
      result.x = (sqrt(15./Pi)*cos(theta)*(pow(cos(theta),2.0) - cos(2.*phi)*pow(sin(theta),2.0)))/2.;
      result.y = -(sqrt(15./Pi)*cos(theta)*cos(phi)*pow(sin(theta),2.0)*sin(phi));
      result.z = (sqrt(15./Pi)*cos(phi)*(sin(theta) - sin(3.*theta)))/4.;

      } // if (m == -1)
    else if (m == 0)
      {
      result.x = (-3.*sqrt(5./Pi)*pow(cos(theta),2.0)*cos(phi)*sin(theta))/2.;
      result.y = (-3.*sqrt(5./Pi)*pow(cos(theta),2.0)*sin(theta)*sin(phi))/2.;
      result.z = (3.*sqrt(5./Pi)*cos(theta)*pow(sin(theta),2.0))/2.;

      } // if (m == 0)
    else if (m == 1)
      {
      result.x = sqrt(15./Pi)*cos(theta)*cos(phi)*pow(sin(theta),2.0)*sin(phi);
      result.y = -(sqrt(15./Pi)*cos(theta)*(pow(cos(theta),2.0) + cos(2.*phi)*pow(sin(theta),2.0)))/2.;
      result.z = (sqrt(15./Pi)*(-sin(theta) + sin(3.*theta))*sin(phi))/4.;

      } // m == 1
    else if (m == 2)
      {
      result.x = (sqrt(15./Pi)*sin(theta)*(pow(cos(theta),2.0) - cos(2.*phi)*pow(sin(theta),2.0))*sin(phi))/2.;
      result.y = (sqrt(15./Pi)*cos(phi)*sin(theta)*(pow(cos(theta),2.0) + cos(2.*phi)*pow(sin(theta),2.0)))/2.;
      result.z = -(sqrt(15./Pi)*cos(theta)*cos(phi)*pow(sin(theta),2.0)*sin(phi));

      } // m == 2
    } // if (l == 2)
  else if (l == 4)
    {
    if (m == -4)
      {
      result.x = (-3.*sqrt(35./Pi)*cos(phi)*pow(sin(theta),3.0)*(1. - 2.*cos(2.*phi) + cos(4.*phi)*pow(sin(theta),2.0)))/4.;
      result.y = (-3.*sqrt(35./Pi)*pow(sin(theta),3.0)*(1. + 2.*cos(2.*phi) + cos(4.*phi)*pow(sin(theta),2.0))*sin(phi))/4.;
      result.z = (-3.*sqrt(35./Pi)*cos(theta)*cos(4.*phi)*pow(sin(theta),4.0))/4.;
      } // if (m == -4)
    else if (m == -3)
      {
      result.x = (3.*sqrt(35./(2.*Pi))*cos(theta)*pow(sin(theta),2.0)*((2. + cos(2.*theta))*cos(2.*phi) - 2.*cos(4.*phi)*pow(sin(theta),2.0)))/4.;
      result.y = (-3.*sqrt(35./(2.*Pi))*cos(theta)*pow(sin(theta),2.0)*(2. + cos(2.*theta) + 4.*cos(2.*phi)*pow(sin(theta),2.0))*sin(2.*phi))/4.;
      result.z = (-3.*sqrt(35./(2.*Pi))*(1. + 2.*cos(2.*theta))*cos(3.*phi)*pow(sin(theta),3.0))/4.;
      } // if (m == -3)
    else if (m == -2)
      {
      result.x = (3.*sqrt(5./Pi)*sin(theta)*((15. + 26.*cos(2.*theta) + 7.*cos(4.*theta))*cos(phi) - 4.*(6. + 7.*cos(2.*theta))*cos(3.*phi)*pow(sin(theta),2.0)))/32.;
      result.y = (-3.*sqrt(5./Pi)*sin(theta)*((15. + 26.*cos(2.*theta) + 7.*cos(4.*theta))*sin(phi) + 4.*(6. + 7.*cos(2.*theta))*pow(sin(theta),2.0)*sin(3.*phi)))/32.;
      result.z = (-3.*sqrt(5./Pi)*(5.*cos(theta) + 7.*cos(3.*theta))*cos(2.*phi)*pow(sin(theta),2.0))/8.;
      } // if (m == -2)
    else if (m == -1)
      {
      result.x = (3.*sqrt(5./(2.*Pi))*cos(theta)*(pow(cos(theta),2.0)*(-3. + 7.*cos(2.*theta)) - (4. + 7.*cos(2.*theta))*cos(2.*phi)*pow(sin(theta),2.0)))/4.;
      result.y = (-3.*sqrt(5./(2.*Pi))*(15.*cos(theta) + 7.*cos(3.*theta))*pow(sin(theta),2.0)*sin(2.*phi))/8.;
      result.z = (3.*sqrt(5./(2.*Pi))*cos(phi)*(sin(theta) + 6.*sin(3.*theta) - 7.*sin(5.*theta)))/16.;
      } // if (m == -1)
    else if (m == 0)
      {
      result.x = (15.*pow(cos(theta),2.0)*cos(phi)*(5.*sin(theta) - 7.*sin(3.*theta)))/(16.*sqrt(Pi));
      result.y = (15.*pow(cos(theta),2.0)*(5.*sin(theta) - 7.*sin(3.*theta))*sin(phi))/(16.*sqrt(Pi));
      result.z = (15.*(9.*cos(theta) + 7.*cos(3.*theta))*pow(sin(theta),2.0))/(16.*sqrt(Pi));
      } // if (m == 0)
    else if (m == 1)
      {
      result.x = (3.*sqrt(5./(2.*Pi))*(15.*cos(theta) + 7.*cos(3.*theta))*pow(sin(theta),2.0)*sin(2.*phi))/8.;
      result.y = (-3.*sqrt(5./(2.*Pi))*cos(theta)*(pow(cos(theta),2.0)*(-3. + 7.*cos(2.*theta)) + (4. + 7.*cos(2.*theta))*cos(2.*phi)*pow(sin(theta),2.0)))/4.;
      result.z = (-3.*sqrt(5./(2.*Pi))*(sin(theta) + 6.*sin(3.*theta) - 7.*sin(5.*theta))*sin(phi))/16.;
      } // if (m == 1)
    else if (m == 2)
      {
      result.x = (3.*sqrt(5./Pi)*sin(theta)*((15. + 26.*cos(2.*theta) + 7.*cos(4.*theta))*sin(phi) - 4.*(6. + 7.*cos(2.*theta))*pow(sin(theta),2.0)*sin(3.*phi)))/32.;
      result.y = (3.*sqrt(5./Pi)*sin(theta)*((15. + 26.*cos(2.*theta) + 7.*cos(4.*theta))*cos(phi) + 4.*(6. + 7.*cos(2.*theta))*cos(3.*phi)*pow(sin(theta),2.0)))/32.;
      result.z = (-3.*sqrt(5./Pi)*(5.*cos(theta) + 7.*cos(3.*theta))*pow(sin(theta),2.0)*sin(2.*phi))/8.;
      } // if (m == 2)
    else if (m == 3)
      {
      result.x = (-3.*sqrt(35./(2.*Pi))*cos(theta)*pow(sin(theta),2.0)*(2. + cos(2.*theta) - 4.*cos(2.*phi)*pow(sin(theta),2.0))*sin(2.*phi))/4.;
      result.y = (-3.*sqrt(35./(2.*Pi))*cos(theta)*pow(sin(theta),2.0)*((2. + cos(2.*theta))*cos(2.*phi) + 2.*cos(4.*phi)*pow(sin(theta),2.0)))/4.;
      result.z = (3.*sqrt(35./(2.*Pi))*(1. + 2.*cos(2.*theta))*pow(sin(theta),3.0)*sin(3.*phi))/4.;
      }
    else if (m == 4)
      {
      result.x = (3.*sqrt(35./Pi)*pow(sin(theta),3.0)*(pow(cos(theta),2.0)*(1. + 2.*cos(2.*phi)) - cos(4.*phi)*pow(sin(theta),2.0))*sin(phi))/4.;
      result.y = (3.*sqrt(35./Pi)*cos(phi)*pow(sin(theta),3.0)*(pow(cos(theta),2.0)*(-1. + 2.*cos(2.*phi)) + cos(4.*phi)*pow(sin(theta),2.0)))/4.;
      result.z = (-3.*sqrt(35./Pi)*cos(theta)*pow(sin(theta),4.0)*sin(4.*phi))/4.;
      } // if (m == 4)
    } // if (l == 4)

  return result;
}

float SHcoefficient(int i)
{
  float coeff;

  // XXX: for testing.
//  if (i == 0) return vSHCoefficients1[i];
//  else return 0.0;

  if (i < 4) coeff = vSHCoefficients1[i];
  else if (i < 8) coeff = vSHCoefficients2[i-4];
  else if (i < 12) coeff = vSHCoefficients3[i-8];
  else if (i < 15) coeff = vSHCoefficients4[i-12];
  else coeff = 0.0; // should not happen.

  return coeff;
}

// returns the radius in the direction given by angles theta and phi
float GetSHRadius(float theta, float phi)
{
  int MaxOrder = 4;
  float r = 0.0; float val;
  int j = 0;
  for (int l=0; l <= MaxOrder; l+=2)
    {
    for (int m=-l; m<=l; m++)
      {
      if(SHcoefficient(j) != 0.0) r+= SHcoefficient(j)*RealSHTransform(l, m, theta, phi);
      j++;
      }
    }

  //return abs(r);

  // Added for min-max normalization:
  r = abs(r);
  return r;
}

float EvaluateSHValue(vec3 SHCenter, vec3 Position)
{
  float dist = length(Position - SHCenter);
  vec3 direction = normalize(Position - SHCenter);
  // TODO: get the angles right. depends on definition.
  float phi;	// angle with XZ-plane
  float theta;	// angle with positive z-axis

  // compute the first eigenvector in spherical coordinates.
  // r == 1, because we have a unit vector.

  theta = atan(length(direction.xy), direction.z);

  // vector along z-axis:
  // TODO: check if it is corrrect to use these signs here. Check definition of atan().
  // (Orange book page 120)

  phi = atan(direction.y, direction.x);

  if (dist < GetSHRadius(theta, phi)) return -1.0;
  else return 1.0;
}

// MaxDist is the maximum distance along the ray to trace.
float RaySHIntersection(vec3 RayPosition, vec3 RayDirection, vec3 SHCenter, float MaxDist)
{
  int i; // current step;

  int MaxNumSteps = int(ceil(MaxDist / StepSize));
  int NumSteps = 0;

  vec3 CurrentPosition = RayPosition;
  vec3 Step = vec3(StepSize) * RayDirection; // assuming that RayDirection was normalized.

  float SHvalue = 100.0;

  while ((SHvalue > 0.0) && (NumSteps < MaxNumSteps))
    {
    CurrentPosition = CurrentPosition + Step;
    SHvalue = EvaluateSHValue(SHCenter, CurrentPosition);
    NumSteps++;
    } // while (NumSteps < MaxNumSteps)

  if (NumSteps == MaxNumSteps) return infinity();

  // linear search is done. now refine using binary search

  // we found a point inside the SH. Now do binary search to refine:
  vec3 lower = CurrentPosition - Step;
  vec3 upper = CurrentPosition;
  int j = 0;
  while (j < NumRefineSteps)
    {
    CurrentPosition = (lower + upper) / vec3(2.0);
    SHvalue = EvaluateSHValue(SHCenter, CurrentPosition);
    if (SHvalue > 0.0) // outside
      {
      lower = CurrentPosition;
      }
    else // inside
      {
      upper = CurrentPosition;
      }
    j++;
    } // while j
  CurrentPosition = (lower + upper) / vec3(2.0);

  return length(CurrentPosition - RayPosition);
}

vec3 SHDerivative(float theta, float phi)
{
  int MaxOrder = 4;
  vec3 d = vec3(0.0);

  int j = 0;
  for (int l=0; l <= MaxOrder; l+=2)
    {
    for (int m=-l; m<=l; m++)
      {
      d += vec3(SHcoefficient(j))*DSHTable(l, m, theta, phi);
      j++;
      }
    }
  return d;
}

vec3 SHNormal(vec3 SHCenter, vec3 Position)
{
  float dist = length(Position - SHCenter);
  vec3 direction = normalize(Position - SHCenter);
  // TODO: get the angles right. depends on definition.
  float phi;	// angle with XZ-plane
  float theta;	// angle with positive z-axis

  // compute the first eigenvector in spherical coordinates.
  // r == 1, because we have a unit vector.

  theta = atan(length(direction.xy), direction.z);
  phi = atan(direction.y, direction.x);

  vec3 GradientR;
  GradientR.x = cos(phi)*sin(theta);
  GradientR.y = sin(theta)*sin(phi);
  GradientR.z = cos(theta);

  vec3 psi = SHDerivative(theta, phi);
  return normalize(GradientR - psi/dist);
}
