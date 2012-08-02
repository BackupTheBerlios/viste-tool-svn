uniform float StepSize;
uniform int NumRefineSteps;
varying float vCameraZ;
varying vec3 vRelativeVertexPosition;

float infinity();
float SHcoefficient(int i);

const float Pi = 3.14159265;

float RealCylindricalHarmonicY0()
{
  return 1.0/(2.0*sqrt(Pi));
}

float RealCylindricalHarmonicY2(int m, float z, float rho, float phi)
{
  float v;
  if (m == -2)
    {
    v = 0.5 * rho*rho * cos(2.0*phi);
    } // m == -2
  else if (m == -1)
    {
    v = z * rho * cos(phi);
    } // m == -1
  else if (m == 0)
    {
    v = (2.0 * z*z - rho*rho)/(2.0 * sqrt(3.0));
    } // m == 0
  else if (m == 1)
    {
    v = -1.0 * z * rho * sin(phi);
    }
  else if (m == 2)
    {
    v = 0.5 * rho*rho * sin(2.0*phi);
    } // m == 2

  v *= 0.5*sqrt(15.0/Pi);
  return v/(z*z+rho*rho);
}

float RealCylindricalHarmonicY4(int m, float z, float rho, float phi)
{
  float v;
  if (m == -4)
    {
    v = 0.25 * sqrt(7.0) * pow(rho, 4.0) * cos(4.0 * phi);
    } // m == -4
  else if (m == -3)
    {
    v = sqrt(7.0/2.0) * z * pow(rho, 3.0) * cos(3.0 * phi);
    }
  else if (m == -2)
    {
    v = 0.5 * rho*rho * (6.0*z*z - rho*rho) * cos(2.0 * phi);
    }
  else if (m == -1)
    {
    v = z * rho * (4.0*z*z - 3.0*rho*rho) * cos(phi);
    v /= sqrt(2.0);
    }
  else if (m == 0)
    {
    v = 8.0 * pow(z, 4.0) - 24.0*z*z*rho*rho + 3.0 * pow(rho, 4.0);
    v /= 4.0 * sqrt(5.0);
    } // m == 0
  else if (m == 1)
    {
    v = -1.0 * z * rho * (4.0*z*z - 3.0*rho*rho) * sin(phi);
    v /= sqrt(2.0);
    } // m == 1
  else if (m == 2)
    {
    v = 0.5 * rho*rho * (6.0*z*z - rho*rho) * sin(2.0 * phi);
    } // m == 2
  else if (m == 3)
    {
    v = -1.0 * sqrt(7.0/2.0) * z * pow(rho, 3.0) * sin(3.0 * phi);
    }
  else if (m == 4)
    {
    v = 0.25 * sqrt(7.0) * pow(rho, 4.0) * sin(4.0 * phi);
    }

  v *= 0.75 * sqrt(5.0 / Pi);
  float zr = (z*z + rho*rho);
  return v/(zr*zr);
}

// returns the radius in the direction given by angles theta and phi
float RealCylindricalHarmonicY(float z, float rho, float phi)
{
  //int MaxOrder = 4;
  //float r = 0.0;

  float r = SHcoefficient(0) * RealCylindricalHarmonicY0();
//  r = 0.0;
  r += SHcoefficient(1) * RealCylindricalHarmonicY2(-2, z, rho, phi);
  r += SHcoefficient(2) * RealCylindricalHarmonicY2(-1, z, rho, phi);
  r += SHcoefficient(3) * RealCylindricalHarmonicY2(0, z, rho, phi);
  r += SHcoefficient(4) * RealCylindricalHarmonicY2(1, z, rho, phi);
  r += SHcoefficient(5) * RealCylindricalHarmonicY2(2, z, rho, phi);
  r += SHcoefficient(6) * RealCylindricalHarmonicY4(-4, z, rho, phi);
  r += SHcoefficient(7) * RealCylindricalHarmonicY4(-3, z, rho, phi);
  r += SHcoefficient(8) * RealCylindricalHarmonicY4(-2, z, rho, phi);
  r += SHcoefficient(9) * RealCylindricalHarmonicY4(-1, z, rho, phi);
  r += SHcoefficient(10) * RealCylindricalHarmonicY4(0, z, rho, phi);
  r += SHcoefficient(11) * RealCylindricalHarmonicY4(1, z, rho, phi);
  r += SHcoefficient(12) * RealCylindricalHarmonicY4(2, z, rho, phi);
  r += SHcoefficient(13) * RealCylindricalHarmonicY4(3, z, rho, phi);
  r += SHcoefficient(14) * RealCylindricalHarmonicY4(4, z, rho, phi);

  return r;
}

bool IsInsideCylindricalHarmonic(float z, float rho, float phi)
{
  bool result;
  float ch = RealCylindricalHarmonicY(z, rho, phi);
  if (ch*ch - abs(z*z) > rho*rho) result = true;
  else result = false;

//  if (rho < 0.2) result = true; else result = false;

  return result;
}


float C0even(float phi)
{
return (SHcoefficient(0) + sqrt(5.0)*SHcoefficient(3) + 3.0*SHcoefficient(10))/(2.0*sqrt(Pi));
}

float C1even(float phi)
{
return (sqrt(5.0/Pi)*((sqrt(3.0)*SHcoefficient(2) + 3.0*sqrt(2.0)*SHcoefficient(9))*cos(phi) - (sqrt(3.0)*SHcoefficient(4) + 3.0*sqrt(2.0)*SHcoefficient(11))*sin(phi)))/2.0;
}

float C2even(float phi)
{
return (4.0*SHcoefficient(0) + sqrt(5.0)*SHcoefficient(3) - 18.0*SHcoefficient(10) + sqrt(5.0)*(sqrt(3.0)*SHcoefficient(1) + 9.0*SHcoefficient(8))*cos(2.0*phi) + sqrt(5.0)*(sqrt(3.0)*SHcoefficient(5) + 9.0*SHcoefficient(12))*sin(2.0*phi))/(4.0*sqrt(Pi));
}

float C3even(float phi)
{
return (sqrt(5.0/Pi)*((4.0*sqrt(3.0)*SHcoefficient(2) - 9.0*sqrt(2.0)*SHcoefficient(9))*cos(phi) + 3.0*sqrt(14.0)*SHcoefficient(7)*cos(3.0*phi) - 4.0*sqrt(3.0)*SHcoefficient(4)*sin(phi) + 9.0*sqrt(2.0)*SHcoefficient(11)*sin(phi) - 3.0*sqrt(14.0)*SHcoefficient(13)*sin(3.0*phi)))/8.0;
}

float C4even(float phi)
{
return (8.0*SHcoefficient(0) - 4.0*sqrt(5.0)*SHcoefficient(3) + 9.0*SHcoefficient(10) + 2.0*sqrt(5.0)*(2.0*sqrt(3.0)*SHcoefficient(1) - 3.0*SHcoefficient(8))*cos(2.0*phi) + 3.0*sqrt(35.0)*SHcoefficient(6)*cos(4.0*phi) + 2.0*sqrt(5.0)*(2.0*sqrt(3.0)*SHcoefficient(5) - 3.0*SHcoefficient(12))*sin(2.0*phi) + 3.0*sqrt(35.0)*SHcoefficient(14)*sin(4.0*phi))/(16.0*sqrt(Pi));
}

float Ceven(int lmax, float phi)
{
  float result = 0.0;
  if (lmax == 0) result = C0even(phi);
  else if (lmax == 1) result = C1even(phi);
  else if (lmax == 2) result = C2even(phi);
  else if (lmax == 3) result = C3even(phi);
  else if (lmax == 4) result = C4even(phi);
  return result;
}

float chPolynomialValue(float Ce[5], float rho, float z)
{
    float sumit = 0.0;
    float zp = abs(z);
    float s;
    for (int n=0; n <=4; n++)
      {
      // XXX: There is a problem for z < 0. The code below fixes this but its not ideal.
      //  It would be better to fix the IsInsideCylindricalHarmonic for negative z.
      if ((z < 0.0) && ((n==1)||(n==3))) s = -1.0;
      else s = 1.0;
      sumit += Ce[n]*pow(rho, float(n))*pow(zp, float(4-n))*s;
      } // for n
    //if (sumit > pow(sqrt(rho*rho+zp*zp), 5.0)) isInside = true;
    //else isInside = false;
    return sumit - pow(sqrt(rho*rho+zp*zp), 5.0);
}

// Do ray casting. Camera is placed along the z-axis with z=vCameraZ.
// Glyph center is (0,0,0).
// bounds contains the start- and end-distances for the ray casting.
//float RayCHIntersection(vec2 bounds)
vec3 RayCHIntersection(vec2 bounds)
{
  if (bounds[0] >= infinity()) discard; //return vec3(-1.0); //infinity();

  int MaxNumSteps = int(ceil((bounds[1]-bounds[0]) / StepSize));
//  MaxNumSteps *=10;
  int NumSteps = 0;

  float z = vCameraZ;
  vec3 RayDirection = normalize(vRelativeVertexPosition.xyz - vec3(0.0, 0.0, z));
  vec3 Step = vec3(StepSize) * RayDirection;
  vec3 CurrentPosition = vec3(0.0, 0.0, z) + vec3(bounds[0])*RayDirection;

  float phi = atan(vRelativeVertexPosition.y, vRelativeVertexPosition.x);
  float rho;

  // ADDED FOR THE NEW REPRESENTATION
  float C[5];
  C[0] = C0even(phi);
  C[1] = C1even(phi);
  C[2] = C2even(phi);
  C[3] = C3even(phi);
  C[4] = C4even(phi);

  int n;
  float polValue;

  bool isInside = false;
  while (!(isInside) && (NumSteps < MaxNumSteps))
    {
    CurrentPosition += Step;
    rho = length(CurrentPosition.xy);
    z = CurrentPosition.z;

    polValue = chPolynomialValue(C, rho, z);
    if (polValue > 0.0) isInside = true;

    NumSteps++;
    }

  if (!(isInside)) discard; //return vec3(-1.0); //infinity();
  if (NumRefineSteps == 0) return CurrentPosition;

  // linear search is done. Now refine using binary search
  vec3 lower = CurrentPosition - Step;
  rho = length(lower.xy); z = lower.z;
  float lowerValue = chPolynomialValue(C, rho, z);

  vec3 upper = CurrentPosition;
  rho = length(upper.xy); z = upper.z;
  float upperValue = chPolynomialValue(C, rho, z);

  int j = 1;
  while (j < NumRefineSteps)
    {
    //CurrentPosition = vec3(0.5)*(upper-lower) + lower;
    // even better, interpolate:
    // assuming: upperValue > 0 && lowerValue < 0
//XXX: this doesn't work well. why?
//    CurrentPosition = vec3(-1.0*lowerValue/(upperValue-lowerValue))*(upper-lower) + lower;
    CurrentPosition = vec3(upperValue/(upperValue-lowerValue))*(lower-upper)+upper;
    rho = length(CurrentPosition.xy);
    z = CurrentPosition.z;

    polValue = chPolynomialValue(C, rho, z);

    if (polValue > 0.0)
      {
      upper = CurrentPosition;
      upperValue = polValue;
      }
    else // outside
      {
      lower = CurrentPosition;
      lowerValue = polValue;
      }
    j++;
    } // while j
//  CurrentPosition = (lower + upper) / vec3(2.0);
  CurrentPosition = vec3(upperValue/(upperValue-lowerValue))*(lower-upper)+upper;

  return CurrentPosition;
}


