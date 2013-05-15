/**
 * Give the rotation matrix for rotation arond
 * the given axis with the given angle.
 * For the axes, we have: 0 --> X, 1 --> Y, 2 --> Z.
 */
mat3 simpleRotationMatrix(int axis, float angle)
{
  mat3 rotation = mat3(0.0);

  if (axis == 0)
    { // rotate around X-axis
    rotation[0] = vec3(1.0, 0.0, 0.0);
    rotation[1] = vec3(0.0, cos(angle), -1.0*sin(angle));
    rotation[2] = vec3(0.0, sin(angle), cos(angle));
    } // if (axis == 0)
  else if (axis == 1)
    { // rotate around Y-axis
    rotation[0] = vec3(cos(angle), 0.0, sin(angle));
    rotation[1] = vec3(0.0, 1.0, 0.0);
    rotation[2] = vec3(-1.0*sin(angle), 0.0, cos(angle));
    } // if (axis == 1)
  else // axis == 2
    { // rotate around Z-axis
    rotation[0] = vec3(cos(angle), -1.0*sin(angle), 0.0);
    rotation[1] = vec3(sin(angle), cos(angle), 0.0);
    rotation[2] = vec3(0.0, 0.0, 1.0);
    } // else

  return rotation;
}

float determinant(mat3 a)
{
 return -1.0*a[0][2]*a[1][1]*a[2][0] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][0]*a[1][2]*a[2][1] - a[0][1]*a[1][0]*a[2][2] + a[0][0]*a[1][1]*a[2][2];
}

mat3 inverse(mat3 matrix)
{
  float a00 = matrix[0][0]; float a01 = matrix[0][1]; float a02 = matrix[0][2];
  float a10 = matrix[1][0]; float a11 = matrix[1][1]; float a12 = matrix[1][2];
  float a20 = matrix[2][0]; float a21 = matrix[2][1]; float a22 = matrix[2][2];

  mat3 inv;

  inv[0] = vec3((a12*a21 - a11*a22)/(a02*a11*a20 - a01*a12*a20 - a02*a10*a21 + 
     a00*a12*a21 + a01*a10*a22 - a00*a11*a22), 
     (a02*a21 - a01*a22)/((-a02)*a11*a20 + a01*a12*a20 + 
     a02*a10*a21 - a00*a12*a21 - a01*a10*a22 + a00*a11*a22), 
     (a02*a11 - a01*a12)/(a02*a11*a20 - a01*a12*a20 - a02*a10*a21 + 
     a00*a12*a21 + a01*a10*a22 - a00*a11*a22));

  inv[1] = vec3((a12*a20 - a10*a22)/((-a02)*a11*a20 + a01*a12*a20 + a02*a10*a21 - 
     a00*a12*a21 - a01*a10*a22 + a00*a11*a22), 
     (a02*a20 - a00*a22)/(a02*a11*a20 - a01*a12*a20 - a02*a10*a21 + 
     a00*a12*a21 + a01*a10*a22 - a00*a11*a22), 
     (a02*a10 - a00*a12)/((-a02)*a11*a20 + a01*a12*a20 + 
     a02*a10*a21 - a00*a12*a21 - a01*a10*a22 + a00*a11*a22));

  inv[2] = vec3((a11*a20 - a10*a21)/(a02*a11*a20 - a01*a12*a20 - a02*a10*a21 + 
     a00*a12*a21 + a01*a10*a22 - a00*a11*a22), 
     (a01*a20 - a00*a21)/((-a02)*a11*a20 + a01*a12*a20 + 
     a02*a10*a21 - a00*a12*a21 - a01*a10*a22 + a00*a11*a22), 
     (a01*a10 - a00*a11)/(a02*a11*a20 - a01*a12*a20 - a02*a10*a21 + 
     a00*a12*a21 + a01*a10*a22 - a00*a11*a22));

  return inv;
}

mat4 TranslateMatrix(vec3 translate)
{
  mat4 m = mat4(0.0);
  //for (int i=0; i < 4; i++) m[i][i] = 1.0;
  m[0][0] = 1.0;
  m[1][1] = 1.0;
  m[2][2] = 1.0;
  m[3][3] = 1.0;
  m[0][3] = translate.x;
  m[1][3] = translate.y;
  m[2][3] = translate.z;
  return m;
}

mat4 IsotropicScaleMatrix(float scale)
{
  mat4 m = mat4(0.0);
  // for some reason the line below doesn't work. :S :S
  //for (int i=0; i < 3; i++) m[i][i] = 1.0/scale;
  // and it DOES work when I write it out like below.
  m[0][0] = 1.0/scale;
  m[1][1] = 1.0/scale;
  m[2][2] = 1.0/scale;
  m[3][3] = 1.0;
  return m;
}

mat4 ScaleMatrix(vec3 scale)
{
  mat4 m = mat4(0.0);
  m[0][0] = 1.0/scale.x;
  m[1][1] = 1.0/scale.y;
  m[2][2] = 1.0/scale.z;
  m[3][3] = 1.0;
  return m;
}

// rotation matrix that aligns the main axis of a glyph with the given vector.
mat3 rotationMatrixV(vec3 direction)
{
  mat3 mat;

  float phi;	// angle with XZ-plane
  float theta;	// angle with positive z-axis

  // compute the first eigenvector in spherical coordinates.
  // r == 1, because we have a unit vector.

  theta = asin(direction.z);
  theta = theta * sign(direction.x) * -1.0;

  // vector along z-axis:
  if (length(direction.xy) < 0.01) phi = 0.0;
  else phi = atan(direction.y, length(direction.xz)*sign(direction.x)*sign(direction.z));
  // TODO: check if it is corrrect to use these signs here. Check definition of atan().
  // (Orange book page 120)

  //float PI = 3.14159;
  // 0 <= theta <= PI
  // 0 <= phi <= 2*PI

  mat3 rotateAroundZ = simpleRotationMatrix(2, phi);
  mat3 rotateAroundY = simpleRotationMatrix(1, -1.0*theta);

  mat = rotateAroundY*rotateAroundZ;

  return mat;
}

mat3 transposeMatrix(mat3 matrix)
{
    mat3 result = mat3(0.0);
    for(int i = 0; i < 3; ++i )
        for(int j = 0; j < 3; ++j )
            result[i][j] = matrix[j][i];
    return result;
}

mat3 rotationMatrix(mat3 Eigenvectors)
{
//  mat3 result = mat3(0.0);
//  for (int i=0; i < 3; i++) result[i][i] = 1.0;
//  return result;

  return transposeMatrix(Eigenvectors);
//  return Eigenvectors;

/*
  vec3 e1 = Eigenvectors[0];
  vec3 e2 = Eigenvectors[1];

  float theta;	// angle with positive X-axis
  float phi;	// angle with XY-plane
  float gamma;  // rotation around e1

  theta = atan(e1.y, e1.x);
  phi = atan(e1.z, length(e1.xy));

  mat3 Ap = simpleRotationMatrix(2, 1.0*theta);
  mat3 Bp = simpleRotationMatrix(1, 1.0*phi);
  mat3 An = simpleRotationMatrix(2, -1.0*theta);
  mat3 Bn = simpleRotationMatrix(1, -1.0*phi);

  mat3 combinedRotationMatrix;
//  combinedRotationMatrix = simpleRotationMatrix(2, -1.0*theta)*simpleRotationMatrix(1, 1.0*phi);
  combinedRotationMatrix = simpleRotationMatrix(1, -1.0*phi)*simpleRotationMatrix(2, 1.0*theta);
  combinedRotationMatrix = An*Bp;
//  combinedRotationMatrix = Bn*Bp;

  vec3 e2transformed = e2*combinedRotationMatrix;
  gamma = atan(e2transformed.z, e2transformed.y);

  mat3 Cp = simpleRotationMatrix(0, 1.0*gamma);
  mat3 Cn = simpleRotationMatrix(0, -1.0*gamma);

//  gamma = atan(e2.z*cos(phi) - (e2.x*cos(theta) + e2.z*sin(theta))*sin(phi),
//		e2.y*cos(theta) - e2.x*sin(theta));

//  combinedRotationMatrix = combinedRotationMatrix * simpleRotationMatrix(0, -1.0*gamma);
//  combinedRotationMatrix = simpleRotationMatrix(0, 1.0*gamma)*combinedRotationMatrix;

  combinedRotationMatrix = An*Bp*Cn;
//  combinedRotationMatrix = Cp*Bn*Ap;

  float d = determinant(combinedRotationMatrix);
//  if ((d > 0.95) && (d < 1.05)) return inverse(combinedRotationMatrix);
//  else return mat3(0.0);

//  if (abs(dot(e1, e2)) > 0.01) return mat3(0.0);
//  else if (abs(dot(e1, Eigenvectors[2])) > 0.01) return mat3(0.0);
//  else if (abs(dot(e2, Eigenvectors[2])) > 0.01) return mat3(0.0);
//  else
  return combinedRotationMatrix;
*/
}

