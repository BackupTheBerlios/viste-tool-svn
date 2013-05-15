uniform float GlyphScaling;
mat4 EigensystemAtWorld(vec3 w);
float infinity();
mat3 rotationMatrix(mat3);

float RaySphereIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius);

// all spheres the same size
float SameSizeSphereDistanceFromEye(vec3 Pos, vec3 RayPosition, vec3 RayDirection)
{
  vec3 NRayDirection = normalize(RayDirection); // XXX: check if this is already normalized
  return RaySphereIntersection(RayPosition, NRayDirection, Pos, GlyphScaling);
}

// varying sphere size, depending on C_l
float VariableSizeSphereDistanceFromEye(vec3 TenPos, vec3 RayPosition, vec3 RayDirection)
{
  mat4 e = EigensystemAtWorld(TenPos);
  float result;
  if (e[0].a == 0.0) // for now assuming that C_l == 0 means we have a nulltensor.
    {
    result = infinity();
    }
  else
    {
    vec3 NRayDirection = normalize(RayDirection); // XXX: check if this is already normalized
    // the 0.5 is/was;) there to convert from scaling (diameter) to radius.
    result = RaySphereIntersection(RayPosition, NRayDirection, TenPos, GlyphScaling*e[0].a*0.5);
    }
  return result;
}

float RayAlignedEllipsoidIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 EllipsoidCenter, vec3 EllipsoidRadius);
uniform vec3 EyePosition;

vec3 EllipsoidIntersectionNormal;
vec3 EllipsoidIntersectionPoint;
mat3 Eigenvectors;
vec3 Eigenvalues;
vec3 TenPosW;
vec3 NEyeDir;
float EllipsoidDistanceFromEye() //vec3 TenPos, vec3 RayPosition, vec3 RayDirection)
{
  float result = infinity();
  vec3 scaling = Eigenvalues*vec3(GlyphScaling);

  mat3 rot = rotationMatrix(Eigenvectors);

  // Transform the eye direction to tensor-space
  vec3 dir;
  dir = rot * NEyeDir; // * rot;

  // Transform the eye position to tensor-space
  vec3 pos = EyePosition - TenPosW;
  pos = rot * pos; // * rot;

    //result = RayAlignedEllipsoidIntersection(pos, dir, TenPosW, scaling);
    result = RayAlignedEllipsoidIntersection(pos, dir, vec3(0.0), scaling);
  return result;
}

float GlyphDistanceFromEye() //vec3 TenPos, vec3 RayPosition, vec3 RayDirection)
{
//  return VariableSizeSphereDistanceFromEye(TenPos, RayPosition, RayDirection);
  return EllipsoidDistanceFromEye(); //TenPos, RayPosition, RayDirection);
}
