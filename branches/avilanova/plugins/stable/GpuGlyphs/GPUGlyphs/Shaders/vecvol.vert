uniform vec3 EyePosition;

uniform vec3 PoiOrigin;
uniform vec3 PoiPoint1;
uniform vec3 PoiPoint2;
uniform vec3 Step1;
uniform vec3 Step2;
varying vec3 PoiNormal;
varying vec4 PoiEquation; // PoiEquation.xyz is the normal of the plane, PoiEquation[3] is the distance from the plane to (0,0,0).

varying vec3 EyeDirection;
uniform float LineLength;
uniform float SeedDistance;

void main()
{
  vec3 PoiTangent = normalize(PoiPoint1 - PoiOrigin);
  vec3 PoiBinormal = normalize(PoiPoint2 - PoiOrigin);
  PoiNormal = cross(PoiTangent, PoiBinormal);
  //PoiDistanceToOrigin = -1.0 * dot(PoiNormal, PoiOrigin);
  PoiEquation.xyz = normalize(cross(PoiTangent, PoiBinormal));
  PoiEquation.w = -1.0 * dot(PoiEquation.xyz, PoiOrigin);

  //EyeDirection = normalize(gl_Vertex.xyz - EyePosition);
  // the normalize above seems to mess everything up.
  // I guess it causes big rounding errors. :S
  // Took me 2 weeks (!) to fix the bug...
  EyeDirection = gl_Vertex.xyz - EyePosition;

  float NdotE = abs(dot(PoiNormal, normalize(EyeDirection)));

  gl_Position = ftransform();
}
