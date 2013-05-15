#extension GL_ARB_draw_buffers : enable

/**
 * Depth fragment shader
 */

uniform vec3 EyePosition;
varying vec3 EyeDirection;
uniform float MaxGlyphRadius;
vec3 NEyeDir;
vec3 TenPosW;
vec3 TenPosT;

float infinity();
float GlyphDistanceFromEye(); 
vec3 WorldToTexture(vec3 w);
vec3 Eigenvalues;
mat3 Eigenvectors;
void EigensystemAtTenPos();
//float FragDepthFromDistance(float distance, vec3 EyePosition, vec3 NEyeDir);

void main()
{
  NEyeDir = normalize(EyeDirection);

  TenPosW = gl_TexCoord[0].xyz;
  TenPosT = WorldToTexture(TenPosW);

  // this doesn't seem to make a lot of difference in performance.
  //if (PointLineDistanceSquared(TenPosW, EyePosition, NEyeDir) > r) discard;//MaxGlyphRadius*MaxGlyphRadius) discard;

  // if we are out of the volume, discard
  bool smaller = any(lessThan(TenPosT, vec3(0.0)));
  bool greater = any(greaterThan(TenPosT, vec3(1.0)));
  if (greater||smaller) discard;

  EigensystemAtTenPos();
  // Eigenvalues and Eigenvectors are set now.
//  if (Eigenvalues[2] < 0.05) discard; // TODO: remove this and avoid numerical errors further on!!!
  if (Eigenvalues[2] < 0.05) Eigenvalues[2] = 0.05;
  if (Eigenvalues[1] < 0.05) Eigenvalues[1] = 0.05;

  vec4 color;
  color.a = 1.0;

  float distance = GlyphDistanceFromEye(); //EllipsoidDistanceFromEye(); //TenPosW, EyePosition, NEyeDir);
  if (distance == infinity()) discard;

  vec4 intersection;
  intersection.xyz  = EyePosition + vec3(distance) * NEyeDir;
  intersection.a = 1.0;
  vec4 iproj = gl_ModelViewProjectionMatrix * intersection;
  float depth = 0.5*(iproj.z/iproj.w + 1.0);
  // or use FragDepthFromDistance();

  intersection.a = depth;
  color.rgb = TenPosW;

  gl_FragData[0] = intersection; //vec4(10.0, 5.0, 0.1.0); //intersection; //vec4(1, 0, 0, 1);//iproj; //WorldToTexture(intersection).xyz;
  gl_FragData[1] = color; //TenPosT;

  // this is needed only if glyphs can intersect.
  // if not, standard z-buffer calculations with the cubes suffices.
  gl_FragDepth = depth;
//  gl_FragDepth = gl_FragCoord.z;

} // main()
