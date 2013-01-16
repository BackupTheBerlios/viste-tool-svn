uniform vec3 EyePosition;
uniform vec3 LightPosition;
varying vec3 vEyeDirection;
varying vec3 vGlyphPosition;
varying vec2 vMinMaxRadius;

//uniform float GlyphScaling;
uniform int Coloring; // 0 = direction, 1 = radius, 2 = vertex color
uniform float RadiusThreshold;

float infinity();
float RaySphereIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius);
vec2 RaySphereIntersections(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius);

float RaySHIntersection(vec3 RayPosition, vec3 RayDirection, vec3 SHCenter, float MaxDist);

float diffuse(in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 n, in float p);

float SHcoefficient(int i);
vec3 SHNormal(vec3 SHCenter, vec3 Position);

float diffuse(in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 n, in float p);

//float MinRadius = 0.3*vMaxRadius;

void main()
{
  float saturation = gl_Color[0];
  ///////////////////////////////////////////////////////////////
  // determine the intersection of the view-ray with the glyph //
  ///////////////////////////////////////////////////////////////
  vec3 NEyeDir = normalize(vEyeDirection);
  
  vec2 sphereIntersections = RaySphereIntersections(EyePosition, NEyeDir, vGlyphPosition, vMinMaxRadius[1]);
  // sphereIntersections[0] is the first intersection. sphereIntersections[1] is the last one (other side of the sphere).
  float dist = sphereIntersections[0];
  if (dist >= infinity()) discard;

  // the intersection with the sphere is the starting point for the ray casting
  vec3 StartPosition = EyePosition + vec3(dist)*NEyeDir;
  dist = RaySHIntersection(StartPosition, NEyeDir, vGlyphPosition, sphereIntersections[1]-sphereIntersections[0]); // SH coefficients are obtained in the function via varying variables.
  if (dist >= infinity()) discard;

  vec4 intersection;
  intersection.xyz = StartPosition + vec3(dist) * NEyeDir;
  intersection.w = 1.0;
  // the following is needed to update gl_FragDepth
  vec4 iproj = gl_ModelViewProjectionMatrix * intersection;
  float depth = 0.5*(iproj.z/iproj.w + 1.0);

  //////////////////////
  // compute lighting //
  //////////////////////
  vec3 normal = SHNormal(vGlyphPosition, intersection.xyz);
  vec3 light_direction = normalize(intersection.xyz - LightPosition);
  vec3 eye_direction = normalize(intersection.xyz - EyePosition);

  // compute a color depending on the eigenvalues and main diffusion direction
  vec4 color; color.a = 1.0;

  if (Coloring == 0)
    { // color encodes direction from center to current point on glyph:
    color.rgb = abs(normalize(vGlyphPosition - intersection.xyz));
    //color.rgb = mix(color.rgb, vec3(sqrt(1.0/3.0)), saturation*saturation);
    }
  else if (Coloring == 1)
    { // color encodes radius of the glyph:
    color.rgb = vec3(0.0, 0.0, 0.0);
    float radius = length(intersection.xyz - vGlyphPosition.xyz);
    float mixer = (radius - vMinMaxRadius[0]) / (vMinMaxRadius[1]- vMinMaxRadius[0]);

    vec3 peak_color = vec3(1.0, 1.0, 0.0); //vec3(1.0, 0.5, 1.0);
    vec3 other_color = vec3(0.8, 0.4, 0.4); //vec3(0.5, 0.5, 0.5);

    if (mixer > RadiusThreshold) color.rgb = peak_color;
//    else color.rgb = other_color;
    else color.rgb = mix(vec3(0.4, 0.4, 0.4), vec3(1.0, 0.0, 0.0), mixer);
    }
  else // Coloring == 2
    { // color is defined by vertex color
    color.rgb = gl_Color.rgb;
    }

  float dp = diffuse(light_direction, normal);
  float sp = specular(eye_direction, light_direction, normal, 8.0);

  color.rgb = vec3(0.2+0.7*dp)*color.rgb + vec3(0.5*sp);

  //color.rgb = abs(normal);

  gl_FragColor = color;
  gl_FragDepth = depth; //FragDepthFromDistance(dist, EyePosition, NEyeDir);
}
