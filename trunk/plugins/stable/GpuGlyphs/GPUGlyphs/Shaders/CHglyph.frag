uniform vec3 EyePosition;
//uniform vec3 LightPosition;
varying vec3 vEyeDirection;
varying vec3 vGlyphPosition;
varying vec4 vMinMaxRadius;
varying vec3 vRelativeVertexPosition;
varying float vCameraZ;

uniform int Coloring; // 0 = direction, 1 = radius, 2 = vertex color
uniform float RadiusThreshold;

float infinity();
float RaySphereIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius);
vec2 RaySphereIntersections(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius);

vec3 RayCHIntersection(vec2 bounds);

float diffuse(in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 n, in float p);

float SHcoefficient(int i);
vec3 SHNormal(vec3 SHCenter, vec3 Position);

float diffuse(in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 n, in float p);

void main()
{
  float CylinderR = vMinMaxRadius[2];
  float CylinderZ = vMinMaxRadius[3];
  vec2 bounds;

  ///////////////////////////////////////////////////////////////
  // determine the intersection of the view-ray with the glyph //
  ///////////////////////////////////////////////////////////////
  vec3 NEyeDir = normalize(vEyeDirection);
  
  // compute intersections with the sphere
  vec2 sphereIntersections = RaySphereIntersections(EyePosition, NEyeDir, vGlyphPosition, vMinMaxRadius[1]);
  // sphereIntersections[0] is the first interesection. sphereIntersections[1] is the last one (back of the sphere).
  float dist = sphereIntersections[0];
  if (dist >= infinity()) discard;

  bounds = sphereIntersections;

  // compute intersections with cylinder. In cylindrical coordinates. Eye position is (0, 0, vCameraZ).
  vec3 RayDirection = normalize(vRelativeVertexPosition.xyz - vec3(0.0, 0.0, vCameraZ)); // in glyph coordinate system.
  vec2 cylinderIntersections;
  float d = length(RayDirection.xy);
  if (d > 0.0)
    {
    cylinderIntersections[0] = -1.0*CylinderR/d; // behind the camera actually. so pointless. Value always smaller than bounds[0] (if there is intersection)
    cylinderIntersections[1] = CylinderR/d;
    bounds[0] = max(bounds[0], cylinderIntersections[0]);
    bounds[1] = min(bounds[1], cylinderIntersections[1]); 
    } // if (d > 0.0)

  float dldz = length(RayDirection/vec3(RayDirection.z));
  bounds[0] = max(bounds[0], dldz*(vCameraZ - CylinderZ));
  bounds[1] = min(bounds[1], dldz*(vCameraZ + CylinderZ));

//  bounds[0] = dldz*(vCameraZ - CylinderZ);
  if (bounds[0] < 0.0) discard; // glyph behind camera or camera inside glyph
//  bounds[1] = dldz*(vCameraZ + CylinderZ);
  if (d > 0.0)
    {
    float cylIntersection = CylinderR/d;
    if (cylIntersection < bounds[0]) discard;
    bounds[1] = min(bounds[1], cylIntersection);
    }

  // the narrowest intersections are the bounds for the ray casting

  //dist = RayCHIntersection(sphereIntersections);
  //dist = RayCHIntersection(bounds);
  vec3 pos = RayCHIntersection(bounds);

  dist = length(pos - vec3(0.0, 0.0, vCameraZ));
  //bvec evec = equal(pos, vec3(-1.0));
  //if all(evec) discard;
//  if (dist >= infinity()) discard;

  vec4 intersection;
  intersection.xyz = EyePosition + vec3(dist) * NEyeDir;
  intersection.w = 1.0;
  // the following is needed to update gl_FragDepth
  vec4 iproj = gl_ModelViewProjectionMatrix * intersection;
  float depth = 0.5*(iproj.z/iproj.w + 1.0);

  //////////////////////
  // compute lighting //
  //////////////////////
//  vec3 normal = SHNormal(vGlyphPosition, intersection.xyz);
//  vec3 light_direction = normalize(intersection.xyz - LightPosition);
//  vec3 eye_direction = normalize(intersection.xyz - EyePosition);

  vec3 normal = SHNormal(vec3(0.0), pos);
  vec3 eye_direction = normalize(pos - vec3(0.0, 0.0, vCameraZ));
  vec3 light_direction = eye_direction;

  // compute a color depending on the eigenvalues and main diffusion direction
  vec4 color; color.a = 1.0;

 // color.rgb = abs(normalize(vGlyphPosition - intersection.xyz));

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
    color.rgb = gl_Color;
    }

  float dp = diffuse(light_direction, normal);
  float sp = specular(eye_direction, light_direction, normal, 8.0);

  color.rgb = vec3(0.2+0.7*dp)*color.rgb + vec3(0.5*sp);

  gl_FragColor = color;
  gl_FragDepth = depth; //FragDepthFromDistance(dist, EyePosition, NEyeDir);
}
