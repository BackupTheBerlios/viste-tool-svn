varying vec3 EyeDirection;
uniform vec3 EyePosition;
uniform vec3 LightPosition;
uniform float SeedDistance;
varying vec4 PoiEquation;
uniform float LineLength;
uniform float LineThickness;
uniform bool UseShadows;

vec3 LineLineDistanceSquared(vec3 lineAPos, vec3 lineADir, vec3 lineBPos, vec3 lineBDir);
float RayLineDistanceToCylinderIntersection(vec3 RayDir, vec3 LineDir, float d);

float LineSegmentDistance3D(vec3 linePos, vec3 lineDir, vec3 segmentPos, vec3 segmentDir);
vec3 ClosestSeedPoint();
float PointLineDistanceSquared(vec3 point, vec3 linePos, vec3 lineDir);

float diffuseLine(vec3 l, vec3 t);
float specularLine(vec3 v, vec3 l, vec3 t, float p);
float diffuse(vec3 l, vec3 n);
float specular(vec3 v, vec3 l, vec3 n, float p);

vec3 WorldToTexture(vec3 w);
vec4 VectorAtWorld(vec3 w);

uniform vec3 PoiOrigin;
uniform vec3 Step1;
uniform vec3 Step2;
float SeedPointWithClosestVector(vec3 RayPosition, vec3 RayDirection, float thickness);

mat3 rotationMatrixV(vec3 direction);

uniform vec3 PoiPoint1;
uniform vec3 PoiPoint2;

uniform vec3 PoiNormal;

float NumSteps;

vec2 MaxSeedNr;

vec3 VecPos;
float infinity();

void main()
{
float NdotE = abs(dot(normalize(PoiNormal), normalize(EyeDirection)));
NumSteps = LineLength / SeedDistance;
NumSteps = NumSteps / NdotE;
MaxSeedNr.x = length(PoiPoint1 - PoiOrigin) / length(Step1);
MaxSeedNr.y = length(PoiPoint2 - PoiOrigin) / length(Step2);


        vec3 VecDir;
	float LineLength2 = LineLength*LineLength;
	vec4 color; color.a = 1.0;

	float r = LineThickness / 100.0;
//	float r = 0.03; // good for fullscreen?
//	float r = 0.025; // good for fullscreen?
//	float r = 0.05; // good for 800*600?

//	r = 0.1;

	float min_dist_along_viewray = SeedPointWithClosestVector(EyePosition, EyeDirection, r);
	// VecPos should also have a value now.

	if (min_dist_along_viewray == infinity()) discard;

	vec4 intersection;
        intersection.xyz  = EyePosition + vec3(min_dist_along_viewray) * normalize(EyeDirection); //NEyeDir;
	intersection.a = 1.0;
//	vec3 intersection = EyePosition + vec3(min_dist_along_viewray)*normalize(EyeDirection);

	////////////////////////////////
        // normal for ellipsoid glyph //
	////////////////////////////////
	VecDir = VectorAtWorld(VecPos).xyz;
	mat3 rotate = rotationMatrixV(VecDir);

	vec3 normal;
	normal = normalize(VecPos.xyz - intersection.xyz)*rotate; //Eigenvectors; // sphere normal
	vec3 orientationcolor = normal;

	vec3 radi = vec3(1.0) / pow(vec3(0.8, 0.1, 0.1), vec3(2.0));
	normal = vec3(2.0) * radi * normal; //(TenPosW.xyz - intersection.xyz);
	normal = normalize(normal);
	normal = normal * transpose(rotate);

	////////////////////////////
	// end of ellipsoid stuff //
	////////////////////////////

	//shadow stuff
	vec3 a = VecPos;

	float min_dist_from_lightray = 0.0;
	if (UseShadows) SeedPointWithClosestVector(LightPosition, intersection.xyz-LightPosition, 4.0*r);
	vec3 b = VecPos;

	vec4 vec_and_scalar = VectorAtWorld(a);

	VecDir = vec_and_scalar.rgb;
	float scalar = vec_and_scalar.a;
	scalar = sqrt(scalar);

	if (all(equal(a,b)))
		{ // in light
	//	color.rgb = vec3(1,1,0.5);
		vec3 vn = normalize(EyeDirection);
		vec3 ln = normalize(intersection.xyz-LightPosition);

		// line lighting
//		color.rgb = vec3(0.2) + vec3(0.6*diffuseLine(ln, VecDir))*abs(VecDir) + vec3(0.4*specularLine(vn, ln, VecDir, 30.0)); // rgb coloring
		color.rgb = vec3(0.2) + vec3(0.6*diffuseLine(ln, VecDir)) + vec3(0.4*specularLine(vn, ln, VecDir, 30.0));	// grey

		// ellipsoid lighting
//		color.rgb = vec3(0.2) + vec3(0.6*diffuse(ln, normal)) + vec3(0.4*specular(vn, ln, normal, 30.0));
//		color.g = color.g * scalar;
//		color.r = color.r * (1.0-scalar);

//		color.rgb = vec3(specular(vn, ln, VecDir, 30.0));
	    	}
	else
		{ // in shadow
	//	color.rgb = vec3(0.5,0.5,1.0);
		vec3 ln = normalize(intersection.xyz-LightPosition);

		// line lighting:
//		color.rgb = vec3(0.1) + vec3(0.3*diffuseLine(ln, VecDir))*abs(VecDir); // rgb coloring
		color.rgb = vec3(0.1) + vec3(0.3*diffuseLine(ln, VecDir));		// grey

		// ellipsoid lighting:
//		color.rgb = vec3(0.1) + vec3(0.25*diffuse(ln, normal));

//		color.g = color.g * scalar;
//		color.r = color.r * (1.0-scalar);
		}

//color.b = 0.0;
//	if (min_dist_from_lightray == infinity()) color.rgb = vec3(1,1,0);

//	color.rgb = vec4(scalar);
//	color.rgb = vec3(noise1(VecPos));
//	color.rgb = abs(VecDir);
//	color.rgb = WorldToTexture(VecPos);



//  vec4 i;
//  i.xyz  = EyePosition + vec3(min_dist_along_viewray) * normalize(EyeDirection); //NEyeDir;
//  i.a = 1.0;
  vec4 iproj = gl_ModelViewProjectionMatrix * intersection;
  float depth = 0.5*(iproj.z/iproj.w + 1.0);

  gl_FragColor = color;
  gl_FragDepth = depth;
}
