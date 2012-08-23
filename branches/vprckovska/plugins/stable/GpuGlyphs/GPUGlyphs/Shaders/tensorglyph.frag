uniform vec3 EyePosition;
uniform vec3 LightPosition;
varying vec3 vEyeDirection;
varying vec3 vEigenvalues;
varying vec3 vGlyphPosition;
//varying vec3 vEigenvector1;
//varying vec3 vEigenvector2;
varying mat3 vEigenvectors;
varying vec3 vColor;
uniform float GlyphScaling;

float infinity();
float RayAlignedEllipsoidIntersection(vec3, vec3, vec3);
//float RayAlignedSuperquadricIntersection(vec3, vec3, vec3, float, float);
vec3 LineIntersectPlaneNoParallelCheck(vec3 LinePos, vec3 LineDir, vec4 PlaneEquation);
float FragDepthFromDistance(float distance, vec3 EyePosition, vec3 NEyeDir);
mat3 rotationMatrix(mat3);
mat3 transposeMatrix(mat3);

float diffuse(in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 n, in float p);

void main()
{
  if (vEigenvalues[0] == 0.0) discard;

  ///////////////////////////////////////////////////////////////
  // determine the intersection of the view-ray with the glyph //
  ///////////////////////////////////////////////////////////////

  // Ralph: in the code below the vector vEigenvalues is assigned a value!
  // It is amazing this is allowed on a Windows platform. The GLSL compiler
  // on my Mac is complaining vEigenvalues is a read-only variable and it
  // should complain!!!

  // So, create a local copy of the eigenvalues and work with that...
  vec3 eigenValues = vEigenvalues;

  vec3 NEyeDir = normalize(vEyeDirection);
  vec3 EllipsoidRadii = eigenValues * vec3(GlyphScaling);

  // transform the eye position into tensor space
  mat3 rot = rotationMatrix(vEigenvectors);
  vec3 tEyePos = EyePosition - vGlyphPosition;
  tEyePos = rot * tEyePos;
  // transform the eye direction into tensor space
  vec3 tEyeDir = rot * NEyeDir;
 
  float dist;
  if(eigenValues[2] > 0.1)
	{
	dist = RayAlignedEllipsoidIntersection(tEyePos, tEyeDir, EllipsoidRadii);
	}
  else
	{	// avoid numerical problems with very small 2nd eigenvalue
		// here, use plane equation and ellipse equation to determine whether there is an
		// intersection, to avoid "exploding" glyphs.
		// but when there is intersection, use RayAlignedEllipsoidIntersection to compute
		// the correct location, otherwise normal computation will go wrong later because
		// of wrong computed distance.
/*
	vec3 intersectionPoint = LineIntersectPlaneNoParallelCheck(tEyePos, tEyeDir, vec4(0.0, 0.0, 1.0, 0.0));
	float x2 = pow(intersectionPoint.x/EllipsoidRadii.x, 2.0);
	float y2 = pow(intersectionPoint.y/EllipsoidRadii.y, 2.0);
	if ((x2 +y2) <= 1.0) dist = RayAlignedEllipsoidIntersection(tEyePos, tEyeDir, EllipsoidRadii); //length(tEyePos-intersectionPoint);
	else dist = infinity();
*/
    eigenValues[2] = 0.1;
    if(eigenValues[1] < 0.1) eigenValues[1] = 0.1;
    EllipsoidRadii = eigenValues * vec3(GlyphScaling);
	dist = RayAlignedEllipsoidIntersection(tEyePos, tEyeDir, EllipsoidRadii);
	}
  //float dist = RayAlignedSuperquadricIntersection(tEyePos, tEyeDir, EllipsoidRadii, 0.5, 0.4);
  if (dist >= infinity()) discard;

  vec4 intersection;
  intersection.xyz = EyePosition + vec3(dist) * NEyeDir;
  intersection.w = 1.0;
  // the following is needed to update gl_FragDepth
  vec4 iproj = gl_ModelViewProjectionMatrix * intersection;
  float depth = 0.5*(iproj.z/iproj.w + 1.0);

  // compute a color depending on the eigenvalues and main diffusion direction
  
  // Evert: We now do this on the CPU, to allow for more customization in coloring. We use
  // the "vColor" vector as the input color. See the file "vtkDTIGlyphMapperVA.cxx" for
  // details on how we compute the color.
  
  vec4 color; //color.a = 1.0;
  color.rgb = vColor;
  color.a = 1.0;
  
//  vec3 NEvals = normalize(vEigenvalues);
//  float cl = vEigenvalues[0] - vEigenvalues[1]; // NEvals[0] should be >= NEvals[1].
//  color.rgb = vec3(cl) * abs(vEigenvectors[0]) + vec3(1.0-cl)*vec3(sqrt(1.0/3.0));
// for rat brain:
//  color.rgb = vec3(cl) * abs(vEigenvectors[0].yzx) + vec3(1.0-cl)*vec3(sqrt(1.0/3.0));
// for rainer goebel brain:
//  color.rgb = vec3(cl) * abs(vEigenvectors[0].zxy) + vec3(1.0-cl)*vec3(sqrt(1.0/3.0));
//  color.rgb = vec3(cl);

  // compute the normal of the ellipsoid
  vec3 normal = rot*normalize(vGlyphPosition - intersection.xyz);
  vec3 r = vec3(1.0) / pow(EllipsoidRadii, vec3(2.0));
  normal = vec3(2.0) * r * normal;
  normal = transposeMatrix(rot)*normalize(normal);
 
  // lighting calculations to determine the pixel's final color
  vec3 light_direction = normalize(intersection.xyz - LightPosition);
  float d = diffuse(light_direction, normal);
  float s = specular(NEyeDir, light_direction, normal, 8.0);
  color.rgb = vec3(0.2+0.7*d)*color.rgb + vec3(0.5*s);
  // For the AZM-Rainer Goebel dataset:
  //color.gbr = vec3(0.2+0.7*d)*color.rgb + vec3(0.5*s);

  // write the output to the buffers 
  gl_FragColor = color;
  gl_FragDepth = FragDepthFromDistance(dist, EyePosition, NEyeDir);
}
