uniform float LineLength;

float infinity();
float determinant(mat3);
mat3 inverse(mat3 matrix);
mat4 TranslateMatrix(vec3 translate);

// from Geometric Tools for Computer Graphics, page 421
float LineSegmentDistance3D(vec3 linePos, vec3 lineDir, vec3 segmentPos, vec3 segmentDir)
{
	float epsilon = 0.00001;
	float sNum;
	float tNum;
	float tDenom;
	float sDenom;

	vec3 u = linePos - segmentPos;
	float a = dot(lineDir, lineDir);
	float b = dot(lineDir, segmentDir);
	float c = dot(segmentDir, segmentDir);
	float d = dot(lineDir, u);
	float e = dot(segmentDir, u);
	float det = a * c - b * b;
	sDenom = det;

	// Check for (near) parallelism
	if (det < epsilon)
	{
		// Arbitrary choice
		sNum = 0.0;
		tNum = e;
		tDenom = c;
	} else {
		// Find parameter values of closest points
		// on each segment's infinite line. Denominator
		// assumed at this point to be "det",
		// which is always positive. We can check
		// value of numerators to see if we're outside
		// the [0,1] x [0,1] domain.
		sNum = b * e - c * d;
		tNum = a * e - b * d;
	} // else

	// Check t
	if (tNum < 0.0)
	{
		tNum = 0.0;	
		sNum = -1.0*d;
		sDenom = a;
	} else {
		tNum = tDenom;
		sNum = b - d;
		sDenom = a;		
	}

	// XXX: BUG! tDenom is not defined if det >= epsilon :s

	// Parameters of nearest points on restricted domain
	float s = sNum / sDenom;
	float t = tNum / tDenom;

	// Dot product of vector between points is squared distance
	// between segments
	vec3 v = linePos + (s * lineDir) - segmentPos + (t * segmentDir);
	return length(segmentPos - linePos)*200.0;
	return dot(v, v);
}

// Returns a vec3 (distance, s, t) where s and t are the parameters
// along line A and B that determine the closest points
// lineAPos + s*lineADir and lineBPos + t*lineBDir.
// XXX: change function name. it is not valid.
// from Geometric Tools for Computer Graphics, page 421
vec3 LineLineDistanceSquared(vec3 lineAPos, vec3 lineADir, vec3 lineBPos, vec3 lineBDir)
{
//	return length(lineAPos - lineBPos);
	float s; float t;

	// note: epsilon > 0 will give strange changes when det < epsilon.
	float epsilon = 0.00001;

	vec3 u = lineAPos - lineBPos;
	float a = dot(lineADir, lineADir);
	float b = dot(lineADir, lineBDir);
	float c = dot(lineBDir, lineBDir);
	float d = dot(lineADir, u);
	float e = dot(lineBDir, u);
	float f = dot(u, u);
	float det = a * c - b * b;

	// Check for (near) parallelism
	if (det < epsilon)
//	if (det == 0.0)
	{
		// Arbitrarily choose the base point of lineA
		s = 0.0;
		// Choose largest denominator to minimize floating-point problems
		if (b > c)
		{
			t = d / b;
		} else {
			t = e / c;
		}
		//return 1.0;
		//return d * s + f;
		return vec3(d * s + f, s, t);
	} else {
		// Nonparallel lines
		float invDet = 1.0 / det;
		s = (b * e - c * d) * invDet;
		t = (a * e - b * d) * invDet;

		// TODO: if the closest points are not really needed; don't compute them.
		// return only s and t if needed. Maybe write a separate function for that.	
		vec3 closestpoints = lineAPos - lineBPos + vec3(s)*lineADir - vec3(t)*lineBDir;
	//	float dist = length(closestpoints);
//		return t*100;
//		return length(closestpoints);
		return vec3(length(closestpoints), s, t);
//		return vec3(dist, 1.0, 1.0);
	}
}

// given the distance and location (no location??) of the shortest distance between ray and line,
// compute where the ray would intersect a cylinder with radius r around the line.
// t is the distance along the ray from its origin to the closest point with distance d.
// XXX: t is not needed :)
//
// returned is the distance s along the ray from the closest point to the intersection
// with the cylinder (so compute t+s and t-s for the two intersections. t-s is the
// one closer to the ray origin).
float RayLineDistanceToCylinderIntersection(vec3 RayDir, vec3 LineDir, float radius, float d)
{
  vec3 n = normalize(cross(RayDir, LineDir)); // TODO: normalize Dirs?
  vec3 o = normalize(cross(n, LineDir));
  float r = radius; // radius of the cylinder.
  float s = abs(sqrt(r*r - d*d) / dot(RayDir, o));
  return s;
}


// PlaneEquation.xyz is the normal of the plane
// PlaneEquation.t is the distance of the plane to (0, 0, 0);
// See also  Geometric Tools for Computer Graphics, pp. 482-485.
vec3 LineIntersectPlaneNoParallelCheck(vec3 LinePos, vec3 LineDir, vec4 PlaneEquation)
{
  // it is assumed that the line and plane are not parallel.
  // so, dot(LineDir, PlaneEquation.xyz) is not (almost) 0.0.

  // Q is the intersection point of the line and plane.
  // Q = LinePos + t * LineDir
  // dot(Q, PlaneEquation.xyz) + PlaneEquation.w = 0

  float denominator = dot(PlaneEquation.xyz, LineDir);
  // denominator should not be (nearly) 0.0.
  float t = -1.0 * (dot(PlaneEquation.xyz, LinePos) + PlaneEquation.w);
  t = t / denominator;

  return LinePos + vec3(t)*LineDir;

/*
  float a = PlaneEquation[0];
  float b = PlaneEquation[1];
  float c = PlaneEquation[2];
  float d = PlaneEquation[3];
  float dx = LineDir.x;
  float dy = LineDir.y;
  float dz = LineDir.z;
  float Px = LinePos.x;
  float Py = LinePos.y;
  float Pz = LinePos.z;
  float t = (-a*Px-b*Py-c*Pz-d) / (a*dx+b*dy+c*dz);

  return LinePos + vec3(t)*LineDir;
*/
}

float PointLineDistanceSquared(vec3 point, vec3 linePos, vec3 lineDir)
{
  // paramization of the line: linePos + t*lineDir
  // P = point
  // Q = projection of P on line.
  // (Q - P).v = 0

  float t = dot(lineDir, point - linePos) / dot(lineDir, lineDir);
  vec3 Q = linePos + vec3(t) * lineDir;

  return length(Q - point);
}

float RayStandardSphereIntersection(vec3 RayOrigin, vec3 RayDirection)
{
  // solve ||RayDirection + t.RayDirection||^2 == 1
  // t^2*(RayDirection.RayDirection) + 2t(RayDirection.RayOrigin) + ((RayDirection.RayDirection)-1) == 0

  float a = dot(RayDirection, RayDirection);
  float b = 2.0 * dot(RayDirection, RayOrigin);
  float c = dot(RayOrigin, RayOrigin) - 1.0;
  float discrm = b*b - 4.0*a*c;
  float t = infinity();
  // a != 0.0 if RayDirection is unit-length.
  if (discrm >= 0.0) t = (-b -sqrt(discrm)) / (2.0 * a);
  return t;
}

vec2 RayStandardSphereIntersections(vec3 RayOrigin, vec3 RayDirection)
{
  float a = dot(RayDirection, RayDirection);
  float b = 2.0 * dot(RayDirection, RayOrigin);
  float c = dot(RayOrigin, RayOrigin) - 1.0;
  float discrm = b*b - 4.0*a*c;
  float close_intersection = infinity();
  float far_intersection = infinity();
  // a != 0.0 if RayDirection is unit-length.
  if (discrm >= 0.0)
    {
    close_intersection = (-b -sqrt(discrm)) / (2.0 * a);
    far_intersection = (-b +sqrt(discrm)) / (2.0 * a);
    }
  return vec2(close_intersection, far_intersection);
}

mat4 IsotropicScaleMatrix(float scale);
mat4 ScaleMatrix(vec3 scale);

// returns the value of t along the line where it intersects,
// also known as the distance from the origin of the view ray
// (if the ray direction is normalized).
// if the ray doesn't intersect the sphere, return infinity.
float RaySphereIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius)
{
  mat4 M = IsotropicScaleMatrix(SphereRadius);
  vec4 TrDirection = vec4(RayDirection, 1.0) * M; TrDirection.xyz /= TrDirection.w;
  M = M * TranslateMatrix(vec3(-1.0/SphereRadius)*SphereCenter);
  vec4 TrOrigin = vec4(RayOrigin, 1.0) * M; TrOrigin.xyz /= TrOrigin.w;
  return RayStandardSphereIntersection(TrOrigin.xyz, TrDirection.xyz);  
}

vec2 RaySphereIntersections(vec3 RayOrigin, vec3 RayDirection, vec3 SphereCenter, float SphereRadius)
{
  mat4 M = IsotropicScaleMatrix(SphereRadius);
  vec4 TrDirection = vec4(RayDirection, 1.0) * M; TrDirection.xyz /= TrDirection.w;
  M = M * TranslateMatrix(vec3(-1.0/SphereRadius)*SphereCenter);
  vec4 TrOrigin = vec4(RayOrigin, 1.0) * M; TrOrigin.xyz /= TrOrigin.w;
  return RayStandardSphereIntersections(TrOrigin.xyz, TrDirection.xyz);
}

//vec3 EllipsoidIntersectionNormal;
//vec3 EllipsoidIntersectionPoint;
//vec3 TenPos;
float RayAlignedEllipsoidIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 EllipsoidCenter, vec3 EllipsoidRadius)
{
  // this ellipsoid is aligned with the axes.

  vec3 r = vec3(1.0) / pow(EllipsoidRadius, vec3(2.0));
  vec3 d = RayOrigin - EllipsoidCenter;
  vec3 av = r * pow(RayDirection, vec3(2.0));
  float a = av.x + av.y + av.z;
  vec3 bv = vec3(2.0) * r * d * RayDirection;
  float b = bv.x + bv.y + bv.z;
  vec3 cv = r * pow(d, vec3(2.0));
  float c = cv.x + cv.y + cv.z - 1.0;

  float discrm = b*b - 4.0*a*c;
  float t = infinity();
  if (discrm >= 0.0)
    {
    t = (-1.0*b - sqrt(discrm)) / (2.0 * a);
    }

  if (t > infinity()) t = infinity();
  return t;
}

// intersection of a ray with an axis-aligned ellipsoid that has its origin in (0,0,0)
float RayAlignedEllipsoidIntersection(vec3 RayOrigin, vec3 RayDirection, vec3 EllipsoidRadius)
{
  return RayAlignedEllipsoidIntersection(RayOrigin, RayDirection, vec3(0.0, 0.0, 0.0), EllipsoidRadius);
}

// rotation matrix that aligns the main axis of a glyph with the given vector.
mat3 rotationMatrixV(vec3 direction);

float RayEllipsoidIntersection(vec3 RayPosition, vec3 NRayDirection, vec3 VecPos, vec3 VecDir)
{
  float result = infinity();
//  vec3 scaling = Eigenvalues*vec3(GlyphScaling);
  vec3 scaling = vec3(0.8, 0.1, 0.1)*vec3(LineLength);//*vec3(GlyphScaling);

  mat3 rot = rotationMatrixV(VecDir); //Eigenvectors);

  // Transform the eye direction to tensor-space
  vec3 dir;
  //dir = NEyeDir * rot;
  dir = NRayDirection * rot; 

  // Transform the eye position to tensor-space
//  vec3 pos = EyePosition - TenPosW;
  vec3 pos = RayPosition - VecPos;
  pos = pos * rot;

    //result = RayAlignedEllipsoidIntersection(pos, dir, TenPosW, scaling);
  result = RayAlignedEllipsoidIntersection(pos, dir, vec3(0.0), scaling);
  return result;

}

float FragDepthFromDistance(float distance, vec3 EyePosition, vec3 NEyeDir)
{
  vec4 intersection;
  intersection.xyz = EyePosition + vec3(distance) * NEyeDir;
  intersection.a = 1.0;
  vec4 iproj = gl_ModelViewProjectionMatrix * intersection;
  return 0.5*(iproj.z/iproj.w + 1.0);
}
