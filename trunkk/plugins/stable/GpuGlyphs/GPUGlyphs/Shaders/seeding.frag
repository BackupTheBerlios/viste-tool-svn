uniform vec3 EyePosition;
varying vec3 EyeDirection;
uniform vec3 PoiOrigin;
uniform vec3 Step1;
uniform vec3 Step2;
varying vec4 PoiEquation;
varying vec3 PoiNormal;

uniform float SeedDistance;

vec3 LineIntersectPlaneNoParallelCheck(vec3 LinePos, vec3 LineDir, vec4 PlaneEquation);
float Length3(vec3 v);
bool WorldCoordInRange(vec3 w);

float RayLineDistanceToCylinderIntersection(vec3 RayDir, vec3 LineDir, float radius, float d);

vec3 ClosestSeedPoint(vec3 RayPosition, vec3 RayDirection)
{
  // compute the intersection of the view ray and the POI
  vec3 w = LineIntersectPlaneNoParallelCheck(RayPosition, RayDirection, PoiEquation);

  // w = O + t*Step1 + u*Step2
  // t = (w-O).Step1
  // u = (w-O).Step2

  float t = dot((w-PoiOrigin), Step1) / dot(Step1, Step1);
  float u = dot((w-PoiOrigin), Step2) / dot(Step2, Step2);

  // I want round(), but GLSL only has floor() and ceil(),
  // so I do it like this:
  float ti = floor(t);
  float ui = floor(u);
  if (t-ti > 0.5) ti = ti+1.0;
  if (u-ui > 0.5) ui = ui+1.0;

  // To get the closes seed point, t and u should be rounded to the nearest integer.
  // TODO: do this later. for now I floor it.
  vec3 seed = PoiOrigin + vec3(ti)*Step1 + vec3(ui)*Step2;
  return seed;
}

vec2 ClosestSeedPointNumber(vec3 RayPosition, vec3 RayDirection)
{
  // compute the intersection of the view ray and the POI
  vec3 w = LineIntersectPlaneNoParallelCheck(RayPosition, RayDirection, PoiEquation);

  // w = O + t*Step1 + u*Step2
  // t = (w-O).Step1
  // u = (w-O).Step2

  float t = dot((w-PoiOrigin), Step1) / dot(Step1, Step1);
  float u = dot((w-PoiOrigin), Step2) / dot(Step2, Step2);

  // I want round(), but GLSL only has floor() and ceil(),
  // so I do it like this:
  float ti = floor(t);
  float ui = floor(u);
  if (t-ti > 0.5) ti = ti+1.0;
  if (u-ui > 0.5) ui = ui+1.0;

  vec2 seednr = vec2(ti, ui);
  return seednr;
}

//varying float NumSteps;
float NumSteps;
//varying vec2 MaxSeedNr;
vec2 MaxSeedNr;

float PointLineDistanceSquared(vec3 point, vec3 linePos, vec3 lineDir);

vec4 VectorAtWorld(vec3 w);

vec3 LineLineDistanceSquared(vec3 lineAPos, vec3 lineADir, vec3 lineBPos, vec3 lineBDir);
float RayEllipsoidIntersection(vec3 RayPosition, vec3 NRayDirection, vec3 VecPos, vec3 VecDir);

uniform float LineLength;

/*
// XXX: put this somewhere ;)
		// compute the intersection with the tube:
		float s = RayLineDistanceToCylinderIntersection(EyeDirection, VecDir, sqrt(dist2));
		vec3 intersection = EyePosition + vec3(t-s)*EyeDirection;
		// do I need this? I just need the value of t-s to determine which vector is closest.
*/




/**
 * Puts the location of the closest seed point in VecPos.
 * Returns the distance from the eye to the vector with origin VecPos.
 */

// TODO: Add view direction and position as parameters so that I can also
// call it with the light direction.
// Same for thickness of the fibers to render? So that I can make them thicker for the shadows.
// RayDirection must not be normalized!
vec3 VecPos;
float infinity();
float SeedPointWithClosestVector(vec3 RayPosition, vec3 RayDirection, float thickness)
{
	float LineLength2 = LineLength*LineLength;

//	float r = 0.05; // the radius of the cylinders.
//	float r = 0.05 * length(RayDirection)/100.0;
	float r = thickness * length(RayDirection)/100.0;

	// check for viewing direction parallel to the plane
	// NOTE: discard is not the good way to deal with this FIXME
//	if (abs(dot(EyeDirection, PoiEquation.xyz)) < 0.00001) discard;

	// TODO: discard asap in areas with no data (or 0-tensors).
	// AND in areas outside of the texture domain.

	// loop through the seeds in range to see which one should
	// be visualized now
	// TODO: compute on CPU which range of seed nrs to use.
	vec2 center_seed_nr = ClosestSeedPointNumber(RayPosition, RayDirection);

	float NdotE = abs(dot(PoiNormal, EyeDirection));
	float NumStepsi = ceil(NumSteps); //ceil(NumSteps/NdotE); // XXX: maybe floor is good enough

//	float NumStepsi = ceil(NumSteps); // XXX: maybe floor is good enough
	vec2 min_seed_nr = center_seed_nr - vec2(NumStepsi);
	vec2 max_seed_nr = center_seed_nr + vec2(NumStepsi);

	// limit the range of seed points to the actual range of the POI.
	// this avoids computations that are not used later on.

	if (min_seed_nr.x < 0.0) min_seed_nr.x = 0.0;
	if (min_seed_nr.y < 0.0) min_seed_nr.y = 0.0;
	if (max_seed_nr.x > MaxSeedNr.x) max_seed_nr.x = floor(MaxSeedNr.x);
	if (max_seed_nr.y > MaxSeedNr.y) max_seed_nr.y = floor(MaxSeedNr.y);

	if (min_seed_nr.x > MaxSeedNr.x) discard;
	if (min_seed_nr.y > MaxSeedNr.y) discard;

	int t = min_seed_nr.x;
	int u = min_seed_nr.y;
	float min_dist_along_viewray = infinity();
//	vec2 closest_seed;
	vec3 closest_point;
	vec3 distance;
//	vec3 VecPos; // now global.
        vec3 VecDir;

	vec3 line_plane_intersection = LineIntersectPlaneNoParallelCheck(RayPosition, RayDirection, PoiEquation);

	vec3 NRayDirection = normalize(RayDirection);
	// loop through all the seed points

	vec3 vec_to_point;

	float ellipdist;
	vec3 seeddist;
	vec2 xrange;

	for (u = min_seed_nr.y; u <= max_seed_nr.y; u = u+1)
	{
	  VecPos = PoiOrigin + vec3(u)*Step2;
	  seeddist = LineLineDistanceSquared(RayPosition, NRayDirection, VecPos, Step1);
	  xrange[0] = min_seed_nr.x; xrange[1] = max_seed_nr.x;
	  if (seeddist[0] > LineLength) xrange[0] = xrange[1] + 1;
	  else
		{ // at least some times the seed points are in range for the ray
		// seeddist[2] is the number of steps with Step1 to come into range.
		float dt = RayLineDistanceToCylinderIntersection(normalize(Step1), NRayDirection, LineLength, seeddist[0]);
		xrange[0] = max(xrange[0], ceil(seeddist[2] - dt/SeedDistance));
		xrange[1] = min(xrange[1], floor(seeddist[2] + dt/SeedDistance));
		//xrange[0] = ceil(seeddist[2]);
		}

	  for (t = xrange[0]; t <= xrange[1]; t = t+1)
	  	{

	// TODO: check if the random generator on the gpu works already.
	// if yes, I can randomize the seed point locations a BIT so that the
	// seeding grid doesn't give wrong information.

		// here I assume that LineLength > Line thickness
		VecPos = PoiOrigin + vec3(t)*Step1 + vec3(u)*Step2;
//		if (WorldCoordInRange(VecPos))
			{
			vec_to_point = line_plane_intersection - VecPos;
//			if (dot(vec_to_point, vec_to_point) * (1.0 - pow(dot(normalize(vec_to_point), NRayDirection), 2.0)) < LineLength2)
//			if (PointLineDistanceSquared(VecPos, RayPosition, RayDirection) < LineLength)
				{
				VecDir = VectorAtWorld(VecPos).rgb; //texture3D(texture, WorldToTexture(VecPos)).rgb;

// here starts ellipsoid stuff
/*
				if (!(all(equal(VecDir, vec3(0.0)))))
					{
					ellipdist = RayEllipsoidIntersection(RayPosition, NRayDirection, VecPos, VecDir);
					if (ellipdist < min_dist_along_viewray)
						{
						min_dist_along_viewray = ellipdist;
						closest_point = VecPos;
						}
					}
// here ends ellipsoid stuff
*/
// here begins line stuff
				if (!(all(equal(VecDir, vec3(0.0)))))
//				if (LineLength / (abs(dot(EyeDirection, VecPos - line_plane_intersection))) 
					{
					distance =  LineLineDistanceSquared(RayPosition, NRayDirection, VecPos, VecDir);
					if ((distance[0] <= r) && (abs(distance[2]) <= LineLength))
					  	{ // the ray goes through the cylinder
						if (distance[1] < min_dist_along_viewray)
							{
							min_dist_along_viewray = distance[1];
							closest_point = VecPos;
							} //if (min_dist_along_viewray)
						  } // if (LineLength)
					} // if (!all(equal(VecDir, vec3(0.0)))
// here ends line stuff

				} // if (point close enough)
			} // if WorldCoordInRange(VecPos)
		} // for t
	} // for u

	if (min_dist_along_viewray != infinity()) VecPos = closest_point;

	// TODO: remove this. its for debugging only!
	//VecPos = LineIntersectPlaneNoParallelCheck(RayPosition, RayDirection, PoiEquation);

	return min_dist_along_viewray;
}

float GlyphDistanceFromEye(vec3 TenPos, vec3 RayPosition, vec3 RayDirection);

vec3 TenPos;
vec3 EllipsoidIntersectionPoint;
vec3 ClosestIntersection;
float SeedPointWithClosestGlyph(vec3 RayPosition, vec3 RayDirection)
{
	// check for viewing direction parallel to the plane
	// NOTE: discard is not the good way to deal with this FIXME
//	if (abs(dot(EyeDirection, PoiEquation.xyz)) < 0.00001) discard;

	// TODO: discard asap in areas with no data (or 0-tensors).
	// AND in areas outside of the texture domain.

	// loop through the seeds in range to see which one should
	// be visualized now
	// TODO: compute on CPU which range of seed nrs to use.
	vec2 center_seed_nr = ClosestSeedPointNumber(RayPosition, RayDirection);

  vec3 w = LineIntersectPlaneNoParallelCheck(RayPosition, RayDirection, PoiEquation);


//	TenPos = vec3(center_seed_nr[0])*Step1 + vec3(center_seed_nr[1])*Step2;

	float NdotE = abs(dot(PoiNormal, normalize(RayPosition-w)));
//	NdotE = 0.25;
	NdotE = 1.0; // fast, but not correct.
	float NumStepsi = ceil(NumSteps/NdotE); // XXX: maybe floor is good enough
//	NumStepsi = 2.0;
	vec2 min_seed_nr = center_seed_nr - vec2(NumStepsi);
	vec2 max_seed_nr = center_seed_nr + vec2(NumStepsi);

	// limit the range of seed points to the actual range of the POI.
	// this avoids computations that are not used later on.

	if (min_seed_nr.x < 0.0) min_seed_nr.x = 0.0;
	if (min_seed_nr.y < 0.0) min_seed_nr.y = 0.0;
	if (max_seed_nr.x > MaxSeedNr.x) max_seed_nr.x = floor(MaxSeedNr.x);
	if (max_seed_nr.y > MaxSeedNr.y) max_seed_nr.y = floor(MaxSeedNr.y);

	if (min_seed_nr.x > MaxSeedNr.x) discard;
	if (min_seed_nr.y > MaxSeedNr.y) discard;

	float t = min_seed_nr.x;
	float u = min_seed_nr.y;
	float min_dist_along_viewray = infinity();
	vec3 closest_point;
	//vec3 distance;
	mat4 Eigen;
	//vec3 TenPos; // its global!

	float distance = infinity();
	float min_point_dist = length(RayPosition-w)+LineLength;//PoiOrigin); //infinity;


	float LineLength2 = LineLength*LineLength;

	vec3 point;

	vec3 NRayDirection = normalize(RayDirection);
	// loop through all the seed points
	for (u = min_seed_nr.y; u <= max_seed_nr.y; u = u+1.0)
	  {
	  for (t = min_seed_nr.x; t <= max_seed_nr.x; t = t+1.0)
	    {
	    // TODO: check if the random generator on the gpu works already.
	    // if yes, I can randomize the seed point locations a BIT so that the
	    // seeding grid doesn't give wrong information.

	    point = PoiOrigin + vec3(t)*Step1 + vec3(u)*Step2;

	    if (length(RayPosition - point) < min_point_dist)
	      {
	      if (PointLineDistanceSquared(point, RayPosition, RayDirection) < LineLength2)
		{
	      	distance = GlyphDistanceFromEye(point, RayPosition, RayDirection);
	     	if (distance != infinity())
	          {
	          if (distance < min_dist_along_viewray)
		    {
		    min_dist_along_viewray = distance;
		    min_point_dist = distance + LineLength;
		    closest_point = point;
		    ClosestIntersection = EllipsoidIntersectionPoint;
		    } // if (distance < min_dist_along_viewray)
	          } // if distance != infinity()
		} // if (PointLineDistance...)
	      } //if (length(...) < min_point_dist)
	    } // for t
	  } // for u

	  if (min_dist_along_viewray != infinity()) TenPos = closest_point;

	// TODO: remove this. its for debugging only!
	//VecPos = LineIntersectPlaneNoParallelCheck(RayPosition, RayDirection, PoiEquation);
//return NdotE;
	return min_dist_along_viewray;
}
