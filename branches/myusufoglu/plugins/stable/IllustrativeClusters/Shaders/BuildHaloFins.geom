/**
 * Geometry Shader: Renders side fins for a given line
 * by Ron Otten
 *
 * 2009-01-07
 *
 */

#version 120
#extension GL_EXT_gpu_shader4 	   : enable
#extension GL_EXT_geometry_shader4 : enable

// ================================================

uniform float finWidth;
uniform float finRecision;

uniform float minStrokeWidth;
uniform float maxStrokeWidth;

uniform vec4  phongParams;

uniform bool inkLines;

#define MAX_RADIUS_OF_CURVATURE 20

// ================================================

struct bisector
{
	vec3 origin;
	vec3 direction;
};

// Compute the bisector of the line segment (p0, p1) in the plane defined by normal n
bisector computeBisector(in vec3 p0, in vec3 p1, in vec3 n)
{
	bisector bs;
	
	if (distance(p0, p1) == 0)
	{
		bs.origin = p0;
		bs.direction = vec3(0, 0, 0);		
	}
	else
	{
		bs.origin = (p0 + p1) / 2;
		bs.direction = normalize(cross(p1 - p0, n));
	}
	
	return bs;
}

// Approximate the curvature at point p1 of the line segment (p0, p1, p2)
// using a geometric approximation of an osculating circle.
float approximateCurvature(in vec3 p0, in vec3 p1, in vec3 p2)
{
	// Zero curvature if two points match
	if ((distance(p0,p1) == 0) || (distance(p1, p2) == 0))
		return 0;

	vec3 n = normalize(cross(p1-p0, p2-p1));
	
	// No curvature on points in a degenerate plane
	if (length(n) == 0)
		return 0;
	
	bisector b0 = computeBisector(p0, p1, n);
	bisector b1 = computeBisector(p1, p2, n);
	
	vec3 w = b1.origin - b0.origin;
	vec3 v_perp = p2 - p1;
	
	float s = dot(-v_perp, w) / dot(v_perp, b0.direction);
	
	vec3 O = b0.origin + s * b0.direction;
	
	float r = distance(p1, O);
	
	return smoothstep(0, 1, 1 - clamp(r / MAX_RADIUS_OF_CURVATURE, 0, 1));
}


vec3 getViewVector()
{
	return normalize(gl_ModelViewMatrixInverse * vec4(0.0, 0.0, 1.0, 0.0)).xyz;
}

void drawVertex(in vec3 pos, in float distFromCore, in float lerpWidth, in float luminosity)
{
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1);
	gl_FrontColor = vec4(distFromCore, lerpWidth, luminosity, 0);

	EmitVertex();
}

// Computes the tangent of point p1, based on the tangents of line segment (p0, p1) and (p1, p2)
vec3 computeTangent(in vec3 p0, in vec3 p1, in vec3 p2)
{
	return normalize(p2 - p0);
}

float calculateLuminosity(in vec3 position, in vec3 direction)
{
	vec3 view = normalize((gl_ModelViewMatrix * vec4(position,1)).xyz);
	vec3 light = normalize(vec3(1, -1, -1));
	vec3 tangent = normalize(gl_NormalMatrix * direction);	

    /* Calculate lighting */
    float dLT = dot(light, tangent);
    float dLN = sqrt(clamp(1 - (dLT * dLT), 0, 1)); // <-- Clamp to effective range of 0-1 to deal with numerical instabilities causing sqrt() to fail.
    float dVT = dot(view, tangent);

    float diffuseContrib = dLN;
    float specularContrib = dLN * sqrt(clamp(1 - (dVT * dVT), 0, 1)) - (dLT * dVT); // <-- Clamp the sqrt() again.
	specularContrib = max(specularContrib, 0.0); // <-- ensure a positive number for use with pow().

    return phongParams.x + phongParams.y * diffuseContrib + phongParams.z * pow(specularContrib, phongParams.w);
}



void main()
{
	vec3 view;
	vec3 recede;
	vec3 pos[2];
	vec3 perp[2];
	vec3 cap[2];
	vec3 dir[2];
	float coreWidth[2];
	float curvature[2];
	float luminosity[2];

	if (gl_VerticesIn != 4)
		return;

	// Build view vector in world space.
	view = getViewVector();

	// Positions of start vertex and adjacency are equal;
	// this is the starting point of a complete fiber and requires a start cap.
	bool startCap = (gl_PositionIn[0].xyz == gl_PositionIn[1].xyz);

	// Positions of end vertex and adjacency are equal;
	// this is the ending point of a complete fiber and requires an end cap.
	bool endCap = (gl_PositionIn[2].xyz == gl_PositionIn[3].xyz);

	// Get positions of start and end vertex for drawing line segment.
	pos[0] = gl_PositionIn[1].xyz;
	pos[1] = gl_PositionIn[2].xyz;

	// Get fiber's tangent and curvature at start and end vertex.
	{
		if (startCap)
		{
			dir[0] = computeTangent(gl_PositionIn[0].xyz, gl_PositionIn[1].xyz, gl_PositionIn[2].xyz);
			curvature[0] = 0;
		}
		else
		{			
			dir[0] = computeTangent(gl_PositionIn[0].xyz, gl_PositionIn[1].xyz, gl_PositionIn[2].xyz);
			curvature[0] = inkLines ? approximateCurvature(gl_PositionIn[0].xyz, gl_PositionIn[1].xyz, gl_PositionIn[2].xyz) : 0;
		}

		if (endCap)
		{
			dir[1] = computeTangent(gl_PositionIn[1].xyz, gl_PositionIn[2].xyz, gl_PositionIn[3].xyz);
			curvature[1] = 0;
		}
		else
		{			
			dir[1] = computeTangent(gl_PositionIn[1].xyz, gl_PositionIn[2].xyz, gl_PositionIn[3].xyz);
			curvature[1] = inkLines ? approximateCurvature(gl_PositionIn[1].xyz, gl_PositionIn[2].xyz, gl_PositionIn[3].xyz) : 0;
		}
	}

	// Construct perpendicular vectors for the fins, taking into account the fins' width.
	perp[0] = normalize(cross(view, dir[0])) * finWidth;
	perp[1] = normalize(cross(view, dir[1])) * finWidth;

	// Turn the tangents into the cap vectors, taking into account the fins' width.
	cap[0] = dir[0] * finWidth;
	cap[1] = dir[1] * finWidth;

	// Calculate luminosity
	luminosity[0] = clamp(calculateLuminosity(pos[0], dir[0]), 0, 1);
	luminosity[1] = clamp(calculateLuminosity(pos[1], dir[1]), 0, 1);

	// Calculate the relative width of the inked core stroke, dependent on line curvature
	coreWidth[0] = mix(minStrokeWidth, maxStrokeWidth, curvature[0]);
	coreWidth[1] = mix(minStrokeWidth, maxStrokeWidth, curvature[1]);

	recede = finRecision * view;


	if (startCap)
	{
		drawVertex(pos[0] + perp[0] - recede, 1, coreWidth[0], luminosity[0]);
		drawVertex(pos[0]                   , 0, coreWidth[0], luminosity[0]);
		drawVertex(pos[0] -  cap[0] - recede, 1, coreWidth[0], luminosity[0]);
		drawVertex(pos[0] - perp[0] - recede, 1, coreWidth[0], luminosity[0]);
		EndPrimitive();
	}

	if (endCap)
	{
		drawVertex(pos[1] - perp[1] - recede, 1, coreWidth[1], luminosity[1]);
		drawVertex(pos[1] +  cap[1] - recede, 1, coreWidth[1], luminosity[1]);
		drawVertex(pos[1]                   , 0, coreWidth[1], luminosity[1]);
		drawVertex(pos[1] + perp[1] - recede, 1, coreWidth[1], luminosity[1]);
		EndPrimitive();
	}


	drawVertex(pos[0] + perp[0] - recede, 1, coreWidth[0], luminosity[0]);
	drawVertex(pos[1] + perp[1] - recede, 1, coreWidth[1], luminosity[1]);
	drawVertex(pos[0]                   , 0, coreWidth[0], luminosity[0]);
	drawVertex(pos[1]                   , 0, coreWidth[1], luminosity[1]);
	drawVertex(pos[0] - perp[0] - recede, 1, coreWidth[0], luminosity[0]);
	drawVertex(pos[1] - perp[1] - recede, 1, coreWidth[1], luminosity[1]);
	EndPrimitive();
}

