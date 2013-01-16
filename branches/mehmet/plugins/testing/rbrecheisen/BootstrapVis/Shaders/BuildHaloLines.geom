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

uniform vec4  phongParams;

// ================================================

vec3 getViewVector()
{
	return normalize(gl_ModelViewMatrixInverse * vec4(0.0, 0.0, 1.0, 0.0)).xyz;
}

void drawVertex(in vec3 pos, in float luminosity)
{
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1);
	gl_FrontColor = vec4(0, 0, luminosity, 0);

	EmitVertex();
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
	vec3 pos[2];
	vec3 dir[2];
	float luminosity[2];

	// Build view vector in model space.
	view = getViewVector();

	// Get positions of start and end vertex for drawing line segment. Add a small amount of
	// bias towards the eye point so underlying fins don't cause z-fighting with these lines.
	pos[0] = gl_PositionIn[1].xyz + 0.01 * view;
	pos[1] = gl_PositionIn[2].xyz + 0.01 * view;

	{
		vec3 vecA = normalize(gl_PositionIn[1].xyz - gl_PositionIn[0].xyz);
		vec3 vecB = normalize(gl_PositionIn[2].xyz - gl_PositionIn[1].xyz);
		vec3 vecC = normalize(gl_PositionIn[3].xyz - gl_PositionIn[2].xyz);

		dir[0] = 0.5 * (vecA + vecB);
		dir[1] = 0.5 * (vecB + vecC);
	}	

	luminosity[0] = calculateLuminosity(pos[0], dir[0]);
	luminosity[1] = calculateLuminosity(pos[1], dir[1]);

	drawVertex(pos[0], luminosity[0]);
	drawVertex(pos[1], luminosity[1]);
	EndPrimitive();
}

