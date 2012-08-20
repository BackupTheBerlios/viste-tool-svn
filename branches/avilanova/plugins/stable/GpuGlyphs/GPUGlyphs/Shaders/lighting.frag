float diffuseLine(	in vec3 l,	// light direction
		in vec3 t)		// tangent direction
{
  float LdotT = dot(l, t);
  float NdotL = sqrt(1.0-pow(LdotT, 2.0));

  return NdotL;
}

float specularLine(	in vec3 v,	// view direction
		in vec3 l,		// light direction
		in vec3 t,		// tangent direction
		in float p)		// specular power
{

  float LdotT = dot(l, t);
  float VdotT = dot(v, t);
  float NdotL = sqrt(1.0-pow(LdotT, 2.0));
  float RdotV = max(0.0, NdotL * sqrt(1.0-pow(VdotT, 2.0)) - LdotT*VdotT);

  float pf;

  if (NdotL < 0.01)
    pf = 0.0;
  else
    pf = pow(RdotV, p);

  return pf;
}

float diffuse(	in vec3 l,	// light direction
		in vec3 n)	// normal
{
  float NdotL = max(0.0, abs(dot(n, l)));
  return NdotL;
}

float specular(	in vec3 v,	// view direction
		in vec3 l,	// light direction
		in vec3 n,	// normal
		in float p)	// specular power
{
  vec3 reflectDir = normalize(dot(2*n, l) * n - l); 

  float NdotL = max(0.0, abs(dot(n, l)));
  float RdotV = max(0.0, dot(reflectDir, v));

  float pf;
  if (NdotL < 0.01)
    pf = 0.0;
  else
    pf = pow(RdotV, p);
    //pf = pow(RdotV, gl_FrontMaterial.shininess);

  return pf;
}
