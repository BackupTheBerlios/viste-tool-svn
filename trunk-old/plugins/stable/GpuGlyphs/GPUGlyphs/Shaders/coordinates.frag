uniform sampler3D Texture1;
//uniform sampler3D Texture2;
uniform ivec3 TextureDimensions;
uniform vec3 TextureSpacing;
// TODO: TextureOriginPosition or sth.

uniform bool NegateX;
uniform bool NegateY;
uniform bool NegateZ;

vec3 WorldToTexture(vec3 w)
{
  // return vector with coordinate components from 0..1.
  // It is possible that the input coordinates are not in the range
  // 0..TextureDimensions*TextureSpacing. In that case a value outside
  // of the texture range of 0..1 is returned.
  return (w+vec3(0.5))/(vec3(TextureDimensions)*TextureSpacing);
}

vec3 TextureToWorld(vec3 t)
{
  // convert coordinate components from 0..1 to world space.
  return t*vec3(TextureDimensions)*TextureSpacing;
}

bool TextureCoordInRange(vec3 t)
{
  bool result = true;

  for (int i=0; i < 3; i++)
    {
    if (t[i] < 0.0) result = false;
    else if (t[i] > 1.0) result = false;
    }

  return result;
}

bool WorldCoordInRange(vec3 w)
{
  return TextureCoordInRange(WorldToTexture(w));
}

vec3 WorldToVoxelNr(vec3 w)
{
  return w/TextureSpacing;
}

vec3 VoxelNrToWorld(vec3 nr)
{
  return nr*TextureSpacing;
}

vec3 VoxelNrToTexture(vec3 nr)
{
  //vec3 w = VoxelNrToWorld(nr);
  //return WorldToTexture(w);
  return nr/vec3(TextureDimensions);
}

// convert from vector in texture with each component
// in range 0..1 to vector with components in range -1..1.
vec3 ConvertVectorFromTexture(vec3 t)
{
  vec3 v;// = vec3(0.0);

  if (all(equal(t, vec3(0.0))))
    {
    v = vec3(0.0);
    }
  else
    {
    v = t * vec3(2.0) - vec3(1.0);
    if (NegateX) v.x *= -1.0;
//    v.y *= -1.0;
    if (NegateY) v.y *= -1.0;	// :S
    if (NegateZ) v.z *= -1.0;	// :s
  }
  
  return v;
}

// return the vector at the specified world coordinate
// the fourth component is the length/strength of the vector
vec4 VectorAtWorld(vec3 w)
{
  vec4 tex;
  vec3 v;
  vec3 t = WorldToTexture(w);
  bool smaller = any(lessThan(t, vec3(0.0)));
  bool greater = any(greaterThan(t, vec3(1.0)));
  if (greater||smaller)
    {
    v = vec3(0.0);
    }
  else
    {
    tex = texture3D(Texture1, WorldToTexture(w)).rgba;
    v = ConvertVectorFromTexture(tex.rgb);
    }

  tex.rgb = v;
  return tex;
}
