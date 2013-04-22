uniform sampler3D Texture1;
uniform sampler3D Texture2;

mat3 Eigenvectors;
vec3 Eigenvalues;
vec3 TenPosT;

vec3 ConvertVectorFromTexture(vec3 t);

// return the eigenvectors, combined with normalized eigenvalues in matrix.
void EigensystemAtTenPos()
{
  const vec4 tex1 = texture3D(Texture1, TenPosT);
  const vec4 tex2 = texture3D(Texture2, TenPosT);

  Eigenvectors[0] = ConvertVectorFromTexture(tex1.rgb);
  Eigenvectors[1] = ConvertVectorFromTexture(tex2.rgb);
  Eigenvectors[2] = cross(Eigenvectors[0], Eigenvectors[1]);

  Eigenvalues[0] = tex1.a;
  Eigenvalues[1] = tex2.a;
  Eigenvalues[2] = 1.0 - tex1.a - tex2.a;
}

/*
void EigensystemAtTenPos()
{

  Eigenvectors[0] = vec3(0.211325, 0.788675, 0.57735);
  Eigenvectors[1] = vec3(-0.788675, -0.211325, 0.57735);
  Eigenvectors[2] = vec3(0.57735, -0.57735, 0.57735);

  Eigenvalues = vec3(5.73205, 2.26795, 1.0);
  Eigenvalues = normalize(Eigenvalues);

}
*/
