uniform vec3 EyePosition;
varying vec3 vEyeDirection;
attribute vec3 Eigenvalues;
attribute vec3 Eigenvector1;
attribute vec3 Eigenvector2;
attribute vec3 Eigenvector3;
attribute vec3 GlyphPosition;
attribute vec3 Color;
varying vec3 vEigenvalues;
//varying vec3 vEigenvector1;
//varying vec3 vEigenvector2;
varying mat3 vEigenvectors;
varying vec3 vGlyphPosition;
varying vec3 vColor;

void main()
{
  vec3 vertex = gl_Vertex.xyz;
  vEyeDirection = vertex - EyePosition;

  vGlyphPosition = GlyphPosition;
  int i;
  vec3 e; for (i=0; i < 3; i++) e[i] = abs(Eigenvalues[i]);
  float evalsum = e[0]+e[1]+e[2];
  e = e / vec3(evalsum);
//  for (int i=1; i < 3; i++) if (e[i] < 0.05) e[i] = 0.05;
  vEigenvalues = e;
  vEigenvectors[0] = normalize(Eigenvector1);
  vEigenvectors[1] = normalize(Eigenvector2);
  vEigenvectors[2] = normalize(Eigenvector3);
  vColor = Color;

  gl_Position = ftransform();
}
