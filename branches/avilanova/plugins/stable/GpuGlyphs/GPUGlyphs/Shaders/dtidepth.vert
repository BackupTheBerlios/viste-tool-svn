uniform vec3 EyePosition;
varying vec3 EyeDirection;

void main()
{
  vec3 vertex = gl_Vertex.xyz;
//  vertex.y = gl_MultiTexCoord0.y + (gl_MultiTexCoord0.y - vertex.y);
//  vertex.x = gl_MultiTexCoord0.x + (gl_MultiTexCoord0.x - vertex.x);
//  vertex.z = gl_MultiTexCoord0.z + (gl_MultiTexCoord0.z - vertex.z);
//  vertex = gl_MultiTexCoord0.xyz + (gl_MultiTexCoord0.xyz - vertex);
  EyeDirection = vertex - EyePosition;
//  EyeDirection.z *= -1.0;
//  EyeDirection = gl_Vertex.xyz- EyePosition;
  gl_TexCoord[0] = gl_MultiTexCoord0;
  gl_Position = ftransform();
}
