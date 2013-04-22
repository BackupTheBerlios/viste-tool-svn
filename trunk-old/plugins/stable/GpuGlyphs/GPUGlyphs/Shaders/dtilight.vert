//uniform vec3 EyePosition;
//varying vec3 EyeDirection;
varying vec2 TexCoord;

void main()
{
//  EyeDirection = gl_Vertex.xyz - EyePosition;
  TexCoord = gl_MultiTexCoord0.st;
  gl_Position = ftransform();
}
