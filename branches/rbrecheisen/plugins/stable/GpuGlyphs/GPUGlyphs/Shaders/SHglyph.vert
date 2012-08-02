uniform vec3 EyePosition;
varying vec3 vEyeDirection;

attribute vec3 GlyphPosition;
attribute vec4 SHCoefficients1;
attribute vec4 SHCoefficients2;
attribute vec4 SHCoefficients3;
attribute vec3 SHCoefficients4;

varying vec3 vGlyphPosition;
varying vec4 vSHCoefficients1;
varying vec4 vSHCoefficients2;
varying vec4 vSHCoefficients3;
varying vec3 vSHCoefficients4;

attribute vec2 MinMaxRadius;
varying vec2 vMinMaxRadius;

void main()
{
  vec3 vertex = gl_Vertex.xyz;
  vEyeDirection = vertex - EyePosition;

  vGlyphPosition = GlyphPosition;

  vSHCoefficients1 = SHCoefficients1;
  vSHCoefficients2 = SHCoefficients2;
  vSHCoefficients3 = SHCoefficients3;
  vSHCoefficients4 = SHCoefficients4;

  vMinMaxRadius = MinMaxRadius;

  gl_FrontColor = gl_Color;
  gl_Position = ftransform();
}
