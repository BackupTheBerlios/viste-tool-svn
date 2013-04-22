uniform vec3 EyePosition;
varying vec3 vEyeDirection;

//attribute float MaxRadius;
attribute vec4 MinMaxRadius;
attribute vec4 SHCoefficients1;
attribute vec4 SHCoefficients2;
attribute vec4 SHCoefficients3;
attribute vec3 SHCoefficients4;

varying vec3 vGlyphPosition;
//varying float vMaxRadius;
varying vec4 vMinMaxRadius;
varying vec4 vSHCoefficients1;
varying vec4 vSHCoefficients2;
varying vec4 vSHCoefficients3;
varying vec3 vSHCoefficients4;
varying float vCameraZ;

varying vec3 vRelativeVertexPosition;

void main()
{
//  vec3 GlyphPosition = gl_Normal.xyz;
//  vGlyphPosition = GlyphPosition;
  vGlyphPosition = gl_Normal.xyz;

  vRelativeVertexPosition = vec3(gl_Color.rgb); //vec3(gl_Color.rg, 0.0);
  vCameraZ = gl_Color.a;
  // XXX: gl_Color.a is still free to use. Maybe for MaxRadius?
  // nah, wait with this because maybe I'll pass MinRadius also at some point.

  vEyeDirection = gl_Vertex.xyz - EyePosition;
  // vEyeDirection will first be interpolated, and then normalized in fragment shader


//  vMaxRadius = MaxRadius;
  vMinMaxRadius = MinMaxRadius;
  vSHCoefficients1 = SHCoefficients1;
  vSHCoefficients2 = SHCoefficients2;
  vSHCoefficients3 = SHCoefficients3;
  vSHCoefficients4 = SHCoefficients4;

//  gl_FrontColor = vec4(gl_Color.xyz, 1.0);
  gl_FrontColor = vec4(gl_MultiTexCoord0.x, gl_MultiTexCoord0.y, gl_MultiTexCoord0.z, 1.0);

  //gl_Position = gl_ModelViewProjectionMatrix * v;
  gl_Position = ftransform();
}
