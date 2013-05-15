/**
 * Vertex Shader: default vertex shader
 * by Ralph Brecheisen
 *
 * 2010-01-31	Ralph Brecheisen
 * - First version 
 */
varying vec3 normal, lightDir, halfVector;
 
void main()
{
	normal = normalize( gl_NormalMatrix * gl_Normal );
	lightDir = normalize( vec3( gl_LightSource[0].position ) );
	halfVector = normalize( vec3( gl_LightSource[0].halfVector.xyz ) );
	
    gl_FrontColor = gl_Color;
    gl_TexCoord[0] = gl_MultiTexCoord0;
    
	gl_Position = ftransform();
}
