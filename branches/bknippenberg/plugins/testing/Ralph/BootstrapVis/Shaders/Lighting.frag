/**
 * Fragment Shader: default vertex shader
 * by Ralph Brecheisen
 *
 * 2010-01-31	Ralph Brecheisen
 * - First version 
 */
uniform float shininess;
uniform vec4  ambient, diffuse;
varying vec3  normal, lightDir, halfVector;

void main()
{
	vec3 n, halfV;
	float NdotL, NdotHV;
	
	vec4 color = ambient;
	n = normalize( normal );
	
	NdotL = max( dot( n, lightDir), 0.0 );
	if( NdotL > 0.0 )
	{
		color += diffuse * NdotL;
		halfV = normalize( halfVector );
		NdotHV = max( dot( n, halfV ), 0.0 );
		color += vec4( 1.0, 1.0, 1.0, 1.0 ) * pow( NdotHV, shininess );
	}
	
	gl_FragColor = color;
}
