uniform sampler2D colorSampler;
uniform sampler2D depthSampler;

void main( void )
{
	gl_FragColor = texture2D( colorSampler, gl_TexCoord[0].st );
	gl_FragDepth = texture2D( depthSampler, gl_TexCoord[0].st ).r;
}
