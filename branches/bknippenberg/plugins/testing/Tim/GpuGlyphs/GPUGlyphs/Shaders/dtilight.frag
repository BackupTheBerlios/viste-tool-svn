#extension GL_ARB_texture_rectangle : enable

/**
 * Lighting fragment shader
 */ 
varying vec2 TexCoord;
uniform sampler2DRect IntersectionTexture;
uniform sampler2DRect GlyphCenterTexture;

uniform vec3 LightPosition;
uniform vec3 EyePosition;

vec3 WorldToTexture(vec3 w);

vec3 TenPosT;
vec3 TenPosW;

vec3 Eigenvalues;
mat3 Eigenvectors;
// Sets Eigenvectors and Eigenvalues
void EigensystemAtTenPos();

// spotlight:
float diffuse(in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 n, in float p);

mat3 rotationMatrix(mat3 Eigenvectors);

void main()
{
vec4 color;
TenPosW = texture2DRect(GlyphCenterTexture, TexCoord).xyz;
if (all(equal(TenPosW, vec3(0.0)))) discard;
TenPosT = WorldToTexture(TenPosW);
vec4 Intersection = texture2DRect(IntersectionTexture, TexCoord);

EigensystemAtTenPos();

//  if (Eigenvalues[2] < 0.1) discard; // TODO: remove this and avoid numerical errors further on!!!
// color.rgb = vec3(0,0,1);
color.a = 1.0;

mat3 rotate = rotationMatrix(Eigenvectors);

vec3 normal;
//normal = normalize(TenPosW.xyz - Intersection.xyz)*rotate; //Eigenvectors; // sphere normal
normal = rotate*normalize(TenPosW.xyz - Intersection.xyz); //Eigenvectors; // sphere normal
vec3 orientationcolor = normal;

vec3 r = vec3(1.0) / pow(Eigenvalues, vec3(2.0));
normal = vec3(2.0) * r * normal; //(TenPosW.xyz - Intersection.xyz);
normal = normalize(normal);
normal = transpose(rotate) * normal;

vec3 light_direction = normalize(Intersection.xyz - LightPosition);
vec3 eye_direction = normalize(Intersection.xyz - EyePosition);

float cl = Eigenvalues[0] - Eigenvalues[1];
//cl = sqrt(cl); // more saturation
// RGB color coding with saturation depending on cl:
color.rgb = vec3(cl) * abs(Eigenvectors[0]) + vec3(1.0-cl)*vec3(sqrt(1.0/3.0));
//color.rgb = vec3(cl) * Eigenvectors[0] + vec3(1.0-cl)*vec3(sqrt(1.0/3.0));
// swap the components for AZM-Rainer Goebel brain for my paper.
//color.gbr = vec3(cl) * abs(Eigenvectors[0]) + vec3(1.0-cl)*vec3(sqrt(1.0/3.0));

// spotlight:
float dp = diffuse(light_direction, normal);
float sp = specular(eye_direction, light_direction, normal, 8.0);

//color.rgb = vec3(0.2+0.7*dp)*vec3(1.0, 0.7, 1.0) + vec3(0.5*sp); // pink

// UNCOMMENT THIS FOR CORRECT LIGHTING
color.rgb = vec3(0.2+0.7*dp)*color.rgb + vec3(0.5*sp);

if (all(equal(TenPosW, vec3(0.0)))) color.rgb = vec3(1, 1, 0);

  gl_FragColor = color;
  gl_FragDepth = Intersection.a;
}
