uniform sampler2D ShadowMap;

uniform float SpecularPower;
uniform float DiffuseContribution;
uniform float SpecularContribution;
uniform float AmbientContribution;
uniform float DiffuseContributionShadow;
uniform float SpecularContributionShadow;
uniform float AmbientContributionShadow;

varying vec4 projCoord;
varying vec3 lightDir, viewDir;
varying vec3 ntang; // normalized tangent

varying vec2 id;

float diffuse(in vec3 l,  in vec3 n);
//vec4 diffuse_cylavg(in vec3 v, in vec3 l, in vec3 n);
float specular(in vec3 v, in vec3 l, in vec3 t, in float p);

uniform bool ToneShading;
uniform vec3 CoolColor;
uniform vec3 WarmColor;

void main()
{
  vec2 idf = vec2(id[0], id[1]);
  idf = mod(idf, 512.0);
  idf = idf / 512.0;

  vec4 color;
  vec3 projectiveBiased = (projCoord.xyz / projCoord.q);

  // convert from [-1.0, 1.0] to [0.0, 1.0]
  projectiveBiased = (projectiveBiased + 1.0) * 0.5;
  
  vec4 shadowValue = texture2D(ShadowMap, projectiveBiased.xy);

  float diff = abs(idf[0] - shadowValue[0]);
  float diff2 = abs(idf[1] - shadowValue[1]);

  float d = diffuse(lightDir, ntang);
  float s = specular(viewDir, lightDir, ntang, SpecularPower);

  if ((diff < 0.01)&&(diff2 < 0.01))
    { // in light
    if (ToneShading)
      {
      float val = AmbientContribution + DiffuseContribution*d + SpecularContribution*s;
      color.rgb = vec3(val)*WarmColor + vec3(1.0-val)*CoolColor;
      } // if (ToneShading)
    else
      { // no tone shading
      color = (vec4(AmbientContribution) + vec4(DiffuseContribution)*d)*gl_Color + vec4(SpecularContribution)*s;
      }
    }
  else
    { // in shadow
    if (ToneShading)
      {
      float val = AmbientContributionShadow + DiffuseContributionShadow*d + SpecularContributionShadow*s;
      color.rgb = vec3(val)*WarmColor + vec3(1.0-val)*CoolColor;
      } // if (ToneShading)
    else
      { // no tone shading
      color = (vec4(AmbientContributionShadow) + vec4(DiffuseContributionShadow)*d)*gl_Color + vec4(SpecularContributionShadow)*s;
      }
    }

  color.a = 1.0;

  gl_FragColor = color;
}
