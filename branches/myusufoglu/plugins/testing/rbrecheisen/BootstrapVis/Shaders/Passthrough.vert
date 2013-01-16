/**
 * Vertex Shader: Direct passthrough of vertices
 * by Ron Otten
 *
 * 2009-01-07
 *
 */

#version 120
#extension GL_EXT_gpu_shader4: enable

// ================================================

void main(void)
{
    gl_Position = gl_Vertex;
    gl_FrontColor = gl_Color;
    gl_TexCoord[0] = gl_MultiTexCoord0;
}

