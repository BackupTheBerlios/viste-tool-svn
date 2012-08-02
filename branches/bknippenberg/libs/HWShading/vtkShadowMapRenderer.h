/**
 * Sets up the rendering context so that all rendering is done as fast as possible
 * (e.g. without lighting etc) and the scene is rendered as a depth-map into a
 * framebuffer object. The ID of this framebuffer object can be requested by
 * other objects so that they can use its contents as a shadow map in a
 * shader.
 *
 * If this is implemented as a subclass of a vtkRenderer, then it is easy to
 * select which actors/volume cast shadows. Those added to this renderer do,
 * and all others do not.
 *
 * Probably no support for transparent objects.
 */ 
