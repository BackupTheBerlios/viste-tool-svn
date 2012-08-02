float infinity();

float qz(vec3 p, float alpha, float beta)
{
  float xp = pow(p.x, 2.0/alpha);
  float yp = pow(p.y, 2.0/alpha);
  float zp = pow(p.z, 2.0/beta);
  return pow(xp+yp, alpha/beta) + zp - 1.0;
}

float RayAlignedSuperquadricIntersection(vec3 RayPos, vec3 RayDir, vec3 Radii, float alpha, float beta)
{
  // first, choose 3 random points
  float t_left = 0.0;
  float t_right = 1000.0;
  float t_middle = 400;
  float t_new;
  vec3 x_left = RayPos + vec3(t_left)*RayDir;
  vec3 x_right = RayPos + vec3(t_right)*RayDir;
  vec3 x_middle = RayPos + vec3(t_middle)*RayDir;
  vec3 x_new;
  float f_left = qz(alpha, beta, x_left);
  float f_right = qz(alpha, beta, x_right);
  float f_middle = qz(alpha, beta, x_middle);
  float f_new;

  // XXX: remove all assumtions (add code)
  // assume qz(x_left) > qz(x_middle) && qz(x_right) > qz(x_middle)
  if ((f_middle >= f_left) || (f_middle >= f_right)) return infinity();

  while ((f_middle > 0.0) && ((t_right - t_left) > 0.5))
    {
    if ((t_middle - t_left) > (t_right - t_middle))
      {
      t_new = (t_middle + t_left) / 2.0;
      x_new = RayPos + vec3(t_new)*RayDir;
      f_new = qz(alpha, beta, x_new);
    
      if (f_new < f_middle)
        {
        f_right = f_middle; x_right = x_middle; t_right = t_middle;
        f_middle = f_new; x_middle = x_new; t_middle = t_new;      
        } // if (f_new < f_middle)
      else
        { // f_new >= f_middle
        f_left = f_new; x_left = x_new; t_left = t_new;
        } // else
      } // if
    else
      { // t_middle - t_left <= t_right - t_middle
      t_new = (t_middle + t_right) / 2.0;
      x_new = RayPos + vec3(t_new)*RayDir;
      f_new = qz(alpha, beta, x_new);

      if (f_new < f_middle)
        {
        f_left = f_middle; x_left = x_middle; t_left = t_middle;
        f_middle = f_new; x_middle = x_new; t_middle = t_new;
        } // if
      else
        { // f_new >= f_middle)
        f_right = f_new; x_right = x_new; t_right = t_new;
        } // else
      } // else
    } // while

  if (f_middle > 0.0) return infinity();
  return t_middle;
}
