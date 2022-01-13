// forward Euler finite difference solution to the heat equation on a 2D grid

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#ifdef WITHOUT_COMPRESSION
  #include "array2d.h"
#else
  #include "zfparray2.h"
  using namespace zfp;
#endif

int main(int argc, char* argv[])
{
  int nx = 0;
  int ny = 0;
  int nt = 0;
  double rate = 64;

  // parse arguments
  switch (argc) {
    case 5:
      if (sscanf(argv[4], "%d", &nt) != 1)
        goto usage;
      // FALLTHROUGH
    case 4:
      if (sscanf(argv[2], "%d", &nx) != 1 ||
          sscanf(argv[3], "%d", &ny) != 1)
        goto usage;
      // FALLTHROUGH
    case 2:
      if (sscanf(argv[1], "%lf", &rate) != 1)
        goto usage;
      // FALLTHROUGH
    case 1:
      break;
    default:
    usage:
      std::cerr << "Usage: diffusion [rate] [nx] [ny] [nt]" << std::endl;
      return EXIT_FAILURE;
  }

  // grid dimensions
  if (nx == 0)
    nx = 100;
  if (ny == 0)
    ny = nx;

  // location of point heat source
  int x0 = (nx - 1) / 2;
  int y0 = (ny - 1) / 2;

  // constants used in the solution
  const double k = 0.04;
  const double dx = 2.0 / (std::max(nx, ny) - 1);
  const double dy = 2.0 / (std::max(nx, ny) - 1);
  const double dt = 0.5 * (dx * dx + dy * dy) / (8 * k);
  const double tfinal = nt ? nt * dt : 1;
  const double pi = 3.14159265358979323846;

  // initialize u (constructor zero-initializes)
  array2d u(nx, ny, rate);
  rate = u.rate();
  u(x0, y0) = 1;

  // iterate until final time
  std::cerr.precision(6);
  double t;
  for (t = 0; t < tfinal; t += dt) {
    std::cerr << "t=" << std::fixed << t << std::endl;
    // compute du/dt
    array2d du(nx, ny, rate);
    for (int y = 1; y < ny - 1; y++) {
      for (int x = 1; x < nx - 1; x++) {
        double uxx = (u(x - 1, y) - 2 * u(x, y) + u(x + 1, y)) / (dx * dx);
        double uyy = (u(x, y - 1) - 2 * u(x, y) + u(x, y + 1)) / (dy * dy);
        du(x, y) = dt * k * (uxx + uyy);
      }
    }
    // take forward Euler step
    for (uint i = 0; i < u.size(); i++)
      u[i] += du[i];
  }

  // compute root mean square error with respect to exact solution
  double e = 0;
  double sum = 0;
  for (int y = 1; y < ny - 1; y++) {
    double py = dy * (y - y0);
    for (int x = 1; x < nx - 1; x++) {
      double px = dx * (x - x0);
      double f = u(x, y);
      double g = dx * dy * std::exp(-(px * px + py * py) / (4 * k * t)) / (4 * pi * k * t);
      e += (f - g) * (f - g);
      sum += f;
    }
  }
  e = std::sqrt(e / ((nx - 2) * (ny - 2)));
  std::cerr.unsetf(std::ios::fixed);
  std::cerr << "rate=" << rate << " sum=" << std::fixed << sum << " error=" << std::setprecision(6) << std::scientific << e << std::endl;

  return 0;
}
