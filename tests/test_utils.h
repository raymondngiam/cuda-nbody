#pragma once
#include <string>
#include <fstream>
#include <fmt/core.h>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include "nbody_common.h"

using Body = nbody_common::Body;

namespace test_utils {
  void PrintNBodies(Body *p, int nBodies);

  bool AlmostEqual(Body *p, Body *q, int nBodies, float epsilon);

  void GetBaseline(Body *baseline, int nBodies);
} // namespace test_utils