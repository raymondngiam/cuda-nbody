#include "test_utils.h"

void test_utils::PrintNBodies(Body *p, int nBodies){
  for (int i = 0 ; i < nBodies; i++) {
    fmt::print("{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n", 
      p[i].pos.x, 
      p[i].pos.y, 
      p[i].pos.z, 
      p[i].vel.x, 
      p[i].vel.y, 
      p[i].vel.z);
  }
}

bool test_utils::AlmostEqual(Body *p, Body *q, int nBodies, float epsilon){
  bool almostEqual = true;
  float *p_float = (float*)p;
  float *q_float = (float*)q;
  for (int i = 0 ; i < 6*nBodies; i++) {
    if (fabs(p_float[i]-q_float[i]) > epsilon)
    {
      almostEqual = false;
      break;
    }      
  }
  return almostEqual;
}

void test_utils::GetBaseline(Body *baseline, int nBodies){
  std::ifstream f(fmt::format("../data/data_{:04d}.txt",nBodies));
  std::string line;
  boost::regex pat( "(.*),(.*),(.*),(.*),(.*),(.*)" );
  std::getline(f,line);  //skip header line

  int index=0;
  while(std::getline(f,line)){
    boost::smatch matches;
    if (boost::regex_match(line, matches, pat)){
      baseline[index].pos.x = boost::lexical_cast<float>(matches[1]);
      baseline[index].pos.y = boost::lexical_cast<float>(matches[2]);
      baseline[index].pos.z = boost::lexical_cast<float>(matches[3]);
      baseline[index].vel.x = boost::lexical_cast<float>(matches[4]);
      baseline[index].vel.y = boost::lexical_cast<float>(matches[5]);
      baseline[index].vel.z = boost::lexical_cast<float>(matches[6]);
      index += 1;
    }
  }  
}