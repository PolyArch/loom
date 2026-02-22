//===-- connectivity_matrix.cpp - ConnectivityMatrix test ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify ConnectivityMatrix outToIn and inToOut operations.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ConnectivityMatrix.h"

using namespace loom;

int main() {
  ConnectivityMatrix m;

  // Add physical edges: out0 -> in1, out2 -> in3.
  m.outToIn[0] = 1;
  m.outToIn[2] = 3;

  TEST_ASSERT(m.outToIn.count(0));
  TEST_ASSERT(m.outToIn[0] == 1);
  TEST_ASSERT(m.outToIn.count(2));
  TEST_ASSERT(m.outToIn[2] == 3);
  TEST_ASSERT(!m.outToIn.count(4));

  // Add routing internals: input 1 can reach outputs 4 and 5.
  m.inToOut[1].push_back(4);
  m.inToOut[1].push_back(5);

  TEST_ASSERT(m.inToOut.count(1));
  TEST_ASSERT(m.inToOut[1].size() == 2);
  TEST_ASSERT(m.inToOut[1][0] == 4);
  TEST_ASSERT(m.inToOut[1][1] == 5);

  // Input 3 has one output reachable.
  m.inToOut[3].push_back(6);
  TEST_ASSERT(m.inToOut[3].size() == 1);

  return 0;
}
