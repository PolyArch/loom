add_executable(loom-dse-objective-test
  ObjectiveTest.cpp
)

add_executable(loom-dse-candidate-generator-test
  CandidateGeneratorTest.cpp
)

add_executable(loom-dse-plan-test
  PlanTest.cpp
)

add_executable(loom-dse-evidence-obligation-test
  EvidenceObligationTest.cpp
)

add_executable(loom-dse-promotion-test
  PromotionTest.cpp
)

target_link_libraries(loom-dse-objective-test PRIVATE
  LoomDSEObjective
  LLVMSupport
)

target_link_libraries(loom-dse-candidate-generator-test PRIVATE
  LoomDSECandidateGenerator
  LLVMSupport
)

target_link_libraries(loom-dse-plan-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-evidence-obligation-test PRIVATE
  LoomDSE
  LLVMSupport
)

target_link_libraries(loom-dse-promotion-test PRIVATE
  LoomDSE
  LLVMSupport
)
