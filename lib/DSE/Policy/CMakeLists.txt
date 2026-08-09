add_library(LoomDSEObjective
  ../Objective.cpp
)

target_include_directories(LoomDSEObjective PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEObjective PUBLIC
  LoomCommon
  LLVMSupport
)

add_library(LoomDSECandidateGenerator
  ../CandidateGenerator.cpp
)

target_include_directories(LoomDSECandidateGenerator PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSECandidateGenerator PUBLIC
  LoomCommon
  LoomExternalTool
  LLVMSupport
)

add_library(LoomDSEPolicy
  ../EvidenceObligation.cpp
  ../Plan.cpp
  ../Promotion.cpp
  ../PromotionAcquisition.cpp
  ../ResolvedConfigView.cpp
)

target_include_directories(LoomDSEPolicy PUBLIC
  $<BUILD_INTERFACE:${LOOM_SOURCE_DIR}/include>
)

target_link_libraries(LoomDSEPolicy PUBLIC
  LoomCommon
  LoomDSECandidateGenerator
  LoomDSEObjective
  LoomEvaluation
  LLVMSupport
)
