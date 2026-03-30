set(SCICOMP_KERNEL_SOURCES
  halo_exchange.c
  stencil_compute.c
  boundary_apply.c
  residual_check.c
  spmv.c
  dot_product.c
  axpy.c
  precondition.c
  convergence_check.c
  force_compute.c
  position_update.c
  neighbor_rebuild.c
  energy_reduce.c
)

add_library(scicomp_kernels OBJECT ${SCICOMP_KERNEL_SOURCES})
target_include_directories(scicomp_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(scicomp_kernels_smoke scicomp_smoke.c)
target_sources(scicomp_kernels_smoke PRIVATE $<TARGET_OBJECTS:scicomp_kernels>)
target_include_directories(scicomp_kernels_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(scicomp_kernels_smoke PRIVATE m)
