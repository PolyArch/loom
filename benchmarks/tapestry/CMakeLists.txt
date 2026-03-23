cmake_minimum_required(VERSION 3.14)
project(tapestry_benchmarks C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common include path
set(COMMON_INC ${CMAKE_CURRENT_SOURCE_DIR}/common)

# Compiler flags
add_compile_options(-O2 -Wall -Wextra)

# Math library
find_library(MATH_LIB m)

# --- AI/LLM Transformer Layer kernels ---
set(AI_LLM_KERNELS
    qkv_proj
    attn_score
    softmax
    attn_output
    ffn1
    gelu
    ffn2
    layernorm
)

foreach(kernel ${AI_LLM_KERNELS})
    add_executable(bench_${kernel} ai_llm/${kernel}.c)
    target_include_directories(bench_${kernel} PRIVATE ${COMMON_INC})
    if(MATH_LIB)
        target_link_libraries(bench_${kernel} ${MATH_LIB})
    endif()
endforeach()

# --- DSP OFDM Receiver Chain kernels ---
set(DSP_KERNELS
    fft_butterfly
    channel_est
    equalizer
    qam_demod
    viterbi
    crc_check
)

foreach(kernel ${DSP_KERNELS})
    add_executable(bench_${kernel} dsp_ofdm/${kernel}.c)
    target_include_directories(bench_${kernel} PRIVATE ${COMMON_INC})
    if(MATH_LIB)
        target_link_libraries(bench_${kernel} ${MATH_LIB})
    endif()
endforeach()

# --- AR/VR Stereo Vision kernels ---
set(ARVR_KERNELS
    harris_corner
    sad_matching
    stereo_disparity
    image_warp
    post_filter
)

foreach(kernel ${ARVR_KERNELS})
    add_executable(bench_${kernel} arvr_stereo/${kernel}.c)
    target_include_directories(bench_${kernel} PRIVATE ${COMMON_INC})
    if(MATH_LIB)
        target_link_libraries(bench_${kernel} ${MATH_LIB})
    endif()
endforeach()

# --- Robotics VIO Pipeline kernels ---
set(VIO_KERNELS
    imu_integration
    fast_detect
    orb_descriptor
    feature_match
    pose_estimate
)

foreach(kernel ${VIO_KERNELS})
    add_executable(bench_${kernel} robotics_vio/${kernel}.c)
    target_include_directories(bench_${kernel} PRIVATE ${COMMON_INC})
    if(MATH_LIB)
        target_link_libraries(bench_${kernel} ${MATH_LIB})
    endif()
endforeach()

# --- Graph Analytics kernels ---
set(GRAPH_KERNELS
    bfs_traversal
    pagerank_spmv
    triangle_count
    label_prop
)

foreach(kernel ${GRAPH_KERNELS})
    add_executable(bench_${kernel} graph_analytics/${kernel}.c)
    target_include_directories(bench_${kernel} PRIVATE ${COMMON_INC})
    if(MATH_LIB)
        target_link_libraries(bench_${kernel} ${MATH_LIB})
    endif()
endforeach()

# --- ZK/STARK Pipeline kernels ---
set(ZK_KERNELS
    ntt
    msm
    poseidon_hash
    poly_eval
    proof_compose
)

foreach(kernel ${ZK_KERNELS})
    add_executable(bench_${kernel} zk_stark/${kernel}.c)
    target_include_directories(bench_${kernel} PRIVATE ${COMMON_INC}
                               ${CMAKE_CURRENT_SOURCE_DIR}/zk_stark)
    if(MATH_LIB)
        target_link_libraries(bench_${kernel} ${MATH_LIB})
    endif()
endforeach()

# --- CTest integration ---
enable_testing()

foreach(kernel ${AI_LLM_KERNELS} ${DSP_KERNELS} ${ARVR_KERNELS}
               ${VIO_KERNELS} ${GRAPH_KERNELS} ${ZK_KERNELS})
    add_test(NAME ${kernel} COMMAND bench_${kernel})
endforeach()
