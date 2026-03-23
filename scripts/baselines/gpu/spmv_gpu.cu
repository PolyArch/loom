// SpMV (Sparse Matrix-Vector multiply) baseline using cuSPARSE.
// CSR format input, measures throughput for varying sparsity levels.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

// -----------------------------------------------------------------------
// Error checking
// -----------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(err));                                    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t st = (call);                                              \
    if (st != CUSPARSE_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", __FILE__,         \
                   __LINE__, static_cast<int>(st));                            \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// -----------------------------------------------------------------------
// Timing helper
// -----------------------------------------------------------------------

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin() { CUDA_CHECK(cudaEventRecord(start, 0)); }
  float end_ms() {
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

struct BenchResult {
  double mean_ms;
  double stddev_ms;
  double min_ms;
};

template <typename Fn>
BenchResult run_benchmark(Fn &&fn, int warmup = 3, int trials = 10) {
  GpuTimer timer;
  std::vector<double> times;

  for (int i = 0; i < warmup; ++i) {
    timer.begin();
    fn();
    timer.end_ms();
  }

  for (int i = 0; i < trials; ++i) {
    timer.begin();
    fn();
    double ms = timer.end_ms();
    times.push_back(ms);
  }

  double sum = 0.0, mn = 1e30;
  for (double t : times) {
    sum += t;
    if (t < mn) mn = t;
  }
  double mean = sum / times.size();
  double sq = 0.0;
  for (double t : times)
    sq += (t - mean) * (t - mean);
  double sd = std::sqrt(sq / (times.size() - 1));
  return {mean, sd, mn};
}

// -----------------------------------------------------------------------
// Generate random sparse matrix in CSR format
// -----------------------------------------------------------------------

static void generate_sparse_csr(int rows, int cols, double density,
                                std::vector<int> &row_ptr,
                                std::vector<int> &col_idx,
                                std::vector<float> &values) {
  row_ptr.resize(rows + 1);
  col_idx.clear();
  values.clear();

  row_ptr[0] = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (static_cast<double>(rand()) / RAND_MAX < density) {
        col_idx.push_back(c);
        values.push_back(static_cast<float>(rand()) / RAND_MAX);
      }
    }
    row_ptr[r + 1] = static_cast<int>(col_idx.size());
  }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
  int rows = 10000, cols = 10000;
  double density = 0.01; // 1% non-zero
  if (argc >= 3) {
    rows = std::atoi(argv[1]);
    cols = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    density = std::atof(argv[3]);
  }

  // Generate sparse matrix.
  std::vector<int> h_row_ptr, h_col_idx;
  std::vector<float> h_values;
  generate_sparse_csr(rows, cols, density, h_row_ptr, h_col_idx, h_values);

  int nnz = static_cast<int>(h_values.size());
  double total_ops = 2.0 * nnz; // Each non-zero: 1 multiply + 1 add.

  // Dense vector.
  std::vector<float> h_x(cols), h_y(rows, 0.0f);
  for (int i = 0; i < cols; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX;

  // Device allocations.
  int *d_row_ptr, *d_col_idx;
  float *d_values, *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (rows + 1) * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float),
                         cudaMemcpyHostToDevice));

  // cuSPARSE setup.
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;

  CUSPARSE_CHECK(cusparseCreateCsr(&matA, rows, cols, nnz, d_row_ptr,
                                    d_col_idx, d_values, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                    CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F));

  float alpha = 1.0f, beta = 0.0f;

  // Determine buffer size.
  size_t buffer_size = 0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
      vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

  void *d_buffer = nullptr;
  if (buffer_size > 0)
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));

  auto spmv_fn = [&]() {
    CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));
  };

  BenchResult res = run_benchmark(spmv_fn);

  std::printf("RESULT kernel=spmv_cusparse mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f rows=%d cols=%d nnz=%d density=%.4f\n",
              res.mean_ms, res.stddev_ms, res.min_ms, total_ops, rows, cols,
              nnz, density);

  // Cleanup.
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
  CUSPARSE_CHECK(cusparseDestroy(handle));
  if (d_buffer)
    CUDA_CHECK(cudaFree(d_buffer));
  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_idx));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  return 0;
}
