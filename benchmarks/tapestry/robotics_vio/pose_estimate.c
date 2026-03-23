/*
 * 8-Point Pose Estimation for Visual-Inertial Odometry.
 * Given point correspondences (x1,y1) <-> (x2,y2), estimates the
 * essential matrix using the 8-point algorithm.
 * Uses power iteration to find the smallest singular vector of the
 * constraint matrix A (simplified SVD).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define NUM_POINTS    20
#define MIN_POINTS    8
#define TILE_PTS      8
#define POWER_ITERS   100
#define MAT_COLS      9  /* flattened 3x3 essential matrix */

/*
 * Build the constraint matrix A from point correspondences.
 * For each point pair (x1,y1,1) and (x2,y2,1), the constraint is:
 * [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] * e = 0
 * where e is the flattened essential matrix.
 */
void build_constraint_matrix(const float *pts1, const float *pts2,
                             int n, float *A) {
    int i;
    for (i = 0; i < n; i++) {
        float x1 = pts1[i * 2 + 0];
        float y1 = pts1[i * 2 + 1];
        float x2 = pts2[i * 2 + 0];
        float y2 = pts2[i * 2 + 1];

        float *row = &A[i * MAT_COLS];
        row[0] = x2 * x1;
        row[1] = x2 * y1;
        row[2] = x2;
        row[3] = y2 * x1;
        row[4] = y2 * y1;
        row[5] = y2;
        row[6] = x1;
        row[7] = y1;
        row[8] = 1.0f;
    }
}

/*
 * Compute A^T * A (9x9 matrix) from A (n x 9).
 * Tiled by rows of A.
 */
void compute_AtA_tiled(const float *A, int n, float *AtA) {
    int i, j, k;
    memset(AtA, 0, MAT_COLS * MAT_COLS * sizeof(float));

    TILE_FOR(tk, 0, n, TILE_PTS) {
        int tk_end = TILE_END(tk, n, TILE_PTS);
        for (k = tk; k < tk_end; k++) {
            for (i = 0; i < MAT_COLS; i++) {
                for (j = 0; j < MAT_COLS; j++) {
                    AtA[i * MAT_COLS + j] +=
                        A[k * MAT_COLS + i] * A[k * MAT_COLS + j];
                }
            }
        }
    }
}

/* Reference A^T * A */
void compute_AtA_ref(const float *A, int n, float *AtA) {
    int i, j, k;
    memset(AtA, 0, MAT_COLS * MAT_COLS * sizeof(float));
    for (k = 0; k < n; k++) {
        for (i = 0; i < MAT_COLS; i++) {
            for (j = 0; j < MAT_COLS; j++) {
                AtA[i * MAT_COLS + j] +=
                    A[k * MAT_COLS + i] * A[k * MAT_COLS + j];
            }
        }
    }
}

/*
 * Power iteration to find the eigenvector corresponding to the
 * smallest eigenvalue of AtA. Uses inverse iteration:
 * For the smallest eigenvalue, we shift and iterate.
 * Simplified: direct power iteration on (maxeval*I - AtA).
 */
void power_iteration_smallest(const float *AtA, float *evec, int n_iter) {
    int i, j, iter;

    /* Find approximate largest eigenvalue (trace gives upper bound) */
    float trace = 0.0f;
    for (i = 0; i < MAT_COLS; i++) {
        trace += AtA[i * MAT_COLS + i];
    }

    /* Construct shifted matrix B = trace*I - AtA */
    float B[MAT_COLS * MAT_COLS];
    for (i = 0; i < MAT_COLS; i++) {
        for (j = 0; j < MAT_COLS; j++) {
            B[i * MAT_COLS + j] = -AtA[i * MAT_COLS + j];
            if (i == j) {
                B[i * MAT_COLS + j] += trace;
            }
        }
    }

    /* Initialize vector */
    for (i = 0; i < MAT_COLS; i++) {
        evec[i] = 1.0f / sqrtf((float)MAT_COLS);
    }

    /* Power iteration on B */
    float tmp[MAT_COLS];
    for (iter = 0; iter < n_iter; iter++) {
        /* tmp = B * evec */
        for (i = 0; i < MAT_COLS; i++) {
            float sum = 0.0f;
            for (j = 0; j < MAT_COLS; j++) {
                sum += B[i * MAT_COLS + j] * evec[j];
            }
            tmp[i] = sum;
        }

        /* Normalize */
        float norm = 0.0f;
        for (i = 0; i < MAT_COLS; i++) {
            norm += tmp[i] * tmp[i];
        }
        norm = sqrtf(norm);
        if (norm < 1e-12f) break;
        for (i = 0; i < MAT_COLS; i++) {
            evec[i] = tmp[i] / norm;
        }
    }
}

/*
 * Extract essential matrix from the solution vector.
 * evec is the 9-element vector corresponding to the null space of A.
 */
void extract_essential(const float *evec, float E[3][3]) {
    int i, j;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            E[i][j] = evec[i * 3 + j];
        }
    }
}

/*
 * Verify essential matrix constraint: x2^T * E * x1 should be ~0
 * for all point correspondences.
 */
float verify_constraint(const float *pts1, const float *pts2, int n,
                        const float E[3][3]) {
    float max_residual = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float x1 = pts1[i * 2 + 0];
        float y1 = pts1[i * 2 + 1];
        float x2 = pts2[i * 2 + 0];
        float y2 = pts2[i * 2 + 1];

        /* E * [x1, y1, 1]^T */
        float ex = E[0][0]*x1 + E[0][1]*y1 + E[0][2];
        float ey = E[1][0]*x1 + E[1][1]*y1 + E[1][2];
        float ez = E[2][0]*x1 + E[2][1]*y1 + E[2][2];

        /* [x2, y2, 1] * (E * [x1,y1,1]^T) */
        float residual = x2*ex + y2*ey + ez;
        if (fabsf(residual) > max_residual) {
            max_residual = fabsf(residual);
        }
    }
    return max_residual;
}

int main(void) {
    int n = NUM_POINTS;

    float *pts1 = (float *)malloc((size_t)n * 2 * sizeof(float));
    float *pts2 = (float *)malloc((size_t)n * 2 * sizeof(float));
    float *A = (float *)malloc((size_t)n * MAT_COLS * sizeof(float));
    float AtA_t[MAT_COLS * MAT_COLS];
    float AtA_r[MAT_COLS * MAT_COLS];
    float evec[MAT_COLS];
    float E[3][3];

    if (!pts1 || !pts2 || !A) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate synthetic point correspondences */
    /* Simulate a small rotation + translation */
    float R_true[3][3] = {
        { 0.9998f, -0.0175f,  0.0087f},
        { 0.0175f,  0.9998f, -0.0044f},
        {-0.0087f,  0.0044f,  0.9999f}
    };
    float t_true[3] = {0.1f, 0.05f, 0.9f};

    int i;
    unsigned int state = 42;
    for (i = 0; i < n; i++) {
        /* Random 3D point */
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        float X = ((state % 1000) - 500) / 100.0f;
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        float Y = ((state % 1000) - 500) / 100.0f;
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF;
        float Z = 5.0f + (state % 500) / 100.0f;

        /* Project to camera 1: normalized coords */
        pts1[i * 2 + 0] = X / Z;
        pts1[i * 2 + 1] = Y / Z;

        /* Transform to camera 2 */
        float X2 = R_true[0][0]*X + R_true[0][1]*Y + R_true[0][2]*Z + t_true[0];
        float Y2 = R_true[1][0]*X + R_true[1][1]*Y + R_true[1][2]*Z + t_true[1];
        float Z2 = R_true[2][0]*X + R_true[2][1]*Y + R_true[2][2]*Z + t_true[2];
        pts2[i * 2 + 0] = X2 / Z2;
        pts2[i * 2 + 1] = Y2 / Z2;
    }

    /* Build constraint matrix */
    build_constraint_matrix(pts1, pts2, n, A);

    /* Compute A^T * A with both methods */
    compute_AtA_tiled(A, n, AtA_t);
    compute_AtA_ref(A, n, AtA_r);

    /* Verify AtA matches */
    float ata_err = 0.0f;
    for (i = 0; i < MAT_COLS * MAT_COLS; i++) {
        float err = fabsf(AtA_t[i] - AtA_r[i]);
        if (err > ata_err) ata_err = err;
    }

    /* Find smallest eigenvector via power iteration */
    power_iteration_smallest(AtA_t, evec, POWER_ITERS);
    extract_essential(evec, E);

    /* Verify epipolar constraint */
    float max_residual = verify_constraint(pts1, pts2, n, E);

    printf("pose_estimate: AtA_err=%e, max_residual=%e\n",
           ata_err, max_residual);

    /* AtA should match exactly, residual should be small */
    int pass = (ata_err < 1e-3f) && (max_residual < 0.1f);
    printf("pose_estimate: %s\n", pass ? "PASS" : "FAIL");

    free(pts1); free(pts2); free(A);
    return pass ? 0 : 1;
}
