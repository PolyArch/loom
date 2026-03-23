/*
 * IMU Pre-Integration for Visual-Inertial Odometry.
 * Integrates accelerometer and gyroscope data at 200 Hz using
 * Euler integration to compute delta position, velocity, and orientation.
 * FP32 precision, sequential accumulation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tile_utils.h"

#define IMU_RATE      200     /* Hz */
#define WINDOW_SIZE   200     /* 1 second of data */
#define DT            (1.0f / IMU_RATE)
#define TILE_SAMPLES  32
#define GRAVITY       9.81f

/* 3D vector type */
typedef struct { float x, y, z; } vec3_t;

/* Simple rotation matrix (3x3 stored row-major) */
typedef struct { float m[9]; } mat3_t;

static inline mat3_t mat3_identity(void) {
    mat3_t r;
    memset(r.m, 0, sizeof(r.m));
    r.m[0] = r.m[4] = r.m[8] = 1.0f;
    return r;
}

/* Rotate vector by matrix: R * v */
static inline vec3_t mat3_mul_vec(const mat3_t *R, vec3_t v) {
    vec3_t out;
    out.x = R->m[0]*v.x + R->m[1]*v.y + R->m[2]*v.z;
    out.y = R->m[3]*v.x + R->m[4]*v.y + R->m[5]*v.z;
    out.z = R->m[6]*v.x + R->m[7]*v.y + R->m[8]*v.z;
    return out;
}

/* Apply small-angle rotation update: R = R * (I + [w]x * dt) */
static inline mat3_t mat3_update_rotation(const mat3_t *R, vec3_t w, float dt) {
    /* Skew-symmetric: [w]x = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]] */
    mat3_t dR;
    dR.m[0] = 1.0f;        dR.m[1] = -w.z * dt;  dR.m[2] = w.y * dt;
    dR.m[3] = w.z * dt;    dR.m[4] = 1.0f;        dR.m[5] = -w.x * dt;
    dR.m[6] = -w.y * dt;   dR.m[7] = w.x * dt;    dR.m[8] = 1.0f;

    /* Result = R * dR */
    mat3_t out;
    int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            float sum = 0.0f;
            for (k = 0; k < 3; k++) {
                sum += R->m[i*3+k] * dR.m[k*3+j];
            }
            out.m[i*3+j] = sum;
        }
    }
    return out;
}

/*
 * Tiled IMU integration: process samples in tiles of TILE_SAMPLES.
 * Accumulates delta_pos, delta_vel, and rotation.
 */
void imu_integrate_tiled(const vec3_t *accel, const vec3_t *gyro,
                         int n_samples, vec3_t *delta_pos,
                         vec3_t *delta_vel, mat3_t *rotation) {
    *delta_pos = (vec3_t){0.0f, 0.0f, 0.0f};
    *delta_vel = (vec3_t){0.0f, 0.0f, 0.0f};
    *rotation = mat3_identity();

    TILE_FOR(ts, 0, n_samples, TILE_SAMPLES) {
        int ts_end = TILE_END(ts, n_samples, TILE_SAMPLES);
        int t;
        for (t = ts; t < ts_end; t++) {
            /* Update rotation from gyroscope */
            *rotation = mat3_update_rotation(rotation, gyro[t], DT);

            /* Rotate acceleration to world frame and remove gravity */
            vec3_t a_world = mat3_mul_vec(rotation, accel[t]);
            a_world.z -= GRAVITY;

            /* Euler integration: v += a*dt, p += v*dt */
            delta_vel->x += a_world.x * DT;
            delta_vel->y += a_world.y * DT;
            delta_vel->z += a_world.z * DT;

            delta_pos->x += delta_vel->x * DT;
            delta_pos->y += delta_vel->y * DT;
            delta_pos->z += delta_vel->z * DT;
        }
    }
}

/* Reference (non-tiled) implementation */
void imu_integrate_ref(const vec3_t *accel, const vec3_t *gyro,
                       int n_samples, vec3_t *delta_pos,
                       vec3_t *delta_vel, mat3_t *rotation) {
    *delta_pos = (vec3_t){0.0f, 0.0f, 0.0f};
    *delta_vel = (vec3_t){0.0f, 0.0f, 0.0f};
    *rotation = mat3_identity();

    int t;
    for (t = 0; t < n_samples; t++) {
        *rotation = mat3_update_rotation(rotation, gyro[t], DT);
        vec3_t a_world = mat3_mul_vec(rotation, accel[t]);
        a_world.z -= GRAVITY;
        delta_vel->x += a_world.x * DT;
        delta_vel->y += a_world.y * DT;
        delta_vel->z += a_world.z * DT;
        delta_pos->x += delta_vel->x * DT;
        delta_pos->y += delta_vel->y * DT;
        delta_pos->z += delta_vel->z * DT;
    }
}

int main(void) {
    int n = WINDOW_SIZE;
    vec3_t *accel = (vec3_t *)malloc((size_t)n * sizeof(vec3_t));
    vec3_t *gyro  = (vec3_t *)malloc((size_t)n * sizeof(vec3_t));
    if (!accel || !gyro) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Generate synthetic IMU data: gentle circular motion */
    int i;
    for (i = 0; i < n; i++) {
        float t = (float)i / IMU_RATE;
        accel[i].x = 0.1f * sinf(2.0f * 3.14159f * t);
        accel[i].y = 0.1f * cosf(2.0f * 3.14159f * t);
        accel[i].z = GRAVITY + 0.05f * sinf(4.0f * 3.14159f * t);
        gyro[i].x = 0.01f * sinf(3.14159f * t);
        gyro[i].y = 0.02f * cosf(3.14159f * t);
        gyro[i].z = 0.05f;
    }

    vec3_t pos_t, vel_t, pos_r, vel_r;
    mat3_t rot_t, rot_r;

    imu_integrate_tiled(accel, gyro, n, &pos_t, &vel_t, &rot_t);
    imu_integrate_ref(accel, gyro, n, &pos_r, &vel_r, &rot_r);

    float pos_err = fabsf(pos_t.x - pos_r.x) + fabsf(pos_t.y - pos_r.y)
                    + fabsf(pos_t.z - pos_r.z);
    float vel_err = fabsf(vel_t.x - vel_r.x) + fabsf(vel_t.y - vel_r.y)
                    + fabsf(vel_t.z - vel_r.z);

    printf("imu_integration: pos_err = %e, vel_err = %e\n", pos_err, vel_err);
    printf("imu_integration: delta_pos = (%.6f, %.6f, %.6f)\n",
           pos_t.x, pos_t.y, pos_t.z);

    int pass = (pos_err < 1e-5f) && (vel_err < 1e-5f);
    printf("imu_integration: %s\n", pass ? "PASS" : "FAIL");

    free(accel);
    free(gyro);
    return pass ? 0 : 1;
}
