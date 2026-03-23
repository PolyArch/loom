#ifndef TILE_UTILS_H
#define TILE_UTILS_H

/* Tiling helper macros for benchmark kernels */

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

/*
 * TILE_FOR: iterate over tiles.
 * var iterates from start to end in steps of tile_size.
 * The actual tile extent is MIN(tile_size, end - var).
 */
#define TILE_FOR(var, start, end, tile_size) \
    for (int var = (start); var < (end); var += (tile_size))

/*
 * TILE_END: compute the actual end of the current tile,
 * handling the boundary case where the last tile may be partial.
 */
#define TILE_END(var, end, tile_size) MIN((var) + (tile_size), (end))

/*
 * TILE_SIZE_AT: compute the actual tile size at position var.
 */
#define TILE_SIZE_AT(var, end, tile_size) (TILE_END(var, end, tile_size) - (var))

#endif /* TILE_UTILS_H */
