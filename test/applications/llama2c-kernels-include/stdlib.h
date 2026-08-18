#include <stddef.h>
void *malloc(size_t);
void *calloc(size_t, size_t);
void free(void *);
void exit(int);
int abs(int);
int atoi(const char *);
double atof(const char *);
long strtol(const char *, char **, int);
void *bsearch(const void *, const void *, size_t, size_t,
              int (*)(const void *, const void *));
void qsort(void *, size_t, size_t, int (*)(const void *, const void *));
#define EXIT_FAILURE 1
