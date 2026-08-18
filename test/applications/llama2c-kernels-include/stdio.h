#include <stddef.h>
typedef struct FILE FILE;
extern FILE *stderr;
extern FILE *stdin;
extern FILE *stdout;
FILE *fopen(const char *, const char *);
size_t fread(void *, size_t, size_t, FILE *);
int fseek(FILE *, long, int);
long ftell(FILE *);
int fclose(FILE *);
int fprintf(FILE *, const char *, ...);
int printf(const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int sprintf(char *, const char *, ...);
int sscanf(const char *, const char *, ...);
int fflush(FILE *);
char *fgets(char *, int, FILE *);
#define SEEK_END 2
