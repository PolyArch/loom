typedef long time_t;
struct timespec {
  time_t tv_sec;
  long tv_nsec;
};
int clock_gettime(int, struct timespec *);
time_t time(time_t *);
#define CLOCK_REALTIME 0
