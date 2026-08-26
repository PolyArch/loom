#line 100 "mapped.c"
#pragma loom candidate
#line 1 "mapped.c"
int hinted_remapped(int value) { return value + 1; }

#line 200 "nonmonotonic.c"
#pragma loom candidate
#line 100 "nonmonotonic.c"
int
#line 1 "nonmonotonic.c"
hinted_nonmonotonic(int value) {
#line 2 "nonmonotonic.c"
  return value + 2;
}
