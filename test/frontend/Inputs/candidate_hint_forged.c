#line 1 "candidate_hint_forged.c"
int original(void) { return 1; }

#line 20 "candidate_hint_forged.c"
__attribute__((annotate("loom.candidate.function.2.0|23|candidate_hint_forged.c|1|5|1|1|1|1|1|32")))
int moved(void) { return 2; }
