__attribute__((annotate("keep.annotation")))
int preserved_annotation(int value) { return value + 1; }

#pragma loom candidate
int projected_candidate(int value) { return value - 1; }
