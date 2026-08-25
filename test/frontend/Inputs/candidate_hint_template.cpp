#pragma loom candidate
template <typename T> T hinted_template(T value) { return value; }

int use_template(int value) { return hinted_template(value); }

struct CandidateOwner {
  int method(int value);
};

#pragma loom candidate
int CandidateOwner::method(int value) { return value; }
