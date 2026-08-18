inline double round(double value) {
  return value >= 0.0
             ? static_cast<double>(static_cast<long long>(value + 0.5))
             : static_cast<double>(static_cast<long long>(value - 0.5));
}
inline float round(float value) {
  return value >= 0.0f
             ? static_cast<float>(static_cast<long long>(value + 0.5f))
             : static_cast<float>(static_cast<long long>(value - 0.5f));
}
