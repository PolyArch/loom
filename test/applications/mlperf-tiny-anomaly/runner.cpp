#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kModelByteCount = 276976;
constexpr std::size_t kDatasetByteCount = 102400;
constexpr std::size_t kSampleElementCount = 640;
constexpr std::size_t kSampleByteCount = kSampleElementCount * sizeof(float);
constexpr std::size_t kLayerCount = 10;
constexpr std::uint8_t kTensorTypeInt32 = 2;
constexpr std::uint8_t kTensorTypeInt8 = 9;
constexpr std::uint8_t kBuiltinFullyConnected = 9;
constexpr std::int32_t kBuiltinCodePlaceholder = 127;
constexpr std::uint8_t kFullyConnectedOptions = 8;
constexpr std::int8_t kActivationNone = 0;
constexpr std::int8_t kActivationRelu = 1;

bool checkedAdd(std::size_t lhs, std::size_t rhs, std::size_t &result) {
  if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMultiply(std::size_t lhs, std::size_t rhs, std::size_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

class FlatBufferReader final {
public:
  struct Table final {
    std::size_t offset = 0;
  };

  struct Vector final {
    std::size_t data = 0;
    std::uint32_t size = 0;
  };

  FlatBufferReader(const std::uint8_t *data, std::size_t size)
      : data_(data), size_(size) {}

  bool hasIdentifier(const char (&identifier)[5]) const {
    return size_ >= 8 && std::memcmp(data_ + 4, identifier, 4) == 0;
  }

  bool root(Table &result) const {
    std::uint32_t offset = 0;
    if (!readU32(0, offset) || offset >= size_)
      return false;
    result.offset = offset;
    return true;
  }

  bool tableU8(const Table &table, unsigned field, std::uint8_t fallback,
               std::uint8_t &result) const {
    std::size_t position = 0;
    if (!fieldPosition(table, field, position)) {
      result = fallback;
      return true;
    }
    return readU8(position, result);
  }

  bool tableI8(const Table &table, unsigned field, std::int8_t fallback,
               std::int8_t &result) const {
    std::uint8_t value = 0;
    if (!tableU8(table, field, static_cast<std::uint8_t>(fallback), value))
      return false;
    result = static_cast<std::int8_t>(value);
    return true;
  }

  bool tableBool(const Table &table, unsigned field, bool fallback,
                 bool &result) const {
    std::uint8_t value = 0;
    if (!tableU8(table, field, fallback ? 1 : 0, value) || value > 1)
      return false;
    result = value != 0;
    return true;
  }

  bool tableU32(const Table &table, unsigned field, std::uint32_t fallback,
                std::uint32_t &result) const {
    std::size_t position = 0;
    if (!fieldPosition(table, field, position)) {
      result = fallback;
      return true;
    }
    return readU32(position, result);
  }

  bool tableI32(const Table &table, unsigned field, std::int32_t fallback,
                std::int32_t &result) const {
    std::uint32_t value = 0;
    if (!tableU32(table, field, static_cast<std::uint32_t>(fallback), value))
      return false;
    std::memcpy(&result, &value, sizeof(result));
    return true;
  }

  bool table(const Table &owner, unsigned field, Table &result) const {
    std::size_t position = 0;
    if (!fieldPosition(owner, field, position))
      return false;
    return indirect(position, result.offset);
  }

  bool vector(const Table &owner, unsigned field, Vector &result) const {
    std::size_t position = 0;
    std::size_t vectorOffset = 0;
    if (!fieldPosition(owner, field, position) ||
        !indirect(position, vectorOffset) ||
        !readU32(vectorOffset, result.size))
      return false;
    return checkedAdd(vectorOffset, sizeof(std::uint32_t), result.data) &&
           result.data <= size_;
  }

  bool tableElement(const Vector &vector, std::uint32_t index,
                    Table &result) const {
    std::size_t increment = 0;
    std::size_t position = 0;
    return index < vector.size &&
           checkedMultiply(index, sizeof(std::uint32_t), increment) &&
           checkedAdd(vector.data, increment, position) &&
           indirect(position, result.offset);
  }

  bool vectorI32(const Vector &vector, std::uint32_t index,
                 std::int32_t &result) const {
    std::size_t increment = 0;
    std::size_t position = 0;
    std::uint32_t value = 0;
    if (index >= vector.size ||
        !checkedMultiply(index, sizeof(std::int32_t), increment) ||
        !checkedAdd(vector.data, increment, position) ||
        !readU32(position, value))
      return false;
    std::memcpy(&result, &value, sizeof(result));
    return true;
  }

  bool vectorI64(const Vector &vector, std::uint32_t index,
                 std::int64_t &result) const {
    std::size_t increment = 0;
    std::size_t position = 0;
    std::uint64_t value = 0;
    if (index >= vector.size ||
        !checkedMultiply(index, sizeof(std::int64_t), increment) ||
        !checkedAdd(vector.data, increment, position) ||
        !readU64(position, value))
      return false;
    std::memcpy(&result, &value, sizeof(result));
    return true;
  }

  bool vectorF32(const Vector &vector, std::uint32_t index,
                 float &result) const {
    std::size_t increment = 0;
    std::size_t position = 0;
    std::uint32_t value = 0;
    if (index >= vector.size ||
        !checkedMultiply(index, sizeof(float), increment) ||
        !checkedAdd(vector.data, increment, position) ||
        !readU32(position, value))
      return false;
    std::memcpy(&result, &value, sizeof(result));
    return true;
  }

  bool byteRange(const Vector &vector, const std::uint8_t *&data,
                 std::size_t &size) const {
    size = vector.size;
    if (vector.data > size_ || size > size_ - vector.data)
      return false;
    data = data_ + vector.data;
    return true;
  }

private:
  bool contains(std::size_t offset, std::size_t length) const {
    return offset <= size_ && length <= size_ - offset;
  }

  bool readU8(std::size_t offset, std::uint8_t &result) const {
    if (!contains(offset, 1))
      return false;
    result = data_[offset];
    return true;
  }

  bool readU16(std::size_t offset, std::uint16_t &result) const {
    if (!contains(offset, 2))
      return false;
    result = static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(data_[offset]) |
        static_cast<std::uint16_t>(data_[offset + 1]) << 8);
    return true;
  }

  bool readU32(std::size_t offset, std::uint32_t &result) const {
    if (!contains(offset, 4))
      return false;
    result = static_cast<std::uint32_t>(data_[offset]) |
             static_cast<std::uint32_t>(data_[offset + 1]) << 8 |
             static_cast<std::uint32_t>(data_[offset + 2]) << 16 |
             static_cast<std::uint32_t>(data_[offset + 3]) << 24;
    return true;
  }

  bool readU64(std::size_t offset, std::uint64_t &result) const {
    if (!contains(offset, 8))
      return false;
    result = 0;
    for (unsigned index = 0; index != 8; ++index)
      result |= static_cast<std::uint64_t>(data_[offset + index])
                << (index * 8);
    return true;
  }

  bool indirect(std::size_t position, std::size_t &result) const {
    std::uint32_t relative = 0;
    return readU32(position, relative) && relative != 0 &&
           checkedAdd(position, relative, result) && result < size_;
  }

  bool fieldPosition(const Table &table, unsigned field,
                     std::size_t &result) const {
    std::uint32_t distanceBits = 0;
    if (!readU32(table.offset, distanceBits))
      return false;
    std::int32_t distance = 0;
    std::memcpy(&distance, &distanceBits, sizeof(distance));
    if (distance == 0)
      return false;
    std::size_t vtable = 0;
    if (distance > 0) {
      if (static_cast<std::size_t>(distance) > table.offset)
        return false;
      vtable = table.offset - static_cast<std::size_t>(distance);
    } else {
      const std::size_t forward =
          static_cast<std::size_t>(-static_cast<std::int64_t>(distance));
      if (!checkedAdd(table.offset, forward, vtable) || vtable >= size_)
        return false;
    }
    std::uint16_t vtableSize = 0;
    if (!readU16(vtable, vtableSize))
      return false;
    std::size_t entryIncrement = 0;
    std::size_t entry = 0;
    if (!checkedMultiply(field, sizeof(std::uint16_t), entryIncrement) ||
        !checkedAdd(entryIncrement, 4, entryIncrement) ||
        entryIncrement > vtableSize || vtableSize - entryIncrement < 2 ||
        !checkedAdd(vtable, entryIncrement, entry))
      return false;
    std::uint16_t objectOffset = 0;
    if (!readU16(entry, objectOffset) || objectOffset == 0)
      return false;
    return checkedAdd(table.offset, objectOffset, result) && result < size_;
  }

  const std::uint8_t *data_;
  std::size_t size_;
};

struct Tensor final {
  std::vector<std::int32_t> shape;
  std::uint8_t type = 0;
  std::uint32_t buffer = 0;
  float scale = 0.0f;
  std::int64_t zeroPoint = 0;
  const std::uint8_t *data = nullptr;
  std::size_t dataSize = 0;
  std::size_t elements = 0;

  std::size_t elementCount() const { return elements; }
};

struct Layer final {
  std::uint32_t input = 0;
  std::uint32_t weights = 0;
  std::uint32_t bias = 0;
  std::uint32_t output = 0;
  bool relu = false;
};

class AnomalyModel final {
public:
  explicit AnomalyModel(const std::vector<std::uint8_t> &bytes)
      : bytes_(bytes), reader_(bytes.data(), bytes.size()) {}

  bool parse() {
    if (bytes_.size() != kModelByteCount)
      return fail("model byte count differs from the pinned model");
    if (!reader_.hasIdentifier("TFL3"))
      return fail("model does not have the TFLite identifier");
    FlatBufferReader::Table model;
    std::int32_t version = 0;
    if (!reader_.root(model) || !reader_.tableI32(model, 0, 0, version) ||
        version != 3)
      return fail("model root or schema version is invalid");

    FlatBufferReader::Vector operatorCodes;
    FlatBufferReader::Table operatorCode;
    std::uint8_t deprecatedBuiltinCode = 0;
    std::int32_t builtinCode = 0;
    std::int32_t operatorVersion = 0;
    if (!reader_.vector(model, 1, operatorCodes) || operatorCodes.size != 1 ||
        !reader_.tableElement(operatorCodes, 0, operatorCode) ||
        !reader_.tableU8(operatorCode, 0, 0, deprecatedBuiltinCode) ||
        !reader_.tableI32(operatorCode, 3, 0, builtinCode) ||
        !reader_.tableI32(operatorCode, 2, 1, operatorVersion) ||
        (builtinCode < kBuiltinCodePlaceholder
             ? deprecatedBuiltinCode
             : builtinCode) != kBuiltinFullyConnected ||
        operatorVersion != 4)
      return fail("model operator catalog is not the expected FC v4 catalog");

    FlatBufferReader::Vector buffers;
    if (!reader_.vector(model, 4, buffers) || buffers.size != 33)
      return fail("model buffer catalog has an unexpected shape");
    buffers_.reserve(buffers.size);
    for (std::uint32_t index = 0; index != buffers.size; ++index) {
      FlatBufferReader::Table buffer;
      FlatBufferReader::Vector data;
      const std::uint8_t *bytes = nullptr;
      std::size_t size = 0;
      if (!reader_.tableElement(buffers, index, buffer))
        return fail("model buffer table is malformed");
      if (reader_.vector(buffer, 0, data)) {
        if (!reader_.byteRange(data, bytes, size))
          return fail("model buffer data escapes the FlatBuffer");
      }
      buffers_.push_back({bytes, size});
    }

    FlatBufferReader::Vector subgraphs;
    FlatBufferReader::Table subgraph;
    if (!reader_.vector(model, 2, subgraphs) || subgraphs.size != 1 ||
        !reader_.tableElement(subgraphs, 0, subgraph))
      return fail("model does not have one valid subgraph");
    if (!parseTensors(subgraph) || !parseGraphEndpoints(subgraph) ||
        !parseLayers(subgraph))
      return false;
    return true;
  }

  bool invoke(const std::uint8_t *sample, std::size_t sampleSize,
              std::vector<std::int8_t> &result) const {
    if (sampleSize != kSampleByteCount || tensors_.empty())
      return fail("sample or parsed model is invalid");
    const Tensor &input = tensors_[inputTensor_];
    std::vector<std::int8_t> current(input.elementCount());
    for (std::size_t index = 0; index != current.size(); ++index) {
      const float value = readFloat(sample + index * sizeof(float));
      if (!std::isfinite(value))
        return fail("sample contains a non-finite input");
      current[index] = quantize(value, input);
    }

    for (const Layer &layer : layers_) {
      const Tensor &inputTensor = tensors_[layer.input];
      const Tensor &weights = tensors_[layer.weights];
      const Tensor &bias = tensors_[layer.bias];
      const Tensor &output = tensors_[layer.output];
      const std::size_t inputCount = inputTensor.elementCount();
      const std::size_t outputCount = output.elementCount();
      std::vector<std::int8_t> next(outputCount);
      for (std::size_t outputIndex = 0; outputIndex != outputCount;
           ++outputIndex) {
        std::int64_t accumulator = 0;
        for (std::size_t inputIndex = 0; inputIndex != inputCount;
             ++inputIndex) {
          const std::int32_t inputValue = current[inputIndex];
          const std::int32_t weightValue =
              signedByte(weights.data[outputIndex * inputCount + inputIndex]);
          accumulator += (inputValue - inputTensor.zeroPoint) *
                         (weightValue - weights.zeroPoint);
        }
        const std::int32_t biasValue =
            readI32(bias.data + outputIndex * sizeof(std::int32_t));
        const double real = static_cast<double>(accumulator) *
                                inputTensor.scale * weights.scale +
                            static_cast<double>(biasValue) * bias.scale;
        std::int64_t quantized =
            clampInt8(std::round(real / output.scale) +
                      static_cast<double>(output.zeroPoint));
        if (layer.relu)
          quantized = std::max<std::int64_t>(quantized, output.zeroPoint);
        next[outputIndex] = static_cast<std::int8_t>(quantized);
      }
      current = std::move(next);
    }
    result = std::move(current);
    return result.size() == tensors_[outputTensor_].elementCount();
  }

private:
  struct Buffer final {
    const std::uint8_t *data;
    std::size_t size;
  };

  bool parseTensors(const FlatBufferReader::Table &subgraph) {
    FlatBufferReader::Vector tensors;
    if (!reader_.vector(subgraph, 0, tensors) || tensors.size != 31)
      return fail("model tensor catalog has an unexpected shape");
    tensors_.reserve(tensors.size);
    for (std::uint32_t index = 0; index != tensors.size; ++index) {
      FlatBufferReader::Table table;
      FlatBufferReader::Vector shape;
      FlatBufferReader::Table quantization;
      FlatBufferReader::Vector scales;
      FlatBufferReader::Vector zeroPoints;
      Tensor tensor;
      if (!reader_.tableElement(tensors, index, table) ||
          !reader_.vector(table, 0, shape) || shape.size == 0 ||
          shape.size > 2 || !reader_.tableU8(table, 1, 0, tensor.type) ||
          !reader_.tableU32(table, 2, 0, tensor.buffer) ||
          tensor.buffer >= buffers_.size() ||
          !reader_.table(table, 4, quantization) ||
          !reader_.vector(quantization, 2, scales) || scales.size != 1 ||
          !reader_.vector(quantization, 3, zeroPoints) ||
          zeroPoints.size != 1 || !reader_.vectorF32(scales, 0, tensor.scale) ||
          !reader_.vectorI64(zeroPoints, 0, tensor.zeroPoint) ||
          !std::isfinite(tensor.scale) || tensor.scale <= 0.0f)
        return fail("model tensor metadata is malformed");
      tensor.shape.reserve(shape.size);
      tensor.elements = 1;
      for (std::uint32_t dimension = 0; dimension != shape.size; ++dimension) {
        std::int32_t extent = 0;
        if (!reader_.vectorI32(shape, dimension, extent) || extent <= 0 ||
            !checkedMultiply(tensor.elements, static_cast<std::size_t>(extent),
                             tensor.elements))
          return fail("model tensor has an invalid shape");
        tensor.shape.push_back(extent);
      }
      tensor.data = buffers_[tensor.buffer].data;
      tensor.dataSize = buffers_[tensor.buffer].size;
      tensors_.push_back(std::move(tensor));
    }
    return true;
  }

  bool parseGraphEndpoints(const FlatBufferReader::Table &subgraph) {
    FlatBufferReader::Vector inputs;
    FlatBufferReader::Vector outputs;
    std::int32_t input = -1;
    std::int32_t output = -1;
    if (!reader_.vector(subgraph, 1, inputs) || inputs.size != 1 ||
        !reader_.vectorI32(inputs, 0, input) || input != 0 ||
        !reader_.vector(subgraph, 2, outputs) || outputs.size != 1 ||
        !reader_.vectorI32(outputs, 0, output) || output != 30)
      return fail("model input or output selection is unexpected");
    inputTensor_ = static_cast<std::uint32_t>(input);
    outputTensor_ = static_cast<std::uint32_t>(output);
    const Tensor &inputTensor = tensors_[inputTensor_];
    const Tensor &outputTensor = tensors_[outputTensor_];
    if (inputTensor.type != kTensorTypeInt8 ||
        outputTensor.type != kTensorTypeInt8 ||
        inputTensor.elementCount() != kSampleElementCount ||
        outputTensor.elementCount() != kSampleElementCount)
      return fail("model endpoint tensor types or shapes are unexpected");
    return true;
  }

  bool parseLayers(const FlatBufferReader::Table &subgraph) {
    FlatBufferReader::Vector operators;
    if (!reader_.vector(subgraph, 3, operators) ||
        operators.size != kLayerCount)
      return fail("model does not have the expected FC layer count");
    layers_.reserve(operators.size);
    std::uint32_t previousOutput = inputTensor_;
    for (std::uint32_t index = 0; index != operators.size; ++index) {
      FlatBufferReader::Table operation;
      FlatBufferReader::Vector inputs;
      FlatBufferReader::Vector outputs;
      FlatBufferReader::Table options;
      std::uint32_t opcode = 0;
      std::uint8_t optionsType = 0;
      std::int8_t activation = 0;
      std::int8_t weightsFormat = 0;
      bool keepDimensions = false;
      bool asymmetricInputs = false;
      std::int32_t input = -1;
      std::int32_t weights = -1;
      std::int32_t bias = -1;
      std::int32_t output = -1;
      if (!reader_.tableElement(operators, index, operation) ||
          !reader_.tableU32(operation, 0, 0, opcode) || opcode != 0 ||
          !reader_.vector(operation, 1, inputs) || inputs.size != 3 ||
          !reader_.vectorI32(inputs, 0, input) ||
          !reader_.vectorI32(inputs, 1, weights) ||
          !reader_.vectorI32(inputs, 2, bias) ||
          !reader_.vector(operation, 2, outputs) || outputs.size != 1 ||
          !reader_.vectorI32(outputs, 0, output) || input < 0 || weights < 0 ||
          bias < 0 || output < 0 ||
          static_cast<std::size_t>(input) >= tensors_.size() ||
          static_cast<std::size_t>(weights) >= tensors_.size() ||
          static_cast<std::size_t>(bias) >= tensors_.size() ||
          static_cast<std::size_t>(output) >= tensors_.size() ||
          !reader_.tableU8(operation, 3, 0, optionsType) ||
          optionsType != kFullyConnectedOptions ||
          !reader_.table(operation, 4, options) ||
          !reader_.tableI8(options, 0, kActivationNone, activation) ||
          !reader_.tableI8(options, 1, 0, weightsFormat) ||
          !reader_.tableBool(options, 2, false, keepDimensions) ||
          !reader_.tableBool(options, 3, false, asymmetricInputs) ||
          weightsFormat != 0 || keepDimensions || asymmetricInputs)
        return fail("model FC operation metadata is malformed");
      const bool relu = index + 1 != operators.size;
      if (activation != (relu ? kActivationRelu : kActivationNone) ||
          static_cast<std::uint32_t>(input) != previousOutput)
        return fail("model FC activation or connectivity is unexpected");
      Layer layer{static_cast<std::uint32_t>(input),
                  static_cast<std::uint32_t>(weights),
                  static_cast<std::uint32_t>(bias),
                  static_cast<std::uint32_t>(output), relu};
      if (!validateLayer(layer))
        return false;
      layers_.push_back(layer);
      previousOutput = layer.output;
    }
    if (previousOutput != outputTensor_)
      return fail("model FC chain does not reach the selected output");
    return true;
  }

  bool validateLayer(const Layer &layer) const {
    const Tensor &input = tensors_[layer.input];
    const Tensor &weights = tensors_[layer.weights];
    const Tensor &bias = tensors_[layer.bias];
    const Tensor &output = tensors_[layer.output];
    const std::size_t inputCount = input.elementCount();
    const std::size_t outputCount = output.elementCount();
    std::size_t expectedWeights = 0;
    std::size_t expectedBiasBytes = 0;
    if (input.type != kTensorTypeInt8 || weights.type != kTensorTypeInt8 ||
        bias.type != kTensorTypeInt32 || output.type != kTensorTypeInt8 ||
        weights.shape.size() != 2 || bias.shape.size() != 1 ||
        weights.shape[0] != static_cast<std::int32_t>(outputCount) ||
        weights.shape[1] != static_cast<std::int32_t>(inputCount) ||
        bias.elementCount() != outputCount ||
        !checkedMultiply(inputCount, outputCount, expectedWeights) ||
        !checkedMultiply(outputCount, sizeof(std::int32_t),
                         expectedBiasBytes) ||
        weights.data == nullptr || weights.dataSize != expectedWeights ||
        bias.data == nullptr || bias.dataSize != expectedBiasBytes ||
        weights.zeroPoint != 0 || bias.zeroPoint != 0 ||
        input.zeroPoint < -128 || input.zeroPoint > 127 ||
        output.zeroPoint < -128 || output.zeroPoint > 127)
      return fail("model FC tensor contract is invalid");
    return true;
  }

  static std::int8_t signedByte(std::uint8_t value) {
    return value < 128 ? static_cast<std::int8_t>(value)
                       : static_cast<std::int8_t>(value - 256);
  }

  static std::int32_t readI32(const std::uint8_t *data) {
    const std::uint32_t value = static_cast<std::uint32_t>(data[0]) |
                                static_cast<std::uint32_t>(data[1]) << 8 |
                                static_cast<std::uint32_t>(data[2]) << 16 |
                                static_cast<std::uint32_t>(data[3]) << 24;
    std::int32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  static float readFloat(const std::uint8_t *data) {
    const std::uint32_t value = static_cast<std::uint32_t>(data[0]) |
                                static_cast<std::uint32_t>(data[1]) << 8 |
                                static_cast<std::uint32_t>(data[2]) << 16 |
                                static_cast<std::uint32_t>(data[3]) << 24;
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  static std::int8_t quantize(float value, const Tensor &tensor) {
    return static_cast<std::int8_t>(
        clampInt8(static_cast<double>(std::round(value / tensor.scale)) +
                  static_cast<double>(tensor.zeroPoint)));
  }

  static std::int64_t clampInt8(double value) {
    if (value <= -128.0)
      return -128;
    if (value >= 127.0)
      return 127;
    return static_cast<std::int64_t>(value);
  }

  static bool fail(const char *message) {
    std::fprintf(stderr, "mlperf-tiny-anomaly-runner: %s\n", message);
    return false;
  }

  const std::vector<std::uint8_t> &bytes_;
  FlatBufferReader reader_;
  std::vector<Buffer> buffers_;
  std::vector<Tensor> tensors_;
  std::vector<Layer> layers_;
  std::uint32_t inputTensor_ = 0;
  std::uint32_t outputTensor_ = 0;
};

bool readFile(const std::string &path, std::size_t expectedSize,
              std::vector<std::uint8_t> &result) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < 0 ||
      static_cast<std::uint64_t>(input.tellg()) != expectedSize) {
    std::fprintf(stderr,
                 "mlperf-tiny-anomaly-runner: '%s' has an unexpected size\n",
                 path.c_str());
    return false;
  }
  result.resize(expectedSize);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(result.data()),
             static_cast<std::streamsize>(result.size()));
  if (!input) {
    std::fprintf(stderr,
                 "mlperf-tiny-anomaly-runner: cannot read '%s' exactly\n",
                 path.c_str());
    return false;
  }
  return true;
}

bool parseCount(const char *text, std::size_t &result) {
  const char *end = text + std::strlen(text);
  const auto parsed = std::from_chars(text, end, result);
  return parsed.ec == std::errc() && parsed.ptr == end;
}

void printOutput(std::size_t ordinal, const std::vector<std::int8_t> &output) {
  std::printf("measured_sample=%zu output=", ordinal);
  for (std::int8_t value : output)
    std::printf("%02x",
                static_cast<unsigned>(static_cast<std::uint8_t>(value)));
  std::printf("\n");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: mlperf-tiny-anomaly-runner <model> <dataset> "
                         "<warmup-samples> <measured-samples>\n");
    return 2;
  }
  std::size_t warmupSamples = 0;
  std::size_t measuredSamples = 0;
  std::size_t totalSamples = 0;
  if (!parseCount(argv[3], warmupSamples) ||
      !parseCount(argv[4], measuredSamples) || measuredSamples == 0 ||
      !checkedAdd(warmupSamples, measuredSamples, totalSamples) ||
      totalSamples > kDatasetByteCount / kSampleByteCount) {
    std::fprintf(stderr,
                 "mlperf-tiny-anomaly-runner: sample profile is invalid\n");
    return 2;
  }

  std::vector<std::uint8_t> modelBytes;
  std::vector<std::uint8_t> datasetBytes;
  if (!readFile(argv[1], kModelByteCount, modelBytes) ||
      !readFile(argv[2], kDatasetByteCount, datasetBytes))
    return 3;
  AnomalyModel model(modelBytes);
  if (!model.parse())
    return 4;

  for (std::size_t sample = 0; sample != totalSamples; ++sample) {
    std::vector<std::int8_t> output;
    if (!model.invoke(datasetBytes.data() + sample * kSampleByteCount,
                      kSampleByteCount, output))
      return 5;
    if (sample >= warmupSamples)
      printOutput(sample - warmupSamples, output);
  }
  return 0;
}
