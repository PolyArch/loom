#include "StructuredProgramNativeExecutionInternal.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>

namespace loom::sim::native_detail {
namespace {

llvm::Error orderedChannelError(
    runtime::OrderedChannelABIError::Kind kind, llvm::StringRef message) {
  return llvm::make_error<runtime::OrderedChannelABIError>(kind,
                                                           message.str());
}

llvm::Error cancelUnjoinedLogicalChannels(NativeExecutionContext &capture) {
  llvm::Error result = llvm::Error::success();
  for (NativeExecutionContext::LogicalChannel &channel :
       capture.logicalChannels)
    if (channel.abi && !channel.abi->generationJoined())
      result =
          llvm::joinErrors(std::move(result), channel.abi->cancelGeneration());
  return result;
}

} // namespace

llvm::Error failLogicalChannelExecution(NativeExecutionContext &capture,
                                        llvm::Error failure) {
  return llvm::joinErrors(std::move(failure),
                          cancelUnjoinedLogicalChannels(capture));
}

llvm::Error finishLogicalChannelExecution(NativeExecutionContext &capture) {
  for (NativeExecutionContext::LogicalChannel &channel :
       capture.logicalChannels) {
    if (!channel.abi || channel.abi->generationJoined())
      continue;
    if (!channel.abi->hasStaticRateContract())
      return failLogicalChannelExecution(
          capture,
          orderedChannelError(
              runtime::OrderedChannelABIError::Kind::InvalidLifecycle,
              "logical channel did not open its finite generation"));
    if (channel.abi->nextSendSequence() < channel.producerMessageCount) {
      if (llvm::Error error = channel.abi->finishProducer())
        return failLogicalChannelExecution(capture, std::move(error));
    }
    for (std::uint32_t receiver = 0; receiver != channel.abi->consumerCount();
         ++receiver) {
      auto finished = channel.abi->consumerFinished(receiver);
      if (!finished)
        return failLogicalChannelExecution(capture, finished.takeError());
      if (*finished)
        continue;
      if (!channel.consumerMessageCounts[receiver])
        return failLogicalChannelExecution(
            capture,
            orderedChannelError(
                runtime::OrderedChannelABIError::Kind::InvalidConfiguration,
                "logical channel endpoint has no finite rate"));
      auto observed = channel.abi->nextReceiveSequence(receiver);
      if (!observed)
        return failLogicalChannelExecution(capture, observed.takeError());
      if (*observed < *channel.consumerMessageCounts[receiver]) {
        if (llvm::Error error = channel.abi->finishConsumer(receiver))
          return failLogicalChannelExecution(capture, std::move(error));
      }
      if (*observed > *channel.consumerMessageCounts[receiver])
        return failLogicalChannelExecution(
            capture,
            orderedChannelError(
                runtime::OrderedChannelABIError::Kind::StaticRateExceeded,
                "logical channel endpoint exceeded its finite rate"));
      auto terminal = channel.abi->receive(receiver);
      if (!terminal)
        return failLogicalChannelExecution(capture, terminal.takeError());
      if (terminal->kind != runtime::OrderedChannelReceiveKind::EndOfGeneration)
        return failLogicalChannelExecution(
            capture,
            orderedChannelError(
                runtime::OrderedChannelABIError::Kind::InvalidLifecycle,
                "logical channel endpoint did not reach its generation "
                "terminal"));
      if (llvm::Error error = channel.abi->finishConsumer(receiver))
        return failLogicalChannelExecution(capture, std::move(error));
    }
    if (llvm::Error error = channel.abi->joinGeneration())
      return failLogicalChannelExecution(capture, std::move(error));
  }
  return llvm::Error::success();
}

} // namespace loom::sim::native_detail
