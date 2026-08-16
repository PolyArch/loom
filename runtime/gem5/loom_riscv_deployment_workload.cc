#include "runtime/gem5/loom_riscv_deployment_workload.hh"

#include "arch/riscv/regs/int.hh"
#include "base/loader/object_file.hh"
#include "base/logging.hh"
#include "cpu/thread_context.hh"
#include "mem/port_proxy.hh"
#include "sim/system.hh"

#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <utility>

namespace gem5 {
namespace {

constexpr std::uint64_t noActiveTarget =
    std::numeric_limits<std::uint64_t>::max();

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

} // namespace

LoomRiscvDeploymentWorkload::LoomRiscvDeploymentWorkload(const Params &params)
    : RiscvISA::BareMetal(params), hostCpuId(params.host_cpu_id),
      hostEntrySymbol(params.host_entry_symbol),
      hostDispatchAddress(params.host_dispatch_address),
      hostMemoryTableAddress(params.host_memory_table_address),
      hostMemoryTableEntries(params.host_memory_table_entries),
      hostResultAddress(params.host_result_address),
      hostResultSize(params.host_result_size),
      hostReturnAddress(params.host_return_address),
      stackBase(params.stack_base), stackStride(params.stack_stride),
      runtimeImagePaths(params.runtime_images),
      runtimeImageAddresses(params.runtime_image_addresses),
      memoryObservationPath(params.memory_observation_path),
      memoryObservationAddresses(params.memory_observation_addresses),
      memoryObservationSizes(params.memory_observation_sizes) {
  fatal_if(hostEntrySymbol.empty(), "Loom host entry symbol is empty");
  fatal_if((hostResultAddress == 0) != (hostResultSize == 0),
           "Loom host result address and size are not both present or absent");
  fatal_if(hostReturnAddress == 0, "Loom host return address is zero");
  fatal_if(stackStride == 0, "Loom per-CPU stack stride is zero");
  fatal_if(runtimeImagePaths.size() != runtimeImageAddresses.size(),
           "Loom runtime image paths and addresses differ in cardinality");
  fatal_if(memoryObservationPath.empty(),
           "Loom memory-observation destination is empty");
  fatal_if(memoryObservationAddresses.size() != memoryObservationSizes.size(),
           "Loom memory-observation arrays differ in cardinality");
  for (std::uint64_t size : memoryObservationSizes)
    fatal_if(size == 0, "Loom memory observation is empty");

  instructionImages.reserve(params.instruction_images.size());
  for (const std::string &path : params.instruction_images) {
    std::unique_ptr<loader::ObjectFile> image(loader::createObjectFile(path));
    fatal_if(!image, "Could not load InstructionCore image %s", path);
    fatal_if(image->getArch() != bootloader->getArch(),
             "InstructionCore image %s has a foreign architecture", path);
    instructionImages.push_back(std::move(image));
  }

  const std::size_t count = params.target_cpu_ids.size();
  fatal_if(params.target_image_ordinals.size() != count ||
               params.target_entry_symbols.size() != count ||
               params.target_bridge_addresses.size() != count ||
               params.target_launch_addresses.size() != count ||
               params.target_launch_sizes.size() != count,
           "Loom Thread Dispatch target arrays differ in cardinality");
  std::set<std::uint64_t> cpuIds;
  targets.reserve(count);
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    const std::uint64_t imageOrdinal = params.target_image_ordinals[ordinal];
    fatal_if(imageOrdinal >= instructionImages.size(),
             "Loom dispatch target %d names an absent image", ordinal);
    fatal_if(params.target_entry_symbols[ordinal].empty(),
             "Loom dispatch target %d has an empty entry symbol", ordinal);
    fatal_if(params.target_launch_sizes[ordinal] == 0,
             "Loom dispatch target %d has an empty Spatial launch", ordinal);
    fatal_if(!cpuIds.insert(params.target_cpu_ids[ordinal]).second,
             "Loom dispatch targets repeat CPU id %d",
             params.target_cpu_ids[ordinal]);
    targets.push_back({params.target_cpu_ids[ordinal], imageOrdinal,
                       params.target_entry_symbols[ordinal],
                       params.target_bridge_addresses[ordinal],
                       params.target_launch_addresses[ordinal],
                       params.target_launch_sizes[ordinal]});
  }
  activeTargets.assign(targets.size(), noActiveTarget);
}

LoomRiscvDeploymentWorkload::~LoomRiscvDeploymentWorkload() = default;

ThreadContext *
LoomRiscvDeploymentWorkload::contextForCpu(std::uint64_t cpuId) const {
  for (ThreadContext *context : contexts)
    if (context->cpuId() >= 0 &&
        static_cast<std::uint64_t>(context->cpuId()) == cpuId)
      return context;
  return nullptr;
}

Addr LoomRiscvDeploymentWorkload::symbolAddress(
    const loader::SymbolTable &symbols, const std::string &name) const {
  const auto found = symbols.find(name);
  fatal_if(found == symbols.end(), "Loom executable omits symbol %s", name);
  return found->address();
}

Addr LoomRiscvDeploymentWorkload::stackPointer(std::uint64_t cpuId) const {
  fatal_if(cpuId > (std::numeric_limits<Addr>::max() - stackBase) / stackStride,
           "Loom CPU id overflows the stack projection");
  return stackBase + (cpuId + 1) * stackStride;
}

void LoomRiscvDeploymentWorkload::initState() {
  Workload::initState();
  fatal_if(!bootloader->buildImage().write(system->physProxy),
           "Could not load the Loom HostCore image");
  for (const auto &image : instructionImages)
    fatal_if(!image->buildImage().write(system->physProxy),
             "Could not load a Loom InstructionCore image");
  for (std::size_t ordinal = 0; ordinal < runtimeImagePaths.size(); ++ordinal) {
    std::ifstream input(runtimeImagePaths[ordinal], std::ios::binary);
    fatal_if(!input, "Could not read Loom runtime image %s",
             runtimeImagePaths[ordinal]);
    const std::vector<char> raw{std::istreambuf_iterator<char>(input),
                                std::istreambuf_iterator<char>()};
    const std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
    fatal_if(bytes.empty(), "Loom runtime image %s is empty",
             runtimeImagePaths[ordinal]);
    system->physProxy.writeBlob(runtimeImageAddresses[ordinal], bytes.data(),
                                bytes.size());
  }

  contexts.assign(system->threads.begin(), system->threads.end());
  fatal_if(!contextForCpu(hostCpuId), "Loom HostCore CPU id %d is absent",
           hostCpuId);
  for (const Target &target : targets)
    fatal_if(!contextForCpu(target.cpuId),
             "Loom InstructionCore CPU id %d is absent", target.cpuId);

  for (ThreadContext *context : contexts) {
    context->getIsaPtr()->resetThread();
    context->setReg(RiscvISA::int_reg::Sp,
                    stackPointer(static_cast<std::uint64_t>(context->cpuId())));
    context->suspend();
  }
  ThreadContext *host = contextForCpu(hostCpuId);
  host->pcState(symbolAddress(bootloaderSymtab, hostEntrySymbol));
  host->setReg(RiscvISA::int_reg::A0, hostDispatchAddress);
  host->setReg(RiscvISA::int_reg::A1, targets.size());
  host->setReg(RiscvISA::int_reg::A2, hostMemoryTableAddress);
  host->setReg(RiscvISA::int_reg::A3, hostMemoryTableEntries);
  host->setReg(RiscvISA::int_reg::A4, hostResultAddress);
  host->setReg(RiscvISA::int_reg::A5, hostResultSize);
  host->setReg(RiscvISA::int_reg::Ra, hostReturnAddress);
  host->activate();
}

void LoomRiscvDeploymentWorkload::writeMemoryObservations() {
  std::vector<std::uint8_t> encoded{'L', 'G', 'M', '1'};
  appendU64(encoded, memoryObservationAddresses.size());
  for (std::size_t ordinal = 0; ordinal < memoryObservationAddresses.size();
       ++ordinal) {
    const Addr address = memoryObservationAddresses[ordinal];
    const std::uint64_t size = memoryObservationSizes[ordinal];
    fatal_if(size > std::numeric_limits<std::size_t>::max(),
             "Loom memory observation is too large");
    appendU64(encoded, address);
    appendU64(encoded, size);
    const std::size_t offset = encoded.size();
    encoded.resize(offset + static_cast<std::size_t>(size));
    system->physProxy.readBlob(address, encoded.data() + offset, size);
  }
  std::ofstream output(memoryObservationPath,
                       std::ios::binary | std::ios::trunc);
  fatal_if(!output, "Could not create Loom memory-observation result %s",
           memoryObservationPath);
  output.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
  fatal_if(!output, "Could not write Loom memory-observation result %s",
           memoryObservationPath);
}

const loader::SymbolTable &
LoomRiscvDeploymentWorkload::symtab(ThreadContext *context) {
  if (context && context->cpuId() >= 0) {
    const std::uint64_t cpuId = static_cast<std::uint64_t>(context->cpuId());
    for (const Target &target : targets)
      if (target.cpuId == cpuId)
        return instructionImages[target.imageOrdinal]->symtab();
  }
  return bootloaderSymtab;
}

bool LoomRiscvDeploymentWorkload::dispatch(std::uint64_t targetOrdinal,
                                           Addr completionAddress,
                                           Addr invocationAddress,
                                           std::uint64_t invocationSize) {
  if (targetOrdinal >= targets.size() ||
      activeTargets[targetOrdinal] != noActiveTarget ||
      ((invocationAddress == 0) != (invocationSize == 0)))
    return false;
  const Target &target = targets[targetOrdinal];
  ThreadContext *context = contextForCpu(target.cpuId);
  if (!context || context->status() == ThreadContext::Active)
    return false;
  context->getIsaPtr()->resetThread();
  context->pcState(symbolAddress(
      instructionImages[target.imageOrdinal]->symtab(), target.entrySymbol));
  context->setReg(RiscvISA::int_reg::Sp, stackPointer(target.cpuId));
  context->setReg(RiscvISA::int_reg::A0, target.bridgeAddress);
  context->setReg(RiscvISA::int_reg::A1, target.launchAddress);
  context->setReg(RiscvISA::int_reg::A2, target.launchSize);
  context->setReg(RiscvISA::int_reg::A3, completionAddress);
  context->setReg(RiscvISA::int_reg::A4, invocationAddress);
  context->setReg(RiscvISA::int_reg::A5, invocationSize);
  activeTargets[targetOrdinal] = target.cpuId;
  context->activate();
  return true;
}

LoomRiscvDeploymentWorkload::CompletionState
LoomRiscvDeploymentWorkload::complete(std::uint64_t targetOrdinal) {
  if (targetOrdinal >= targets.size() ||
      activeTargets[targetOrdinal] == noActiveTarget)
    return CompletionState::Invalid;
  ThreadContext *context = contextForCpu(activeTargets[targetOrdinal]);
  if (!context)
    return CompletionState::Invalid;
  if (context->status() == ThreadContext::Active)
    return CompletionState::Pending;
  if (context->status() != ThreadContext::Suspended)
    return CompletionState::Invalid;
  activeTargets[targetOrdinal] = noActiveTarget;
  return CompletionState::Complete;
}

} // namespace gem5
