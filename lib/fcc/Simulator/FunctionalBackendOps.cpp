#include "FunctionalBackendImpl.h"

namespace fcc {
namespace sim {

// ===----------------------------------------------------------------------===
// Node executor methods
// ===----------------------------------------------------------------------===

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeModuleOutput(IdIndex nodeId, const Node *node) {
  Action action;
  if (node->inputPorts.empty() || !hasInputToken(node->inputPorts.front()))
    return action;
  unsigned ordinal = swOutputToHwOrdinal[nodeId];
  if (ordinal == kInvalidOrdinal || ordinal >= outputCollectors.size()) {
    action.error = "module output is not mapped to a hardware boundary";
    return action;
  }
  Token token = popInputToken(node->inputPorts.front());
  outputCollectors[ordinal].push_back(token);
  action.progress = true;
  action.tokensIn = 1;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeJoin(IdIndex, const Node *node) {
  Action action;
  for (IdIndex portId : node->inputPorts) {
    if (!hasInputToken(portId))
      return action;
  }
  for (IdIndex portId : node->inputPorts)
    (void)popInputToken(portId);
  if (!node->outputPorts.empty())
    pushToken(node->outputPorts.front(), Token{});
  action.progress = true;
  action.tokensIn = static_cast<unsigned>(node->inputPorts.size());
  action.tokensOut = node->outputPorts.empty() ? 0u : 1u;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeForkLike(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.empty() || !hasInputToken(node->inputPorts.front()))
    return action;
  Token token = popInputToken(node->inputPorts.front());
  for (IdIndex outPort : node->outputPorts)
    pushToken(outPort, token);
  action.progress = true;
  action.tokensIn = 1;
  action.tokensOut = static_cast<unsigned>(node->outputPorts.size());
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeSink(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.empty() || !hasInputToken(node->inputPorts.front()))
    return action;
  (void)popInputToken(node->inputPorts.front());
  action.progress = true;
  action.tokensIn = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeSource(IdIndex nodeId, const Node *node) {
  Action action;
  NodeRuntime &runtime = nodeRuntime[nodeId];
  if (runtime.emittedOnce || node->outputPorts.empty())
    return action;
  pushToken(node->outputPorts.front(), Token{});
  runtime.emittedOnce = true;
  action.progress = true;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeMerge(IdIndex, const Node *node, bool withIndex) {
  Action action;
  for (unsigned i = 0; i < node->inputPorts.size(); ++i) {
    IdIndex inPort = node->inputPorts[i];
    if (!hasInputToken(inPort))
      continue;
    Token token = popInputToken(inPort);
    if (!node->outputPorts.empty())
      pushToken(node->outputPorts.front(), token);
    if (withIndex && node->outputPorts.size() > 1) {
      Token indexToken;
      indexToken.data = i;
      pushToken(node->outputPorts[1], indexToken);
    }
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut =
        1 + static_cast<unsigned>(withIndex && node->outputPorts.size() > 1);
    return action;
  }
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeHandshakeConstant(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.empty() || node->outputPorts.empty() ||
      !hasInputToken(node->inputPorts.front()))
    return action;
  (void)popInputToken(node->inputPorts.front());
  Token token;
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
          getNodeAttr(node, "value")))
    token.data = intAttr.getValue().getZExtValue();
  pushToken(node->outputPorts.front(), token);
  action.progress = true;
  action.tokensIn = 1;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeArithConstant(IdIndex nodeId, const Node *node) {
  Action action;
  NodeRuntime &runtime = nodeRuntime[nodeId];
  if (runtime.emittedOnce || node->outputPorts.empty())
    return action;
  Token token;
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
          getNodeAttr(node, "value")))
    token.data = intAttr.getValue().getZExtValue();
  pushToken(node->outputPorts.front(), token);
  runtime.emittedOnce = true;
  action.progress = true;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeStream(IdIndex nodeId, const Node *node) {
  Action action;
  NodeRuntime &runtime = nodeRuntime[nodeId];
  if (node->inputPorts.size() < 3 || node->outputPorts.size() < 2)
    return action;

  if (!runtime.stream.active) {
    for (IdIndex portId : node->inputPorts) {
      if (!hasInputToken(portId))
        return action;
    }
    runtime.stream.nextIdx = popInputToken(node->inputPorts[0]).data;
    runtime.stream.step = popInputToken(node->inputPorts[1]).data;
    runtime.stream.bound = popInputToken(node->inputPorts[2]).data;
    runtime.stream.active = true;
    action.progress = true;
    action.tokensIn = 3;
    return action;
  }

  bool willContinue = evaluateStreamCond(
      runtime.stream.nextIdx, runtime.stream.bound, runtime.stream.contCond);
  Token idxToken;
  idxToken.data = runtime.stream.nextIdx;
  Token condToken;
  condToken.data = willContinue ? 1 : 0;
  pushToken(node->outputPorts[0], idxToken);
  pushToken(node->outputPorts[1], condToken);
  if (willContinue)
    runtime.stream.nextIdx = applyStreamStep(
        runtime.stream.nextIdx, runtime.stream.step, runtime.stream.stepOp);
  else
    runtime.stream.active = false;
  action.progress = true;
  action.tokensOut = 2;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeGate(IdIndex nodeId, const Node *node) {
  Action action;
  NodeRuntime &runtime = nodeRuntime[nodeId];
  if (node->inputPorts.size() < 2 || node->outputPorts.size() < 2)
    return action;
  if (!hasInputToken(node->inputPorts[0]) ||
      !hasInputToken(node->inputPorts[1]))
    return action;

  Token value = popInputToken(node->inputPorts[0]);
  Token cond = popInputToken(node->inputPorts[1]);
  bool condBit = (cond.data & 1) != 0;
  action.progress = true;
  action.tokensIn = 2;

  if (runtime.gate.phase == GateState::NeedHead) {
    if (condBit) {
      pushToken(node->outputPorts[0], value);
      action.tokensOut = 1;
      runtime.gate.phase = GateState::NeedNext;
    }
    return action;
  }

  if (condBit) {
    pushToken(node->outputPorts[0], value);
    Token afterCond;
    afterCond.data = 1;
    pushToken(node->outputPorts[1], afterCond);
    action.tokensOut = 2;
  } else {
    Token afterCond;
    afterCond.data = 0;
    pushToken(node->outputPorts[1], afterCond);
    action.tokensOut = 1;
    runtime.gate.phase = GateState::NeedHead;
  }
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeCarry(IdIndex nodeId, const Node *node) {
  Action action;
  NodeRuntime &runtime = nodeRuntime[nodeId];
  if (node->inputPorts.size() < 3 || node->outputPorts.empty())
    return action;

  switch (runtime.carry.phase) {
  case CarryState::NeedInit:
    if (!hasInputToken(node->inputPorts[1]))
      return action;
    runtime.carry.initValue = popInputToken(node->inputPorts[1]).data;
    pushToken(node->outputPorts.front(), Token{runtime.carry.initValue});
    runtime.carry.phase = CarryState::NeedCond;
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  case CarryState::NeedCond: {
    if (!hasInputToken(node->inputPorts[0]))
      return action;
    bool cond = (popInputToken(node->inputPorts[0]).data & 1) != 0;
    runtime.carry.phase = cond ? CarryState::NeedLoop : CarryState::NeedInit;
    action.progress = true;
    action.tokensIn = 1;
    return action;
  }
  case CarryState::NeedLoop:
    if (!hasInputToken(node->inputPorts[2]))
      return action;
    pushToken(node->outputPorts.front(), popInputToken(node->inputPorts[2]));
    runtime.carry.phase = CarryState::NeedCond;
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  }
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeInvariant(IdIndex nodeId, const Node *node) {
  Action action;
  NodeRuntime &runtime = nodeRuntime[nodeId];
  if (node->inputPorts.size() < 2 || node->outputPorts.empty())
    return action;

  switch (runtime.invariant.phase) {
  case InvariantState::NeedInit:
    if (!hasInputToken(node->inputPorts[1]))
      return action;
    runtime.invariant.storedValue = popInputToken(node->inputPorts[1]).data;
    pushToken(node->outputPorts.front(),
              Token{runtime.invariant.storedValue});
    runtime.invariant.phase = InvariantState::NeedCond;
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  case InvariantState::NeedCond:
    if (!hasInputToken(node->inputPorts[0]))
      return action;
    if ((popInputToken(node->inputPorts[0]).data & 1) != 0) {
      pushToken(node->outputPorts.front(),
                Token{runtime.invariant.storedValue});
      action.tokensOut = 1;
    } else {
      runtime.invariant.phase = InvariantState::NeedInit;
    }
    action.progress = true;
    action.tokensIn = 1;
    return action;
  }
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeCondBr(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.size() < 2 || node->outputPorts.size() < 2)
    return action;
  if (!hasInputToken(node->inputPorts[0]) ||
      !hasInputToken(node->inputPorts[1]))
    return action;
  bool cond = (popInputToken(node->inputPorts[0]).data & 1) != 0;
  Token data = popInputToken(node->inputPorts[1]);
  pushToken(node->outputPorts[cond ? 0 : 1], data);
  action.progress = true;
  action.tokensIn = 2;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeSelect(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.size() < 3 || node->outputPorts.empty())
    return action;
  for (IdIndex portId : node->inputPorts) {
    if (!hasInputToken(portId))
      return action;
  }
  bool cond = (popInputToken(node->inputPorts[0]).data & 1) != 0;
  Token trueToken = popInputToken(node->inputPorts[1]);
  Token falseToken = popInputToken(node->inputPorts[2]);
  pushToken(node->outputPorts.front(), cond ? trueToken : falseToken);
  action.progress = true;
  action.tokensIn = 3;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeHandshakeMux(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.size() < 2 || node->outputPorts.empty())
    return action;
  if (!hasInputToken(node->inputPorts[0]))
    return action;
  unsigned select = getSelectIndex(
      edgeQueues[inputPortEdge[node->inputPorts[0]]].front().data,
      dfg.getPort(node->inputPorts[0])->type);
  if (select + 1 >= node->inputPorts.size()) {
    action.error = "mux select out of range";
    return action;
  }
  IdIndex selectedPort = node->inputPorts[select + 1];
  if (!hasInputToken(selectedPort))
    return action;
  (void)popInputToken(node->inputPorts[0]);
  Token selected = popInputToken(selectedPort);
  pushToken(node->outputPorts.front(), selected);
  action.progress = true;
  action.tokensIn = 2;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeFabricMux(IdIndex, const Node *node) {
  Action action;
  unsigned numInputs = static_cast<unsigned>(node->inputPorts.size());
  unsigned numOutputs = static_cast<unsigned>(node->outputPorts.size());
  if (numInputs == 0 || numOutputs == 0)
    return action;

  bool disconnect = getNodeAttrBool(node, "disconnect");
  bool discard = getNodeAttrBool(node, "discard");
  unsigned sel = 0;
  if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
          getNodeAttr(node, "sel")))
    sel = static_cast<unsigned>(intAttr.getInt());

  if (numInputs == 1 && numOutputs == 1) {
    if (disconnect || discard) {
      action.error = "1:1 mux cannot set discard or disconnect";
      return action;
    }
    if (!hasInputToken(node->inputPorts.front()))
      return action;
    Token token = popInputToken(node->inputPorts.front());
    pushToken(node->outputPorts.front(), token);
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
    return action;
  }

  if (disconnect)
    return action;

  if (numOutputs == 1) {
    if (sel >= numInputs) {
      action.error = "mux select out of range";
      return action;
    }

    unsigned discardedInputs = 0;
    if (discard) {
      for (unsigned i = 0; i < numInputs; ++i) {
        if (i == sel || !hasInputToken(node->inputPorts[i]))
          continue;
        (void)popInputToken(node->inputPorts[i]);
        ++discardedInputs;
      }
    }

    if (!hasInputToken(node->inputPorts[sel])) {
      if (discardedInputs > 0) {
        action.progress = true;
        action.tokensIn = discardedInputs;
      }
      return action;
    }

    Token token = popInputToken(node->inputPorts[sel]);
    pushToken(node->outputPorts.front(), token);
    action.progress = true;
    action.tokensIn = discardedInputs + 1;
    action.tokensOut = 1;
    return action;
  }

  if (numInputs == 1) {
    if (sel >= numOutputs) {
      action.error = "mux select out of range";
      return action;
    }
    if (!hasInputToken(node->inputPorts.front()))
      return action;
    Token token = popInputToken(node->inputPorts.front());
    action.progress = true;
    action.tokensIn = 1;
    if (!discard) {
      pushToken(node->outputPorts[sel], token);
      action.tokensOut = 1;
    }
    return action;
  }

  action.error = "mux must be either M:1 or 1:M";
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeLoad(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.size() < 3 || node->outputPorts.size() < 2)
    return action;
  unsigned numAddr = static_cast<unsigned>(node->outputPorts.size() - 1);
  unsigned dataInputIdx = numAddr;
  unsigned ctrlInputIdx = numAddr + 1;
  if (ctrlInputIdx >= node->inputPorts.size())
    return action;

  bool canIssueAddr = hasInputToken(node->inputPorts[ctrlInputIdx]);
  for (unsigned i = 0; i < numAddr && canIssueAddr; ++i)
    canIssueAddr = hasInputToken(node->inputPorts[i]);

  if (canIssueAddr) {
    std::vector<Token> addrs;
    addrs.reserve(numAddr);
    for (unsigned i = 0; i < numAddr; ++i)
      addrs.push_back(popInputToken(node->inputPorts[i]));
    (void)popInputToken(node->inputPorts[ctrlInputIdx]);
    for (unsigned i = 0; i < numAddr; ++i)
      pushToken(node->outputPorts[i + 1], addrs[i]);
    action.progress = true;
    action.tokensIn = numAddr + 1;
    action.tokensOut = numAddr;
    return action;
  }

  if (dataInputIdx < node->inputPorts.size() &&
      hasInputToken(node->inputPorts[dataInputIdx])) {
    pushToken(node->outputPorts.front(),
              popInputToken(node->inputPorts[dataInputIdx]));
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 1;
  }
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeStore(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.size() < 3 || node->outputPorts.size() < 2)
    return action;
  unsigned numAddr = static_cast<unsigned>(node->outputPorts.size() - 1);
  unsigned dataInputIdx = numAddr;
  unsigned ctrlInputIdx = numAddr + 1;
  if (ctrlInputIdx >= node->inputPorts.size())
    return action;

  bool ready = hasInputToken(node->inputPorts[dataInputIdx]) &&
               hasInputToken(node->inputPorts[ctrlInputIdx]);
  for (unsigned i = 0; i < numAddr && ready; ++i)
    ready = hasInputToken(node->inputPorts[i]);
  if (!ready)
    return action;

  std::vector<Token> addrs;
  addrs.reserve(numAddr);
  for (unsigned i = 0; i < numAddr; ++i)
    addrs.push_back(popInputToken(node->inputPorts[i]));
  Token data = popInputToken(node->inputPorts[dataInputIdx]);
  (void)popInputToken(node->inputPorts[ctrlInputIdx]);
  pushToken(node->outputPorts.front(), data);
  for (unsigned i = 0; i < numAddr; ++i)
    pushToken(node->outputPorts[i + 1], addrs[i]);
  action.progress = true;
  action.tokensIn = numAddr + 2;
  action.tokensOut = numAddr + 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeMemory(IdIndex nodeId, const Node *node) {
  Action action;
  auto regionId = getMemoryRegionId(nodeId);
  if (!regionId) {
    action.error = "memory node is missing a bound simulation region";
    return action;
  }

  unsigned ldCount = static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(node, "ldCount", 0)));
  unsigned stCount = static_cast<unsigned>(
      std::max<int64_t>(0, getNodeAttrInt(node, "stCount", 0)));
  bool hasMemrefInput =
      !node->inputPorts.empty() &&
      mlir::isa<mlir::MemRefType>(dfg.getPort(node->inputPorts.front())->type);
  unsigned inputBase = hasMemrefInput ? 1u : 0u;

  unsigned ldAddrPort = inputBase;
  unsigned stAddrPort = inputBase + (ldCount > 0 ? 1u : 0u);
  unsigned stDataPort = stAddrPort + (stCount > 0 ? 1u : 0u);

  unsigned ldDataPort = 0;
  unsigned ldDonePort = ldDataPort + (ldCount > 0 ? 1u : 0u);
  unsigned stDonePort = ldDonePort + (ldCount > 0 ? 1u : 0u);

  if (ldCount > 0 && ldAddrPort < node->inputPorts.size() &&
      hasInputToken(node->inputPorts[ldAddrPort])) {
    Token addrToken = popInputToken(node->inputPorts[ldAddrPort]);
    uint64_t loadedValue = 0;
    if (!readMemory(*regionId, addrToken.data, loadedValue, action.error))
      return action;
    if (ldDataPort >= node->outputPorts.size() ||
        ldDonePort >= node->outputPorts.size()) {
      action.error = "memory load family index out of range";
      return action;
    }
    Token dataToken = addrToken;
    dataToken.data = loadedValue;
    pushToken(node->outputPorts[ldDataPort], dataToken);
    Token doneToken = addrToken;
    doneToken.data = 0;
    pushToken(node->outputPorts[ldDonePort], doneToken);
    action.progress = true;
    action.tokensIn = 1;
    action.tokensOut = 2;
    return action;
  }

  if (stCount > 0 && stAddrPort < node->inputPorts.size() &&
      stDataPort < node->inputPorts.size() &&
      hasInputToken(node->inputPorts[stAddrPort]) &&
      hasInputToken(node->inputPorts[stDataPort])) {
    Token addrToken = popInputToken(node->inputPorts[stAddrPort]);
    Token dataToken = popInputToken(node->inputPorts[stDataPort]);
    if (addrToken.hasTag && dataToken.hasTag && addrToken.tag != dataToken.tag) {
      action.error = "memory store address/data tag mismatch";
      return action;
    }
    if (!writeMemory(*regionId, addrToken.data, dataToken.data, action.error))
      return action;
    if (stDonePort >= node->outputPorts.size()) {
      action.error = "memory store done family index out of range";
      return action;
    }
    Token doneToken = addrToken.hasTag ? addrToken : dataToken;
    doneToken.data = 0;
    pushToken(node->outputPorts[stDonePort], doneToken);
    action.progress = true;
    action.tokensIn = 2;
    action.tokensOut = 1;
    return action;
  }

  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeIndexCast(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.empty() || node->outputPorts.empty() ||
      !hasInputToken(node->inputPorts.front()))
    return action;
  Token input = popInputToken(node->inputPorts.front());
  const Port *srcPort = dfg.getPort(node->inputPorts.front());
  const Port *dstPort = dfg.getPort(node->outputPorts.front());
  unsigned srcWidth = srcPort ? getTypeBitWidth(srcPort->type) : 64;
  uint64_t value = input.data;
  if (dstPort && dstPort->type.isIndex() &&
      srcWidth < fcc::fabric::getConfiguredIndexBitWidth())
    value = static_cast<uint64_t>(signExtendToI64(input.data, srcWidth));
  Token output = input;
  output.data = dstPort ? coerceValueToType(value, dstPort->type) : value;
  pushToken(node->outputPorts.front(), output);
  action.progress = true;
  action.tokensIn = 1;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeCmpi(IdIndex, const Node *node) {
  Action action;
  if (node->inputPorts.size() < 2 || node->outputPorts.empty())
    return action;
  if (!hasInputToken(node->inputPorts[0]) ||
      !hasInputToken(node->inputPorts[1]))
    return action;
  auto predicate = getCmpPredicate(node);
  if (!predicate) {
    action.error = "cmpi is missing predicate attribute";
    return action;
  }
  Token lhs = popInputToken(node->inputPorts[0]);
  Token rhs = popInputToken(node->inputPorts[1]);
  const Port *lhsPort = dfg.getPort(node->inputPorts[0]);
  unsigned width = lhsPort ? getTypeBitWidth(lhsPort->type) : 64;
  Token resultToken;
  resultToken.data =
      evaluateCmpPredicate(*predicate, lhs.data, rhs.data, width) ? 1 : 0;
  pushToken(node->outputPorts.front(), resultToken);
  action.progress = true;
  action.tokensIn = 2;
  action.tokensOut = 1;
  return action;
}

FunctionalSimulationBackend::Impl::Action
FunctionalSimulationBackend::Impl::executeBinaryArith(IdIndex, const Node *node,
                                      llvm::StringRef opName) {
  Action action;
  if (node->inputPorts.size() < 2 || node->outputPorts.empty())
    return action;
  if (!hasInputToken(node->inputPorts[0]) ||
      !hasInputToken(node->inputPorts[1]))
    return action;
  Token lhs = popInputToken(node->inputPorts[0]);
  Token rhs = popInputToken(node->inputPorts[1]);
  const Port *dstPort = dfg.getPort(node->outputPorts.front());
  const Port *lhsPort = dfg.getPort(node->inputPorts[0]);
  unsigned width = lhsPort ? getTypeBitWidth(lhsPort->type) : 64;
  uint64_t resultValue = 0;

  if (opMatches(opName, "addi"))
    resultValue = lhs.data + rhs.data;
  else if (opMatches(opName, "subi"))
    resultValue = lhs.data - rhs.data;
  else if (opMatches(opName, "muli"))
    resultValue = lhs.data * rhs.data;
  else if (opMatches(opName, "divsi"))
    resultValue =
        rhs.data == 0
            ? 0
            : static_cast<uint64_t>(signExtendToI64(lhs.data, width) /
                                    signExtendToI64(rhs.data, width));
  else if (opMatches(opName, "divui"))
    resultValue = rhs.data == 0 ? 0 : lhs.data / rhs.data;
  else if (opMatches(opName, "remsi"))
    resultValue =
        rhs.data == 0
            ? 0
            : static_cast<uint64_t>(signExtendToI64(lhs.data, width) %
                                    signExtendToI64(rhs.data, width));
  else if (opMatches(opName, "remui"))
    resultValue = rhs.data == 0 ? 0 : lhs.data % rhs.data;
  else if (opMatches(opName, "andi"))
    resultValue = lhs.data & rhs.data;
  else if (opMatches(opName, "ori"))
    resultValue = lhs.data | rhs.data;
  else if (opMatches(opName, "xori"))
    resultValue = lhs.data ^ rhs.data;
  else if (opMatches(opName, "shli"))
    resultValue = lhs.data << rhs.data;
  else if (opMatches(opName, "shrsi"))
    resultValue =
        static_cast<uint64_t>(signExtendToI64(lhs.data, width) >> rhs.data);
  else if (opMatches(opName, "shrui"))
    resultValue = lhs.data >> rhs.data;
  else
    action.error = "unsupported binary arithmetic op";

  if (!action.error.empty())
    return action;

  Token output;
  output.data = dstPort ? coerceValueToType(resultValue, dstPort->type)
                        : resultValue;
  pushToken(node->outputPorts.front(), output);
  action.progress = true;
  action.tokensIn = 2;
  action.tokensOut = 1;
  return action;
}

} // namespace sim
} // namespace fcc
