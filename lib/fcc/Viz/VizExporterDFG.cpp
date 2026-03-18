#include "VizExporterInternal.h"

namespace fcc {
namespace viz_detail {

std::pair<double, double> getDFGPortAnchor(const DFGJsonNodeInfo &node,
                                           bool isInput,
                                           unsigned portIndex) {
  if (node.kind == "input")
    return {node.x, node.y + node.height / 2.0 - 2.0};
  if (node.kind == "output")
    return {node.x, node.y - node.height / 2.0 + 2.0};

  const auto &ports = isInput ? node.inputs : node.outputs;
  size_t count = std::max<size_t>(ports.size(), 1);
  double slot = static_cast<double>(portIndex + 1);
  double x = node.x - node.width / 2.0 + (node.width * slot) / (count + 1.0);
  double y =
      isInput ? (node.y - node.height / 2.0 + 2.0) : (node.y + node.height / 2.0 - 2.0);
  return {x, y};
}

void simplifyDFGPolyline(
    std::vector<std::pair<double, double>> &points) {
  if (points.size() <= 2)
    return;
  std::vector<std::pair<double, double>> out;
  out.reserve(points.size());
  out.push_back(points.front());
  for (size_t i = 1; i + 1 < points.size(); ++i) {
    const auto &a = out.back();
    const auto &b = points[i];
    const auto &c = points[i + 1];
    bool sameX = std::abs(a.first - b.first) < 0.01 &&
                 std::abs(b.first - c.first) < 0.01;
    bool sameY = std::abs(a.second - b.second) < 0.01 &&
                 std::abs(b.second - c.second) < 0.01;
    if (sameX || sameY)
      continue;
    out.push_back(b);
  }
  out.push_back(points.back());
  points.swap(out);
}

std::vector<std::pair<double, double>>
buildFallbackDFGEdgePolyline(const DFGJsonModel &model,
                              const DFGJsonEdgeInfo &edge) {
  const auto &src = model.nodes[edge.from];
  const auto &dst = model.nodes[edge.to];
  auto srcPt = getDFGPortAnchor(src, /*isInput=*/false, edge.fromPort);
  auto dstPt = getDFGPortAnchor(dst, /*isInput=*/true, edge.toPort);
  bool isForward = srcPt.second < dstPt.second - 8.0;
  double laneSpread = static_cast<double>((edge.id % 5) - 2) * 18.0;
  std::vector<std::pair<double, double>> points;
  if (isForward) {
    double stub = 18.0;
    double midY = (srcPt.second + dstPt.second) / 2.0 + laneSpread;
    double minMid = srcPt.second + stub + 8.0;
    double maxMid = dstPt.second - stub - 8.0;
    if (minMid <= maxMid)
      midY = std::max(minMid, std::min(maxMid, midY));
    points = {
        srcPt,
        {srcPt.first, srcPt.second + stub},
        {srcPt.first, midY},
        {dstPt.first, midY},
        {dstPt.first, dstPt.second - stub},
        dstPt,
    };
  } else {
    double bendX =
        std::max(srcPt.first, dstPt.first) + 84.0 + static_cast<double>(edge.id % 4) * 18.0;
    points = {
        srcPt,
        {bendX, srcPt.second},
        {bendX, dstPt.second},
        dstPt,
    };
  }
  simplifyDFGPolyline(points);
  return points;
}

std::pair<double, double> estimateDFGNodeSize(const DFGJsonNodeInfo &node) {
  if (node.kind == "input" || node.kind == "output")
    return {132.0, 62.0};

  auto maxLabelLen = [&](llvm::StringRef text) -> size_t {
    size_t maxLen = 0;
    size_t start = 0;
    while (start <= text.size()) {
      size_t end = text.find('\n', start);
      llvm::StringRef line =
          end == llvm::StringRef::npos ? text.substr(start) : text.slice(start, end);
      maxLen = std::max(maxLen, line.size());
      if (end == llvm::StringRef::npos)
        break;
      start = end + 1;
    }
    return maxLen;
  };

  size_t maxPortCount =
      std::max(node.inputs.size(), std::max(node.outputs.size(), size_t{1}));
  double width = std::max<double>(
      128.0,
      std::max<double>(static_cast<double>(maxLabelLen(node.display)) * 8.0 + 36.0,
                       static_cast<double>(maxPortCount) * 58.0));
  return {width, 82.0};
}

bool routeDFGEdgesOffline(DFGJsonModel &model) {
  if (model.nodes.empty() || model.edges.empty())
    return false;
  for (const auto &node : model.nodes) {
    if (!node.hasLayout)
      return false;
  }

  constexpr int kGridStep = 10;
  constexpr double kNodePad = 10.0;
  constexpr double kCanvasPad = 90.0;
  constexpr double kStubLen = 22.0;

  double minX = std::numeric_limits<double>::infinity();
  double minY = std::numeric_limits<double>::infinity();
  double maxX = -std::numeric_limits<double>::infinity();
  double maxY = -std::numeric_limits<double>::infinity();
  for (const auto &node : model.nodes) {
    minX = std::min(minX, node.x - node.width / 2.0);
    minY = std::min(minY, node.y - node.height / 2.0);
    maxX = std::max(maxX, node.x + node.width / 2.0);
    maxY = std::max(maxY, node.y + node.height / 2.0);
  }
  if (!std::isfinite(minX) || !std::isfinite(minY) || !std::isfinite(maxX) ||
      !std::isfinite(maxY))
    return false;

  double gridMinX = minX - kCanvasPad;
  double gridMinY = minY - kCanvasPad;
  double gridMaxX = maxX + kCanvasPad;
  double gridMaxY = maxY + kCanvasPad;
  int gridCols =
      std::max(1, static_cast<int>(std::llround((gridMaxX - gridMinX) / kGridStep)));
  int gridRows =
      std::max(1, static_cast<int>(std::llround((gridMaxY - gridMinY) / kGridStep)));

  auto cellKey = [](int col, int row) -> int64_t {
    return (static_cast<int64_t>(col) << 32) ^
           static_cast<uint32_t>(row);
  };
  auto edgeKey = [&](int c0, int r0, int c1, int r1) -> std::tuple<int, int, int, int> {
    if (c0 < c1 || (c0 == c1 && r0 <= r1))
      return {c0, r0, c1, r1};
    return {c1, r1, c0, r0};
  };
  auto clampGrid = [&](int value, int upper) {
    return std::max(0, std::min(upper, value));
  };
  auto toGrid = [&](double x, double y) -> std::pair<int, int> {
    int col = clampGrid(static_cast<int>(std::llround((x - gridMinX) / kGridStep)),
                        gridCols);
    int row = clampGrid(static_cast<int>(std::llround((y - gridMinY) / kGridStep)),
                        gridRows);
    return {col, row};
  };
  auto toPixel = [&](int col, int row) -> std::pair<double, double> {
    return {gridMinX + static_cast<double>(col) * kGridStep,
            gridMinY + static_cast<double>(row) * kGridStep};
  };

  std::unordered_set<int64_t> blocked;
  for (const auto &node : model.nodes) {
    double left = node.x - node.width / 2.0 - kNodePad;
    double right = node.x + node.width / 2.0 + kNodePad;
    double top = node.y - node.height / 2.0 - kNodePad;
    double bottom = node.y + node.height / 2.0 + kNodePad;
    int c0 = static_cast<int>(std::floor((left - gridMinX) / kGridStep));
    int c1 = static_cast<int>(std::ceil((right - gridMinX) / kGridStep));
    int r0 = static_cast<int>(std::floor((top - gridMinY) / kGridStep));
    int r1 = static_cast<int>(std::ceil((bottom - gridMinY) / kGridStep));
    for (int col = c0; col <= c1; ++col) {
      for (int row = r0; row <= r1; ++row) {
        if (col < 0 || col > gridCols || row < 0 || row > gridRows)
          continue;
        blocked.insert(cellKey(col, row));
      }
    }
  }

  auto nearestUnblockedCell = [&](std::pair<int, int> seed) {
    if (!blocked.count(cellKey(seed.first, seed.second)))
      return seed;
    for (int radius = 1; radius < 24; ++radius) {
      for (int dc = -radius; dc <= radius; ++dc) {
        for (int dr = -radius; dr <= radius; ++dr) {
          if (std::abs(dc) != radius && std::abs(dr) != radius)
            continue;
          int col = seed.first + dc;
          int row = seed.second + dr;
          if (col < 0 || col > gridCols || row < 0 || row > gridRows)
            continue;
          if (!blocked.count(cellKey(col, row)))
            return std::pair<int, int>{col, row};
        }
      }
    }
    return seed;
  };

  struct RouteState {
    int col = 0;
    int row = 0;
    int dir = -1;
    bool operator<(const RouteState &other) const {
      return std::tie(col, row, dir) < std::tie(other.col, other.row, other.dir);
    }
  };
  struct QueueItem {
    RouteState state;
    double f = 0.0;
  };
  struct QueueItemGreater {
    bool operator()(const QueueItem &lhs, const QueueItem &rhs) const {
      return lhs.f > rhs.f;
    }
  };

  std::map<std::tuple<int, int, int, int>, unsigned> usedSegments;
  std::vector<size_t> edgeOrder(model.edges.size());
  std::iota(edgeOrder.begin(), edgeOrder.end(), 0);
  std::sort(edgeOrder.begin(), edgeOrder.end(),
            [&](size_t lhsIdx, size_t rhsIdx) {
              const auto &lhs = model.edges[lhsIdx];
              const auto &rhs = model.edges[rhsIdx];
              double lhsY = model.nodes[lhs.from].y + model.nodes[lhs.to].y;
              double rhsY = model.nodes[rhs.from].y + model.nodes[rhs.to].y;
              if (lhsY != rhsY)
                return lhsY < rhsY;
              return lhs.id < rhs.id;
            });

  auto routeGrid = [&](std::pair<int, int> start,
                       std::pair<int, int> goal) -> std::optional<std::vector<std::pair<int, int>>> {
    static constexpr std::array<std::pair<int, int>, 4> kDirs = {
        std::pair<int, int>{1, 0}, std::pair<int, int>{-1, 0},
        std::pair<int, int>{0, 1}, std::pair<int, int>{0, -1}};

    std::priority_queue<QueueItem, std::vector<QueueItem>, QueueItemGreater> open;
    std::map<RouteState, double> cost;
    std::map<RouteState, RouteState> prev;
    std::set<RouteState> closed;

    RouteState startState{start.first, start.second, -1};
    cost[startState] = 0.0;
    open.push({startState, 0.0});

    std::optional<RouteState> goalState;
    while (!open.empty()) {
      RouteState cur = open.top().state;
      open.pop();
      if (closed.count(cur))
        continue;
      closed.insert(cur);
      if (cur.col == goal.first && cur.row == goal.second) {
        goalState = cur;
        break;
      }
      double baseCost = cost[cur];
      for (int dir = 0; dir < 4; ++dir) {
        int nextCol = cur.col + kDirs[dir].first;
        int nextRow = cur.row + kDirs[dir].second;
        if (nextCol < 0 || nextRow < 0 || nextCol > gridCols || nextRow > gridRows)
          continue;
        if (blocked.count(cellKey(nextCol, nextRow)))
          continue;

        double stepCost = 1.0;
        if (cur.dir >= 0 && dir != cur.dir)
          stepCost += 0.9;
        if (cur.dir >= 0 && dir == (cur.dir ^ 1))
          stepCost += 1.6;
        auto seg = edgeKey(cur.col, cur.row, nextCol, nextRow);
        auto segIt = usedSegments.find(seg);
        if (segIt != usedSegments.end())
          stepCost += 2.4 * static_cast<double>(segIt->second);

        RouteState nextState{nextCol, nextRow, dir};
        double newCost = baseCost + stepCost;
        auto costIt = cost.find(nextState);
        if (costIt != cost.end() && newCost >= costIt->second)
          continue;
        cost[nextState] = newCost;
        prev[nextState] = cur;
        double heuristic =
            static_cast<double>(std::abs(nextCol - goal.first) + std::abs(nextRow - goal.second));
        open.push({nextState, newCost + heuristic});
      }
    }

    if (!goalState)
      return std::nullopt;

    std::vector<std::pair<int, int>> path;
    for (RouteState cur = *goalState;;) {
      path.push_back({cur.col, cur.row});
      auto prevIt = prev.find(cur);
      if (prevIt == prev.end())
        break;
      cur = prevIt->second;
    }
    std::reverse(path.begin(), path.end());
    return path;
  };

  auto reservePath = [&](const std::vector<std::pair<int, int>> &path) {
    for (size_t i = 0; i + 1 < path.size(); ++i)
      usedSegments[edgeKey(path[i].first, path[i].second, path[i + 1].first,
                           path[i + 1].second)]++;
  };

  for (size_t edgeIdx : edgeOrder) {
    auto &edge = model.edges[edgeIdx];
    const auto &src = model.nodes[edge.from];
    const auto &dst = model.nodes[edge.to];
    auto srcPt = getDFGPortAnchor(src, /*isInput=*/false, edge.fromPort);
    auto dstPt = getDFGPortAnchor(dst, /*isInput=*/true, edge.toPort);
    std::pair<double, double> srcStub{srcPt.first, srcPt.second + kStubLen};
    std::pair<double, double> dstStub{dstPt.first, dstPt.second - kStubLen};
    auto startCell = nearestUnblockedCell(toGrid(srcStub.first, srcStub.second));
    auto goalCell = nearestUnblockedCell(toGrid(dstStub.first, dstStub.second));
    auto gridPath = routeGrid(startCell, goalCell);
    if (!gridPath) {
      edge.points = buildFallbackDFGEdgePolyline(model, edge);
      continue;
    }

    reservePath(*gridPath);
    edge.points.clear();
    edge.points.push_back(srcPt);
    edge.points.push_back(srcStub);
    for (const auto &cell : *gridPath)
      edge.points.push_back(toPixel(cell.first, cell.second));
    edge.points.push_back(dstStub);
    edge.points.push_back(dstPt);
    simplifyDFGPolyline(edge.points);
  }
  return true;
}

void buildDFGJsonModel(DFGJsonModel &model, mlir::ModuleOp topModule) {
  circt::handshake::FuncOp funcOp;
  topModule->walk([&](circt::handshake::FuncOp func) {
    if (!funcOp) funcOp = func;
  });

  if (!funcOp) {
    model.funcName.clear();
    return;
  }

  auto &body = funcOp.getBody().front();
  auto argNames = funcOp->getAttrOfType<mlir::ArrayAttr>("argNames");
  auto resNames = funcOp->getAttrOfType<mlir::ArrayAttr>("resNames");
  model.funcName = funcOp.getName().str();

  llvm::SmallVector<mlir::Operation *, 16> ops;
  circt::handshake::ReturnOp returnOp;
  for (auto &op : body.getOperations()) {
    if (auto ret = mlir::dyn_cast<circt::handshake::ReturnOp>(op)) {
      returnOp = ret;
      continue;
    }
    ops.push_back(&op);
  }

  llvm::DenseMap<mlir::Value, std::pair<int, unsigned>> valueToNodePort;
  int nodeId = 0;

  for (auto arg : body.getArguments()) {
    std::string argName;
    if (argNames && arg.getArgNumber() < argNames.size()) {
      if (auto str =
              mlir::dyn_cast<mlir::StringAttr>(argNames[arg.getArgNumber()]))
        argName = str.getValue().str();
    }
    std::string typeStr = printType(arg.getType());
    DFGJsonNodeInfo node;
    node.id = nodeId;
    node.kind = "input";
    node.label = "arg" + std::to_string(arg.getArgNumber());
    node.argIndex = arg.getArgNumber();
    node.name = argName;
    node.type = typeStr;
    node.outputs.push_back({0, !argName.empty() ? argName : std::string("value"),
                            typeStr});
    auto [w, h] = estimateDFGNodeSize(node);
    node.width = w;
    node.height = h;
    model.nodes.push_back(std::move(node));
    valueToNodePort[arg] = {nodeId, 0};
    nodeId++;
  }

  for (auto *op : ops) {
    std::string opName = op->getName().getStringRef().str();
    std::string displayName = opName;
    size_t dotPos = displayName.find('.');
    if (dotPos != std::string::npos)
      displayName =
          displayName.substr(0, dotPos) + "\n" + displayName.substr(dotPos + 1);

    DFGJsonNodeInfo node;
    node.id = nodeId;
    node.kind = "op";
    node.label = opName;
    node.display = displayName;
    node.op = opName;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      node.inputs.push_back(
          {i, dfgOperandName(op, i), printType(op->getOperand(i).getType())});
    }
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      node.outputs.push_back(
          {i, dfgResultName(op, i), printType(op->getResult(i).getType())});
    }
    auto [w, h] = estimateDFGNodeSize(node);
    node.width = w;
    node.height = h;
    model.nodes.push_back(std::move(node));

    for (unsigned i = 0; i < op->getNumResults(); ++i)
      valueToNodePort[op->getResult(i)] = {nodeId, i};
    nodeId++;
  }

  if (returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      std::string resName;
      if (resNames && i < resNames.size()) {
        if (auto str = mlir::dyn_cast<mlir::StringAttr>(resNames[i]))
          resName = str.getValue().str();
      }
      std::string typeStr = printType(returnOp.getOperand(i).getType());
      DFGJsonNodeInfo node;
      node.id = nodeId;
      node.kind = "output";
      node.label = "return";
      node.resultIndex = static_cast<int>(i);
      node.name = resName;
      node.type = typeStr;
      node.inputs.push_back({0, !resName.empty() ? resName : std::string("value"),
                             typeStr});
      auto [w, h] = estimateDFGNodeSize(node);
      node.width = w;
      node.height = h;
      model.nodes.push_back(std::move(node));
      nodeId++;
    }
  }

  llvm::DenseMap<mlir::Operation *, int> opToNodeId;
  int firstOpNodeId = static_cast<int>(body.getNumArguments());
  for (unsigned i = 0; i < ops.size(); ++i)
    opToNodeId[ops[i]] = firstOpNodeId + static_cast<int>(i);

  int firstOutputNodeId = firstOpNodeId + static_cast<int>(ops.size());
  int edgeId = 0;

  for (auto *op : ops) {
    int dstNodeId = opToNodeId[op];
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto it = valueToNodePort.find(op->getOperand(i));
      if (it == valueToNodePort.end())
        continue;
      std::string typeStr = printType(op->getOperand(i).getType());
      model.edges.push_back({edgeId, it->second.first, it->second.second,
                             dstNodeId, i,
                             dfgEdgeType(op->getOperand(i).getType()),
                             typeStr});
      edgeId++;
    }
  }

  if (returnOp) {
    for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
      auto it = valueToNodePort.find(returnOp.getOperand(i));
      if (it == valueToNodePort.end())
        continue;
      std::string typeStr = printType(returnOp.getOperand(i).getType());
      model.edges.push_back({edgeId, it->second.first, it->second.second,
                             firstOutputNodeId + static_cast<int>(i), 0,
                             dfgEdgeType(returnOp.getOperand(i).getType()),
                             typeStr});
      edgeId++;
    }
  }
}

bool applyDotLayoutToDFGModel(DFGJsonModel &model) {
  if (model.nodes.size() <= 1)
    return false;

  auto dotPath = llvm::sys::findProgramByName("dot");
  if (!dotPath)
    return false;

  llvm::SmallString<128> srcPath;
  llvm::SmallString<128> plainPath;
  int srcFd = -1;
  int plainFd = -1;
  if (llvm::sys::fs::createTemporaryFile("fcc_dfg_layout", "dot", srcFd, srcPath))
    return false;
  if (llvm::sys::fs::createTemporaryFile("fcc_dfg_layout", "plain", plainFd,
                                         plainPath)) {
    llvm::sys::fs::remove(srcPath);
    return false;
  }

  auto cleanup = [&]() {
    llvm::sys::fs::remove(srcPath);
    llvm::sys::fs::remove(plainPath);
  };

  {
    llvm::raw_fd_ostream dotOS(srcFd, true);
    auto dotNum = [&](double value) {
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(6) << value;
      return ss.str();
    };
    constexpr double kPxPerIn = 72.0;
    dotOS << "digraph G {\n";
    dotOS << "  graph [rankdir=TB, overlap=false, splines=ortho, "
             "nodesep=1.25, ranksep=1.7, outputorder=edgesfirst, pad=0.2];\n";
    dotOS << "  node [fixedsize=true, margin=0, label=\"\"];\n";
    dotOS << "  edge [arrowsize=0.8, penwidth=1.2];\n";
    for (const auto &node : model.nodes) {
      llvm::StringRef shape =
          (node.kind == "op") ? llvm::StringRef("ellipse") : llvm::StringRef("box");
      dotOS << "  n" << node.id << " [shape=" << shape
            << ", width=" << dotNum(std::max(0.25, node.width / kPxPerIn))
            << ", height=" << dotNum(std::max(0.25, node.height / kPxPerIn))
            << "];\n";
    }
    for (const auto &edge : model.edges)
      dotOS << "  n" << edge.from << " -> n" << edge.to << ";\n";
    dotOS << "}\n";
  }
  {
    llvm::raw_fd_ostream plainOS(plainFd, true);
    plainOS.flush();
  }

  std::string errMsg;
  bool execFailed = false;
  std::string dotPathStr = *dotPath;
  std::string srcPathStr = srcPath.str().str();
  std::string plainPathStr = plainPath.str().str();
  llvm::SmallVector<llvm::StringRef, 5> args = {
      llvm::StringRef(dotPathStr), llvm::StringRef("-Tplain"),
      llvm::StringRef(srcPathStr), llvm::StringRef("-o"),
      llvm::StringRef(plainPathStr)};
  int rc = llvm::sys::ExecuteAndWait(dotPathStr, args, std::nullopt, {}, 0, 0,
                                     &errMsg, &execFailed);
  if (rc != 0 || execFailed) {
    cleanup();
    return false;
  }

  auto plainBuf = llvm::MemoryBuffer::getFile(plainPath);
  if (!plainBuf) {
    cleanup();
    return false;
  }

  double graphHeightIn = 0.0;
  std::map<int, std::pair<double, double>> nodePosIn;
  std::map<std::pair<int, int>, std::vector<size_t>> edgeBuckets;
  std::map<std::pair<int, int>, size_t> edgeBucketCursor;
  for (size_t idx = 0; idx < model.edges.size(); ++idx)
    edgeBuckets[{model.edges[idx].from, model.edges[idx].to}].push_back(idx);
  std::istringstream input((*plainBuf)->getBuffer().str());
  std::string lineStr;
  while (std::getline(input, lineStr)) {
    llvm::StringRef line(lineStr);
    line = line.trim();
    if (line.empty())
      continue;
    std::istringstream ls(line.str());
    std::string kind;
    ls >> kind;
    if (kind == "graph") {
      double scale = 0.0, width = 0.0;
      ls >> scale >> width >> graphHeightIn;
      (void)scale;
      (void)width;
      continue;
    }
    if (kind == "node") {
      std::string nodeName;
      double xIn = 0.0, yIn = 0.0, widthIn = 0.0, heightIn = 0.0;
      ls >> nodeName >> xIn >> yIn >> widthIn >> heightIn;
      if (nodeName.size() <= 1 || nodeName[0] != 'n')
        continue;
      try {
        int nodeId = std::stoi(nodeName.substr(1));
        nodePosIn[nodeId] = {xIn, yIn};
      } catch (...) {
      }
      continue;
    }
    if (kind == "edge") {
      std::string tailName, headName;
      int numPoints = 0;
      ls >> tailName >> headName >> numPoints;
      if (tailName.size() <= 1 || headName.size() <= 1 || tailName[0] != 'n' ||
          headName[0] != 'n')
        continue;
      int fromId = -1;
      int toId = -1;
      try {
        fromId = std::stoi(tailName.substr(1));
        toId = std::stoi(headName.substr(1));
      } catch (...) {
        continue;
      }
      auto key = std::make_pair(fromId, toId);
      auto bucketIt = edgeBuckets.find(key);
      if (bucketIt == edgeBuckets.end())
        continue;
      size_t bucketIdx = edgeBucketCursor[key];
      if (bucketIdx >= bucketIt->second.size())
        continue;
      auto &edge = model.edges[bucketIt->second[bucketIdx]];
      edgeBucketCursor[key] = bucketIdx + 1;
      for (int i = 0; i < numPoints; ++i) {
        double xIn = 0.0, yIn = 0.0;
        ls >> xIn >> yIn;
        edge.points.push_back({xIn, yIn});
      }
    }
  }
  cleanup();

  if (graphHeightIn <= 0.0 || nodePosIn.empty())
    return false;

  constexpr double kPxPerIn = 72.0;
  double minLeft = 0.0;
  double minTop = 0.0;
  bool haveBounds = false;
  for (auto &node : model.nodes) {
    auto it = nodePosIn.find(node.id);
    if (it == nodePosIn.end())
      continue;
    node.x = it->second.first * kPxPerIn;
    node.y = (graphHeightIn - it->second.second) * kPxPerIn;
    node.hasLayout = true;
    double left = node.x - node.width / 2.0;
    double top = node.y - node.height / 2.0;
    if (!haveBounds) {
      minLeft = left;
      minTop = top;
      haveBounds = true;
    } else {
      minLeft = std::min(minLeft, left);
      minTop = std::min(minTop, top);
    }
  }
  if (!haveBounds)
    return false;

  for (auto &edge : model.edges) {
    for (auto &pt : edge.points) {
      pt.first *= kPxPerIn;
      pt.second = (graphHeightIn - pt.second) * kPxPerIn;
      minLeft = std::min(minLeft, pt.first);
      minTop = std::min(minTop, pt.second);
    }
  }

  double shiftX = 80.0 - minLeft;
  double shiftY = 80.0 - minTop;
  for (auto &node : model.nodes) {
    if (node.hasLayout) {
      node.x += shiftX;
      node.y += shiftY;
    }
  }
  for (auto &edge : model.edges) {
    for (auto &pt : edge.points) {
      pt.first += shiftX;
      pt.second += shiftY;
    }
  }
  return true;
}

void writeDFGJson(llvm::raw_ostream &os, mlir::ModuleOp topModule) {
  DFGJsonModel model;
  buildDFGJsonModel(model, topModule);
  applyDotLayoutToDFGModel(model);
  routeDFGEdgesOffline(model);

  os << "{\n";
  if (model.nodes.empty()) {
    os << "  \"dot\": null, \"nodes\": [], \"edges\": []\n}";
    return;
  }

  os << "  \"func\": \"" << jsonEsc(model.funcName) << "\",\n";
  os << "  \"nodes\": [\n";
  for (size_t ni = 0; ni < model.nodes.size(); ++ni) {
    const auto &node = model.nodes[ni];
    if (ni > 0)
      os << ",\n";
    os << "    {\"id\": " << node.id << ", \"kind\": \"" << node.kind << "\""
       << ", \"label\": \"" << jsonEsc(node.label) << "\"";
    if (!node.display.empty())
      os << ", \"display\": \"" << jsonEsc(node.display) << "\"";
    if (!node.op.empty())
      os << ", \"op\": \"" << jsonEsc(node.op) << "\"";
    if (node.argIndex >= 0)
      os << ", \"arg_index\": " << node.argIndex;
    if (node.resultIndex >= 0)
      os << ", \"result_index\": " << node.resultIndex;
    os << ", \"name\": \"" << jsonEsc(node.name) << "\""
       << ", \"type\": \"" << jsonEsc(node.type) << "\""
       << ", \"w\": " << node.width
       << ", \"h\": " << node.height;
    if (node.hasLayout)
      os << ", \"x\": " << node.x << ", \"y\": " << node.y;
    auto emitPorts = [&](llvm::StringRef key,
                         const std::vector<DFGJsonPortInfo> &ports) {
      os << ", \"" << key << "\": [";
      for (size_t pi = 0; pi < ports.size(); ++pi) {
        if (pi > 0)
          os << ", ";
        os << "{\"index\": " << ports[pi].index
           << ", \"name\": \"" << jsonEsc(ports[pi].name)
           << "\", \"type\": \"" << jsonEsc(ports[pi].type) << "\"}";
      }
      os << "]";
    };
    emitPorts("inputs", node.inputs);
    emitPorts("outputs", node.outputs);
    os << "}";
  }
  os << "\n  ],\n";

  os << "  \"edges\": [\n";
  for (size_t ei = 0; ei < model.edges.size(); ++ei) {
    const auto &edge = model.edges[ei];
    if (ei > 0)
      os << ",\n";
    os << "    {\"id\": " << edge.id
       << ", \"from\": " << edge.from
       << ", \"from_port\": " << edge.fromPort
       << ", \"to\": " << edge.to
       << ", \"to_port\": " << edge.toPort
       << ", \"edge_type\": \"" << jsonEsc(edge.edgeType)
       << "\", \"value_type\": \"" << jsonEsc(edge.valueType) << "\"";
    if (!edge.points.empty()) {
      os << ", \"points\": [";
      for (size_t pi = 0; pi < edge.points.size(); ++pi) {
        if (pi > 0)
          os << ", ";
        os << "{\"x\": " << edge.points[pi].first
           << ", \"y\": " << edge.points[pi].second << "}";
      }
      os << "]";
    }
    os << "}";
  }
  os << "\n  ],\n";
  os << "  \"dot\": " << (model.nodes.empty() ? "null" : "\"dot\"") << "\n}";
}

} // namespace viz_detail
} // namespace fcc
