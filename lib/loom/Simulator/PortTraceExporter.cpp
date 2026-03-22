#include "loom/Simulator/PortTraceExporter.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

namespace loom {
namespace sim {

PortTraceExporter::PortTraceExporter(const std::string &outputDir)
    : outputDir_(outputDir) {}

void PortTraceExporter::addTracedModule(
    unsigned moduleIndex, const std::string &moduleName,
    const std::vector<TracedPort> &ports) {
  TracedModule mod;
  mod.moduleIndex = moduleIndex;
  mod.moduleName = moduleName;
  mod.ports = ports;
  tracedModules_.push_back(std::move(mod));

  // Add empty per-port trace vectors.
  traceData_.emplace_back();
  auto &modTraces = traceData_.back();
  modTraces.resize(ports.size());
}

void PortTraceExporter::recordCycle(
    uint64_t cycle, const std::vector<SimChannel> &portState) {
  for (size_t mi = 0; mi < tracedModules_.size(); ++mi) {
    const auto &mod = tracedModules_[mi];
    for (size_t pi = 0; pi < mod.ports.size(); ++pi) {
      const auto &port = mod.ports[pi];
      if (port.portIndex >= portState.size())
        continue;
      const auto &ch = portState[port.portIndex];
      PortTraceEntry entry;
      entry.cycle = cycle;
      entry.valid = ch.valid;
      entry.ready = ch.ready;
      entry.data = ch.data;
      entry.tag = ch.tag;
      entry.hasTag = ch.hasTag;
      entry.transferred = ch.didTransfer;
      traceData_[mi][pi].push_back(entry);
    }
  }
}

bool PortTraceExporter::flush() const {
  // Create output directory if it does not exist.
  if (auto ec = llvm::sys::fs::create_directories(outputDir_)) {
    llvm::errs() << "PortTraceExporter: cannot create directory " << outputDir_
                 << ": " << ec.message() << "\n";
    return false;
  }

  bool success = true;

  for (size_t mi = 0; mi < tracedModules_.size(); ++mi) {
    const auto &mod = tracedModules_[mi];

    for (size_t pi = 0; pi < mod.ports.size(); ++pi) {
      const auto &port = mod.ports[pi];
      const auto &entries = traceData_[mi][pi];
      std::string dirStr =
          (port.dir == StaticPortDirection::Input) ? "in" : "out";

      // Token-only trace (transfers only) for TB driver/golden comparison.
      // Format: one packed hex value per line = {tag, data} or just data.
      {
        llvm::SmallString<256> path(outputDir_);
        llvm::sys::path::append(path, mod.moduleName + "_" + dirStr +
                                          std::to_string(pi) + "_tokens.hex");
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec);
        if (ec) {
          llvm::errs() << "PortTraceExporter: cannot open " << path << ": "
                       << ec.message() << "\n";
          success = false;
          continue;
        }

        for (const auto &e : entries) {
          if (!e.transferred)
            continue;
          if (port.isTagged && port.tagWidth > 0) {
            // Write packed {tag, data} as single hex value.
            // The packed width is tagWidth + valueWidth bits.
            // We output it as a wide hex number.
            unsigned totalBits = port.tagWidth + port.valueWidth;
            unsigned hexDigits = (totalBits + 3) / 4;
            __uint128_t packed =
                (static_cast<__uint128_t>(e.tag) << port.valueWidth) | e.data;
            // Format manually for wide values.
            char buf[33]; // max 128 bits = 32 hex digits
            for (int d = hexDigits - 1; d >= 0; --d) {
              unsigned nibble = static_cast<unsigned>(packed >> (d * 4)) & 0xF;
              buf[hexDigits - 1 - d] =
                  (nibble < 10) ? ('0' + nibble) : ('a' + nibble - 10);
            }
            os.write(buf, hexDigits);
            os << "\n";
          } else {
            // Data only.
            unsigned hexDigits = (port.valueWidth + 3) / 4;
            if (hexDigits == 0)
              hexDigits = 1;
            char buf[17]; // max 64 bits = 16 hex digits
            for (int d = hexDigits - 1; d >= 0; --d) {
              unsigned nibble =
                  static_cast<unsigned>(e.data >> (d * 4)) & 0xF;
              buf[hexDigits - 1 - d] =
                  (nibble < 10) ? ('0' + nibble) : ('a' + nibble - 10);
            }
            os.write(buf, hexDigits);
            os << "\n";
          }
        }
      }

      // Full signal trace (all cycles) for debugging.
      {
        llvm::SmallString<256> path(outputDir_);
        llvm::sys::path::append(path, mod.moduleName + "_" + dirStr +
                                          std::to_string(pi) + "_full.hex");
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec);
        if (ec) {
          llvm::errs() << "PortTraceExporter: cannot open " << path << ": "
                       << ec.message() << "\n";
          success = false;
          continue;
        }

        // Header comment.
        os << "// cycle valid ready data tag transferred\n";
        for (const auto &e : entries) {
          os << llvm::format_hex_no_prefix(e.cycle, 8) << " "
             << (e.valid ? "1" : "0") << " " << (e.ready ? "1" : "0") << " "
             << llvm::format_hex_no_prefix(e.data, 16) << " "
             << llvm::format_hex_no_prefix(e.tag, 4) << " "
             << (e.transferred ? "1" : "0") << "\n";
        }
      }

      // Write a .count file with the number of transfer tokens.
      {
        llvm::SmallString<256> path(outputDir_);
        llvm::sys::path::append(path, mod.moduleName + "_" + dirStr +
                                          std::to_string(pi) + "_tokens.count");
        std::error_code ec;
        llvm::raw_fd_ostream os(path, ec);
        if (ec) {
          success = false;
          continue;
        }
        unsigned count = 0;
        for (const auto &e : entries)
          if (e.transferred)
            ++count;
        os << count << "\n";
      }
    }
  }

  return success;
}

} // namespace sim
} // namespace loom
