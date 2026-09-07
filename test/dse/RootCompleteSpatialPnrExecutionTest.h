#pragma once

namespace loom::test {

void candidateWorkerCountPreservesFormalResult();
void sharedFrontierWorkersPreserveFormalResult();
void finalizedRestartSurvivesUnfinishedPeer();

} // namespace loom::test
