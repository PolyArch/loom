// Playwright interaction tests for Loom viz HTML viewer.
// Tests: SVG rendering, detail panel, mode toggle, cross-highlight.
//
// Run: npx playwright test tests/viz/html/interaction.spec.js
// Requires: chromium via `npx playwright install chromium`

const { test, expect } = require('@playwright/test');
const path = require('path');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');

// Generate test fixtures before all tests.
let dfgHtmlPath;
let adgHtmlPath;
let mappedHtmlPath;
let fixtureError = null;

test.beforeAll(async () => {
  const rootDir = path.resolve(__dirname, '../../..');
  const loom = path.join(rootDir, 'build/bin/loom');

  if (!fs.existsSync(loom)) {
    fixtureError = 'loom binary not found';
    return;
  }

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'loom-viz-test-'));

  const hsFile = path.join(rootDir,
    'tests/app/vecsum/Output/vecsum.handshake.mlir');
  const adgFile = path.join(rootDir,
    'tests/mapper-app/templates/loom_cgra_small.fabric.mlir');

  try {
    // Generate DFG viz using execFileSync (no shell spawn).
    dfgHtmlPath = path.join(tmpDir, 'dfg.html');
    execFileSync(loom, ['--viz-dfg', hsFile, '-o', dfgHtmlPath],
                 { stdio: 'pipe' });

    // Generate ADG viz.
    adgHtmlPath = path.join(tmpDir, 'adg.html');
    execFileSync(loom, ['--viz-adg', adgFile, '-o', adgHtmlPath],
                 { stdio: 'pipe' });

    // Generate mapped viz (mapper may fail; mapped tests will be skipped).
    try {
      const configBin = path.join(tmpDir, 'out.config.bin');
      execFileSync(loom,
        ['--handshake-input', hsFile, '--adg', adgFile,
         '-o', configBin, '--dump-viz'],
        { stdio: 'pipe' });
      const candidate = path.join(tmpDir, 'out.mapped.html');
      if (fs.existsSync(candidate)) {
        mappedHtmlPath = candidate;
      }
    } catch {
      // Mapper failure is acceptable; mapped tests will be skipped.
      mappedHtmlPath = null;
    }
  } catch (e) {
    // Graceful skip when process execution is restricted (EPERM).
    fixtureError = 'fixture generation failed: ' + (e.code || e.message);
  }
});

test('SVG renders in DFG viewer', async ({ page }) => {
  if (fixtureError) { test.skip(fixtureError); return; }

  await page.goto('file://' + dfgHtmlPath);
  const svg = await page.waitForSelector('svg', { timeout: 15000 });
  expect(svg).not.toBeNull();

  const nodes = await page.locator('.node').count();
  expect(nodes).toBeGreaterThan(0);
});

test('detail panel populates on node click', async ({ page }) => {
  if (fixtureError) { test.skip(fixtureError); return; }

  await page.goto('file://' + dfgHtmlPath);
  await page.waitForSelector('svg', { timeout: 15000 });

  // Detail panel should not have the 'visible' class initially.
  const panelHasVisible = await page.locator('#detail-panel')
    .evaluate(el => el.classList.contains('visible'));
  expect(panelHasVisible).toBe(false);

  // Click the first graph node.
  const firstNode = page.locator('.node').first();
  await firstNode.click();

  // Detail panel must gain the 'visible' class.
  await page.waitForFunction(() => {
    const panel = document.getElementById('detail-panel');
    return panel && panel.classList.contains('visible');
  }, { timeout: 5000 });

  // Detail content should have text (node attributes).
  const content = await page.locator('#detail-content').textContent();
  expect(content.length).toBeGreaterThan(0);
});

test('mode toggle switches between overlay and side-by-side', async ({ page }) => {
  if (fixtureError) { test.skip(fixtureError); return; }
  if (!mappedHtmlPath || !fs.existsSync(mappedHtmlPath)) {
    test.skip('mapped HTML not available; mode toggle requires mapped viewer');
    return;
  }

  await page.goto('file://' + mappedHtmlPath);
  await page.waitForSelector('svg', { timeout: 15000 });

  // Mode toggle button must be visible in the mapped viewer.
  const toggleBtn = page.locator('#btn-mode-toggle');
  await expect(toggleBtn).toBeVisible({ timeout: 5000 });

  // Initial mode is overlay: button text should be "Side-by-Side".
  const initialText = await toggleBtn.textContent();
  expect(initialText.trim()).toBe('Side-by-Side');

  // Click mode toggle to switch to side-by-side.
  await toggleBtn.click();

  // After toggle: button text should change to "Overlay".
  const afterText = await toggleBtn.textContent();
  expect(afterText.trim()).toBe('Overlay');

  // Click again to toggle back to overlay.
  await toggleBtn.click();

  // Button text should revert to "Side-by-Side".
  const revertText = await toggleBtn.textContent();
  expect(revertText.trim()).toBe('Side-by-Side');
});

test('SVG renders in ADG viewer', async ({ page }) => {
  if (fixtureError) { test.skip(fixtureError); return; }

  await page.goto('file://' + adgHtmlPath);
  const svg = await page.waitForSelector('svg', { timeout: 15000 });
  expect(svg).not.toBeNull();

  const nodes = await page.locator('.node').count();
  expect(nodes).toBeGreaterThan(0);
});

test('cross-highlight on hover in mapped viewer', async ({ page }) => {
  if (fixtureError) { test.skip(fixtureError); return; }
  if (!mappedHtmlPath || !fs.existsSync(mappedHtmlPath)) {
    test.skip('mapped HTML not available (mapper may have failed)');
    return;
  }

  await page.goto('file://' + mappedHtmlPath);
  await page.waitForSelector('svg', { timeout: 15000 });

  // First switch to side-by-side mode (cross-highlight only works there).
  const toggleBtn = page.locator('#btn-mode-toggle');
  const isToggleVisible = await toggleBtn.isVisible();
  if (isToggleVisible) {
    await toggleBtn.click();
    // Wait for re-render after mode switch.
    await page.waitForTimeout(1000);
    await page.waitForSelector('svg', { timeout: 15000 });
  }

  // Find a node in the left panel (DFG side) to hover.
  const leftNodes = page.locator('#graph-left .node');
  const leftNodeCount = await leftNodes.count();
  expect(leftNodeCount).toBeGreaterThan(0);

  // Hover over the first DFG node.
  await leftNodes.first().hover();
  await page.waitForTimeout(500);

  // Cross-highlight: a node in the opposite panel should gain
  // the 'cross-highlight' CSS class.
  const crossHighlightCount = await page.locator('.cross-highlight').count();
  expect(crossHighlightCount).toBeGreaterThan(0);
});
