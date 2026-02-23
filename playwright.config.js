// Playwright configuration for Loom viz interaction tests.
const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './tests/viz/html',
  timeout: 30000,
  use: {
    browserName: 'chromium',
    headless: true,
  },
});
