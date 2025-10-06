
// @ts-check
const { defineConfig } = require('@playwright/test');
const path = require('path');

const browserMatrix = (process.env.BROWSERS || 'chromium,firefox,webkit')
  .split(',')
  .map(b => b.trim())
  .filter(Boolean);

const projects = browserMatrix.map(browserName => ({
  name: browserName,
  use: { browserName }
}));

module.exports = defineConfig({
  testDir: './tests',
  timeout: 45000,
  expect: { timeout: 8000 },
  reporter: [
    ['html', { outputFolder: path.resolve(__dirname, '../reports/playwright'), open: 'never' }],
    ['json', { outputFile: path.resolve(__dirname, '../reports/playwright/report.json') }]
  ],
  projects,
  use: {
    baseURL: 'https://va-a.botplatform.liveperson.net/#/account/signin',
    storageState: process.env.STORAGE_STATE || 'auth/session_state.json',
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    headless: true,
  },
});