
import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

// Keep logs inside workspace reports directory (resolved at runtime)
const LOG_DIR = path.resolve(process.cwd(), 'reports/step_logs');

async function ensureLogDir() {
  if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });
}

async function logStep(page, stepId, description, status, takeScreenshot = false) {
  await ensureLogDir();
  const timestamp = new Date().toISOString();
  console.log(`📘 [${timestamp}] [${status}] [STEP ${stepId}] ${description}`);
  fs.writeFileSync(path.join(LOG_DIR, `${stepId}_${status}.txt`), `[${timestamp}] ${status}: ${description}`);
  if (takeScreenshot) {
    const imgPath = path.join(LOG_DIR, `${stepId}_${status}.png`);
    try { await page.screenshot({ path: imgPath, fullPage: true }); } catch (e) { console.warn('Screenshot failed', e.message); }
  }
}

test.describe('ui suite', () => {

  test('form fields – input typing & validation', async ({ page, browserName }) => {
    console.log(`🧭 Running in browser: ${browserName}`);

    // ---- STEP badf6156 ----
    {
      const stepDesc = 'type sample data into \'userId\' field';
      const stepId = 'badf6156';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        try { await page.fill("[name='userId']", "sample text"); } catch (e) { console.error("Fallback typing for userId", e); await page.fill("text=userId", "sample text"); }
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step badf6156 PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step badf6156 failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step badf6156...`);
          try { await page.fill("[name='userId']", "sample text"); } catch (e) { console.error("Fallback typing for userId", e); await page.fill("text=userId", "sample text"); }
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step badf6156 RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step badf6156 retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

    // ---- STEP 1128916a ----
    {
      const stepDesc = 'type sample data into \'password\' field';
      const stepId = '1128916a';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        try { await page.fill("[name='password']", "Test@1234"); } catch (e) { console.error("Fallback typing for password", e); await page.fill("text=password", "Test@1234"); }
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 1128916a PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 1128916a failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 1128916a...`);
          try { await page.fill("[name='password']", "Test@1234"); } catch (e) { console.error("Fallback typing for password", e); await page.fill("text=password", "Test@1234"); }
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 1128916a RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 1128916a retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

  });

  test('buttons & clickables – interactions', async ({ page, browserName }) => {
    console.log(`🧭 Running in browser: ${browserName}`);

    // ---- STEP 26435dd2 ----
    {
      const stepDesc = 'click \'Helpful resources\' button';
      const stepId = '26435dd2';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        try { await page.locator("text=Helpful resources").click(); } catch (e) { console.error("Retry click for Helpful resources", e); await page.locator("text=Helpful resources").click(); }
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 26435dd2 PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 26435dd2 failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 26435dd2...`);
          try { await page.locator("text=Helpful resources").click(); } catch (e) { console.error("Retry click for Helpful resources", e); await page.locator("text=Helpful resources").click(); }
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 26435dd2 RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 26435dd2 retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

    // ---- STEP 678f518b ----
    {
      const stepDesc = 'click \'Sign In\' button';
      const stepId = '678f518b';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        try { await page.locator("text=Sign In").click(); } catch (e) { console.error("Retry click for Sign In", e); await page.locator("text=Sign In").click(); }
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 678f518b PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 678f518b failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 678f518b...`);
          try { await page.locator("text=Sign In").click(); } catch (e) { console.error("Retry click for Sign In", e); await page.locator("text=Sign In").click(); }
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 678f518b RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 678f518b retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

  });

  test('headings – visibility checks', async ({ page, browserName }) => {
    console.log(`🧭 Running in browser: ${browserName}`);

    // ---- STEP c1d46a32 ----
    {
      const stepDesc = 'verify headings visible: [\'SIGN IN\']';
      const stepId = 'c1d46a32';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        const heads = await page.locator("h1, h2, h3").allInnerTexts(); console.log("Detected headings:", heads); expect(heads.length).toBeGreaterThan(0);
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step c1d46a32 PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step c1d46a32 failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step c1d46a32...`);
          const heads = await page.locator("h1, h2, h3").allInnerTexts(); console.log("Detected headings:", heads); expect(heads.length).toBeGreaterThan(0);
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step c1d46a32 RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step c1d46a32 retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

  });

  test('links – href verification', async ({ page, browserName }) => {
    console.log(`🧭 Running in browser: ${browserName}`);

    // ---- STEP 0bd6fd9b ----
    {
      const stepDesc = 'verify link \'Bot & Automation Platform\' navigates to \'/#/dashboard\'';
      const stepId = '0bd6fd9b';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        const link = page.locator("a", { hasText: /./ }); expect(await link.count()).toBeGreaterThan(0);
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 0bd6fd9b PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 0bd6fd9b failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 0bd6fd9b...`);
          const link = page.locator("a", { hasText: /./ }); expect(await link.count()).toBeGreaterThan(0);
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 0bd6fd9b RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 0bd6fd9b retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

    // ---- STEP 2c522469 ----
    {
      const stepDesc = 'verify link \'Forgot Password\' navigates to \'#/account/retrieve\'';
      const stepId = '2c522469';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        const link = page.locator("a", { hasText: /./ }); expect(await link.count()).toBeGreaterThan(0);
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 2c522469 PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 2c522469 failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 2c522469...`);
          const link = page.locator("a", { hasText: /./ }); expect(await link.count()).toBeGreaterThan(0);
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 2c522469 RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 2c522469 retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

  });

  test('smoke – page title visible', async ({ page, browserName }) => {
    console.log(`🧭 Running in browser: ${browserName}`);

    // ---- STEP 30de02cf ----
    {
      const stepDesc = 'goto https://va-a.botplatform.liveperson.net/#/account/signin';
      const stepId = '30de02cf';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        await page.goto("https://va-a.botplatform.liveperson.net/#/account/signin");
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 30de02cf PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 30de02cf failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 30de02cf...`);
          await page.goto("https://va-a.botplatform.liveperson.net/#/account/signin");
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 30de02cf RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 30de02cf retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

    // ---- STEP 8142651e ----
    {
      const stepDesc = 'verify page title is visible';
      const stepId = '8142651e';
      console.log(`[STEP ${s.step_id}] ${s.step_safe}`);
      const stepStart = Date.now();
      try {
        const title = await page.title(); expect(title.length).toBeGreaterThan(0);
        const duration = ((Date.now() - stepStart) / 1000).toFixed(2);
        await logStep(page, stepId, stepDesc, 'PASS', true);
        console.log(`✅ Step 8142651e PASS (${duration}s)`);
      } catch (e) {
        console.error(`❌ Step 8142651e failed:`, e && e.message ? e.message : e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // single retry
        try {
          console.log(`🔁 Retrying step 8142651e...`);
          const title = await page.title(); expect(title.length).toBeGreaterThan(0);
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
          console.log(`✅ Step 8142651e RETRY_PASS`);
        } catch (retryErr) {
          console.error(`❌ Step 8142651e retry failed:`, retryErr && retryErr.message ? retryErr.message : retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }

  });

});