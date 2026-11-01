#!/usr/bin/env node
// Exercise real offline teaching controls in Chromium (requires Playwright).
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
const { chromium } = createRequire(import.meta.url)('playwright');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const slides = join(root,'llm-compute-systems-course','slides');
const browser = await chromium.launch();
const errors = [];
const watch = page => {
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(/^https?:/.test(r.url()))errors.push('unexpected network request: '+r.url());});
};
try {
  let deckCount=0;
  for (const dir of [slides,join(slides,'zh')]) for (const file of readdirSync(dir).filter(x=>x.endsWith('.html'))) {
    const text=readFileSync(join(dir,file),'utf8');
    assert.equal((text.match(/<script id="course-presenter">/g)||[]).length,1,file);
    deckCount++;
  }
  for (const zh of [false,true]) {
    const context=await browser.newContext({viewport:{width:1280,height:720}});
    const page=await context.newPage();watch(page);
    await page.goto(pathToFileURL(join(slides,zh?'zh':'','01_workloads_rooflines.html')).href);
    const panel=page.locator('#course-notes-panel');
    assert.equal(await panel.isVisible(),false);
    await page.keyboard.press('s');assert.equal(await panel.isVisible(),true);
    assert.match(await panel.locator('.course-note-content').innerText(),zh?/板书/:/Draw work/);
    const initial=await panel.locator('h3').innerText();
    await page.keyboard.press('ArrowRight');await page.waitForTimeout(50);
    assert.notEqual(await panel.locator('h3').innerText(),initial);
    await page.keyboard.press('s');assert.equal(await panel.isVisible(),false);
    const popupPromise=page.waitForEvent('popup');await page.keyboard.press('p');
    const popup=await popupPromise;watch(popup);await popup.waitForSelector('h3');
    assert.match(await popup.locator('h3').innerText(),/^2 \/ 20/);
    assert.equal(await panel.isVisible(),false,'P must not reveal notes on the audience window');
    await popup.getByRole('button',{name:zh?'下一页':'Next',exact:true}).click();
    await page.waitForTimeout(50);assert.match(await popup.locator('h3').innerText(),/^3 \/ 20/);
    await popup.getByRole('button',{name:zh?'开始':'Start',exact:true}).click();
    await popup.waitForTimeout(1300);
    await popup.getByRole('button',{name:zh?'暂停':'Pause',exact:true}).click();
    const paused=await popup.locator('time').innerText();assert.notEqual(paused,'00:00');
    await popup.waitForTimeout(1100);assert.equal(await popup.locator('time').innerText(),paused);
    await popup.getByRole('button',{name:zh?'重置':'Reset',exact:true}).click();
    assert.equal(await popup.locator('time').innerText(),'00:00');
    await page.keyboard.press('End');await page.waitForTimeout(50);
    assert.match(await popup.locator('h3').innerText(),/^20 \/ 20/);
    assert.equal(await popup.getByRole('button',{name:zh?'下一页':'Next',exact:true}).isDisabled(),true);
    await page.keyboard.press('Home');await page.keyboard.press('s');
    await page.setViewportSize({width:960,height:540});
    await page.waitForFunction(()=>document.getElementById('stage').getBoundingClientRect().bottom<=325);
    const rect=await page.locator('#stage').boundingBox();assert(rect.y+rect.height<=325);
    await context.close();
  }
  const context=await browser.newContext();
  await context.addInitScript(()=>{window.open=()=>null;});
  const page=await context.newPage();watch(page);
  await page.goto(pathToFileURL(join(slides,'40_inference_accelerator_architectures.html')).href);
  await page.keyboard.press('p');
  assert.equal(await page.locator('#course-notes-panel').isVisible(),true);
  assert.match(await page.locator('.course-note-content').innerText(),/No speaker notes/);
  assert.match(await page.locator('#course-notes-panel').innerText(),/Presenter window blocked/);
  await context.close();
  assert.deepEqual(errors,[]);
  console.log(`Presenter checks passed: ${deckCount} embedded copies; bilingual navigation, hidden audience notes, timer, resize and popup fallback.`);
} finally { await browser.close(); }
