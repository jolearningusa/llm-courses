#!/usr/bin/env node
// Open every slide deck (<course>/slides/*.html) in headless Chromium and fail
// on console errors, uncaught exceptions, or broken slide numbering.
//
// Each deck draws "N / TOTAL" on its slides. Within a deck the numbers must
// step by one, and every slide must show the same TOTAL. A deck that starts
// above 1 continues another deck of the same course (a master deck split over
// several files), so some deck there must end on the slide just before it.
// A TOTAL that disagrees between such parts is reported as a warning.
//
// Usage: node scripts/check_slides.mjs   (needs the playwright package)
import { readdirSync, existsSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

// require() also finds a globally installed playwright (NODE_PATH); import does not.
const { chromium } = createRequire(import.meta.url)('playwright');

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const skip = new Set(['.git', '.github', 'docs', 'scripts', '_site', 'node_modules']);

const decks = readdirSync(root)
  .filter(d => !skip.has(d) && !d.startsWith('.') && statSync(join(root, d)).isDirectory())
  .filter(d => existsSync(join(root, d, 'README.md')) && existsSync(join(root, d, 'slides')))
  .flatMap(d => readdirSync(join(root, d, 'slides'))
    .filter(f => /\.html?$/i.test(f)).sort().map(f => join(root, d, 'slides', f)));

const problems = [];
const seen = []; // {name, course, first, last, total}
const warnings = [];
const browser = await chromium.launch();
for (const file of decks) {
  const name = relative(root, file);
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(e.message));
  page.on('requestfailed', r => errors.push(`request failed: ${r.url()}`));
  await page.goto(pathToFileURL(file).href, { waitUntil: 'load' });
  await page.waitForTimeout(300);
  const slides = await page.$$eval('.slide', els => els.map(s => {
    const m = (s.querySelector('.num')?.textContent || '').match(/(\d+)\s*\/\s*(\d+)/);
    return m ? [Number(m[1]), Number(m[2])] : null;
  }));
  await page.close();

  for (const e of errors) problems.push(`${name}: console error: ${e}`);
  if (!slides.length) { problems.push(`${name}: no .slide elements`); continue; }
  if (slides.some(s => !s)) { problems.push(`${name}: a slide has no "N / TOTAL" number`); continue; }
  const totals = new Set(slides.map(s => s[1]));
  if (totals.size > 1) problems.push(`${name}: slides disagree on the total (${[...totals].join(', ')})`);
  slides.forEach((s, i) => {
    if (i && s[0] !== slides[i - 1][0] + 1) problems.push(`${name}: slide ${slides[i - 1][0]} is followed by ${s[0]}`);
  });
  const [first, last] = [slides[0][0], slides.at(-1)[0]];
  seen.push({ name, course: name.split('/')[0], first, last, total: slides[0][1] });
  console.log(`${name}: slides ${first}-${last} of ${slides[0][1]}${errors.length ? `, ${errors.length} error(s)` : ''}`);
}
await browser.close();

for (const d of seen.filter(d => d.first > 1)) {
  const prev = seen.find(p => p.course === d.course && p.last === d.first - 1);
  if (!prev) problems.push(`${d.name}: starts at ${d.first}, but no deck in the course ends at ${d.first - 1}`);
  else if (prev.total !== d.total) warnings.push(`${prev.name} says ${prev.total} slides in total, ${d.name} says ${d.total}`);
}

for (const w of warnings) console.log(`warning  ${w}`);
for (const p of problems) console.log(`problem  ${p}`);
console.log(`Checked ${decks.length} deck(s): ${problems.length} problem(s)`);
process.exit(problems.length ? 1 : 0);
