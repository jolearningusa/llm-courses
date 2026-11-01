/* Offline teaching controls. Embedded by build_presenter_mode.py; no network or storage. */
(() => {
  const zh = document.documentElement.lang.startsWith('zh');
  const words = zh ? {
    notes:'讲者备注', presenter:'讲者窗口', missing:'本页暂无讲者备注。请结合本讲来源与实验手册备课。',
    prev:'上一页', next:'下一页', start:'开始', pause:'暂停', reset:'重置',
    help:'S：备注 · P：讲者窗口 · ←/→：翻页。投影窗口默认不显示备注。',
    blocked:'讲者窗口未能打开；请允许弹出窗口，或按 S 使用本页备注。',
    elapsed:'授课计时', nextTitle:'下一页', end:'本讲结束'
  } : {
    notes:'Speaker notes', presenter:'Presenter window', missing:'No speaker notes for this slide yet. Prepare with the lecture sources and lab handout.',
    prev:'Previous', next:'Next', start:'Start', pause:'Pause', reset:'Reset',
    help:'S: notes · P: presenter · ←/→: navigate. Notes stay hidden on the projection window by default.',
    blocked:'Presenter window blocked. Allow popups, or press S for notes in this window.',
    elapsed:'Lecture timer', nextTitle:'Next slide', end:'End of lecture'
  };
  const style = document.createElement('style');
  style.id = 'course-presenter-style';
  style.textContent = `
    #course-notes-panel{display:none;position:fixed;inset:60% 0 0;z-index:100;background:#f4f6fa;color:#152238;padding:16px 28px;overflow:auto;border-top:3px solid #4a719f;font:18px/1.5 system-ui}
    #course-notes-panel h3{font-size:22px;margin:8px 0}#course-notes-panel p{font-size:18px;line-height:1.5;margin:8px 0}
    .course-presenter-tools{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.course-presenter-tools button{font:16px system-ui;padding:6px 12px;cursor:pointer}
    .course-presenter-tools time{font:20px ui-monospace,monospace;min-width:70px}.course-presenter-tools small{font-size:14px}
    @media print{#course-notes-panel{display:none!important}}`;
  document.head.append(style);
  const panel = document.createElement('section');
  panel.id = 'course-notes-panel';
  panel.setAttribute('aria-label', words.notes);
  document.body.append(panel);
  let visible = false, popup = null, elapsed = 0, started = 0, running = false;
  const stage = document.getElementById('stage');
  const slides = [...document.querySelectorAll('.slide')];
  if (!stage || !slides.length) return;
  const index = () => Math.max(0, slides.findIndex(s => s.classList.contains('active')));
  const clock = () => elapsed + (running ? performance.now() - started : 0);
  const stamp = () => {
    const seconds = Math.floor(clock() / 1000);
    return String(Math.floor(seconds / 60)).padStart(2,'0') + ':' + String(seconds % 60).padStart(2,'0');
  };
  const move = delta => { if (typeof show === 'function') show(index() + delta); };
  function makeView(doc, mount) {
    const bar = doc.createElement('div'); bar.className = 'course-presenter-tools';
    const button = (label, action) => {
      const b = doc.createElement('button'); b.type = 'button'; b.textContent = label;
      b.addEventListener('click', action); bar.append(b); return b;
    };
    const prev = button(words.prev, () => move(-1));
    const next = button(words.next, () => move(1));
    const time = doc.createElement('time'); time.setAttribute('aria-label',words.elapsed); bar.append(time);
    const toggle = button(words.start, () => {
      if (running) { elapsed = clock(); running = false; }
      else { started = performance.now(); running = true; }
      render();
    });
    button(words.reset, () => { elapsed = 0; started = performance.now(); render(); });
    const help = doc.createElement('small'); help.textContent = words.help; bar.append(help);
    const title = doc.createElement('h3');
    const body = doc.createElement('div'); body.className = 'course-note-content';
    const upcoming = doc.createElement('p');
    mount.append(bar,title,body,upcoming);
    return {prev,next,time,toggle,title,body,upcoming,slide:-1};
  }
  const inline = makeView(document,panel);
  let external = null;
  function update(view) {
    const i = index(), slide = slides[i];
    view.time.textContent = stamp();
    view.toggle.textContent = running ? words.pause : words.start;
    view.prev.disabled = i === 0; view.next.disabled = i === slides.length - 1;
    if (view.slide === i) return;
    view.slide = i;
    view.title.textContent = `${i+1} / ${slides.length} · ${slide.querySelector('h1,h2')?.textContent || ''}`;
    const note = slide.querySelector('aside.notes');
    view.body.replaceChildren();
    if (note && note.textContent.trim()) {
      // Clone authored, repository-owned notes; do not fetch or execute external content.
      for (const child of note.childNodes) view.body.append(child.cloneNode(true));
    } else {
      const p = view.body.ownerDocument.createElement('p'); p.textContent = words.missing; view.body.append(p);
    }
    view.upcoming.textContent = words.nextTitle + ': ' + (slides[i+1]?.querySelector('h1,h2')?.textContent || words.end);
  }
  function render() {
    update(inline);
    if (popup && !popup.closed && external) update(external);
  }
  function fit() {
    if (!visible) { stage.style.top = '50%'; if (typeof resize === 'function') resize(); return; }
    stage.style.top = '30%';
    stage.style.transform = `translate(-50%,-50%) scale(${Math.min(innerWidth/1280,innerHeight*0.6/720)})`;
  }
  function toggleNotes() {
    visible = !visible; panel.style.display = visible ? 'block' : 'none'; render(); fit();
  }
  function openPresenter() {
    if (popup && !popup.closed) { popup.focus(); return; }
    popup = window.open('', 'llm-course-presenter', 'width=960,height=800');
    if (!popup) { if (!visible) toggleNotes(); inline.upcoming.textContent = words.blocked; return; }
    const doc = popup.document;
    doc.open(); doc.write('<!doctype html><html><head><meta charset="utf-8"><title></title></head><body></body></html>'); doc.close();
    doc.documentElement.lang = document.documentElement.lang;
    doc.title = words.presenter + ' · ' + document.title;
    const css = doc.createElement('style');
    css.textContent = 'body{margin:24px;background:#f4f6fa;color:#152238;font:20px/1.6 system-ui}h3{font-size:26px}p,li{font-size:20px}.course-presenter-tools{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.course-presenter-tools button{font:16px system-ui;padding:8px;cursor:pointer}.course-presenter-tools time{font:24px monospace}.course-presenter-tools small{width:100%;font-size:15px}a{color:#24548c}';
    doc.head.append(css); external = makeView(doc,doc.body);
    popup.addEventListener('keydown', e => {
      if (['ArrowRight','PageDown'].includes(e.key)) {e.preventDefault();move(1);}
      if (['ArrowLeft','PageUp'].includes(e.key)) {e.preventDefault();move(-1);}
    });
    render();
  }
  addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.target.isContentEditable) return;
    if (e.key.toLowerCase() === 's') {e.preventDefault();toggleNotes();}
    if (e.key.toLowerCase() === 'p') {e.preventDefault();openPresenter();}
  });
  addEventListener('resize',fit);
  new MutationObserver(render).observe(stage,{subtree:true,attributes:true,attributeFilter:['class']});
  setInterval(render,1000);
  const hint = document.getElementById('hint');
  if (hint) hint.textContent += zh ? ' · S 备注 · P 讲者窗口' : ' · S notes · P presenter';
  render();
})();
