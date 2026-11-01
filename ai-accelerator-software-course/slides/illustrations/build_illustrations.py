"""Editable, offline SVG teaching figures and idempotent HTML slide insertion.
Run from any directory with Python 3.9+. All figures are conceptual, not benchmarks.
"""
from html import escape
from pathlib import Path
import json
import re

HERE=Path(__file__).resolve().parent
SLIDES=HERE.parent
INK='#eef4ff';MUTED='#b7c7df';TEAL='#8de0cf';BLUE='#8bbcff';AMBER='#f1c979';GRAY='#50617a';BG='#101927'
class SVG:
    def __init__(self,slug,title,description):
        self.slug=slug;self.title=title;self.description=description;self.parts=[]
    def text(self,x,y,s,size=22,color=INK,anchor='start'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')
    def rect(self,x,y,w,h,fill=BG,stroke=BLUE,r=5):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    def box(self,x,y,w,h,label,color=BLUE):
        self.rect(x,y,w,h,stroke=color)
        lines=label.split('|')
        for j,line in enumerate(lines):self.text(x+w/2,y+h/2+(j-(len(lines)-1)/2)*25+7,line,22,color,anchor='middle')
    def line(self,x1,y1,x2,y2,color=MUTED,arrow=False,dash=False):
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.5"'+(' stroke-dasharray="6 5"' if dash else '')+(f' marker-end="url(#{self.slug}-arrow)"' if arrow else '')+'/>')
    def path(self,points,color=MUTED,arrow=True):
        self.parts.append('<path d="'+' '.join(('M' if j==0 else 'L')+f'{x},{y}' for j,(x,y) in enumerate(points))+f'" fill="none" stroke="{color}" stroke-width="2.5"'+(f' marker-end="url(#{self.slug}-arrow)"' if arrow else '')+'/>')
    def chain(self,labels,y=120,colors=None):
        gap=28;w=(1040-gap*(len(labels)-1))/len(labels)
        for i,label in enumerate(labels):
            x=30+i*(w+gap);self.box(x,y,w,64,label,(colors or [BLUE]*len(labels))[i])
            if i<len(labels)-1:self.line(x+w+3,y+32,x+w+gap-4,y+32,arrow=True)
    def grid(self,x,y,rows,cols,cell=25,active=None):
        for i in range(rows):
            for j in range(cols):
                color=TEAL if active and active(i,j) else GRAY
                self.rect(x+j*cell,y+i*cell,cell-3,cell-3,fill=color,stroke=color,r=1)
    def out(self):
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 340" role="img" aria-labelledby="{self.slug}-title {self.slug}-desc" style="display:block;width:100%;height:auto"><title id="{self.slug}-title">{escape(self.title)}</title><desc id="{self.slug}-desc">{escape(self.description)}</desc><defs><marker id="{self.slug}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="{MUTED}"/></marker></defs><rect width="1100" height="340" fill="#101927"/><g font-family="system-ui,sans-serif">'+''.join(self.parts)+'</g></svg>'

def build():
    specs=json.loads((HERE/'figure_specs.json').read_text())
    for spec in specs:
        slug='accelerator-'+spec['prefix']
        for zh in (False,True):
            s=SVG(slug,spec['anchor_cn'] if zh else spec['anchor'],
                  '概念顺序与责任边界，不是实测时序或硬件图。' if zh else 'Conceptual order and responsibility boundaries, not measured timing or a hardware schematic.')
            s.chain(spec['cnlabels'] if zh else spec['labels'],110)
            svg=s.out()
            if not zh:(HERE/(slug+'.svg')).write_text(svg)
            path=SLIDES/('zh' if zh else '')/spec['name']
            source=path.read_text()
            source=re.sub(r'<section class="slide"[^>]* data-illustration="[^"]+".*?</section>','',source,flags=re.S)
            anchor=spec['anchor_cn'] if zh else spec['anchor']
            match=next((m for m in re.finditer(r'<section class="slide".*?</section>',source,re.S) if '<h2>'+anchor+'</h2>' in m.group()),None)
            if match is None:raise ValueError('missing anchor '+anchor)
            cite=re.search(r'<div class="cite">(.*?)</div>',match.group(),re.S).group(1)
            caption='箭头表示教学中的职责顺序；不表示真实设备时序。' if zh else 'Arrows show the teaching responsibility sequence; they do not establish physical timing.'
            title=anchor+('示意图' if zh else ' — diagram')
            addition='<section class="slide" data-illustration="'+slug+'"><h2>'+escape(title)+'</h2><div class="content">'+svg+'<p class="note">'+caption+'</p></div><div class="cite">'+cite+'</div><div class="num"></div></section>'
            source=source[:match.end()]+addition+source[match.end():]
            total=len(re.findall(r'<section class="slide"',source));index=[0]
            def number(m):
                index[0]+=1;t=re.sub(r'<section class="slide"(?: id="[^"]*")?',f'<section class="slide" id="s{index[0]}"',m.group(),count=1)
                return re.sub(r'<div class="num">.*?</div>',f'<div class="num">{index[0]} / {total}</div>',t,count=1)
            path.write_text(re.sub(r'<section class="slide".*?</section>',number,source,flags=re.S))
    print('Generated eight editable figures and sixteen bilingual companion slides')

if __name__=='__main__':build()
