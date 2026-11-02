"""Attributed bilingual mechanism pages, in place of existing text pages.

Called by build_illustrations.py after the legacy companion figures are built.
Geometry is original. No upstream artwork, network or plotting dependencies.
"""
from html import escape, unescape
from pathlib import Path
import json
import re
import textwrap

HERE=Path(__file__).resolve().parent
SLIDES=HERE.parent
BG='#101927'; INK='#eef4ff'; MUTED='#b7c7df'; BLUE='#8bbcff'; TEAL='#8de0cf'; AMBER='#f1c979'; RED='#ff9e9e'; GRAY='#50617a'

class Drawing:
    def __init__(self,spec,zh):
        self.spec=spec;self.zh=zh;self.slug=spec['slug'];self.parts=[]
        self.labels=spec['zh' if zh else 'en']
    def t(self,x,y,value,size=23,color=INK,anchor='start'):
        for j,line in enumerate(value.split('|')):
            self.parts.append(f'<text x="{x}" y="{y+j*28}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(line)}</text>')
    def rect(self,x,y,w,h,color=BLUE,fill=BG,dash=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{color}" stroke-width="2"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
    def box(self,x,y,w,h,value,color=BLUE):
        lines=[]
        for line in value.split('|'):
            width=int((w-22)/(23 if self.zh and any(ord(c)>255 for c in line) else 12.5))
            lines.extend(textwrap.wrap(line,width=max(5,width),break_long_words=True,break_on_hyphens=False) or [''])
        if len(lines)*28>h+10:raise ValueError(f'{self.slug}: label too long: {value}')
        self.rect(x,y,w,h,color)
        self.t(x+w/2,y+h/2+8-(len(lines)-1)*14,'|'.join(lines),23,color,'middle')
    def path(self,pts,color=MUTED,arrow=True,dash=False):
        self.parts.append('<path d="'+' '.join(('M' if i==0 else 'L')+f'{x},{y}' for i,(x,y) in enumerate(pts))+f'" fill="none" stroke="{color}" stroke-width="2.5"'+(' stroke-dasharray="6 5"' if dash else '')+(f' marker-end="url(#{self.slug}-arrow)"' if arrow else '')+'/>')
    def line(self,x1,y1,x2,y2,color=MUTED,arrow=True,dash=False):self.path([(x1,y1),(x2,y2)],color,arrow,dash)
    def grid(self,x,y,rows,cols,active,cell=30):
        for r in range(rows):
            for c in range(cols):
                col=TEAL if active(r,c) else GRAY
                self.rect(x+c*cell,y+r*cell,cell-3,cell-3,col,col)
    def render(self):
        a=self.labels;k=self.spec['kind']
        if k=='flow':
            for i,label in enumerate(a[:4]):
                x=25+i*270;self.box(x,110,240,85,label,TEAL if i==3 else BLUE)
                if i<3:self.line(x+240,153,x+270,153)
            self.t(550,295,self.spec.get('formula',''),26,AMBER,'middle')
        elif k=='fork':
            for x,y,w,h,idx,c in [(25,125,230,80,0,BLUE),(410,20,295,80,1,TEAL),(410,235,295,80,2,AMBER),(855,125,220,80,3,BLUE)]:self.box(x,y,w,h,a[idx],c)
            self.path([(255,165),(330,165),(330,60),(410,60)])
            self.path([(330,165),(330,275),(410,275)])
            self.path([(705,60),(785,60),(785,165),(855,165)])
            self.path([(705,275),(785,275),(785,165)])
            if len(a)>4:
                wrapped=textwrap.wrap(a[4],width=17 if self.zh else 32,break_long_words=True)
                self.t(550,160-(len(wrapped)-1)*14,'|'.join(wrapped),23,MUTED,'middle')
        elif k=='states':
            coords=[(25,30),(665,30),(665,230),(25,230)]
            for i,(x,y) in enumerate(coords):self.box(x,y,400,75,a[i],TEAL if i==2 else BLUE)
            self.line(425,67,665,67);self.line(865,105,865,230);self.line(665,267,425,267)
            if not self.spec.get('terminal'):self.line(225,230,225,105,AMBER)
            if len(a)>4:self.t(550,170,a[4],23,AMBER,'middle')
        elif k in ['serial','overlap']:
            self.t(15,67,a[0],22,BLUE);self.t(15,187,a[1],22,TEAL)
            if k=='serial':
                bars=[(0,0,1),(1,1,2),(0,2,3),(1,3,4)]
            else:bars=[(0,0,1),(1,1,2),(0,1,3),(1,2,4)]
            for row,step,index in bars:
                self.box(230+step*205,30+row*120,195,76,a[index+1],BLUE if row==0 else TEAL)
            self.line(210,290,1080,290);self.t(650,330,'logical order / schematic' if not self.zh else '逻辑顺序 / 示意',21,MUTED,'middle')
        elif k=='lifetime':
            events=a[:5]
            for i,v in enumerate(events):
                x=150+i*205;self.t(x,40,v,22,INK,'middle');self.line(x,65,x,260,GRAY,False,True)
            self.line(150,115,1030,115)
            self.rect(150,180,205*self.spec.get('retain_until',4),54,AMBER,'#273142');self.t(150+205*self.spec.get('retain_until',4)/2,215,a[5],23,AMBER,'middle')
            if len(a)>6:self.t(550,315,a[6],23,TEAL,'middle')
        elif k=='budget':
            # Equal blocks identify additive ledger terms, not proportional quantities.
            for i,v in enumerate(a[:4]):
                self.box(25+i*270,50,240,85,v,BLUE if i%2==0 else TEAL)
                if i<3:self.t(280+i*270,105,self.spec.get('join','+'),24,AMBER,'middle')
            self.t(550,215,a[4],26,AMBER,'middle')
            self.t(550,300,self.spec.get('formula',''),25,INK,'middle')
        elif k=='equation':
            self.box(25,20,330,80,a[0]);self.box(745,20,330,80,a[1],TEAL)
            self.path([(190,100),(190,155),(550,155),(550,190)])
            self.path([(910,100),(910,155),(550,155)])
            self.t(550,230,self.spec.get('formula',''),27,AMBER,'middle')
            self.t(550,310,a[2],24,TEAL,'middle')
        elif k in ['compare','compare3']:
            for row in range(3 if k=='compare3' else 2):
                y=10+row*110 if k=='compare3' else 35+row*165;self.t(20,y+40,a[row*4],22,TEAL if row else BLUE)
                for j in range(3):
                    x=260+j*280;self.box(x,y,255,80,a[row*4+j+1],TEAL if row else BLUE)
                    if j<2:self.line(x+255,y+40,x+280,y+40)
        elif k=='mapping':
            # Explicit mapping is stored per figure (not a vendor address or rank order).
            destinations=self.spec.get('map',[0,1,0,2])
            for i in range(4):self.box(25,10+i*82,250,62,a[i])
            n=max(destinations)+1
            target_y=[25+i*110 if n==3 else 60+i*150 for i in range(n)]
            for i,y in enumerate(target_y):self.box(815,y,260,75,a[4+i],TEAL)
            for i,j in enumerate(destinations):self.path([(275,41+i*82),(420+i*65,41+i*82),(420+i*65,target_y[j]+37),(815,target_y[j]+37)])
            if n==2 and len(a)>6:self.t(825,335,a[6],21,AMBER)
        elif k=='tree':
            self.box(25,120,200,75,a[0],AMBER)
            for i in range(3):
                y=10+i*110;self.box(430,y,230,75,a[i+1],TEAL if i==0 else BLUE)
                self.path([(225,157),(330,157),(330,y+37),(430,y+37)])
                self.line(660,y+37,805,y+37);self.box(805,y,270,75,a[i+4],TEAL if i==0 else MUTED)
        elif k=='accumulation':
            self.box(25,25,330,80,a[0]);self.box(25,235,330,80,a[1])
            self.box(445,125,300,80,a[2],AMBER);self.box(855,125,220,80,a[3],TEAL)
            self.path([(355,65),(400,65),(400,165),(445,165)]);self.path([(355,275),(400,275),(400,165)])
            self.line(745,165,855,165);self.t(550,335,a[4],23,MUTED,'middle')
        elif k=='cache_layers':
            for i in range(6):
                y=10+i*55;self.box(25,y,450,44,a[i*2],TEAL if i%2 else BLUE)
                self.line(475,y+22,590,y+22);self.t(620,y+30,a[i*2+1],23)
        elif k=='zero_stages':
            for j,title in enumerate(a[:3]):self.t(380+j*280,30,title,23,INK,'middle')
            for stage in range(1,4):
                y=55+(stage-1)*88;self.t(25,y+47,'ZeRO-'+str(stage),25,AMBER)
                for j in range(3):
                    x=260+j*280;sharded=(j>=3-stage)
                    self.rect(x,y,240,70,TEAL if sharded else BLUE)
                    for rank in range(3):
                        self.rect(x+12+(rank*72 if sharded else 0),y+9+rank*19,68 if sharded else 216,12,TEAL if sharded else BLUE,TEAL if sharded else BLUE)
            self.t(550,340,a[3],21,MUTED,'middle')
        elif k=='spec_tokens':
            for i in range(4):
                self.box(25+i*270,45,240,80,a[i],TEAL if i<2 else RED if i==2 else MUTED)
                if i<3:self.line(265+i*270,85,295+i*270,85)
            self.box(25,235,500,75,a[4],TEAL);self.box(620,235,455,75,a[5],AMBER)
            self.path([(145,125),(145,180),(275,180),(275,235)],TEAL)
            self.path([(685,125),(685,180),(845,180),(845,235)],AMBER)
        elif k=='test':
            self.box(25,120,250,80,a[0]);self.box(425,20,300,80,a[1],TEAL);self.box(425,235,300,80,a[2],AMBER)
            self.path([(275,160),(350,160),(350,60),(425,60)]);self.path([(350,160),(350,275),(425,275)])
            self.box(870,120,205,80,a[3],TEAL)
            self.path([(725,60),(800,60),(800,160),(870,160)]);self.path([(725,275),(800,275),(800,160)])
        elif k=='tiles':
            for x,title,rows,cols,pattern in [(50,a[0],5,6,lambda r,c:1<=r<3),(445,a[1],6,5,lambda r,c:2<=c<4),(840,a[2],5,5,lambda r,c:1<=r<3 and 2<=c<4)]:
                self.t(x,35,title,23);self.grid(x,65,rows,cols,pattern)
            self.line(260,140,415,140);self.line(655,140,810,140)
            self.t(550,300,a[3],24,AMBER,'middle')
        elif k=='mask':
            self.t(45,30,a[0],24,BLUE);self.grid(45,60,7,7,lambda r,c:c<=r,32)
            self.t(410,65,a[1],25,TEAL);self.t(410,125,a[2],24);self.t(410,200,a[3],24,AMBER)
            if len(a)>4:self.t(410,280,a[4],23,MUTED)
        elif k=='lanes':
            for row in range(2):
                y=40+row*155;self.t(20,y+30,a[row*3],23,BLUE if row==0 else TEAL)
                for j in range(8):
                    x=320+j*85;mode=self.spec.get('lane_mode','plain')
                    color=(TEAL if j<5 else GRAY) if mode=='tail' else BLUE if row==0 else TEAL
                    value=(str(j) if row==0 else str(int(j<5))) if mode=='tail' else str(j if row==0 else 4*j) if mode=='stride' else str(j)
                    self.rect(x,y,72,50,color);self.t(x+36,y+33,value,22,INK,'middle')
                self.t(320,y+98,a[row*3+1],23,AMBER)
            self.t(550,335,a[2]+' / '+a[5],22,MUTED,'middle')
        elif k=='pages':
            self.t(25,25,a[0],24,BLUE);self.t(650,25,a[1],24,TEAL)
            for i in range(3):
                self.box(25,55+i*90,240,60,a[2+i])
                j=self.spec.get('map',[2,0,3])[i];self.path([(265,85+i*90),(350+i*55,85+i*90),(350+i*55,80+j*62),(650,80+j*62)])
            for j in range(4):self.box(650,50+j*62,400,52,a[5+j],TEAL)
        elif k=='numberline':
            self.line(50,170,1050,170)
            for i,t in enumerate(a[:5]):
                x=90+i*225;self.line(x,155,x,185,TEAL,False);self.t(x,130,t,23,INK,'middle')
            self.t(550,265,a[5],25,AMBER,'middle');self.t(550,325,self.spec.get('formula',''),23,MUTED,'middle')
        elif k=='schedule_interference':
            self.t(20,70,a[0],23,BLUE);self.t(20,205,a[1],23,TEAL)
            self.box(230,30,600,80,a[2],BLUE)
            self.box(230,165,600,80,a[3],AMBER)
            self.box(845,165,230,80,a[5],TEAL)
            self.line(215,290,1080,290);self.t(650,330,'schematic event order' if not self.zh else '事件顺序示意',21,MUTED,'middle')
        elif k=='tile_occupancy':
            self.t(35,35,'Constructed 3 × 5 output / 4 × 6 tile' if not self.zh else '构造示例：3 × 5 输出 / 4 × 6 块',24)
            self.grid(35,75,4,6,lambda r,c:r<3 and c<5,48)
            self.t(440,100,'Useful cells: 15' if not self.zh else '有效单元：15',25,TEAL)
            self.t(440,165,'Tile cells: 24' if not self.zh else '块单元：24',25,AMBER)
            self.t(440,240,'Gray cells: outside the output' if not self.zh else '灰色单元：输出范围之外',23,MUTED)
            self.t(550,330,'Schematic shape occupancy, not measured device utilization' if not self.zh else '形状占用示意，并非实测设备利用率',22,MUTED,'middle')
        elif k=='quant_blocks':
            self.t(550,30,a[0]+' + '+a[1],24,TEAL,'middle')
            for i in range(4):
                x=25+i*270;self.rect(x,65,240,145,TEAL if i%2 else BLUE)
                self.box(x+15,80,210,60,('Values ' if not self.zh else '数值块 ')+str(i))
                self.box(x+15,155,210,43,'scale s'+str(i),AMBER)
                self.path([(x+120,210),(x+120,255),(550,255),(550,280)])
            self.t(550,310,a[3],24,TEAL,'middle')
        elif k=='energy':
            self.rect(25,60,300,245,TEAL,dash=True);self.t(175,105,a[0],23,TEAL,'middle');self.box(55,155,240,90,a[1])
            self.line(435,280,1060,280);self.line(435,280,435,40);self.t(400,35,'P(t)',23);self.t(1060,320,'t',23)
            self.parts.append('<path d="M490,280 L490,220 L590,150 L690,185 L790,100 L890,135 L1000,200 L1000,280 Z" fill="#244747"/>')
            self.path([(490,220),(590,150),(690,185),(790,100),(890,135),(1000,200)],TEAL,False)
            self.t(745,60,'E = ∫ P(t) dt',26,AMBER,'middle');self.t(745,260,a[2],22,INK,'middle');self.t(490,320,'t₀',23);self.t(1000,320,'t₁',23)
        else:raise ValueError(k)
        title=self.spec['title_cn' if self.zh else 'title']
        desc=self.spec['caption_cn' if self.zh else 'caption']+' '+' '.join(unescape(x) for x in re.findall(r'<text[^>]*>(.*?)</text>',''.join(self.parts)))
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 350" role="img" aria-labelledby="{self.slug}-title {self.slug}-desc" style="display:block;width:100%;height:auto"><title id="{self.slug}-title">{escape(title)}</title><desc id="{self.slug}-desc">{escape(desc)}</desc><defs><marker id="{self.slug}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="{MUTED}"/></marker></defs><rect width="1100" height="350" fill="{BG}"/><g font-family="system-ui,sans-serif">'+''.join(self.parts)+'</g></svg>'


def apply():
    data=json.loads((HERE/'visual_specs.json').read_text());manifest=json.loads((HERE/'manifest.json').read_text())
    for spec in data['figures']:
        for zh in (False,True):
            svg=Drawing(spec,zh).render();assets=HERE/('zh' if zh else '');assets.mkdir(exist_ok=True)
            (assets/(spec['slug']+'.svg')).write_text(svg+'\n')
            path=SLIDES/('zh' if zh else '')/spec['deck'];text=path.read_text()
            sections=list(re.finditer(r'<section class="slide".*?</section>',text,re.S))
            anchor=spec['title_cn' if zh else 'title']
            target=next((m for m in sections if '<h2>'+anchor+'</h2>' in m.group()),None)
            if not target:raise ValueError('missing anchor '+anchor)
            old=target.group();note=re.search(r'<aside class="notes">(.*?)</aside>',old,re.S)
            notes=note.group(1) if note else ''
            if 'data-illustration=' not in old:
                original=re.search(r'<div class="content">(.*?)</div><div class="cite">',old,re.S).group(1)
                # Retain both detailed original content and any previously authored notes.
                original_cite=re.search(r'<div class="cite">(.*?)</div>',old,re.S).group(1)
                notes='<div data-original-explanation="true">'+original+'</div><p data-original-sources="true">'+original_cite+'</p>'+notes
            src=data['sources'][spec['source']];url=src['url'];label=src['label']
            if 'data-visual-reading' not in notes:
                prompt='读图：先辨认数据、状态和参与者，再沿箭头解释依赖或回收边界。预测一个失效条件，再用原页解释核对。' if zh else 'Read the data, states and actors first, then trace dependencies or reclamation boundaries. Predict one failure condition and check the retained explanation.'
                notes+=f'<p data-visual-reading="true">{prompt}</p><p><a href="{escape(url,quote=True)}">{escape(label)}</a></p>'
            notes=re.sub(r'(<p data-visual-reading="true">.*?</p>)<p><a href="[^"]+">.*?</a></p>',lambda m:m.group(1)+f'<p><a href="{escape(url,quote=True)}">{escape(label)}</a></p>',notes,flags=re.S)
            caption=spec['caption_cn' if zh else 'caption']
            index_url=('../../' if zh else '../')+'references/figure_sources.html'
            cite=f'<a href="{escape(url,quote=True)}">'+('原创示意图，参考：' if zh else 'Original schematic, reference: ')+escape(label)+f'</a> · <a href="{index_url}">'+('图源与适用范围' if zh else 'Figure sources and scope')+'</a>'
            new=f'<section class="slide" data-illustration="{spec["slug"]}"><h2>{escape(anchor)}</h2><div class="content">{svg}<p class="note">{escape(caption)}</p></div><div class="cite">{cite}</div><div class="num"></div><aside class="notes">{notes}</aside></section>'
            text=text[:target.start()]+new+text[target.end():];path.write_text(text)
        manifest.append({k:spec[k] for k in ['prefix','slug','title','caption','deck','source']}|{'after':spec['title'],'asset':spec['slug']+'.svg','mode':'in-place'})
    # Export localized legacy diagrams too, and keep page numbering unchanged.
    for path in sorted(SLIDES.glob('*.html'))+sorted((SLIDES/'zh').glob('*.html')):
        text=path.read_text();sections=re.findall(r'<section class="slide".*?</section>',text,re.S);total=len(sections)
        for i,section in enumerate(sections,1):
            replacement=re.sub(r'<section class="slide"(?: id="[^"]*")?',f'<section class="slide" id="s{i}"',section,count=1)
            replacement=re.sub(r'<div class="num">.*?</div>',f'<div class="num">{i} / {total}</div>',replacement,count=1)
            text=text.replace(section,replacement,1)
            if path.parent.name=='zh' and 'data-illustration=' in section:
                slug=re.search('data-illustration="([^"]+)"',section).group(1);svg=re.search(r'<svg.*?</svg>',section,re.S).group()
                (HERE/'zh'/(slug+'.svg')).write_text(svg+'\n')
        path.write_text(text)
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Updated {len(data["figures"])} bilingual mechanism pages in place')
