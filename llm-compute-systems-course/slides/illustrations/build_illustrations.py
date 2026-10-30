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

FIGURES=[]
def figure(prefix,after,slug,title,caption,description):
    s=SVG(slug,title,description);FIGURES.append(dict(prefix=prefix,after=after,slug=slug,title=title,caption=caption,svg=s));return s

def build():
    FIGURES.clear()
    s=figure('00','The execution stack','execution-stack','Where the work runs','Choose the layer from evidence: faster kernels cannot remove an exposed host or network wait.','Framework and compiler feed library or custom kernels; execution depends on CPU, GPU memory and network.')
    s.chain(['Framework','Graph / compiler','Library or kernel','GPU execution'],65)
    s.box(30,225,240,64,'CPU / OS / input',AMBER);s.box(430,225,240,64,'HBM / cache',TEAL);s.box(830,225,240,64,'Network / peers',AMBER)
    s.path([(150,225),(150,175),(150,129)],AMBER);s.path([(550,225),(550,175),(940,175),(940,129)],TEAL);s.line(940,225,940,129,AMBER,True)
    s=figure('01','A roofline lower bound','roofline','Compute and bandwidth ceilings','Schematic roofline: changing shape or reuse moves a workload; peak rates do not establish achieved performance.','A bandwidth slope meets a compute ceiling; decode-like low intensity and prefill-like higher intensity workloads occupy different regions.')
    s.line(100,270,1040,270,arrow=True);s.line(100,270,100,30,arrow=True)
    s.path([(110,260),(570,75),(1010,75)],TEAL,False);s.text(660,58,'compute ceiling',24,TEAL);s.text(260,145,'bandwidth ceiling',22,TEAL)
    s.text(170,245,'decode-like',23,AMBER);s.text(650,180,'prefill-like',23,BLUE)
    s.text(530,318,'Arithmetic intensity →',24);s.text(115,35,'Achievable rate',23);s.text(770,240,'more reuse →',22,MUTED)
    s=figure('02','MoE dataflow','moe-primer','Active work and resident experts','A token activates selected experts; unused experts can still occupy memory or require sharded placement.','Two tokens route to selected experts; one expert is idle, illustrating that active work differs from resident parameters.')
    s.box(30,50,150,60,'Token A',TEAL);s.box(30,225,150,60,'Token B',BLUE)
    for i,label in enumerate(['Expert 0','Expert 1','Expert 2','Expert 3']):s.box(360+i*175,120,155,65,label,GRAY if i==3 else TEAL)
    s.path([(180,80),(260,80),(260,140),(360,140)],TEAL);s.path([(180,100),(290,100),(290,70),(785,70),(785,120)],TEAL)
    s.path([(180,245),(612,245),(612,185)],BLUE);s.path([(180,265),(785,265),(785,185)],BLUE)
    s.text(460,310,'All expert weights still need a placement strategy',23,MUTED)
    s=figure('03','Host and device clocks','timing-boundaries','Submission is not completion','Host submission timing can end while GPU work is still running. Synchronize at the boundary appropriate to the question.','Host launch bars end before device compute bars; an end synchronization aligns wall timing with completed work.')
    s.text(30,85,'Host',24);s.text(30,190,'Device',24)
    for x in [190,300,410]:s.box(x,50,85,50,'launch',BLUE)
    for j,x in enumerate([230,440,650]):s.box(x,155,190,65,'kernel '+str(j),TEAL)
    s.line(180,285,1010,285,arrow=True);s.text(220,320,'time →',22)
    s.line(520,35,520,245,AMBER,False,True);s.text(450,265,'host stops',20,AMBER)
    s.line(880,35,880,245,AMBER,False,True);s.text(860,265,'device done',20,AMBER)
    s=figure('04','The data path','host-data-path','Feed the device without losing ownership','Pipeline throughput is limited by the slowest stage; pinning or asynchronous APIs alone do not guarantee overlap.','Storage and CPU preprocessing feed pinned host memory, DMA and device memory; each boundary has a lifetime and capacity constraint.')
    s.chain(['Storage','CPU decode','Pinned buffer','DMA / copy','GPU / HBM'],110,[GRAY,AMBER,AMBER,BLUE,TEAL])
    s.text(90,65,'read / faults',21,MUTED);s.text(390,65,'bounded queue',21,MUTED);s.text(770,65,'copy completion',21,MUTED)
    s.text(160,260,'Do not overwrite a source buffer before its consumer finishes.',25,AMBER)
    s=figure('05','Autograd records dependencies','autograd-graph','Saved values connect forward and backward','Backward needs the declared saved state. Fusion, views and in-place changes must preserve the autograd contract.','Forward x and W produce y and loss; backward uses saved x and W to compute input and weight gradients.')
    s.box(30,50,170,60,'x, W',BLUE);s.box(350,50,170,60,'y = xW',TEAL);s.box(700,50,170,60,'loss',TEAL)
    s.line(200,80,350,80,arrow=True);s.line(520,80,700,80,arrow=True)
    s.box(350,220,300,65,'dX = dY Wᵀ | dW = Xᵀ dY',BLUE);s.box(750,220,160,65,'dY',AMBER)
    s.path([(785,110),(785,220)],AMBER);s.line(750,252,650,252,AMBER,True)
    s.path([(110,110),(110,252),(350,252)],BLUE);s.text(45,180,'saved state',21,BLUE)
    s=figure('06','A conceptual compiler path','compiler-path','Compilation chooses execution paths','Generated kernels and library calls are alternative implementation choices. Shape guards and graph breaks affect the cost.','Dynamo capture and guards lead to FX and Inductor, which can choose library calls or generated kernels; a guard miss returns to capture.')
    s.chain(['Capture / guards','FX graph','Inductor'],60)
    s.box(620,235,190,60,'Library call',BLUE);s.box(860,235,210,60,'Generated kernel',TEAL)
    s.path([(900,124),(900,180),(715,180),(715,235)],BLUE);s.path([(930,124),(930,180),(965,180),(965,235)],TEAL)
    s.path([(210,60),(210,25),(30,25),(30,60)],AMBER);s.text(270,28,'guard miss: capture / specialize again',21,AMBER)
    s=figure('07','Launch hierarchy','cuda-hierarchy','Grid, blocks and storage scopes','A block owns shared memory; registers belong to threads. Cross-block synchronization needs an explicit valid mechanism.','A grid contains blocks, each with warps and per-block shared memory; global HBM is accessible across blocks.')
    for j in range(2):
        x=30+j*530;s.rect(x,30,500,220,stroke=BLUE);s.text(x+20,62,'Block '+str(j),24,BLUE)
        for i in range(4):s.box(x+20+i*117,85,105,55,'warp '+str(i),TEAL)
        s.box(x+20,170,460,55,'shared memory: this block only',AMBER)
    s.box(300,280,500,48,'Global memory / HBM',BLUE)
    s.line(280,250,420,280,arrow=True);s.line(810,250,680,280,arrow=True)
    s=figure('08','Warp-level reductions','reduction-tree','A reduction is a dependency tree','Parallel order changes rounding. Masks, synchronization and participation rules remain part of correctness.','Eight partial values combine in pairs then into one result; every stage depends on the previous partial sums.')
    widths=[8,4,2,1];last=[]
    for level,n in enumerate(widths):
        y=20+level*77;positions=[(i+0.5)*1040/n+30 for i in range(n)]
        for i,x in enumerate(positions):
            s.box(x-45,y,90,42,('x'+str(i)) if level==0 else 'sum',TEAL if level==3 else BLUE)
            if level>0:
                for old in last[i*2:i*2+2]:s.path([(old,y-35),(old,y-17),(x,y-17),(x,y)],MUTED)
        last=positions
    s=figure('09','A shared-memory tile','gemm-tiles','Reuse tiles before returning to HBM','The highlighted K slice contributes to one output tile. Reuse improves intensity but consumes registers and shared memory.','A row tile of A and column tile of B share a K slice and accumulate into an output tile C.')
    s.text(70,35,'A: M × K',25);s.grid(70,60,6,8,28,lambda i,j:2<=i<4 and 2<=j<4)
    s.text(420,35,'B: K × N',25);s.grid(420,60,8,6,28,lambda i,j:2<=i<4 and 2<=j<4)
    s.text(800,35,'C: M × N',25);s.grid(800,60,6,6,28,lambda i,j:2<=i<4 and 2<=j<4)
    s.line(310,140,400,140,arrow=True);s.text(340,125,'×',25);s.line(620,140,780,140,arrow=True);s.text(650,120,'accumulate',20)
    s.text(90,315,'Stream K tiles; retain the output accumulator across the K loop.',24,MUTED)
    s=figure('10','Double buffering','double-buffer','Overlap movement and matrix work','The next tile must be ready before consumption, and a buffer cannot be overwritten while compute still reads it.','Two buffers alternate: copy tile one while computing tile zero, then reuse the released buffer for tile two.')
    for y,label in [(35,'Copy'),(135,'Compute')]:s.text(30,y+35,label,23)
    for i,x in enumerate([200,360,560]):s.box(x,35,160,58,'tile '+str(i)+' → '+('A' if i%2==0 else 'B'),BLUE)
    for i,x in enumerate([360,560,760]):s.box(x,135,200,58,'tile '+str(i)+' from '+('A' if i%2==0 else 'B'),TEAL)
    s.text(180,270,'Fill',23,AMBER);s.text(460,270,'Steady overlap',23,TEAL);s.text(900,270,'Drain',23,AMBER)
    s.line(170,310,1060,310,arrow=True)
    s=figure('11','Tiled execution','online-attention','Stream K/V and retain online row state','Running maximum, denominator and weighted sum are rescaled together; dense attention arithmetic remains quadratic.','A query tile consumes multiple key/value tiles while updating m, l and u, then emits a normalized output without materializing all scores.')
    s.box(30,110,160,70,'Query tile',BLUE)
    for i in range(3):s.box(300+i*230,30,185,55,'K/V tile '+str(i),BLUE)
    s.box(320,160,500,80,'running m, l, u | rescale + merge',TEAL)
    s.line(190,145,320,195,arrow=True)
    for i in range(3):s.line(392+i*230,85,392+i*150,160,arrow=True)
    s.box(890,170,180,60,'O = u / l',TEAL);s.line(820,200,890,200,arrow=True)
    s.text(305,305,'No full S × S score/probability array in this fused dataflow',23,MUTED)
    s=figure('12','A balanced pipeline','attention-pipeline','A faster matrix stage can expose softmax','Hardware co-design balances movement, matrix operations and non-matmul work. Diagram lengths are conceptual.','A serial matrix-plus-softmax path is compared with overlapped stages; non-matmul work remains on the critical path.')
    s.text(30,55,'Serial',24);s.box(200,25,330,55,'matrix',BLUE);s.box(540,25,220,55,'softmax',AMBER);s.box(770,25,280,55,'matrix',BLUE)
    s.text(30,145,'Pipeline',24)
    s.box(200,115,290,55,'matrix tile 0',BLUE);s.box(500,115,290,55,'matrix tile 1',BLUE)
    s.box(350,200,220,55,'softmax tile 0',AMBER);s.box(650,200,220,55,'softmax tile 1',AMBER)
    s.line(190,300,1060,300,arrow=True);s.text(550,328,'conceptual time →',21,MUTED)
    s=figure('13','Worked byte budget','quantized-bytes','Packing saves bytes; scales consume some','For 1,024 weights, group size 32 and FP32 scales, int4 uses 512 + 128 = 640 array bytes before headers and alignment.','A BF16 array of 2048 bytes becomes 512 bytes of packed int4 values plus 128 bytes of group scales, a 3.2-fold raw-array reduction.')
    s.text(30,45,'BF16 baseline: 2,048 B',24,BLUE);s.rect(30,70,1024,65,fill='#24456a',stroke=BLUE)
    s.text(30,190,'Packed int4',24,TEAL);s.rect(30,215,512,65,fill='#25564d',stroke=TEAL);s.text(286,255,'512 B',25,TEAL,'middle')
    s.rect(560,215,128,65,fill='#5f4b2a',stroke=AMBER);s.text(720,252,'128 B FP32 scales',24,AMBER)
    s.text(650,115,'3.2× raw-array reduction',24)
    s=figure('14','Post-training quantization','precision-lifecycle','Different precision paths need different evidence','AMP, training recipes and post-training quantization operate at different points. Conversion alone does not prove quality or speed.','A training path includes mixed-precision forward/backward and state, while a serving path calibrates and quantizes a frozen model then evaluates it.')
    s.text(30,40,'Training',24,BLUE);s.chain(['Initialize','AMP / TE recipe','Loss + backward','Optimizer state'],65)
    s.text(30,205,'Serving',24,TEAL);s.chain(['Freeze weights','Calibrate','Quantize / pack','Held-out evaluation'],230,[TEAL]*4)
    s=figure('25','Fully sharded dataflow','fsdp-shards','Shards become transient full parameters','Persistent sharding reduces one budget; gathered layers, activations and prefetch buffers still contribute to peak memory.','Four ranks hold different parameter shards. All-gather materializes a layer for compute; reduce-scatter partitions reduced gradients afterward.')
    for i in range(4):s.box(30+i*265,25,235,60,'Rank '+str(i)+': shard '+str(i),BLUE)
    s.box(260,145,580,65,'all-gather → full layer → compute',TEAL)
    for i in range(4):s.line(147+i*265,85,330+i*150,145,arrow=True)
    s.box(270,270,560,55,'reduce-scatter → gradient shards',AMBER);s.line(550,210,550,270,arrow=True)
    s=figure('26','All-reduce decomposition','collective-ring','Collective semantics and physical traffic','Reduce-scatter followed by all-gather produces an all-reduce result; the library can select another physical algorithm.','Four ranks form a conceptual communication ring, with reduce-scatter and all-gather identified as two phases.')
    for x,y,label in [(100,30,'rank 0'),(750,30,'rank 1'),(750,230,'rank 2'),(100,230,'rank 3')]:s.box(x,y,230,60,label,BLUE)
    s.line(330,60,750,60,arrow=True);s.line(865,90,865,230,arrow=True);s.line(750,260,330,260,arrow=True);s.line(215,230,215,90,arrow=True)
    s.text(550,145,'1. reduce-scatter',25,TEAL,'middle');s.text(550,190,'2. all-gather',25,AMBER,'middle')
    s=figure('27','An explicit rank mesh','parallel-mesh','A concrete eight-rank layout','DP=2, TP=2, PP=2, CP=1 gives eight ranks. This drawing does not establish expert-group legality or physical placement.','Two rows are data-parallel replicas; pairs of columns belong to different pipeline stages and each pair contains two tensor shards.')
    s.text(275,35,'Pipeline stage 0',24,TEAL);s.text(700,35,'Pipeline stage 1',24,AMBER)
    for row in range(2):
        s.text(20,120+row*120,'DP '+str(row),23,BLUE)
        for col in range(4):s.box(160+col*225,65+row*120,190,70,'rank '+str(row*4+col)+' | TP '+str(col%2),TEAL if col<2 else AMBER)
    s.text(270,325,'Logical groups require a separate topology/communication mapping.',22,MUTED)
    s=figure('28','TorchTitan entry chain','source-walk','Trace the decision through code boundaries','Pin source revisions, follow inputs and constraints, then distinguish a documented path from an observed runtime dispatch.','Configuration feeds rank-mesh construction, model transformations, compilation and the training step. Checkpoint state connects model, optimizer and data.')
    s.chain(['Config','Rank mesh','Transform model','Compile','Train step'],65)
    s.box(230,235,640,60,'Checkpoint: model + optimizer + RNG + data',AMBER)
    s.path([(550,129),(550,235)],AMBER);s.text(260,195,'source path ≠ measured kernel dispatch',23,MUTED)
    s=figure('19','Worked routing budget','expert-skew','The hottest expert can determine the tail','Both cases contain 34 assignments. Skew changes expert shapes and balance, even though the average assignment count is unchanged.','Uniform expert loads of 9,9,8,8 are compared with skewed loads of 17,17,0,0 using a common scale.')
    for panel,(label,counts) in enumerate([('Balanced',[9,9,8,8]),('Skewed',[17,17,0,0])]):
        base=30+panel*550;s.text(base,35,label,25)
        for i,n in enumerate(counts):
            x=base+40+i*112;h=n*10
            if h:s.rect(x,250-h,65,h,fill='#25564d' if panel==0 else '#5f4b2a',stroke=TEAL if panel==0 else AMBER)
            s.text(x+32,275,'E'+str(i),22,MUTED,'middle');s.text(x+32,240-h,str(n),22,INK,'middle')
        s.line(base+10,250,base+490,250)
    s.text(355,325,'34 token–expert assignments in each case',23,MUTED)
    s=figure('29','Recovery validation','restart-state','Recover the same training trajectory','The local CPU lab checks losses, parameters and optimizer tensors; omitting RNG restoration is a deliberate failing control.','An uninterrupted six-step run is compared with a run saved after three steps and resumed after restoring model, optimizer, RNG and data cursor.')
    s.text(30,40,'Uninterrupted',23)
    for i in range(6):s.box(210+i*140,10,120,55,'step '+str(i+1),TEAL)
    s.text(30,145,'Interrupted',23)
    for i in range(3):s.box(210+i*140,115,120,55,'step '+str(i+1),BLUE)
    s.box(210,245,500,65,'Restore model / optimizer / RNG / cursor',AMBER)
    s.path([(570,170),(570,210),(460,210),(460,245)],AMBER)
    s.path([(710,277),(740,277),(740,205),(690,205),(690,170)],AMBER)
    for i in range(3,6):s.box(210+i*140,115,120,55,'step '+str(i+1),TEAL)
    s=figure('15','Shared prefix ownership','shared-blocks','One cancellation must preserve the other owner','Logical sequences can share compatible prefix blocks. References and in-flight work determine when storage is reclaimable.','Two requests point to the same two prefix blocks and distinct suffix blocks; the shared blocks have reference count two.')
    s.box(30,50,180,60,'Request A',BLUE);s.box(30,230,180,60,'Request B',TEAL)
    for i,(label,color) in enumerate([('Prefix 0',AMBER),('Prefix 1',AMBER),('Suffix A',BLUE),('Suffix B',TEAL)]):s.box(370+i*175,130,155,65,label,color)
    s.path([(210,80),(300,80),(300,155),(370,155)],BLUE);s.path([(210,260),(320,260),(320,175),(370,175)],TEAL)
    s.line(525,162,545,162,arrow=True);s.path([(620,130),(620,90),(795,90),(795,130)],BLUE);s.path([(620,195),(620,250),(970,250),(970,195)],TEAL)
    s.text(385,305,'Prefix references: 2 → 1 when one request cancels',23,AMBER)
    s=figure('16','Causal history matters','prefix-branches','A matching suffix is not a matching history','Identity includes the compatible computation and allowed sharing domain. A semantic answer cache has a different validity rule.','A radix-style prefix branches into two suffixes; a changed early token begins another history rather than reusing a matching suffix.')
    s.box(30,120,180,65,'Identity + root',AMBER);s.box(350,120,210,65,'Shared prefix A B',TEAL)
    s.line(210,152,350,152,arrow=True);s.box(790,35,270,60,'A B → C D',BLUE);s.box(790,225,270,60,'A B → E F',BLUE)
    s.path([(560,152),(670,152),(670,65),(790,65)],BLUE);s.path([(560,152),(700,152),(700,255),(790,255)],BLUE)
    s.text(330,310,'Change A → new history; equal later text is insufficient.',23,AMBER)
    s=figure('17','Chunked prefill','chunked-prefill','Bound interruptions to existing decode','A smaller prefill chunk creates more decode opportunities but can add overhead and postpone prompt completion. Bars are schematic.','An unbounded prefill blocks several decode opportunities, while chunked prefill alternates shorter prefill and decode segments.')
    s.text(30,60,'Unbounded',23);s.box(230,25,650,65,'long prefill',BLUE);s.box(895,25,150,65,'decode',TEAL)
    s.text(30,195,'Chunked',23)
    for i in range(4):s.box(230+i*205,160,125,65,'prefill',BLUE);s.box(365+i*205,160,60,65,'D',TEAL)
    s.line(220,290,1060,290,arrow=True);s.text(575,328,'conceptual time →',22,MUTED)
    s=figure('18','Equal hardware comparison','pd-placement','Compare the same total resources','Two colocated workers and one prefill plus one decode worker each use two devices. State movement and queueing can change the result.','A two-device colocated baseline is compared with a two-device split prefill/decode design joined by a KV transfer path.')
    s.text(30,40,'Colocated baseline: 2 devices',24,BLUE);s.box(80,70,340,65,'GPU 0: prefill + decode',BLUE);s.box(660,70,340,65,'GPU 1: prefill + decode',BLUE)
    s.text(30,205,'Split alternative: 2 devices',24,TEAL);s.box(80,240,340,65,'GPU 0: prefill',TEAL);s.box(660,240,340,65,'GPU 1: decode',TEAL)
    s.line(420,272,660,272,AMBER,True);s.text(465,252,'KV transfer',21,AMBER)



    if list(SLIDES.glob('25_*.html')):
        s=figure('20','Draft, verify, correct','speculative-correction','Verify the draft before keeping its state','Keep the accepted prefix; correct the first rejected token and discard later speculative state. The lab validates one categorical step.','Draft tokens are verified by the target; an accepted prefix is retained and the first rejected position is sampled from a corrected distribution.')
        s.chain(['Draft q','Target scores p','Accept prefix','Correct / bonus'],65)
        for i,(label,color) in enumerate([('keep',TEAL),('keep',TEAL),('reject',AMBER),('discard',GRAY)]):s.box(170+i*205,235,175,60,label,color)
        s.text(200,195,'Accepted history ends at the first rejection',24,AMBER)
    if list(SLIDES.glob('26_*.html')):
        s=figure('21','The retained-state decision','agent-wait','State has a lifetime while the agent waits','Compare resume work and displaced capacity across a complete task; idle byte-seconds alone do not select a policy.','A model call enters a tool wait. During the wait a policy retains HBM state, moves compatible state, or evicts it for recomputation on resume.')
        s.box(20,130,180,65,'Tool wait',BLUE)
        for y,label,color in [(20,'Retain in HBM',TEAL),(130,'Offload / restore',BLUE),(240,'Evict / recompute',AMBER)]:
            s.box(340,y,330,65,label,color);s.path([(200,162),(260,162),(260,y+32),(340,y+32)],color);s.path([(670,y+32),(750,y+32),(750,162),(830,162)],color)
        s.box(830,130,240,65,'Validate + resume',AMBER)


    if list(SLIDES.glob('27_*.html')):
        s=figure('22','The deployment boundary','aoti-artifact','The producer and loader meet at an artifact','Record input, target and runtime contracts. Exported graph archives and compiled packages can share a suffix but need different loaders.','A producer exports a contracted graph and compiles a native package; a separate runtime loader validates inputs and executes the artifact.')
        s.chain(['Model + contract','Export graph','Compile package','Checked loader'],70)
        s.box(320,245,450,60,'Build / load / first / warm are separate',AMBER)
    if list(SLIDES.glob('28_*.html')):
        s=figure('23','Delegation is graph partitioning','executorch-partitions','Delegation does not remove boundary costs','Count conversion, transfer and fallback between graph partitions. The validated lab uses one unquantized CPU XNNPACK segment.','An exported graph contains two supported regions with an unsupported region between; delegate and fallback transitions may require conversion or transfer.')
        s.chain(['Supported region','Fallback region','Supported region'],45,[TEAL,AMBER,TEAL])
        s.box(60,235,300,60,'Target delegate',TEAL);s.box(400,235,300,60,'Eligible CPU kernels',AMBER);s.box(740,235,300,60,'Target delegate',TEAL)
        for x in [210,550,890]:s.line(x,109,x,235,arrow=True)


    if list(SLIDES.glob('29_*.html')):
        s=figure('24','Three communication boundaries','rubin-data-paths','CPU, scale-up and scale-out are different paths','Conceptual traffic boundaries, not a complete board/rack layout. Local HBM, coherent links and network transfers have different costs.','Vera CPU connects to two Rubin GPUs with local HBM; GPU scale-up uses NVLink while external communication uses NIC/fabric paths.')
        s.box(20,130,215,70,'Vera CPU',AMBER)
        for y in [35,235]:s.box(350,y,260,65,'Rubin GPU|local HBM4',TEAL);s.path([(235,165),(290,165),(290,y+32),(350,y+32)],AMBER)
        s.box(800,35,275,65,'NVLink 6|scale-up peers',BLUE);s.box(800,235,275,65,'NIC / DPU|scale-out fabric',BLUE)
        s.line(610,67,800,67,BLUE,True);s.line(610,267,800,267,BLUE,True)
        s.path([(610,267),(675,267),(675,67),(800,67)],BLUE);s.path([(610,67),(720,67),(720,267),(800,267)],MUTED)
        s.text(30,285,'CPU–GPU: C2C',22,AMBER)
    if list(SLIDES.glob('30_*.html')):
        s=figure('30','Worked 2× kernel example','amdahl-ledger','A 2× component improvement becomes 1.43× overall','Constructed disjoint phase example: 60 s compute + 40 s other becomes 30 + 40 = 70 s for the same useful work.','Two proportional stacked time bars show 100 seconds before and 70 seconds after halving only a 60-second compute component.')
        s.text(20,65,'Before',24);s.box(200,30,480,70,'compute 60 s',BLUE);s.box(680,30,320,70,'other 40 s',AMBER)
        s.text(20,225,'After',24);s.box(200,190,240,70,'compute 30 s',BLUE);s.box(440,190,320,70,'other 40 s',AMBER)
        s.text(250,325,'Same useful tokens; nonoverlapping phase costs; no vendor benchmark',22,MUTED)


    if list(SLIDES.glob('31_*.html')):
        s=figure('31','Proposal to evidence','capstone-evidence','The claim must survive every evidence gate','Keep raw outcomes, adverse cases and rejected candidates. A well-supported rejection is a useful engineering result.','A workload hypothesis passes through a credible baseline, correctness checks and measurement before a retain, revise or reject decision.')
        s.chain(['Hypothesis','Baseline','Correctness','Measure','Decision'],75)
        s.box(200,245,700,60,'Raw records + ablation + adverse / held-out cases',AMBER)
        s.line(550,139,550,245,AMBER,True)

    if list(SLIDES.glob('32_*.html')):
        s=figure('32','Three implementation choices','optimization-choices','Choose an implementation, then validate it','Libraries, graph compilation and custom kernels are alternative or composable paths; the target and numerical contract decide eligibility.','An exported workload branches to a library, graph compiler or custom kernel path, then all candidates converge on the same correctness and measurement gate.')
        s.box(20,135,180,65,'Workload|contract',AMBER)
        for y,label,color in [(20,'Existing library|cuBLAS / cuDNN',BLUE),(135,'Graph compiler|Inductor / TensorRT',TEAL),(250,'Custom kernel|Triton / TileLang / CUDA',BLUE)]:
            s.box(330,y,350,65,label,color)
            s.path([(200,167),(265,167),(265,y+32),(330,y+32)],color)
            s.path([(680,y+32),(745,y+32),(745,167),(805,167)],color)
        s.box(805,135,275,65,'Build / validate|measure / retain',AMBER)

    if list(SLIDES.glob('33_*.html')):
        s=figure('33','Layers: graph compilers, kernel DSLs, vendor targets','programming-model-layers','Graph compilers, kernel DSLs and vendor targets','Arrows show kernel-DSL paths only; graph compilers also call vendor libraries and emit their own code. Dashed: an opaque custom call.','TorchInductor generates Triton; XLA reaches Triton and TileLang kernels through custom calls and Pallas through JAX; Triton targets NVIDIA and AMD, TileLang targets NVIDIA, AMD and Apple Metal, Pallas targets TPU and NVIDIA, cuTile targets NVIDIA.')
        s.box(180,20,330,58,'TorchInductor',TEAL);s.box(600,20,330,58,'XLA (StableHLO in)',TEAL)
        xs=[30,300,570,840]
        for x,label in zip(xs,['Triton','TileLang','Pallas','cuTile']):s.box(x,140,230,58,label,BLUE)
        for x,label in zip(xs,['NVIDIA (PTX)','AMD (AMDGCN)','Apple Metal','TPU (Mosaic)']):s.box(x,262,230,58,label,AMBER)
        s.line(260,78,150,140,TEAL,True)
        s.line(700,78,180,140,MUTED,True,True);s.line(740,78,420,140,MUTED,True,True);s.line(800,78,685,140,TEAL,True)
        for a,b in [(0,0),(0,1),(1,0),(1,1),(1,2),(2,3),(2,0),(3,0)]:
            s.line(xs[a]+40+b*50,198,xs[b]+40+a*50,262,BLUE,True)

    if list(SLIDES.glob('34_*.html')):
        s=figure('34','Lowering: a model becomes an artifact','lowering-pipeline','Partition, then library or generated kernels','Anchor operators usually reach a library or BYOC code generator; the remaining operators are generated. Both end in one runtime artifact.','A framework graph becomes a graph IR, is partitioned and fused, then anchor operators go to library or vendor kernels and the rest to generated kernels; both feed a runtime artifact such as an IREE module or TVM library.')
        s.chain(['Framework graph','Graph IR','Partition + fuse'],40)
        s.box(150,170,360,64,'Anchor ops: library / BYOC',BLUE);s.box(590,170,360,64,'Long tail: generated kernels',TEAL)
        s.line(950,104,330,170,MUTED,True);s.line(960,104,770,170,MUTED,True)
        s.box(240,272,620,58,'Runtime artifact: host schedule + device binaries',AMBER)
        s.line(330,234,500,272,BLUE,True);s.line(770,234,600,272,TEAL,True)
        s=figure('35','Two design points, one request path','engine-design-points','One API, two families of engines','Clients can keep one OpenAI-style API; behind it, local engines optimize one device at batch size near 1 and serving engines optimize many requests per GPU.','An OpenAI-style HTTP API sits above two families: local engines (llama.cpp, Ollama, MLX) on CPU, Metal and consumer GPUs, and serving engines (vLLM, SGLang, TensorRT-LLM) on data-center GPUs with paged KV and continuous batching.')
        s.box(300,20,500,58,'OpenAI-style HTTP API',AMBER)
        s.box(40,130,470,64,'Local: llama.cpp · Ollama · MLX',TEAL);s.box(590,130,470,64,'Serving: vLLM · SGLang · TensorRT-LLM',BLUE)
        s.line(450,78,275,130,MUTED,True);s.line(650,78,825,130,MUTED,True)
        s.box(40,250,470,64,'GGUF / MLX quants · offload · batch ≈ 1',TEAL);s.box(590,250,470,64,'Paged KV · continuous batching · TP/EP',BLUE)
        s.line(275,194,275,250,TEAL,True);s.line(825,194,825,250,BLUE,True)


def apply():
    build();manifest=[]
    for entry in FIGURES:
        matches=list(SLIDES.glob(entry['prefix']+'_*.html'))
        if len(matches)!=1:raise ValueError('one deck required for '+entry['prefix'])
        path=matches[0];svg=entry['svg'].out();(HERE/(entry['slug']+'.svg')).write_text(svg)
        source=path.read_text()
        # Remove only previously generated illustration sections; keep every authored slide.
        source=re.sub(r'<section class="slide"[^>]* data-illustration="[^"]+".*?</section>','',source,flags=re.S)
        sections=list(re.finditer(r'<section class="slide".*?</section>',source,re.S))
        target=next((m for m in sections if '<h2>'+entry['after']+'</h2>' in m.group()),None)
        if target is None:raise ValueError('missing teaching anchor '+entry['after'])
        cite=re.search(r'<div class="cite">(.*?)</div>',target.group(),re.S).group(1)
        addition='<section class="slide" data-illustration="'+entry['slug']+'"><h2>'+escape(entry['title'])+'</h2><div class="content">'+svg+'<p class="note">'+escape(entry['caption'])+'</p></div><div class="cite">'+cite+' · Original conceptual figure; not measured performance</div><div class="num"></div></section>'
        source=source[:target.end()]+addition+source[target.end():]
        total=source.count('class="slide"');index=[0]
        def renumber(m):
            index[0]+=1;section=m.group()
            section=re.sub(r'<section class="slide"(?: id="[^"]*")?',f'<section class="slide" id="s{index[0]}"',section,count=1)
            return re.sub(r'<div class="num">.*?</div>',f'<div class="num">{index[0]} / {total}</div>',section,count=1)
        source=re.sub(r'<section class="slide".*?</section>',renumber,source,flags=re.S)
        path.write_text(source)
        manifest.append({k:v for k,v in entry.items() if k!='svg'}|{'deck':path.name,'asset':entry['slug']+'.svg'})
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Embedded {len(manifest)} original SVG figures in {len(manifest)} decks')
if __name__=='__main__':apply()
