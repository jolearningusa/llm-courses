"""Build attributed, editable mechanism diagrams and replace matching slides in place.

Python 3.9+, no external dependencies. Run from any working directory.
figure_specs.json records original source figures, scope and bilingual captions.
Existing text is retained in speaker notes. Generation is idempotent.
"""
from html import escape, unescape
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
SLIDES = HERE.parent
INK = '#eef4ff'; MUTED = '#b7c7df'; TEAL = '#8de0cf'; BLUE = '#8bbcff'
AMBER = '#f1c979'; RED = '#ff9e9e'; GRAY = '#50617a'; BG = '#101927'


class SVG:
    def __init__(self, spec, zh):
        self.spec = spec; self.zh = zh; self.slug = spec['slug']; self.parts = []

    def l(self, en, zh):
        return zh if self.zh else en

    def text(self, x, y, value, size=23, color=INK, anchor='start'):
        for i, line in enumerate(value.split('|')):
            self.parts.append(f'<text x="{x}" y="{y+i*29}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(line)}</text>')

    def rect(self, x, y, w, h, color=BLUE, fill=BG, dash=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{color}" stroke-width="2"'+(' stroke-dasharray="7 5"' if dash else '')+'/>')

    def box(self, x, y, w, h, value, color=BLUE):
        self.rect(x,y,w,h,color)
        self.text(x+w/2,y+h/2+8-(len(value.split('|'))-1)*14.5,value,23,color,'middle')

    def path(self, points, color=MUTED, arrow=True, dash=False):
        d=' '.join(('M' if j==0 else 'L')+f'{x},{y}' for j,(x,y) in enumerate(points))
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"'+(' stroke-dasharray="6 5"' if dash else '')+(f' marker-end="url(#{self.slug}-arrow)"' if arrow else '')+'/>')

    def line(self, x1,y1,x2,y2,color=MUTED,arrow=True,dash=False):
        self.path([(x1,y1),(x2,y2)],color,arrow,dash)

    def cross(self, x,y):
        self.line(x-9,y-9,x+9,y+9,RED,False); self.line(x-9,y+9,x+9,y-9,RED,False)

    def bar(self, x,y,w,label,color=TEAL):
        self.rect(x,y,w,36,color,fill='#192d3d');self.text(x+w/2,y+26,label,21,color,'middle')

    def out(self):
        title=self.spec['title_cn' if self.zh else 'title']
        desc=self.spec['caption_cn' if self.zh else 'caption'] + ' ' + ' '.join(unescape(t) for t in re.findall(r'<text[^>]*>(.*?)</text>', ''.join(self.parts)))
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 340" role="img" aria-labelledby="{self.slug}-title {self.slug}-desc" style="display:block;width:100%;height:auto"><title id="{self.slug}-title">{escape(title)}</title><desc id="{self.slug}-desc">{escape(desc)}</desc><defs><marker id="{self.slug}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="{MUTED}"/></marker></defs><rect width="1100" height="340" fill="{BG}"/><g font-family="system-ui,sans-serif">'+''.join(self.parts)+'</g></svg>'


def draw(s):
    l=s.l; k=s.spec['kind']
    if k=='stack':
        s.text(25,30,l('HOST','主机'),24,TEAL);s.text(25,270,l('DEVICE','设备'),24,TEAL)
        for y,label in [(15,l('Framework / compiler','框架 / 编译器')),(92,l('Runtime: buffers + submissions','Runtime：缓冲区与提交')),(169,l('Driver interface','驱动接口')),(246,l('Firmware / execution engine','固件 / 执行引擎'))]:
            s.box(265,y,550,57,label)
        for y in [72,149,226]: s.line(425,y,425,y+20);s.line(670,y+20,670,y,TEAL)
        s.text(115,133,l('Commands','命令'),22);s.text(850,133,l('Completion','完成通知'),22,TEAL)
        s.line(20,236,1080,236,GRAY,False,True)
    elif k=='spine':
        s.box(30,25,230,70,l('Descriptor|24 bytes','描述符|24 字节'))
        s.box(375,25,300,70,l('FIFO + one worker','FIFO + 单个执行单元'))
        s.line(260,60,375,60);s.box(800,25,260,70,l('Completion|observe + reap','完成记录|观察 + 收取'))
        s.line(675,60,800,60)
        for x,label,vals in [(110,l('Source','源缓冲区'),['0','254','255']),(715,l('Destination','目标缓冲区'),['1','255','0'])]:
            s.text(x,163,label,24,TEAL)
            for i,v in enumerate(vals):s.box(x+i*78,185,65,52,v)
        s.line(365,210,690,210);s.text(525,180,l('add-one, mod 256','加一，模 256'),23,AMBER,'middle')
        s.bar(110,280,840,l('Both buffers pinned until safe reap','两个缓冲区都固定到安全收取完成'))
    elif k=='scope':
        labels=[l('CPU model','CPU 模型'),l('Documentation','文档'),l('Target device','目标设备'),l('Instruments','仪器')]
        claims=[l('Packet / state / ownership','描述符 / 状态 / 所有权'),l('API obligations','API 职责与约束'),l('Named revision behavior','指定版本的设备行为'),l('Power / thermal boundary','指定功耗 / 热测量边界')]
        for i,(a,b) in enumerate(zip(labels,claims)):
            y=20+i*78;s.text(20,y+33,a,24,TEAL);s.line(245,y+25,340,y+25);s.text(375,y+33,b,24)
        s.line(1030,20,1030,315,GRAY,False);s.text(1070,52,l('No|scope|jump','不能|跨越|证据|范围'),21,AMBER,'middle')
    elif k=='independent':
        s.box(30,120,220,70,l('AI proposer','AI 提议者'));s.box(380,120,235,70,l('Candidate patch','候选修改'));s.line(250,155,380,155)
        s.box(760,120,290,70,l('Independent evaluator','独立验收器'),TEAL);s.line(615,155,760,155)
        s.box(760,10,290,60,l('Frozen contract + tests','冻结契约与测试'),AMBER);s.line(905,70,905,120,AMBER)
        s.line(905,190,905,245,TEAL);s.text(905,280,l('Accept / reject + raw logs','接受 / 拒绝 + 原始日志'),24,TEAL,'middle')
        s.path([(490,120),(490,40),(735,40)],RED,False,True);s.cross(630,40)
        s.text(380,315,l('Keep prompts, diffs and failed controls','保留提示词、修改与错误对照'),23)
    elif k=='contracts':
        for x,w,label in [(25,260,l('Tensor operation|copy / add-one','张量操作|复制 / 加一')),(420,260,l('Runtime resources|buffers + dependency','Runtime 资源|缓冲区 + 依赖')),(815,260,l('Device command|opcode + handles','设备命令|操作码 + 句柄'))]:s.box(x,50,w,90,label)
        s.line(285,95,420,95);s.line(680,95,815,95)
        for x,label in [(155,l('Meaning','语义')),(550,l('Lifetime','生命周期')),(945,l('Encoding','编码'))]:s.text(x,205,label,26,TEAL,'middle')
        s.line(25,245,1075,245,GRAY,False);s.text(550,297,l('Same intent, different contracts at each boundary','意图相同，各层契约不同'),25,AMBER,'middle')
    elif k=='addresses':
        s.box(20,35,220,70,l('CPU virtual|address X','CPU 虚拟|地址 X'))
        s.box(435,120,230,85,l('RAM buffer|physical Y','RAM 缓冲区|物理地址 Y'),TEAL)
        s.box(855,35,225,70,l('Device DMA|address Z','设备 DMA|地址 Z'))
        s.box(855,220,225,60,'IOMMU',AMBER)
        s.path([(240,70),(550,70),(550,120)]);s.text(310,48,l('CPU MMU','CPU MMU'),22)
        s.line(968,105,968,220);s.line(855,250,665,250);s.line(665,250,665,190)
        s.text(720,232,l('Z maps to Y','Z 映射到 Y'),21,AMBER,'middle')
        s.text(25,305,l('dma_map_* sets up the device view; IOMMU translation is platform-dependent.','dma_map_* 建立设备视图；是否经过 IOMMU 取决于平台。'),22)
    elif k=='packet':
        s.text(25,28,l('Byte offsets (little-endian)','字节偏移（小端序）'),23,TEAL)
        unit=40;x0=65
        fields=[(0,1,'ABI'),(1,1,'OP'),(2,2,'RSV'),(4,4,'generation'),(8,4,'src'),(12,4,'dst'),(16,4,'length'),(20,4,'cookie')]
        for off,width,label in fields:
            x=x0+off*unit;s.rect(x,95,width*unit,78,TEAL if off>=4 else BLUE)
            s.text(x,77,str(off),21,MUTED);s.text(x+width*unit/2,140,label,19 if width<4 else 23,INK,'middle')
        s.text(x0+24*unit,77,'24',21,MUTED)
        s.text(65,225,'<BBHIIIII',26,AMBER);s.text(400,225,l('1 + 1 + 2 + 5 × 4 = 24 bytes','1 + 1 + 2 + 5 × 4 = 24 字节'),24)
        s.text(65,290,l('ABI=1   OP=1/2   RSV=0   handles are model IDs','ABI=1   OP=1/2   RSV=0   句柄是模型 ID'),23)
    elif k=='admission':
        s.text(25,35,l('Packet arrives','描述符到达'),25,TEAL)
        for i,label in enumerate([l('Fields + bounds','字段与边界'),l('Generation + owners','代次与所有权'),l('Capacity','容量')]):
            x=25+i*365;s.box(x,85,300,65,label);s.line(x+150,150,x+150,205,RED)
            if i<2:s.line(x+300,117,x+365,117)
        s.line(175,205,905,205,RED,False);s.text(550,245,l('Any rejection: no pins, no slot mutation','任一拒绝：不固定、不改变槽位'),23,RED,'middle')
        s.path([(1055,117),(1080,117),(1080,290),(825,290)],TEAL)
        s.text(800,300,l('All pass: commit pins + slot','全通过：固定并占用槽位'),23,TEAL,'end')
    elif k=='ownership':
        xs=[190,350,510,670,830,990]
        for x,label in zip(xs,[l('Write','写入'),l('Submit','提交'),l('Run','执行'),l('Done','完成'),l('Observe','观察'),l('Reap','收取')]):s.text(x,38,label,22,INK,'middle');s.line(x,55,x,278,GRAY,False,True)
        s.text(15,113,l('Host','主机'),24,BLUE);s.text(15,193,l('Device','设备'),24,TEAL);s.text(15,268,l('Pins','固定'),24,AMBER)
        s.line(190,105,350,105,BLUE);s.path([(350,105),(350,185),(670,185),(670,105),(990,105)],TEAL)
        s.bar(350,240,640,l('Exclusive ownership retained','保持独占所有权'),AMBER)
        s.text(550,325,l('Logical event order; distances are not elapsed time','逻辑事件顺序；距离不代表实际耗时'),22,MUTED,'middle')
    elif k=='fifo':
        for x,label,color in [(30,l('A: done|unreaped','A：已完成|未收取'),AMBER),(330,l('B: running','B：执行中'),TEAL),(630,l('C: queued','C：排队中'),BLUE)]:s.box(x,60,250,85,label,color)
        s.line(280,102,330,102);s.line(580,102,630,102)
        s.text(30,200,l('Example capacity: 3 outstanding slots','示例容量：3 个未收取槽位'),26)
        s.box(35,250,290,60,l('D: submit rejected','D：提交被拒绝'),RED)
        s.text(385,287,l('reap(A) frees one slot','收取 A 后才空出一个槽位'),24,TEAL)
        s.text(980,93,l('One|FIFO|worker','单个|FIFO|执行单元'),23,TEAL,'middle')
    elif k=='visibility':
        s.text(20,50,l('Producer','生产者'),24,BLUE);s.text(20,220,l('Observer','观察者'),24,TEAL)
        s.box(205,20,280,65,l('Write output','写入输出'));s.box(700,20,340,65,l('Publish completion','发布完成记录'))
        s.line(485,52,700,52);s.text(590,115,l('ordering / visibility','顺序 / 可见性'),22,AMBER,'middle')
        s.box(205,195,280,65,l('Read output','读取输出'),TEAL);s.box(700,195,340,65,l('Observe completion','观察到完成'),TEAL)
        s.line(700,227,485,227,TEAL);s.line(870,85,870,195,TEAL)
        s.text(550,315,l('Real target: documented DMA / barrier / coherency contract','真机：使用文档规定的 DMA、屏障与一致性契约'),23,AMBER,'middle')
    elif k=='lostirq':
        for x,label in [(155,l('Host','主机')),(580,l('Device','设备')),(980,l('Notification','通知'))]:s.text(x,25,label,24,TEAL,'middle');s.line(x,45,x,320,GRAY,False,True)
        s.line(155,75,580,75);s.text(350,63,l('submit','提交'),22,INK,'middle')
        s.text(580,135,l('write + done','写入并完成'),23,TEAL,'middle')
        s.line(580,165,980,165,RED,False,True);s.cross(850,165);s.text(800,150,l('IRQ lost','中断丢失'),22,RED,'middle')
        s.text(155,205,l('wait: timeout','等待：超时'),22,AMBER,'middle')
        s.line(155,242,580,242);s.text(350,230,l('poll','轮询'),22,INK,'middle')
        s.line(580,285,155,285,TEAL);s.text(350,276,l('done observed; reap allowed','观察到完成；可收取'),22,TEAL,'middle')
    elif k=='firmware':
        s.box(25,100,230,70,'RESET');s.box(435,100,230,70,'READY',TEAL);s.box(845,100,230,70,'FAULT',RED)
        s.line(255,135,435,135);s.text(345,85,l('supported boot','受支持的启动'),22,TEAL,'middle')
        s.line(665,135,845,135,RED);s.text(755,85,l('watchdog','看门狗'),22,RED,'middle')
        s.path([(140,100),(140,28),(960,28),(960,100)],RED);s.text(550,24,l('boot rejected','启动被拒绝'),21,RED,'middle')
        s.path([(960,170),(960,260),(140,260),(140,170)],AMBER)
        s.text(550,245,l('quiesce + reset, generation += 1','确认静止并复位，generation += 1'),22,AMBER,'middle')
        s.text(550,320,l('READY may also reset through the same quiescence contract','READY 也可按相同静止契约复位'),21,MUTED,'middle')
    elif k=='watchdog':
        xs=[100,340,610,980];labels=[l('Start','启动'),l('No progress','无进展'),l('Watchdog → FAULT','看门狗 → FAULT'),l('Reset','复位')]
        s.line(70,110,1040,110)
        for x,t in zip(xs,labels):s.line(x,90,x,130,TEAL,False);s.text(x,70,t,22,INK,'middle')
        s.bar(100,180,880,l('Pins retained during hang and FAULT','挂起与 FAULT 期间仍固定缓冲区'),AMBER)
        s.line(610,125,610,165,RED);s.line(610,225,610,260,RED);s.text(610,300,l('Timeout / FAULT does not reclaim live memory','超时 / FAULT 不会回收仍可能被访问的内存'),24,RED,'middle')
    elif k=='quiesce':
        s.box(25,30,300,70,l('Old command active','旧命令仍在执行'),RED)
        s.box(415,30,270,70,l('Stop device access','停止设备访问'),AMBER);s.line(325,65,415,65)
        s.box(795,30,280,70,l('Reclaim buffers','回收缓冲区'),TEAL);s.line(685,65,795,65)
        s.line(550,100,550,185,AMBER);s.box(380,185,340,65,l('Quiescence established','已确认设备静止'),AMBER)
        s.path([(175,100),(175,280),(930,280),(930,100)],RED,False,True);s.cross(800,280)
        s.text(500,320,l('Generation tagging alone cannot stop old writes','仅改变代次无法停止旧写入'),23,RED,'middle')
    elif k=='generation':
        s.text(25,40,l('Before reset','复位前'),24);s.box(25,80,320,70,l('ticket (g=4, cookie=7)','票据 (g=4, cookie=7)'))
        s.box(420,80,260,70,l('reset → g=5','复位 → g=5'),AMBER);s.line(345,115,420,115)
        s.box(765,80,310,70,l('ticket (g=5, cookie=7)','票据 (g=5, cookie=7)'),TEAL);s.line(680,115,765,115)
        s.path([(185,150),(185,245),(740,245)],RED);s.cross(710,245);s.text(790,253,l('old event rejected','拒绝旧事件'),23,RED)
        s.line(920,150,920,190,TEAL);s.text(920,213,l('new identity','新的身份'),23,TEAL,'middle')
        s.text(25,320,l('The pair identifies work; cookie alone is insufficient.','通过二元组标识工作；仅靠 cookie 不足。'),24)
    elif k=='actors':
        for x,label in [(140,'Runtime'),(540,l('Driver interface','驱动接口')),(965,l('Firmware / device','固件 / 设备'))]:s.text(x,30,label,24,TEAL,'middle');s.line(x,55,x,320,GRAY,False,True)
        for y,a,b,t in [(90,140,540,l('submit + resources','提交与资源')),(150,540,965,l('execute command','执行命令')),(220,965,540,l('completion + status','完成与状态')),(285,540,140,l('observe + safe reap','观察与安全收取'))]:s.line(a,y,b,y);s.text((a+b)/2,y-13,t,22,INK,'middle')
    elif k=='pins':
        s.box(25,30,230,65,l('Host API','主机 API'));s.box(760,30,310,65,l('Device command','设备命令'),TEAL)
        for y,t in [(140,l('Source buffer','源缓冲区')),(250,l('Destination buffer','目标缓冲区'))]:
            s.box(390,y,270,65,t,AMBER);s.path([(915,95),(915,y+32),(660,y+32)],TEAL)
            s.path([(140,95),(140,y+32),(365,y+32)],RED,False,True);s.cross(330,y+32)
        s.text(165,217,l('read / write / free|blocked while pinned','固定期间禁止|读 / 写 / 释放'),21,RED)
        s.text(795,220,l('exclusive pins','独占固定'),23,AMBER)
    elif k=='cancel':
        for y,state,action,color in [(30,l('QUEUED','排队中'),l('cancel → observe → reap','取消 → 观察 → 收取'),TEAL),(135,l('IN FLIGHT','执行中'),l('cancel rejected; complete or reset','拒绝取消；等待完成或复位'),RED),(240,l('DONE','已完成'),l('observe → reap','观察 → 收取'),TEAL)]:
            s.box(25,y,255,65,state,color);s.line(280,y+32,420,y+32,color);s.text(450,y+40,action,24,color)
    elif k=='reap':
        for i,(title,own) in enumerate([(l('Device done','设备完成'),l('pins + slot held','仍占固定资源与槽位')),(l('Host observed','主机观察到'),l('pins + slot held','仍占固定资源与槽位')),(l('Reaped','已收取'),l('pins + slot released','释放固定资源与槽位'))]):
            x=25+i*365;s.box(x,45,300,65,title,TEAL if i==2 else BLUE)
            s.text(x+150,175,own,21,TEAL if i==2 else AMBER,'middle')
            if i<2:s.line(x+300,78,x+365,78)
        s.bar(25,215,665,l('Ownership persists after device completion','设备完成后，所有权仍然保留'),AMBER)
        s.text(865,300,l('double reap: reject','重复收取：拒绝'),24,RED,'middle')
    elif k=='testbench':
        s.box(20,120,220,70,l('Stimulus + faults','激励与故障'))
        s.box(410,30,290,70,l('Implementation','被测实现'));s.box(410,220,290,70,l('Reference model','参考模型'),TEAL)
        s.path([(240,155),(320,155),(320,65),(410,65)]);s.path([(320,155),(320,255),(410,255)])
        s.box(855,120,225,70,l('Compare +|invariants','结果对比 +|不变量检查'),AMBER)
        s.path([(700,65),(785,65),(785,155),(855,155)]);s.path([(700,255),(785,255),(785,155)])
        s.text(560,170,l('Independent expected behavior','独立的预期行为'),22,TEAL,'middle')
    elif k=='controls':
        for y,candidate,expected,color in [(25,l('Valid baseline','有效基线'),l('accept','接受'),TEAL),(130,l('Early-free mutant','过早释放的错误变体'),l('reject','拒绝'),RED),(235,l('Legal cleanup','合法的清理变更'),l('accept','接受'),TEAL)]:
            s.box(25,y,340,65,candidate,color);s.line(365,y+32,540,y+32);s.box(540,y,230,65,l('Same evaluator','相同验收器'));s.line(770,y+32,900,y+32);s.text(940,y+40,expected,25,color)
    elif k=='faults':
        nodes=[(30,l('Admission','准入')),(400,l('Execution','执行')),(770,l('Notification','通知'))]
        for x,t in nodes:s.box(x,95,300,65,t)
        s.line(330,127,400,127);s.line(700,127,770,127)
        for x,top,bottom in [(180,l('No slots / allocation','槽位 / 分配不足'),l('Reject, ledger unchanged','拒绝，资源记录不变')),(550,l('DMA error / hang','DMA 错误 / 挂起'),l('Error reap / watchdog reset','错误收取 / 看门狗复位')),(920,l('Dropped IRQ','丢失中断'),l('Done → poll → reap','完成 → 轮询 → 收取'))]:
            s.text(x,40,top,21,RED,'middle');s.line(x,50,x,95,RED);s.line(x,160,x,230);s.text(x,275,bottom,21,AMBER,'middle')
    elif k=='bounded':
        s.box(25,30,230,65,l('Idle + live','空闲 + 内存有效'))
        s.box(425,30,290,65,l('Submitted + live','已提交 + 内存有效'));s.line(255,62,425,62);s.text(340,40,'submit',21,INK,'middle')
        s.box(795,235,290,65,l('Outstanding + freed','工作未结束 + 已释放'),RED)
        s.path([(715,62),(940,62),(940,235)],RED);s.text(960,140,l('unsafe|release','错误规则|允许释放'),22,RED)
        s.text(555,175,l('Safe policy: reject release','安全规则：拒绝释放'),23,TEAL,'middle')
        s.line(365,10,365,320,GRAY,False,True);s.text(175,315,l('depth 0','深度 0'),21);s.text(480,315,l('depth 1','深度 1'),21);s.text(805,330,l('depth 2: witness','深度 2：反例'),21,RED)
    elif k=='matrix':
        s.text(20,25,l('Released evidence','已发布的证据'),24,TEAL)
        cols=[l('Identity','身份'),l('Functional','功能'),l('Device','设备'),l('Physical','物理')]
        for i,t in enumerate(cols):s.text(345+i*210,80,t,23,INK,'middle')
        s.text(25,150,l('CPU report','CPU 报告'),23)
        for i,t in enumerate([l('RECORDED','已记录'),l('PASS','通过'),l('NOT RUN','未运行'),l('NOT RUN','未运行')]):s.box(245+i*210,110,200,60,t,TEAL if i<2 else AMBER)
        s.text(25,245,l('Hardware gate','硬件门槛'),23);s.text(670,245,l('Promotion blocked: missing target evidence','缺少目标证据，不能放行'),25,RED,'middle')
        s.text(550,315,l('A functional pass does not fill unexecuted target cells.','功能通过不会自动填补未执行的目标测试。'),23,MUTED,'middle')
    elif k=='hil':
        s.box(20,20,280,65,l('Test controller','测试控制器'))
        s.box(415,20,270,65,l('Board + fixture map','板卡与夹具映射'),AMBER);s.line(300,52,415,52)
        s.box(800,130,280,85,l('Reserved test board','已预约的测试板'),TEAL)
        s.path([(685,52),(940,52),(940,130)]);s.text(760,35,l('runner','执行器'),21)
        s.path([(800,170),(540,170),(540,255),(300,255)],TEAL);s.text(575,155,l('serial / results','串口 / 结果'),22,TEAL,'middle')
        s.box(20,220,280,70,l('Logs + decision','日志与决策'),TEAL)
        s.path([(160,85),(160,200),(800,200)],AMBER);s.text(350,225,l('deploy / recover','部署 / 恢复'),21,AMBER,'middle')
        s.text(580,320,l('Match test fixture, board identity and supported recovery.','匹配测试夹具、板卡身份与受支持的恢复流程。'),22,MUTED,'middle')
    elif k=='energy':
        s.text(30,30,l('Named measurement boundary','指定测量边界'),23,TEAL)
        s.rect(25,65,310,235,TEAL,dash=True);s.box(70,130,220,70,l('Board / device','板卡 / 设备'))
        s.text(180,260,l('Sensor scope','传感器范围'),22,AMBER,'middle')
        s.line(445,285,1060,285);s.line(445,285,445,45)
        s.text(415,40,'P(t)',24);s.text(1065,320,'t',24)
        pts=[(505,240),(570,170),(650,190),(735,110),(820,145),(895,130),(1000,225)]
        s.parts.append('<path d="M505,285 L505,240 L570,170 L650,190 L735,110 L820,145 L895,130 L1000,225 L1000,285 Z" fill="#244747"/>')
        s.path(pts,TEAL,False);s.text(760,65,l('E = ∫ P(t) dt','E = ∫ P(t) dt'),26,AMBER,'middle')
        s.text(505,320,'t₀',23);s.text(1000,320,'t₁',23);s.text(760,265,l('matched completion window','对齐的工作完成窗口'),21,INK,'middle')
    elif k=='rollback':
        labels=[l('Stop admission','停止准入'),l('Drain or quiesce','排空或确认静止'),l('Restore artifact','恢复产物'),l('Verify fresh work','验证新工作')]
        coords=[(25,30),(640,30),(640,235),(25,235)]
        for (x,y),t in zip(coords,labels):s.box(x,y,400,65,t,AMBER if y==30 else TEAL)
        s.line(425,62,640,62);s.line(840,95,840,235);s.line(640,267,425,267,TEAL)
        s.text(550,170,l('No old DMA access across this boundary','跨越此边界时，旧 DMA 访问必须已停止'),24,RED,'middle')
    elif k=='capstone':
        s.box(25,120,230,70,l('Bounded change','有界修改'))
        s.box(420,20,280,65,l('Frozen contract','冻结契约'),AMBER);s.box(420,225,280,65,l('Tests + raw logs','测试与原始日志'),TEAL)
        s.path([(255,155),(340,155),(340,52),(420,52)]);s.path([(340,155),(340,257),(420,257)])
        s.box(855,120,225,70,l('Retain / revise|or reject','保留 / 修订|或拒绝'),TEAL)
        s.path([(700,52),(775,52),(775,155),(855,155)]);s.path([(700,257),(775,257),(775,155)])
        s.text(555,170,l('Reproducible identity','可复现身份'),23,AMBER,'middle')
    elif k=='boundedpatch':
        layers=[l('Packet ABI','命令 ABI'),l('Queue / ownership','队列 / 所有权'),l('Firmware lifecycle','固件生命周期'),l('Evidence / release','证据 / 发布')]
        for i,t in enumerate(layers):s.box(25,10+80*i,400,60,t,TEAL if i==1 else MUTED)
        s.rect(600,70,460,100,TEAL);s.text(830,109,l('Example: queued cancel cleanup','示例：排队取消的清理'),24,TEAL,'middle');s.text(830,143,l('one invariant, one patch','一个不变量，一个修改'),22,INK,'middle')
        s.line(425,120,600,120,TEAL)
        s.text(640,225,l('Preserve ABI + in-flight policy|Keep real DMA outside CPU claims','保留 ABI 与执行中取消策略|CPU 结论不涵盖真实 DMA'),22)
    elif k=='baseline':
        s.box(25,30,280,65,l('Baseline revision','基线版本'));s.box(25,230,280,65,l('Candidate revision','候选版本'),TEAL)
        s.box(450,120,270,75,l('Frozen evaluator','冻结的验收器'),AMBER)
        s.path([(305,62),(365,62),(365,157),(450,157)]);s.path([(305,262),(365,262),(365,157)])
        s.box(820,20,260,65,l('Held-out cases','留出用例'),AMBER);s.path([(950,85),(950,105),(585,105),(585,120)],AMBER)
        s.box(820,230,260,65,l('Raw paired results','原始成对结果'),TEAL);s.path([(720,157),(950,157),(950,230)],TEAL)
        s.text(480,315,l('Same contract, seeds and accounting','契约、种子与统计口径相同'),23)
    elif k=='dossier':
        s.box(25,130,260,70,l('Manifest + identities','清单与身份'),AMBER)
        items=[l('Source / config / prompts','源码 / 配置 / 提示词'),l('Contract + tests','契约与测试'),l('Raw results + reproducer','原始结果与复现用例'),l('Status + rollback + decision','状态 / 回滚 / 决策')]
        s.line(285,165,405,165,AMBER);s.line(405,40,405,290,AMBER,False)
        for i,t in enumerate(items):
            y=10+i*82;s.line(405,y+30,530,y+30,AMBER);s.text(560,y+38,t,25,TEAL if i==3 else INK)
    else:
        raise ValueError('unknown drawing '+k)
    return s.out()


def build():
    manifest=json.loads((HERE/'figure_specs.json').read_text())
    presenter=(SLIDES.parents[1]/'llm-compute-systems-course/slides/presenter_mode.js').read_text()
    for spec in manifest['figures']:
        for zh in (False,True):
            svg=draw(SVG(spec,zh));assets=HERE/('zh' if zh else '')
            assets.mkdir(exist_ok=True);(assets/(spec['slug']+'.svg')).write_text(svg+'\n')
            path=SLIDES/('zh' if zh else '')/spec['name'];source=path.read_text()
            anchor=spec['anchor_cn' if zh else 'anchor']
            sections=list(re.finditer(r'<section class="slide".*?</section>',source,re.S))
            match=next((m for m in sections if f'data-illustration="{spec["slug"]}"' in m.group()),None)
            if match is None:match=next((m for m in sections if '<h2>'+anchor+'</h2>' in m.group()),None)
            if match is None:raise ValueError('missing anchor '+anchor)
            old=match.group();prior=re.search(r'<aside class="notes">(.*?)</aside>',old,re.S)
            notes=prior.group(1) if prior else ''
            if not notes.strip() and 'data-illustration=' not in old:
                notes=re.search(r'<div class="content">(.*?)</div><div class="cite">',old,re.S).group(1)
            if 'data-visual-notes' not in notes:
                prompt=('读图顺序：先辨认参与者、地址或状态，再沿连线解释事件，最后指出何时允许回收或放行。请学生预测一个失败分支。' if zh else 'Read the actors, addresses or states first, then trace the events and identify the safe reclamation or acceptance boundary. Ask learners to predict one failure branch.')
                notes+='<p data-visual-notes="true">'+prompt+'</p>'
            src=manifest['sources'][spec['source']]
            url=src.get('url') or ('../../' if zh else '../')+src['path']
            if 'data-figure-source' not in notes:
                notes+='<p data-figure-source="true">'+('图示参考：' if zh else 'Figure reference: ')+f'<a href="{escape(url,quote=True)}">{escape(src["label"])}</a>'+'</p>'
            label=('原创建模图，参考：' if zh else 'Original schematic, reference: ')+src['label']
            title=spec['title_cn' if zh else 'title'];caption=spec['caption_cn' if zh else 'caption']
            replacement=f'<section class="slide" data-illustration="{spec["slug"]}"><h2>{escape(title)}</h2><div class="content">{svg}<p class="note">{escape(caption)}</p></div><div class="cite"><a href="{escape(url,quote=True)}">{escape(label)}</a></div><div class="num"></div><aside class="notes">{notes}</aside></section>'
            source=source[:match.start()]+replacement+source[match.end():]
            path.write_text(source)
    for path in sorted(SLIDES.glob('*.html'))+sorted((SLIDES/'zh').glob('*.html')):
        source=path.read_text();total=len(re.findall(r'<section class="slide"',source));index=[0]
        def number(m):
            index[0]+=1;t=re.sub(r'<section class="slide"(?: id="[^"]*")?',f'<section class="slide" id="s{index[0]}"',m.group(),count=1)
            return re.sub(r'<div class="num">.*?</div>',f'<div class="num">{index[0]} / {total}</div>',t,count=1)
        source=re.sub(r'<section class="slide".*?</section>',number,source,flags=re.S)
        source=re.sub(r'<script id="course-presenter">.*?</script>','',source,flags=re.S)
        source=source.replace('</body>','<script id="course-presenter">'+presenter+'</script></body>')
        path.write_text(source)
    print(f'Built {len(manifest["figures"])} original bilingual mechanisms, 64 SVG assets and 16 offline decks')


if __name__=='__main__':
    build()
