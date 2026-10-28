"""Logical block/prefix ownership simulation; stores NO KV tensors or engine state."""
from collections import OrderedDict
from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Identity:
    model: str
    tokenizer: str = 'fixture-v1'
    adapter: str = 'none'
    positions_mask: str = 'causal-from-zero'
    dtype_layout: str = 'fixture-fp16'
    modality: str = 'text'
    sharing_domain: str = 'classroom'


class PrefixCache:
    def __init__(self, capacity=4, block_size=4):
        if capacity<1 or block_size<1:raise ValueError('positive capacity/block size required')
        self.capacity,self.block_size=capacity,block_size
        self.blocks=OrderedDict()
        self.requests={}

    def acquire(self, request_id, tokens, identity):
        if request_id in self.requests:raise ValueError('request already owns state')
        if not isinstance(identity,Identity) or not all(isinstance(t,int) and t>=0 for t in tokens):
            raise ValueError('declared identity and nonnegative integer token IDs required')
        owned=[]
        hits=0
        contiguous=True
        try:
            for end in range(self.block_size,len(tokens)+1,self.block_size):
                key=(identity,tuple(tokens[:end]))
                present=key in self.blocks
                if contiguous and present:hits+=self.block_size
                else:contiguous=False
                if not present:
                    if len(self.blocks)>=self.capacity:
                        victim=next((k for k,v in self.blocks.items() if v==0),None)
                        if victim is None:raise MemoryError('all blocks pinned; reject admission')
                        del self.blocks[victim]
                    self.blocks[key]=0
                self.blocks[key]+=1
                self.blocks.move_to_end(key)
                owned.append(key)
        except MemoryError:
            for key in owned:self.blocks[key]-=1
            raise
        self.requests[request_id]=owned
        return {'matched_tokens':hits,'recomputed_tokens':len(tokens)-hits,'full_blocks_owned':len(owned)}

    def release(self, request_id):
        for key in self.requests.pop(request_id,[]):
            self.blocks[key]-=1

    def snapshot(self):
        return {'resident_blocks':len(self.blocks),'pinned_blocks':sum(v>0 for v in self.blocks.values()),'references':sum(self.blocks.values()),'live_requests':len(self.requests)}


def main():
    cache=PrefixCache(capacity=4)
    identity=Identity('model-v1')
    rows=[]
    for name,tokens,ident in [('cold',list(range(8)),identity),('warm',list(range(8)),identity),('late_mismatch',list(range(4))+[99,5,6,7],identity),('early_mismatch',[99]+list(range(1,8)),identity),('model_change',list(range(8)),Identity('model-v2'))]:
        rows.append({'case':name,**cache.acquire(name,tokens,ident)})
        cache.release(name)
    print(json.dumps({'lab':'L7','provenance':'logical token/block simulation; no engine inference or KV tensors','cases':rows,'final':cache.snapshot()},indent=2))

if __name__=='__main__':main()
