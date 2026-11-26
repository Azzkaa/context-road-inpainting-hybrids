"""Source-only scene/sign conditioning after LaMa Fourier bottleneck concatenation."""
import torch
from torch import nn
from torch.nn import functional as F
from source_data import R
from adapters import SpatialRoadAdapter,VisualContextTokenizer,ContextResidual,bounded_residual_add
import sys
sys.path.insert(0,str(R/'work/road_restoration'))
from models import load_lama
from deterministic_padding import install
class LamaHybrid(nn.Module):
 def __init__(self):
  super().__init__();self.base=load_lama();self.deterministic_padding_modules=install(self.base);self.local=nn.Sequential(nn.Conv2d(512,64,1),nn.SiLU(),nn.Conv2d(64,512,1));nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
  self.spatial=SpatialRoadAdapter();self.spatial.out=nn.Conv2d(64,512,1);nn.init.zeros_(self.spatial.out.weight);nn.init.zeros_(self.spatial.out.bias);self.tokenizer=VisualContextTokenizer();self.context=ContextResidual(512);self.context.relative_rms_limit=.1;self.source=None;self.variant='A1';self.handle=self.base.model[23].register_forward_hook(self.attach)
 def attach(self,module,args,out):
  out=bounded_residual_add(out,self.local(out),.1)
  if self.variant=='A1':return out
  s=self.source;planes=F.interpolate(s['planes'],out.shape[-2:],mode='nearest');out=bounded_residual_add(out,self.spatial(s['features'],planes),.1);tok,keep=self.tokenizer(s['features'][-1],s['signs'] if self.variant=='C' else None,s['boxes'] if self.variant=='C' else None,s['confidence'] if self.variant=='C' else None,s['valid'] if self.variant=='C' else None,scene_valid=s['planes'][:,3:4]);return self.context(out,tok,keep)
 def forward(self,x,mask,source,variant='A1'):
  assert variant in ['A1','B','C'];self.source=source;self.variant=variant;self.base.eval();return self.base(torch.cat([x*(1-mask),mask],1))
