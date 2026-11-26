"""Prompt-free conditioning interfaces; actual U-Net attachment is a separate gate."""
import torch
from torch import nn
from torch.nn import functional as F


def bounded_residual_add(base,delta,relative_limit=None):
 if relative_limit is None:return base+delta.to(base.dtype)
 assert 0<relative_limit<=1
 dims=tuple(range(1,base.ndim));b=base.detach().float().square().mean(dims,keepdim=True).sqrt();r=delta.detach().float().square().mean(dims,keepdim=True).sqrt();gain=(relative_limit*b/(r+1e-6)).clamp(max=1).to(delta.dtype);return base+(gain*delta).to(base.dtype)

class SpatialRoadAdapter(nn.Module):
 def __init__(self):
  super().__init__();self.projects=nn.ModuleList([nn.Conv2d(c,16,1) for c in [32,64,160,256]]);self.fuse=nn.Sequential(nn.Conv2d(68,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU());self.out=nn.Conv2d(64,256,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,features,road_planes):
  assert len(features)==4 and road_planes.shape[1]==4
  size=road_planes.shape[-2:];maps=[F.interpolate(p(x),size,mode='bilinear',align_corners=False) for p,x in zip(self.projects,features)];return self.out(self.fuse(torch.cat(maps+[road_planes],1)))

class VisualContextTokenizer(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Linear(256,192);self.position=nn.Linear(2,192);self.box=nn.Sequential(nn.Linear(5,48),nn.SiLU(),nn.Linear(48,192));self.kind=nn.Parameter(torch.zeros(2,192))
 def grid(self,device,dtype,size):
  yy,xx=torch.meshgrid(torch.linspace(0,1,size,device=device,dtype=dtype),torch.linspace(0,1,size,device=device,dtype=dtype),indexing='ij');return torch.stack([xx,yy],-1).reshape(size*size,2)
 def forward(self,scene,signs=None,boxes=None,confidence=None,valid=None,scene_valid=None):
  b=scene.shape[0];tokens=self.project(F.adaptive_avg_pool2d(scene,(8,8)).flatten(2).transpose(1,2))+self.position(self.grid(scene.device,scene.dtype,8))[None]+self.kind[0];keep=torch.ones((b,64),dtype=torch.bool,device=scene.device)
  if scene_valid is not None:
   keep=F.adaptive_avg_pool2d(scene_valid.float(),(8,8)).flatten(1)>.5;assert keep.any(1).all(), 'No valid source-scene tokens'
  if signs is None:return tokens,keep
  assert signs.shape[0]==b and signs.shape[1]<=4 and signs.shape[2]==256
  k=signs.shape[1];assert boxes.shape==(b,k,4) and confidence.shape==valid.shape==(b,k)
  z=F.adaptive_avg_pool2d(signs.reshape(b*k,*signs.shape[2:]),(4,4)).flatten(2).transpose(1,2).reshape(b,k,16,256);extra=self.project(z)+self.position(self.grid(scene.device,scene.dtype,4))[None,None]+self.box(torch.cat([boxes,confidence[...,None]],-1))[:,:,None]+self.kind[1];return torch.cat([tokens,extra.reshape(b,k*16,192)],1),torch.cat([keep,valid.bool().repeat_interleave(16,1)],1)

class ContextResidual(nn.Module):
 def __init__(self,channels=1024):
  super().__init__();self.relative_rms_limit=None;self.norm=nn.GroupNorm(32,channels);self.q=nn.Linear(channels,128);self.kv=nn.Linear(192,256);self.out=nn.Linear(128,channels);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x,tokens,valid):
  b,c,h,w=x.shape;assert tokens.shape[:2]==valid.shape and valid.any(1).all();q=self.q(self.norm(x).flatten(2).transpose(1,2)).reshape(b,h*w,8,16).transpose(1,2);k,v=self.kv(tokens).chunk(2,-1);k=k.reshape(b,-1,8,16).transpose(1,2);v=v.reshape(b,-1,8,16).transpose(1,2);a=(q@k.transpose(-2,-1))*(16**-.5);a=a.masked_fill(~valid[:,None,None,:],torch.finfo(a.dtype).min);z=(a.softmax(-1)@v).transpose(1,2).reshape(b,h*w,128);return bounded_residual_add(x,self.out(z).transpose(1,2).reshape(b,c,h,w),self.relative_rms_limit)
