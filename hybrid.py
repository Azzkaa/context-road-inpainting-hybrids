"""Real CompVis U-Net attachments; source-only conditioning, no language path."""
from pathlib import Path
import sys,torch
from torch import nn
from adapters import SpatialRoadAdapter,VisualContextTokenizer,ContextResidual,bounded_residual_add
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'work/road_restoration'));from models import load_ldm
class ContextInpainting(nn.Module):
 def __init__(self,relative_rms_limit=None):
  super().__init__();self.relative_rms_limit=relative_rms_limit;self.ldm=load_ldm('direct');self.ldm.model.diffusion_model.input_blocks[0][0].requires_grad_(False);self.visual=nn.ModuleDict({'spatial':SpatialRoadAdapter(),'tokenizer':VisualContextTokenizer(),'context':ContextResidual()});self.visual.load_state_dict(torch.load(R/'outputs/lane_geometry/compvis_hybrid/interface_initialization.pt',map_location='cpu',weights_only=True));self.visual['context'].relative_rms_limit=relative_rms_limit;self._condition=None;net=self.ldm.model.diffusion_model;self.handles=[net.input_blocks[0].register_forward_hook(self._input_hook),net.middle_block.register_forward_hook(self._mid_hook)]
 def _input_hook(self,module,args,out):
  if self._condition is None:return out
  return bounded_residual_add(out,self._condition[0],self.relative_rms_limit)
 def _mid_hook(self,module,args,out):
  if self._condition is None:return out
  return self.visual['context'](out,self._condition[1],self._condition[2])
 def forward(self,z,t,masked_condition,features=None,planes=None,signs=None,boxes=None,confidence=None,valid=None,variant='A1'):
  assert variant in ['A1','B','C'];assert masked_condition.shape[1]==4 and z.shape[1]==3
  if variant=='A1':self._condition=None
  else:
   assert features is not None and planes is not None
   delta=self.visual['spatial'](features,planes);tok,keep=self.visual['tokenizer'](features[-1],signs if variant=='C' else None,boxes if variant=='C' else None,confidence if variant=='C' else None,valid if variant=='C' else None,scene_valid=planes[:,3:4]);self._condition=(delta,tok,keep)
  # Retain conditioning until backward completes; checkpointed modules may re-run hooks.
  return self.ldm.apply_model(z,t,masked_condition)
