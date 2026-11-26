"""One documented LaMa correction: final RGB projection and verified TRAIN lane weighting."""
from lama_hybrid import LamaHybrid
from source_data import R
import torch,json
from torch import nn
from torch.nn import functional as F
class CorrectedLama(LamaHybrid):
 def __init__(self):
  super().__init__();convs=[(n,m) for n,m in self.base.named_modules() if isinstance(m,nn.Conv2d)];self.decoder_projection_name,self.decoder_projection=convs[-1];assert self.decoder_projection.out_channels==3;self.decoder_projection.requires_grad_(True)
 def optimizer_groups(self):
  final=set(id(p) for p in self.decoder_projection.parameters());return [dict(params=[p for p in self.parameters() if p.requires_grad and id(p) not in final],lr=1e-4),dict(params=list(self.decoder_projection.parameters()),lr=1e-5)]
def lane_weight(ck,sid):
 assert ck['split']=='train';road=ck['supervision']['decoded_road_weight'].cuda();inventory=json.loads((R/'work/road_restoration/lane_weight_verification_extended.json').read_text())['records'];verified={r['id'] for r in inventory if 'annotation' in r}
 if sid not in verified:return road
 lane=torch.load(R/'work/road_restoration/lane_weights_extended'/(sid+'.pt'),map_location='cpu',weights_only=False).float().reshape(1,1,128,128).cuda();return road*(1+4*F.interpolate(lane,road.shape[-2:],mode='nearest'))
