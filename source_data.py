"""Source-only conditioning separated from TRAIN-only clean target preparation."""
from pathlib import Path
import json,hashlib,torch,numpy as np
from PIL import Image
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation
R=Path(__file__).resolve().parents[3]
def canvas(im,mask=False):
 im=im.convert('L' if mask else 'RGB');w,h=im.size;s=min(512/w,512/h);nw,nh=round(w*s),round(h*s);l,t=(512-nw)//2,(512-nh)//2;raw=np.array(im.resize((nw,nh),Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS));pads=((t,512-nh-t),(l,512-nw-l))+(() if mask else ((0,0),));a=np.pad(raw,pads,mode='constant' if mask else 'edge');valid=np.zeros((512,512),np.float32);valid[t:t+nh,l:l+nw]=1;return a,valid,(l,t,nw,nh)
class SourceEncoder:
 def __init__(self):
  p=R/'work/snow_replacements820/cityscapes_model';self.net=SegformerForSemanticSegmentation.from_pretrained(p,local_files_only=True).cuda().eval().requires_grad_(False);cfg=json.loads((p/'preprocessor_config.json').read_text());self.mean=torch.tensor(cfg['image_mean'],device='cuda')[None,:,None,None];self.std=torch.tensor(cfg['image_std'],device='cuda')[None,:,None,None]
 @torch.no_grad()
 def prepare(self,row):
  assert hashlib.sha256(Path(row['snowy']).read_bytes()).hexdigest()==row['snowy_sha256'];native=Image.open(row['snowy']).convert('RGB');a,valid,box=canvas(native);x=torch.from_numpy(a.copy()).permute(2,0,1)[None].cuda().float()/255;v=torch.from_numpy(valid)[None,None].cuda();signs=[];boxes=[];record=R/'outputs/lane_geometry/klein_hybrid/train_sign_cue_audit'/row['id']/'record.json'
  with torch.autocast('cuda',dtype=torch.float16):
   e=self.net((x-self.mean)/self.std,output_hidden_states=True);features=[z.detach() for z in e.hidden_states];prob=F.interpolate(e.logits,(512,512),mode='bilinear',align_corners=False).float().softmax(1);labels=prob.argmax(1);protected=torch.zeros_like(labels,dtype=torch.bool)
   for c in [11,12,13,14,15,16,17,18]:protected|=labels==c
   permit=((labels==0)&(prob[:,0]>=.6)&~protected)[:,None].float()*v;planes=F.interpolate(torch.cat([permit,protected[:,None].float(),prob[:,0:1],v],1),(128,128),mode='nearest')
   new_record=R/'outputs/lane_geometry/compvis_hybrid/source_sign_proposals'/(row['id']+'.json')
   sign_boxes=[]
   if new_record.exists():
    rec=json.loads(new_record.read_text());assert rec['source_sha256']==row['snowy_sha256'];sign_boxes=rec['boxes']
   elif record.exists():
    rec=json.loads(record.read_text());assert rec['source_sha256']==row['snowy_sha256'];sign_boxes=list(dict.fromkeys(tuple(c['sign_native_bbox']) for c in rec['candidates']))[:4]
   for b in sign_boxes:
     crop=native.crop(b).resize((256,256),Image.Resampling.LANCZOS);cx=torch.from_numpy(np.array(crop).copy()).permute(2,0,1)[None].cuda().float()/255;signs.append(self.net((cx-self.mean)/self.std,output_hidden_states=True).hidden_states[-1][0].detach());boxes.append([b[0]/native.width,b[1]/native.height,b[2]/native.width,b[3]/native.height])
  k=len(signs);return dict(id=row['id'],source_sha256=row['snowy_sha256'],x=x,permit=permit,planes=planes,features=features,signs=torch.stack(signs)[None] if k else None,boxes=torch.tensor(boxes,device='cuda',dtype=torch.float32)[None] if k else None,confidence=torch.full((1,k),.5,device='cuda') if k else None,valid=torch.ones((1,k),dtype=torch.bool,device='cuda') if k else None,working_valid=v,content_box=box,clean_input_used=False)

def prepare_train_supervision(row,ldm,source):
 assert row['split']=='train';assert hashlib.sha256(Path(row['clean']).read_bytes()).hexdigest()==row['clean_sha256'];assert hashlib.sha256(Path(row['mask']).read_bytes()).hexdigest()==row['mask_sha256'];clean_im=Image.open(row['clean']);target,valid,box=canvas(clean_im);assert box==source['content_box'];mask_im=Image.open(row['mask']).convert('L').resize(clean_im.size,Image.Resampling.NEAREST);edit,_,m_box=canvas(mask_im,mask=True);assert m_box==box;clean=torch.from_numpy(target.copy()).permute(2,0,1)[None].cuda().float()/127.5-1
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):z=ldm.encode_first_stage(clean);z=ldm.get_first_stage_encoding(z).detach();cond_z=ldm.cond_stage_model.encode(source['x']*(1-source['permit'])*2-1).detach()
 mask=torch.from_numpy((edit>127).astype(np.float32))[None,None].cuda();val=source['working_valid'];weight=F.avg_pool2d((mask+.05*val)*val,4);lane_path=R/'work/road_restoration/lane_weights'/ (row['id']+'.pt')
 # Gate example may lack lane annotation. Do not substitute model pseudo labels.
 if lane_path.exists():
  lane=torch.load(lane_path,map_location='cpu',weights_only=False).cuda().float();lane=lane.reshape(1,1,*lane.shape[-2:]);assert lane.shape==weight.shape;weight+=4*lane*F.avg_pool2d(mask,4)*F.avg_pool2d(val,4)
 cond=torch.cat([cond_z,F.interpolate(source['permit']*2-1,cond_z.shape[-2:],mode='nearest')],1);return z,cond,weight
