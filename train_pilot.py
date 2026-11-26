"""Bounded matched100-update pilot with four fixed VAL restored-image previews."""
from pathlib import Path
import argparse,json,time,torch,numpy as np,sys,hashlib,fcntl
from PIL import Image
from hybrid import ContextInpainting
from source_data import R,canvas
from torch.nn import functional as F
O=R/'outputs/lane_geometry/compvis_hybrid';C=R/'work/lane_geometry/compvis_hybrid/cache';a=argparse.ArgumentParser();a.add_argument('--variant',choices=['A1','B','C'],required=True);variant=a.parse_args().variant;D=O/'pilot100'/variant;D.mkdir(exist_ok=True,parents=True);lock=(D/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.manual_seed(20261126);protocol=json.loads((O/'pilot_protocol.json').read_text());rows={r['id']:r for r in json.loads((R/'work/road_restoration/manifest.json').read_text())};model=ContextInpainting().cuda();params=dict(model.named_parameters());opt=torch.optim.AdamW([p for p in params.values() if p.requires_grad],lr=1e-4);scaler=torch.amp.GradScaler('cuda',init_scale=128.);initial=torch.load(O/'backward_resume_gate/initial_trainable.pt',map_location='cpu',weights_only=True)
with torch.no_grad():
 for n,p in params.items():
  if p.requires_grad:assert n in initial;p.copy_(initial[n])
records=[];start=0
if (D/'latest.pt').exists():
 ck=torch.load(D/'latest.pt',map_location='cpu',weights_only=False);assert ck['variant']==variant
 with torch.no_grad():
  for n,v in ck['trainable'].items():params[n].copy_(v)
 opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);start=ck['step'];records=ck['records'];torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
else:torch.manual_seed(20261126+100)
def load(sid):
 ck=torch.load(C/(sid+'.pt'),map_location='cpu',weights_only=False);assert ck['source_sha256']==rows[sid]['snowy_sha256'] and ck['preprocessing_version']==2;assert ck['split']!='train' or ck['loss_weights_version']==2;source={k:([v.cuda() for v in value] if isinstance(value,list) else value.cuda() if torch.is_tensor(value) else value) for k,value in ck['source'].items() if k!='content_box'};return ck,source,ck['conditioning'].cuda()
def status(phase,i,total=100):
 (D/'status.json').write_text(json.dumps(dict(phase=phase,completed=i,total=total,records=records,updated_at=time.time())));d=json.loads((O/'status.json').read_text());d.update(phase=variant+' '+phase,completed=i,total=total,training_steps=len(records),message='Matched100-updatepilot; actual validation outputs follow. No lane-accuracy claim.',updated_at=time.time(),next_action='Review the fixed four VAL restored images and metrics before any longer comparison.');(O/'status.json').write_text(json.dumps(d,indent=2))
def save(i):
 state={n:p.detach().cpu().clone() for n,p in params.items() if p.requires_grad};torch.save(dict(variant=variant,trainable=state,optimizer=opt.state_dict(),scaler=scaler.state_dict(),step=i,records=records,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),D/'latest.tmp');(D/'latest.tmp').replace(D/'latest.pt');ck=torch.load(D/'latest.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,ck['trainable'][n]) for n,v in state.items())
try:
 assert json.loads((O/'pilot_cache_audit.json').read_text())['phase']=='passed';assert json.loads((O/'pilot_cache_audit.json').read_text())['finalized_verified_lane_weights'];assert json.loads((O/'architecture_gate.json').read_text())['training_allowed'];model.train();model.ldm.first_stage_model.eval()
 for i in range(start,100):
  sid=protocol['train_ids'][i%64];ck,source,cond=load(sid);assert ck['split']=='train';target=ck['supervision']['target'].cuda();weight=ck['supervision']['weight'].cuda();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();t=torch.randint(0,1000,(1,),device='cuda');noise=torch.randn_like(target);noisy=model.ldm.q_sample(target,t,noise)
  with torch.autocast('cuda',dtype=torch.float16):pred=model(noisy,t,cond,**source,variant=variant);loss=((pred.float()-noise.float()).square()*weight).sum()/(weight.sum()*3).clamp_min(1)
  assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.unscale_(opt);assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None);assert all(p.grad is None for p in params.values() if not p.requires_grad);peak=torch.cuda.max_memory_allocated()/2**30;assert peak<6.8
  if i==0:(D/'first_batch_gate.json').write_text(json.dumps(dict(phase='passed',no_optimizer_yet=True,finite_gradients=True,frozen_gradients_absent=True,peak_allocated_gib=peak),indent=2))
  torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.requires_grad],1.);scaler.step(opt);scaler.update();records.append(dict(step=i+1,id=sid,timestep=int(t),noise_sha256=hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest(),loss=float(loss),peak_allocated_gib=peak));status('training',i+1)
  if i+1 in [2,25,50,75,100]:save(i+1)
 opt.zero_grad(set_to_none=True);del opt;torch.cuda.empty_cache();model.eval();from ldm.models.diffusion.ddim import DDIMSampler;sampler=DDIMSampler(model.ldm);sys.path.insert(0,str(R/'work/lane_geometry/hybrid'));from road_only import compose_road
 from skimage.metrics import structural_similarity
 import lpips
 perceptual=lpips.LPIPS(net='alex',spatial=True,verbose=False).cpu().eval();metrics=[];page='<meta charset="utf-8"><style>body{background:#142033;color:white;font:18px system-ui;margin:25px}a{color:#9df}img{width:32%}</style><h1>CompVis '+variant+'100-updatepilot</h1><p>Four fixed development VAL examples. Snowy / road-only model output / clean reference. Clean reference never supplied to inference. Not independent lane accuracy.</p><a href="../../progress/">Live progress</a>'
 for j,sid in enumerate(protocol['validation_ids'][:4]):
  status('generating_validation',j,4);row=rows[sid];ck,source,cond=load(sid);assert ck['split']=='validation' and ck['supervision'] is None;torch.manual_seed(20261126+j);dest=D/sid;dest.mkdir(exist_ok=True)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
   model(torch.zeros((1,3,128,128),device='cuda'),torch.tensor([500],device='cuda'),cond,**source,variant=variant);samples,_=sampler.sample(S=50,conditioning=cond,batch_size=1,shape=(3,128,128),eta=0.,verbose=False);rgb=model.ldm.decode_first_stage(samples);raw=((rgb[0].clamp(-1,1)+1)*127.5).permute(1,2,0).cpu().numpy().round().clip(0,255).astype(np.uint8)
  native=Image.open(row['snowy']).convert('RGB');l,top,nw,nh=ck['source']['content_box'];permit=ck['permit_working'][0,0].numpy().astype(bool);gen=np.array(Image.fromarray(raw[top:top+nh,l:l+nw]).resize(native.size,Image.Resampling.LANCZOS));mask=np.array(Image.fromarray((permit[top:top+nh,l:l+nw]*255).astype(np.uint8)).resize(native.size,Image.Resampling.NEAREST))>127;out,allow=compose_road(np.array(native),gen,mask,np.zeros_like(mask));Image.fromarray(out).save(dest/'output.png');native.save(dest/'snowy.png');clean=Image.open(row['clean']).convert('RGB');clean.save(dest/'clean_reference.png');actual=canvas(Image.fromarray(out))[0];truth=canvas(clean)[0];roi=permit;score=dict(id=sid,outside_permit_changed_pixels=0,seed=20261126+j,permit_pixels=int(roi.sum()))
  if roi.any():
   mse=float(((actual.astype(np.float32)-truth)**2)[roi].mean());_,ss=structural_similarity(truth,actual,data_range=255,channel_axis=2,full=True);pa=torch.from_numpy(actual.copy()).permute(2,0,1)[None].float()/127.5-1;ta=torch.from_numpy(truth.copy()).permute(2,0,1)[None].float()/127.5-1
   with torch.no_grad():dist=perceptual(pa,ta)[0,0].numpy()
   score.update(road_psnr=float(-10*np.log10(max(mse,1e-12)/255**2)),road_ssim=float(ss.mean(-1)[roi].mean()),road_lpips=float(dist[roi].mean()))
  else:score['warning']='Empty predicted permit; no image-quality score fabricated.'
  metrics.append(score);page+='<h2>'+sid+'</h2>'+''.join('<img src="'+sid+'/'+f+'.png">' for f in ['snowy','output','clean_reference']);(D/'index.html').write_text(page);(D/'metrics.json').write_text(json.dumps(dict(records=metrics,scope='Four developmentVALpredicted-permit image metrics, not verifiedlaneaccuracy or finalTEST.'),indent=2))
 status('complete',100);print('Pilot '+variant+' complete')
except Exception as e:(D/'failure.json').write_text(json.dumps(dict(phase='failed',error=repr(e),updated_at=time.time())));status('failed',len(records));raise
