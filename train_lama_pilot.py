"""Finite100-update matched LaMa pilot; source-only conditioning and TRAIN-only targets."""
from lama_hybrid import LamaHybrid
from source_data import R,canvas
from PIL import Image
import torch,json,time,argparse,numpy as np,fcntl
v=argparse.ArgumentParser();v.add_argument('--variant',choices=['A1','B','C'],required=True);variant=v.parse_args().variant;O=R/'outputs/lane_geometry/compvis_hybrid';P=O/'lama_fallback';D=P/'pilot100'/variant;D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert json.loads((P/'backward_resume_gate.json').read_text())['phase']=='passed';torch.set_num_threads(4);torch.manual_seed(20261126);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);m=LamaHybrid().cuda();params=dict(m.named_parameters());initial=torch.load(P/'initial_trainable.pt',weights_only=True)
with torch.no_grad():
 for n,w in initial.items():params[n].copy_(w)
opt=torch.optim.AdamW([p for p in params.values() if p.requires_grad],lr=1e-4);protocol=json.loads((O/'pilot_protocol.json').read_text());rows={r['id']:r for r in json.loads((R/'work/road_restoration/manifest.json').read_text())};records=[];start=0
def weights():return {n:p.detach().cpu().clone() for n,p in params.items() if p.requires_grad}
if (D/'latest.pt').exists():
 ck=torch.load(D/'latest.pt',weights_only=False)
 with torch.no_grad():
  for n,w in ck['weights'].items():params[n].copy_(w)
 opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);start=ck['step'];records=ck['records']
def rgb(path):return torch.from_numpy(canvas(Image.open(path).convert('RGB'))[0].copy()).permute(2,0,1)[None].cuda().float()/255
def load(sid):
 ck=torch.load(R/'work/lane_geometry/compvis_hybrid/cache_ego_v3'/(sid+'.pt'),weights_only=False);assert ck['source_sha256']==rows[sid]['snowy_sha256'];s={k:([v.cuda() for v in value] if isinstance(value,list) else value.cuda() if torch.is_tensor(value) else value) for k,value in ck['source'].items() if k!='content_box'};return ck,s,rgb(rows[sid]['snowy']),ck['permit_working'].cuda().float()
def status(phase,i,total=100):
 report=dict(phase=phase,completed=i,total=total,records=records,updated_at=time.time());(D/'status.json').write_text(json.dumps(report));s=json.loads((O/'status.json').read_text());s.update(phase='LaMa '+variant+' '+phase,completed=i,total=total,training_steps=len(records),message='Finite matched100-update LaMa pilot; real validation images follow. No lane accuracy claim.',updated_at=time.time());(O/'status.json').write_text(json.dumps(s,indent=2))
for i in range(start,100):
 sid=protocol['train_ids'][i%64];ck,s,x,mask=load(sid);assert ck['split']=='train';target=rgb(rows[sid]['clean']);road=ck['supervision']['decoded_road_weight'].cuda();opt.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();pred=m(x,mask,s,variant);l1=((pred-target).abs()*road).sum()/(3*road.sum()).clamp_min(1);wx=road[:,:,:,1:]*road[:,:,:,:-1];wy=road[:,:,1:,:]*road[:,:,:-1,:];dx=(pred[:,:,:,1:]-pred[:,:,:,:-1])-(target[:,:,:,1:]-target[:,:,:,:-1]);dy=(pred[:,:,1:,:]-pred[:,:,:-1,:])-(target[:,:,1:,:]-target[:,:,:-1,:]);grad=(dx.abs()*wx).sum()/(3*wx.sum()).clamp_min(1)+(dy.abs()*wy).sum()/(3*wy.sum()).clamp_min(1);loss=l1+.05*grad;assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in params.values() if p.grad is not None);assert all(p.grad is None for p in m.base.parameters());peak=torch.cuda.max_memory_allocated()/2**30;assert peak<6.8 and torch.cuda.max_memory_reserved()/2**30<7.4
 if i==0:(D/'first_batch_gate.json').write_text(json.dumps(dict(phase='passed',peak_allocated_gib=peak,finite_gradients=True,no_optimizer_yet=True)))
 torch.nn.utils.clip_grad_norm_([p for p in params.values() if p.requires_grad],1);opt.step();records.append(dict(step=i+1,id=sid,loss=float(loss),road_l1=float(l1),gradient_l1=float(grad),peak_allocated_gib=peak));status('training',i+1)
 if i+1 in [2,25,50,75,100]:
  torch.save(dict(weights=weights(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),step=i+1,records=records),D/'latest.tmp');(D/'latest.tmp').replace(D/'latest.pt')
page='<meta charset="utf-8"><h1>LaMa '+variant+'100-update pilot</h1><p>Snowy / restored output / clean reference. Four fixed VAL images; not verified lane accuracy.</p>'
for j,sid in enumerate(protocol['validation_ids'][:4]):
 ck,s,x,mask=load(sid);assert ck['supervision'] is None;status('generating',j,4)
 with torch.no_grad():pred=m(x,mask,s,variant);assert torch.isfinite(pred).all()
 l,t,w,h=ck['source']['content_box'];native=Image.open(rows[sid]['snowy']).convert('RGB');raw=(pred[0].clamp(0,1)*255).permute(1,2,0).cpu().numpy().round().astype(np.uint8);gen=np.array(Image.fromarray(raw[t:t+h,l:l+w]).resize(native.size,Image.Resampling.LANCZOS));allow=np.array(Image.fromarray((ck['permit_working'][0,0,t:t+h,l:l+w].numpy()*255).astype(np.uint8)).resize(native.size,Image.Resampling.NEAREST))>127;out=np.array(native).copy();out[allow]=gen[allow];assert np.array_equal(out[~allow],np.array(native)[~allow]);dest=D/sid;dest.mkdir(exist_ok=True);native.save(dest/'snowy.png');Image.fromarray(out).save(dest/'output.png');Image.open(rows[sid]['clean']).save(dest/'clean_reference.png');page+='<h2>'+sid+'</h2>'+''.join('<img width="32%" src="'+sid+'/'+f+'.png">' for f in ['snowy','output','clean_reference']);(D/'index.html').write_text(page)
status('complete',100)
