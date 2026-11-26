"""CPU image metrics on fixed predicted permit; not lane accuracy."""
from source_data import R,canvas
from pathlib import Path
from PIL import Image
import json,numpy as np,torch,lpips
from skimage.metrics import structural_similarity
O=R/'outputs/lane_geometry/compvis_hybrid';D=O/'lama_fallback/eval100';torch.set_num_threads(4);metric=lpips.LPIPS(net='alex',spatial=True,verbose=False).cpu().eval();ids=json.loads((O/'pilot_protocol.json').read_text())['validation_ids'];results={}
for v in ['A1','B','C']:
 records=[]
 for sid in ids:
  d=D/v/sid;a=canvas(Image.open(d/'output.png').convert('RGB'))[0];b=canvas(Image.open(d/'clean_reference.png').convert('RGB'))[0];ck=torch.load(R/'work/lane_geometry/compvis_hybrid/cache_ego_v3'/(sid+'.pt'),weights_only=False);assert ck['supervision'] is None;roi=ck['permit_working'][0,0].numpy().astype(bool);assert roi.any();mse=((a.astype(float)-b)**2)[roi].mean();_,ss=structural_similarity(a,b,data_range=255,channel_axis=2,full=True)
  with torch.no_grad():dist=metric(torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/127.5-1,torch.from_numpy(b.copy()).permute(2,0,1)[None].float()/127.5-1)[0,0].numpy()
  records.append(dict(id=sid,road_lpips=float(dist[roi].mean()),road_psnr=float(-10*np.log10(max(mse,1e-12)/255**2)),road_ssim=float(ss.mean(-1)[roi].mean())))
 results[v]=dict(records=records,means={k:float(np.mean([r[k] for r in records])) for k in ['road_lpips','road_psnr','road_ssim']});(D/v/'metrics.json').write_text(json.dumps(results[v],indent=2))
report=dict(scope='Twenty frozen development validation images; same source-predicted permit across variants. Not verified lane accuracy or finalTEST.',variants=results,relative_lpips_improvement={v:1-results[v]['means']['road_lpips']/results['A1']['means']['road_lpips'] for v in ['B','C']});(D/'comparison_metrics.json').write_text(json.dumps(report,indent=2));print({v:r['means'] for v,r in results.items()})
