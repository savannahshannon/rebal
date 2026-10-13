from pathlib import Path
import hashlib,json
import numpy as np
rawbase=Path('work/data/nonimage/covertype/natural');dest=Path('work/experiments/rebal_next_tabular/data');raw=np.load(rawbase/'features.npy',mmap_mode='r');labels=np.load(dest/'labels.npy',mmap_mode='r');parts=dict(np.load(dest/'split.npz',allow_pickle=False));scaler=dict(np.load(dest/'scaler.npz',allow_pickle=False));meta=json.loads((dest/'metadata.json').read_text())
fit=np.asarray(raw[parts['fit'],:10],dtype=np.float64);mean=fit.sum(0)/len(fit);scale=np.sqrt(((fit-mean)**2).sum(0)/len(fit));scale[scale==0]=1
if not np.allclose(mean,scaler['mean'],rtol=0,atol=1e-12) or not np.allclose(scale,scaler['scale'],rtol=0,atol=1e-12):raise ValueError('Fitonlypopulationstatisticsmismatch')
numeric=np.load(dest/'numeric.npy',mmap_mode='r');flat=np.load(dest/'flat.npy',mmap_mode='r');cat=np.load(dest/'categorical.npy',mmap_mode='r');maxdiff=0.
for begin in range(0,len(labels),65536):
 end=min(len(labels),begin+65536);x=np.asarray(raw[begin:end]);expected=((x[:,:10].astype(np.float64)-mean)/scale).astype(np.float32);maxdiff=max(maxdiff,float(np.max(np.abs(expected-numeric[begin:end]))))
 if not np.allclose(expected,numeric[begin:end],rtol=0,atol=1e-6) or not np.array_equal(flat[begin:end,:10],numeric[begin:end]) or not np.array_equal(flat[begin:end,10:],x[:,10:]):raise ValueError('Preparedmodelinputmismatch')
 if not np.array_equal(np.eye(4,dtype=np.float32)[cat[begin:end,0]],x[:,10:14]) or not np.array_equal(np.eye(40,dtype=np.float32)[cat[begin:end,1]],x[:,14:54]):raise ValueError('Losslesscategoricaltokenmismatch')
counts=np.bincount(labels[parts['fit']],minlength=7);rank=np.argsort(-counts,kind='stable').tolist()
if rank!=meta['class_rank']:raise ValueError('Fitclassrankmismatch')
r=dict(status='fit_only_population_scaler_and_all_prepared_inputs_independently_verified',rows=len(labels),fit_rows=len(parts['fit']),tolerance=1e-6,maximum_numeric_absolute_difference=maxdiff,scaler_mean_difference_max=float(np.max(np.abs(mean-scaler['mean']))),scaler_scale_difference_max=float(np.max(np.abs(scale-scaler['scale']))),categorical_reconstruction_exact=True,flat_binary_preserved_exactly=True,class_rank_from_fit=rank,class_names=json.loads((rawbase/'metadata.json').read_text())['classes'],verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path('work/next_round/tabular/transform_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
