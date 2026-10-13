from pathlib import Path
import gzip,hashlib,io,json,time
import numpy as np
source=Path('work/data/nonimage/sources/covertype/covtype.data.gz');base=Path('work/data/nonimage/covertype/natural');out=Path('work/next_round/tabular/source_row_audit.json')
x=np.load(base/'features.npy',mmap_mode='r');y=np.load(base/'labels.npy',mmap_mode='r');start=time.perf_counter();row=0;blocks=0
with gzip.open(source,'rt') as f:
 while True:
  lines=[]
  for i in range(4096):
   line=f.readline()
   if not line:break
   lines.append(line)
  if not lines:break
  a=np.loadtxt(io.StringIO(''.join(lines)),delimiter=',',dtype=np.int32,ndmin=2)
  if a.shape[1]!=55 or not np.array_equal(a[:,:54].astype(np.float32),x[row:row+len(a)]) or not np.array_equal(a[:,54]-1,y[row:row+len(a)]):raise ValueError('Originalgzip/sourcearraysrowalignmentmismatch')
  row+=len(a);blocks+=1
if row!=581012 or row!=len(y):raise ValueError('Canonicalsourcecountmismatch')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
r=dict(status='canonical_original_gzip_every_row_and_label_verified',rows=row,blocks=blocks,feature_columns=54,label_rule='originalsourcecolumn54minus1',source_gzip_sha256=sha(source),feature_array_sha256=sha(base/'features.npy'),label_array_sha256=sha(base/'labels.npy'),seconds=time.perf_counter()-start,verification_source_sha256=sha(__file__))
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
