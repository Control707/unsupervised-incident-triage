"""Rebuild spatial plot from cached embeddings; preserve geometric aspect."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'poster_work/mpl-cache'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache-q2/numba'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hdbscan,umap
OUT=ROOT/'poster_work/hires_figures';OUT.mkdir(exist_ok=True)
z=np.load(ROOT/'paper_q2_results/joint_embeddings.npz')
labels=hdbscan.HDBSCAN(min_cluster_size=16,min_samples=5,max_cluster_size=120,metric='euclidean',cluster_selection_method='eom').fit_predict(z['umap_5d'])
coords=umap.UMAP(n_components=2,n_neighbors=15,min_dist=.12,metric='cosine',random_state=42,n_jobs=1).fit_transform(z['embeddings'])
counts={int(c):int((labels==c).sum()) for c in sorted(set(labels))}
assert counts[5]==92 and counts[7]==35 and counts[3]==23 and counts[2]==31 and counts[8]==21,counts
assert counts[-1]==336,counts
np.savez(OUT/'spatial_plot_data.npz',coords=coords,labels=labels,ticket_ids=z['ticket_ids'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':24,'axes.labelsize':27,'axes.labelweight':'bold','xtick.labelsize':23,'ytick.labelsize':23,'axes.linewidth':1.1,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(1032/72,780/72),facecolor='white')
ax=fig.add_axes([.12,.16,.85,.76])
noise=labels==-1
ax.scatter(coords[noise,0],coords[noise,1],s=65,color='#D9D9D9',alpha=.65,zorder=1)
cmap=plt.get_cmap('tab20')
for c in sorted(set(labels)-{-1}):
 p=coords[labels==c]; ax.scatter(p[:,0],p[:,1],s=100,color=cmap(c/13),alpha=.85,edgecolors='white',linewidths=.25,zorder=2)
# Place concise labels in reserved gaps rather than enlarging paper callouts.
callouts=[(12,'C12 · Speech',(3.95,4.70)),(4,'C4 · Task cast',(5.15,2.22)),(10,'C10 · Checksums',(7.5,5.45)),(3,'C3 · PyArrow',(9.0,6.15)),(5,'C5 · Pool.map',(11.3,4.25)),(0,'C0 · Duplicate keys',(6.45,1.00)),(1,'C1 · Columns',(10.8,2.13))]
for cid,label,xytext in callouts:
 xy=np.median(coords[labels==cid],axis=0)
 ax.annotate(label,xy=xy,xytext=xytext,fontsize=23,fontweight='bold',ha='center',va='center',bbox={'boxstyle':'round,pad=.22','fc':'white','ec':'#666666','alpha':.96,'lw':1.1},arrowprops={'arrowstyle':'->','color':'#444444','lw':1.6},zorder=5)
ax.set(xlim=(2.8,12.8),ylim=(.7,7.3),xlabel='UMAP dimension 1',ylabel='UMAP dimension 2')
ax.set_aspect('equal',adjustable='box')
ax.grid(color='#E4E4E4',lw=1);ax.set_axisbelow(True)
for s in ax.spines.values():s.set_color('#999999')
fig.text(.12,.96,'GitHub tickets · Bounded EOM',fontsize=28,fontweight='bold',va='top')
fig.text(.12,.035,'Gray = noise (38.1%)   •   Colored points = subsystem clusters',fontsize=22,color='#555555')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'spatial.{ext}',dpi=240,facecolor='white')
(OUT/'spatial-provenance.json').write_text(json.dumps({'input':'paper_q2_results/joint_embeddings.npz','counts':counts,'projection':{'n_components':2,'n_neighbors':15,'min_dist':.12,'metric':'cosine','random_state':42},'canvas_points':[1032,780],'dpi':240,'equal_data_aspect':True},indent=2))
print('Verified counts',counts); print('Saved spatial PNG/PDF/SVG')
