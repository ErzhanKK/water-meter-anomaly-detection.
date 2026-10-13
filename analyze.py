import pandas as pd, numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent
DATA=P/'data'
OUT=P/'results'
OUT.mkdir(exist_ok=True)
r=pd.read_csv(DATA/'pokazaniya.csv',sep=';',parse_dates=['data']).sort_values(['anon_id','data']).reset_index(drop=True)
e=pd.read_csv(DATA/'sobytiya.csv',sep=';',parse_dates=['data'])
r['days']=r.groupby('anon_id').data.diff().dt.days
r['delta']=r.groupby('anon_id').pokazanie.diff()
r['daily']=r.delta.where(r.days.eq(1))
r['valid']=r.days.eq(1)
valid=r[r.valid].copy(); positive=valid[valid.daily.ge(0)].copy()
summary=[]
def add(name,df,sort=None):
 ids=df.anon_id.nunique();examples=(df.sort_values(sort,ascending=False).anon_id.drop_duplicates().head(10).astype(str).tolist() if sort else df.anon_id.drop_duplicates().head(10).astype(str).tolist())
 summary.append({'test':name,'devices':ids,'records':len(df),'examples':', '.join(examples)})
neg=valid[valid.daily.lt(0)].copy();neg['severity']=-neg.daily;add('negative',neg,'severity')
# baseline median positive nonzero daily per device, minimum 10 observations
base=positive[positive.daily.gt(0)].groupby('anon_id').daily.agg(['median','count']);base.loc[base['count']<10,'median']=np.nan
valid=valid.join(base['median'].rename('baseline'),on='anon_id')
valid['ratio']=valid.daily/valid.baseline
spike=valid[(valid.daily>2)&(valid.ratio>=5)&(valid.baseline>=0.01)].copy();add('spike >2m3 and >=5x median',spike,'ratio')
impossible=valid[valid.daily>100].copy();add('extreme >100m3 daily',impossible,'daily')
# runs continuous calendar days, positive >.01, packets >0, at least 14 days
r['leak_ok']=r.valid & r.daily.gt(.01) & r.paketov_za_sutki.gt(0) & r.daily.le(10)
r['zero_ok']=r.valid & r.daily.abs().le(.001) & r.paketov_za_sutki.gt(0)
for label,flag,n in [('leak','leak_ok',14),('zero','zero_ok',14)]:
 group=(~r[flag] | ~r.valid).groupby(r.anon_id).cumsum()
 # group ids reset on false; true records sharing group indicate continuous days
 runs=r[r[flag]].groupby(['anon_id',group[r[flag]]]).agg(start=('data','min'),end=('data','max'),length=('data','size'),median_daily=('daily','median')).reset_index(level=1,drop=True).reset_index()
 runs=runs[runs.length>=n].sort_values('length',ascending=False)
 runs.to_csv(OUT/f'{label}_runs.csv',index=False);add(f'{label} run >= {n} days',runs,'length')
# step change: compare first vs second halves of each device calendar span, positive daily capped <=10, >=20 days each, medians and ratio
v=valid[(valid.daily>=0)&(valid.daily<=10)].copy()
limits=r.groupby('anon_id').data.agg(['min','max']);mid=limits['min']+(limits['max']-limits['min'])/2
v=v.join(mid.rename('mid'),on='anon_id');v['half']=np.where(v.data<=v.mid,'first','second')
a=v.groupby(['anon_id','half']).daily.agg(['median','count']).unstack('half');a.columns=['_'.join(x) for x in a.columns];a=a.dropna()
a=a[(a.count_first>=20)&(a.count_second>=20)&(a.median_first>=.02)&(a.median_second>=.02)].copy();a['fold']=np.maximum(a.median_first/a.median_second,a.median_second/a.median_first)
steps=a[a.fold>=3].sort_values('fold',ascending=False).reset_index();add('step >=3x half medians',steps,'fold')
# events magnet day and adjacent; test negative, spikes, low consumption association
mag=e[e.sobytie.str.contains('магнит',case=False,na=False)][['anon_id','data']].drop_duplicates();mag['magnet']=True
valid=valid.merge(mag,on=['anon_id','data'],how='left');valid['magnet']=valid.magnet.fillna(False)
valid['negative']=valid.daily.lt(0);valid['spike']=valid.daily.gt(2)&valid.ratio.ge(5)&valid.baseline.ge(.01)
for x in ['negative','spike']:
 tab=pd.crosstab(valid.magnet,valid[x]);print('magnet',x,'\n',tab)
 # relative rate
 print('rates',valid.groupby('magnet')[x].mean().to_dict())
valid.to_csv(OUT/'daily_calculated.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print('total valid',len(valid),'gap',int(r.days.gt(1).sum()),'magnet dates',len(mag))
print('event types',e.sobytie.value_counts().to_dict())
