"""Reproducible HistGradientBoosting baseline for the PathSnap road matcher.

Route labels are ordered arcs, not one label per GPS row. They are projected to
the nearest GPS sample; if multiple route arcs project to one sample, the
geometrically closest arc is retained. Candidate generation is trip-local.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

FEATURES = ["distance_m", "heading_error_rad", "gps_speed_mps", "gps_bearing_sin", "gps_bearing_cos", "segment_length_m", "segment_bearing_sin", "segment_bearing_cos", "candidate_degree_from", "candidate_degree_to", "endpoint_min_distance_m", "endpoint_max_distance_m", "time_delta_s"]
LAT_M = 111195.0

def read_trip(path):
    s=path.name
    track=np.loadtxt(path/f"{s}.track",dtype=float,ndmin=2)
    nodes=np.loadtxt(path/f"{s}.nodes",dtype=float,ndmin=2)
    arcs=np.loadtxt(path/f"{s}.arcs",dtype=int,ndmin=2)
    rp=path/f"{s}.route"
    route=np.loadtxt(rp,dtype=int,ndmin=1).ravel() if rp.exists() else np.empty(0,int)
    return track,nodes,arcs,route

def xy(lonlat, lat0):
    return np.column_stack((lonlat[:,0]*LAT_M*math.cos(math.radians(lat0)),lonlat[:,1]*LAT_M))

def features_for_trip(path,radius_m=80.0,max_candidates=40):
    track,nodes,arcs,route=read_trip(path)
    if len(track)<2 or len(arcs)==0:
        return np.empty((0,len(FEATURES))),np.empty(0,int),{"trip":path.name,"gps":len(track),"route":len(route),"rows":0,"miss_rate":1.0}
    lat0=float(np.nanmedian(track[:,1])); gps=xy(track[:,:2],lat0); nxy=xy(nodes,lat0)
    valid=(arcs[:,0]>=0)&(arcs[:,1]>=0)&(arcs[:,0]<len(nodes))&(arcs[:,1]<len(nodes))
    arc_ids=np.flatnonzero(valid); ends=arcs[valid]; p0,p1=nxy[ends[:,0]],nxy[ends[:,1]]
    vec=p1-p0; seglen=np.maximum(np.linalg.norm(vec,axis=1),.1); bearing=np.arctan2(vec[:,1],vec[:,0]); mid=(p0+p1)/2
    degree=np.bincount(ends.ravel(),minlength=len(nodes)); ts=track[:,2]
    dt=np.diff(ts,prepend=ts[0]); dt[~np.isfinite(dt)|(dt<=0)]=1
    motion=np.diff(gps,axis=0,prepend=gps[:1]); gb=np.arctan2(motion[:,1],motion[:,0]); speed=np.linalg.norm(motion,axis=1)/dt
    gps_tree=cKDTree(gps); mid_tree=cKDTree(mid)
    labels_at={}
    for aid in route:
        if 0<=aid<len(arcs) and valid[aid]:
            j=int(np.searchsorted(arc_ids,aid)); d,oi0=gps_tree.query(mid[j]); oi=int(oi0)
            if oi not in labels_at or float(d)<labels_at[oi][0]: labels_at[oi]=(float(d),int(aid))
    rows=[]; ys=[]; missed=0
    for oi,(_,aid) in labels_at.items():
        nearby=mid_tree.query_ball_point(gps[oi],radius_m)
        c=np.asarray(nearby,dtype=int)
        if not len(c): c=np.array([int(mid_tree.query(gps[oi])[1])])
        dist=np.linalg.norm(mid[c]-gps[oi],axis=1)
        if len(c)>max_candidates:
            keep=np.argsort(dist)[:max_candidates]; c=c[keep]; dist=dist[keep]
        if aid not in arc_ids[c]: missed+=1; continue
        delta=np.arctan2(np.sin(bearing[c]-gb[oi]),np.cos(bearing[c]-gb[oi]))
        e0=np.linalg.norm(p0[c]-gps[oi],axis=1); e1=np.linalg.norm(p1[c]-gps[oi],axis=1)
        rows.extend(np.column_stack((dist,abs(delta),np.full(len(c),speed[oi]),np.full(len(c),np.sin(gb[oi])),np.full(len(c),np.cos(gb[oi])),seglen[c],np.sin(bearing[c]),np.cos(bearing[c]),degree[ends[c,0]],degree[ends[c,1]],np.minimum(e0,e1),np.maximum(e0,e1),np.full(len(c),dt[oi]))))
        ys.extend((arc_ids[c] == aid).astype(int).tolist())
    meta={"trip":path.name,"gps":int(len(track)),"route":int(len(route)),"labelled_observations":int(len(labels_at)),"rows":len(ys),"candidate_misses":int(missed),"miss_rate":float(missed/max(len(labels_at),1)),"median_interval_s":float(np.median(np.diff(ts))) if len(ts)>1 else None}
    return np.asarray(rows,dtype=np.float32).reshape(-1,len(FEATURES)),np.asarray(ys,dtype=int),meta

def collect(partition):
    xs=[];ys=[];groups=[];info=[]
    for trip in sorted(p for p in partition.iterdir() if p.is_dir()):
        x,y,m=features_for_trip(trip); info.append(m)
        if len(y): xs.append(x);ys.append(y);groups.extend([trip.name]*len(y))
    if not xs: raise RuntimeError(f"No usable labels in {partition}")
    return np.vstack(xs),np.concatenate(ys),np.asarray(groups),info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--trials',type=int,default=0);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); x,yr,g,train_info=collect(a.data/'train'); xt,ytr,gt,test_info=collect(a.data/'student_test')
    y=yr.astype(int); yt=ytr.astype(int)
    cv=GroupKFold(n_splits=min(5,len(np.unique(g))))
    base=dict(loss='log_loss',learning_rate=.08,max_iter=160,max_leaf_nodes=31,l2_regularization=1.,min_samples_leaf=20,random_state=42)
    folds=[]
    for fold,(tr,va) in enumerate(cv.split(x,y,g),1):
        m=HistGradientBoostingClassifier(**base); start=time.perf_counter();m.fit(x[tr],y[tr]); secs=time.perf_counter()-start
        pr=m.predict(x[va]); pp=m.predict_proba(x[va])[:,1];folds.append({"fold":fold,"validation_trips":sorted(set(g[va])),"accuracy":float(accuracy_score(y[va],pr)),"macro_f1":float(f1_score(y[va],pr,average='macro',zero_division=0)),"log_loss":float(log_loss(y[va],pp,labels=[0,1])),"fit_seconds":secs})
    best=base.copy(); trials=[]
    if a.trials:
        from skopt import gp_minimize
        from skopt.space import Integer,Real
        space=[Real(.03,.18,name='learning_rate'),Integer(80,260,name='max_iter'),Integer(15,63,name='max_leaf_nodes'),Integer(10,80,name='min_samples_leaf'),Real(.01,10.,prior='log-uniform',name='l2_regularization')]
        def objective(v):
            params=dict(base,learning_rate=v[0],max_iter=v[1],max_leaf_nodes=v[2],min_samples_leaf=v[3],l2_regularization=v[4]); scores=[]
            for tr,va in cv.split(x,y,g):
                m=HistGradientBoostingClassifier(**params).fit(x[tr],y[tr]);scores.append(f1_score(y[va],m.predict(x[va]),average='macro',zero_division=0))
            score=float(np.mean(scores));trials.append({"params":{s.name:(float(value) if isinstance(s,Real) else int(value)) for s,value in zip(space,v)},"mean_cv_macro_f1":score});return -score
        opt=gp_minimize(objective,space,n_calls=a.trials,n_initial_points=min(5,a.trials),random_state=42)
        best=dict(base,learning_rate=float(opt.x[0]),max_iter=int(opt.x[1]),max_leaf_nodes=int(opt.x[2]),min_samples_leaf=int(opt.x[3]),l2_regularization=float(opt.x[4]))
        folds=[]
        for fold,(tr,va) in enumerate(cv.split(x,y,g),1):
            m=HistGradientBoostingClassifier(**best); start=time.perf_counter();m.fit(x[tr],y[tr]); secs=time.perf_counter()-start
            pp=m.predict_proba(x[va])[:,1]; pr=(pp>=.5).astype(int)
            folds.append({"fold":fold,"validation_trips":sorted(set(g[va])),"accuracy":float(accuracy_score(y[va],pr)),"macro_f1":float(f1_score(y[va],pr,average='macro',zero_division=0)),"log_loss":float(log_loss(y[va],pp,labels=[0,1])),"fit_seconds":secs})
    final=HistGradientBoostingClassifier(**best);t0=time.perf_counter();final.fit(x,y);fit_s=time.perf_counter()-t0
    t0=time.perf_counter();probs=final.predict_proba(xt)[:,1];infer_s=time.perf_counter()-t0;pred=(probs>=.5).astype(int)
    student={"candidate_accuracy":float(accuracy_score(yt,pred)),"candidate_macro_f1":float(f1_score(yt,pred,average='macro',zero_division=0)),"candidate_log_loss":float(log_loss(yt,probs,labels=[0,1])),"inference_seconds":infer_s,"candidate_rows":int(len(yt)),"positive_candidate_rate":float(np.mean(yt))}
    joblib.dump({"model":final,"features":FEATURES,"parameters":best,"label":"reference_candidate"},a.output/'histgradientboosting.joblib',compress=3)
    report={"model":"HistGradientBoostingClassifier","seed":42,"features":FEATURES,"parameters":best,"training_rows":len(y),"training_trips":len(set(g)),"student_test_rows":len(yt),"cv":{"split":"GroupKFold by trip","folds":folds,"mean_macro_f1":float(np.mean([q['macro_f1'] for q in folds]))},"student_test":student,"training_seconds":fit_s,"bayesian_trials":trials,"data_quality":{"train_trip_inventory":train_info,"student_test_trip_inventory":test_info,"train_mean_candidate_miss_rate":float(np.mean([q['miss_rate'] for q in train_info])),"student_test_mean_candidate_miss_rate":float(np.mean([q['miss_rate'] for q in test_info]))},"label_note":"Binary candidate-level classification: a candidate arc is positive when it matches the trip-local reference route arc projected to that GPS observation. Trip-local arc IDs are never treated as global classes. Metrics are candidate classification metrics, not observation-level route accuracy."}
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='data_quality'},indent=2))
if __name__=='__main__':main()
