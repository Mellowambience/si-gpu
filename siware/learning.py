"""Bounded reconstruction learning experiments, nomination, promotion and rollback."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
from .__main__ import pair
from .core import LearnedUpscaler, psnr
from .motion_lab import evaluate_case


def stamp():return datetime.now(timezone.utc).isoformat()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as file:
        json.dump(value,file,indent=2,allow_nan=False)
        file.flush();os.fsync(file.fileno())
    os.replace(temporary,path)


@contextmanager
def writer_lock(root):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/"writer.lock"
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError("Learning lab already has a writer; stale locks require manual inspection")
    try:
        with os.fdopen(fd,"w") as file:file.write(json.dumps({"pid":os.getpid(),"started":stamp()}))
        yield
    finally:lock.unlink()


def model_path(root,identifier):
    if not isinstance(identifier,str) or not re.fullmatch(r"[a-z0-9-]{1,80}",identifier):
        raise ValueError("invalid model identifier")
    return Path(root)/"models"/(identifier+".npz")


def config_default():
    return {"scale":2,"anchor_seeds":list(range(12)),"validation_seeds":list(range(100,112)),
        "regression_seeds":list(range(200,212)),"audit_seeds":list(range(5000,5008)),
        "new_pairs_per_cycle":8,"replay_capacity":32,"ridges":[.001,.003,.008,.009,.0095,.01,.03,.1],
        "min_mean_gain_db":.01,"max_scene_regression_db":.05,
        "max_temporal_mse_ratio":1.02,"max_flicker_ratio":1.05,
        "cpu_reference_p95_ratio":1.5,"max_dataset_mib":512,
        "gpu_combined_p95_limits_ms":{"1080p":3,"1440p":4},
        "gpu_idle_utilization_max":20,"gpu_min_free_mib":768}


def validate_config(config):
    if config["scale"]!=2:raise ValueError("First learning release supports 2x only")
    groups=[config[k] for k in ("anchor_seeds","validation_seeds","regression_seeds","audit_seeds")]
    flattened=[seed for group in groups for seed in group]
    if any(type(seed) is not int or not 0<=seed<10000 for seed in flattened) or len(flattened)!=len(set(flattened)):
        raise ValueError("scene roles must be distinct, nonnegative seeds below 10000")
    if any(not group for group in groups):raise ValueError("empty split")
    audit=config["audit_seeds"]
    if audit!=list(range(min(audit),min(audit)+len(audit))) or max(seed for group in groups[:3] for seed in group)>=min(audit):
        raise ValueError("Audit seed pool must be contiguous and above all development scene seeds")
    if not 1<=config["new_pairs_per_cycle"]<=16 or not 8<=config["replay_capacity"]<=64:
        raise ValueError("invalid bounded replay settings")
    if not 1<=len(config["ridges"])<=8 or any(not np.isfinite(x) or x<=0 for x in config["ridges"]):
        raise ValueError("ridge candidates must be finite positive values")
    for key in ("min_mean_gain_db","max_scene_regression_db","max_temporal_mse_ratio",
                "max_flicker_ratio","cpu_reference_p95_ratio","max_dataset_mib"):
        if not np.isfinite(config[key]) or config[key]<=0:raise ValueError(f"invalid gate {key}")


def cached_pair(root,seed,scale):
    path=Path(root)/"dataset"/f"seed-{seed}-scale-{scale}.npz"
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        low,high=pair(seed,scale)
        np.savez_compressed(path,low=low,high=high,seed=seed,scale=scale)
    with np.load(path,allow_pickle=False) as data:
        return data["low"].copy(),data["high"].copy()


def load_model(root,state,identifier):
    path=model_path(root,identifier)
    if identifier not in state["models"] or digest(path)!=state["models"][identifier]["sha256"]:
        raise RuntimeError("model checksum mismatch")
    return LearnedUpscaler.load(path)


def initialize(root):
    root=Path(root).resolve()
    with writer_lock(root):
        if (root/"registry.json").exists():raise ValueError("Lab already initialized")
        config=config_default();validate_config(config)
        model=LearnedUpscaler.fit([cached_pair(root,s,2) for s in config["anchor_seeds"]],2)
        path=model_path(root,"baseline-0000");path.parent.mkdir(parents=True,exist_ok=True);model.save(path)
        state={"schema":1,"created":stamp(),"cycle":0,"accepted_lab":"baseline-0000","accepted_runtime":None,
            "previous_lab":[],"previous_runtime":[],"recent_seeds":[],"models":{"baseline-0000":{
                "sha256":digest(path),"kind":"linear residual","training_seeds":config["anchor_seeds"]}},"events":[]}
        atomic_json(root/"config.json",config);atomic_json(root/"registry.json",state)
        atomic_json(root/"DATA-PROVENANCE.json",{"rights":"Original procedural geometry generated locally by SI-GPU scene()",
            "source":"Synthetic static pairs only; not gameplay captures or a broad generalization dataset",
            "generator_sha256":digest(Path(__file__).with_name("__main__.py")),
            "roles_frozen_before_search":{k:config[k] for k in ("anchor_seeds","validation_seeds","regression_seeds","audit_seeds")},
            "fresh_training_seeds":"10000 + cycle*new_pairs_per_cycle; no evaluation-scene overlap"})
    return state


def static_scores(root,model,seeds):
    return [{"seed":seed,"psnr_db":psnr(high,model.upscale(low))}
            for seed in seeds for low,high in [cached_pair(root,seed,model.scale)]]


def quality_gate(reference,candidate,config):
    if len(reference)!=len(candidate) or not reference:raise ValueError("incomparable quality results")
    gains=[]
    for old,new in zip(reference,candidate):
        if old["seed"]!=new["seed"]:raise ValueError("quality scene mismatch")
        a,b=old["psnr_db"],new["psnr_db"]
        if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
            return {"pass":False,"reason":"nonfinite/perfect-score comparisons need explicit handling"}
        gains.append(b-a)
    mean=float(np.mean(gains));worst=float(min(gains))
    return {"pass":mean>=config["min_mean_gain_db"] and worst>=-config["max_scene_regression_db"],
            "mean_gain_db":mean,"worst_scene_gain_db":worst,"per_scene_gain_db":gains}


def regression_gate(reference,candidate,config):
    relaxed={**config,"min_mean_gain_db":1e-12}
    result=quality_gate(reference,candidate,relaxed)
    # Old scenes may stay flat; require no substantial per-scene loss, not an improvement on every category.
    result["pass"]=result.get("worst_scene_gain_db",-math.inf)>=-config["max_scene_regression_db"]
    return result


def temporal_scores(model):
    return [evaluate_case(seed,kind,noise,model=model)
            for seed in (300,301) for kind in ("translation","silhouette") for noise in (0,.025)]


def temporal_gate(reference,candidate,config):
    if len(reference)!=len(candidate):raise ValueError("temporal case mismatch")
    failures=[];worst_mse=0.;worst_flicker=0.
    for old,new in zip(reference,candidate):
        if (old["seed"],old["kind"],old["noise_std"])!=(new["seed"],new["kind"],new["noise_std"]):
            raise ValueError("temporal case mismatch")
        for group,limit in (("interior_mse",config["max_temporal_mse_ratio"]),
                            ("interior_aligned_error_change_mse",config["max_flicker_ratio"]),
                            ("boundary_aligned_error_change_mse",config["max_flicker_ratio"]),
                            ("disocclusion_mse",config["max_temporal_mse_ratio"])):
            a,b=old[group]["temporal"],new[group]["temporal"]
            if a is None and b is None:continue
            if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):failures.append(group+":invalid");continue
            # Absolute floor prevents numerical near-zero error from becoming an arbitrary huge ratio.
            ratio=(b+1e-10)/(a+1e-10)
            if group in ("interior_mse","disocclusion_mse"):worst_mse=max(worst_mse,ratio)
            else:worst_flicker=max(worst_flicker,ratio)
            if b>a*limit+1e-10:failures.append(f"{old['seed']}/{old['kind']}/{old['noise_std']}/{group}")
    return {"pass":not failures,"failures":failures,"worst_mse_ratio":worst_mse,"worst_flicker_ratio":worst_flicker}


def timing_compare(reference,candidate,low,samples=30):
    # Interleave evaluations rather than timing every baseline sample before every candidate sample.
    reference.upscale(low);candidate.upscale(low)
    times={"reference":[],"candidate":[]}
    for i in range(samples):
        entries=(("reference",reference),("candidate",candidate)) if i%2==0 else (("candidate",candidate),("reference",reference))
        for label,model in entries:
            start=time.perf_counter();model.upscale(low);times[label].append((time.perf_counter()-start)*1000)
    return {key:{"p50_ms":float(np.median(value)),"p95_ms":float(np.percentile(value,95)),"samples_ms":value}
            for key,value in times.items()}


def run_cycle(root,project,work,gpu=False,max_seconds=300):
    root=Path(root).resolve();project=Path(project).resolve();work=Path(work).resolve()
    started=time.monotonic()
    with writer_lock(root):
        if (root/"PAUSE").exists():raise RuntimeError("Lab paused by PAUSE marker")
        config=read_json(root/"config.json");validate_config(config)
        data_bytes=sum(p.stat().st_size for p in (root/"dataset").glob("*.npz"))
        if data_bytes>config["max_dataset_mib"]*2**20:raise RuntimeError("Dataset storage budget exceeded")
        state=read_json(root/"registry.json");reference=load_model(root,state,state["accepted_lab"])
        config_hash=digest(root/"config.json")
        cycle=state["cycle"]+1
        new=list(range(10000+cycle*config["new_pairs_per_cycle"],10000+(cycle+1)*config["new_pairs_per_cycle"]))
        replay=(state["recent_seeds"]+new)[-config["replay_capacity"]:]
        training=config["anchor_seeds"]+replay
        run=root/"experiments"/f"cycle-{cycle:06d}";run.mkdir(parents=True,exist_ok=False)
        state["cycle"]=cycle;state["recent_seeds"]=replay;atomic_json(root/"registry.json",state)
        audit_seeds=[s+(cycle-1)*len(config["audit_seeds"]) for s in config["audit_seeds"]]
        if max(audit_seeds)>=10000:raise RuntimeError("Predeclared audit seed pool exhausted; define a new release split")
        manifest={"cycle":cycle,"started":stamp(),"reference":state["accepted_lab"],"config_sha256":config_hash,
            "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,
            "training_seeds":training,"new_seeds":new,"retained_anchor_seeds":config["anchor_seeds"],
            "validation_seeds":config["validation_seeds"],"regression_seeds":config["regression_seeds"],
            "audit_seeds":audit_seeds,"candidate_results":[],"nominee":None,
            "promotion_scope":"Lab nomination only; explicit promotion command required","gpu_requested":gpu}
        try:
            pairs=[cached_pair(root,s,2) for s in training]
            validation=static_scores(root,reference,config["validation_seeds"])
            regression=static_scores(root,reference,config["regression_seeds"])
            candidates=[]
            for i,ridge in enumerate(config["ridges"]):
                if time.monotonic()-started>max_seconds:raise RuntimeError("Learning cycle time budget exhausted")
                candidate=LearnedUpscaler.fit(pairs,2,ridge)
                identifier=f"cycle-{cycle:06d}-candidate-{i+1:02d}"
                path=model_path(root,identifier);candidate.save(path)
                result={"id":identifier,"ridge":ridge,"sha256":digest(path),
                    "validation":quality_gate(validation,static_scores(root,candidate,config["validation_seeds"]),config),
                    "old_scenes":regression_gate(regression,static_scores(root,candidate,config["regression_seeds"]),config),
                    "lab_eligible":False,"runtime_eligible":False}
                state["models"][identifier]={"sha256":result["sha256"],"training_seeds":training,"ridge":ridge,"kind":"linear residual"}
                manifest["candidate_results"].append(result)
                if result["validation"]["pass"] and result["old_scenes"]["pass"]:candidates.append((result,candidate))
            # Temporal/cost results are development checks. The audit is opened only for the frozen winner.
            if candidates:
                reference_temporal=temporal_scores(reference)
                eligible=[]
                for result,candidate in candidates:
                    if time.monotonic()-started>max_seconds:raise RuntimeError("Temporal evaluation time budget exhausted")
                    result["temporal"]=temporal_gate(reference_temporal,temporal_scores(candidate),config)
                    result["cpu_timing"]=timing_compare(reference,candidate,pairs[0][0])
                    result["cpu_cost_pass"]=result["cpu_timing"]["candidate"]["p95_ms"]<=result["cpu_timing"]["reference"]["p95_ms"]*config["cpu_reference_p95_ratio"]
                    if result["temporal"]["pass"] and result["cpu_cost_pass"]:eligible.append((result,candidate))
                if eligible:
                    result,candidate=max(eligible,key=lambda x:x[0]["validation"]["mean_gain_db"])
                    # Freeze/nominate based on validation before opening the predeclared audit.
                    manifest["audit_candidate"]=result["id"]
                    result["audit"]=quality_gate(static_scores(root,reference,audit_seeds),
                                                  static_scores(root,candidate,audit_seeds),config)
                    result["lab_eligible"]=result["audit"]["pass"]
                if eligible and result["lab_eligible"]:
                    manifest["nominee"]=result["id"]
                    if gpu:
                        if time.monotonic()-started>max_seconds:raise RuntimeError("No remaining time for GPU evidence")
                        result["gpu"]=gpu_evidence(project,work,run/result["id"],candidate,max_seconds-(time.monotonic()-started),config)
                        result["runtime_eligible"]=result["gpu"]["pass"]
            manifest["dataset_hashes"]={str(s):digest(root/"dataset"/f"seed-{s}-scale-2.npz") for s in training}
            manifest["completed"]=stamp();manifest["elapsed_seconds"]=time.monotonic()-started
            manifest["status"]="nomination_ready" if manifest["nominee"] else "no_candidate_passed"
            state["events"].append({"at":stamp(),"action":"cycle","cycle":cycle,"nominee":manifest["nominee"]})
            atomic_json(run/"report.json",manifest);atomic_json(root/"registry.json",state)
            write_dashboard(root)
            return manifest
        except Exception as error:
            manifest.update(status="failed",error=str(error),completed=stamp())
            atomic_json(run/"report.json",manifest);atomic_json(root/"registry.json",state)
            raise


def gpu_evidence(project,work,out,candidate,remaining,config):
    from .gpu_temporal import run_sequence,stress_frames,gpu_snapshot
    if remaining<30:raise RuntimeError("Insufficient cycle budget for GPU evidence")
    before=gpu_snapshot()
    if before is None:return {"pass":False,"status":"deferred_unknown_gpu_state"}
    import csv,io
    devices=[{key.strip():value.strip() for key,value in row.items()} for row in csv.DictReader(io.StringIO(before))]
    if not devices:return {"pass":False,"status":"deferred_unknown_gpu_state"}
    try:
        activity=[float(x["utilization.gpu [%]"].split()[0]) for x in devices]
        free=[float(x["memory.free [MiB]"].split()[0]) for x in devices]
        if any(not math.isfinite(x) for x in activity+free):raise ValueError("nonfinite GPU status")
        busy=any(x>config.get("gpu_idle_utilization_max",20) for x in activity)
        memory_low=any(x<config.get("gpu_min_free_mib",768) for x in free)
    except (ValueError,KeyError):return {"pass":False,"status":"deferred_unknown_gpu_state","nvidia_before":before}
    if busy or memory_low:
        return {"pass":False,"status":"deferred_busy_or_low_memory","nvidia_before":before,
                "reason":"GPU evaluation deferred; existing model and applications unchanged"}
    deadline=time.monotonic()+remaining
    # Training is CPU-only. GPU evaluation is opt-in and must not silently run in CPU mode.
    parity=run_sequence(project,out,work,"candidate-parity",stress_frames(2),model=candidate,debug=True,
                        timeout_seconds=min(180,max(1,deadline-time.monotonic())))
    rng=np.random.default_rng(800);profiles={}
    for label,h,w in (("1080p",540,960),("1440p",720,1280)):
        if time.monotonic()>=deadline:raise RuntimeError("GPU evidence time budget exhausted")
        frames=[(rng.random((h,w,3),dtype=np.float32),np.ones((h,w),np.float32),
                 np.full((h,w,2),-.125,np.float32),None,i==0) for i in range(2)]
        result=run_sequence(project,out,work,f"candidate-{label}",frames,model=candidate,
                            repeats=1000,warmups=100,capture_all=False,
                            timeout_seconds=min(180,max(1,deadline-time.monotonic())))
        profiles[label]={"p95_ms":result["timings"]["combined"]["steady_history_p95_ms"],
                         "gpu_buffer_payload_mib":result["gpu_buffer_payload_mib"],"parity_pass":result["parity_pass"]}
    return {"pass":parity["parity_pass"] and all(p["parity_pass"] and p["p95_ms"]<=config["gpu_combined_p95_limits_ms"][label] for label,p in profiles.items()),
            "profiles":profiles,"parity_frames":8,"nvidia_before":before,"nvidia_after":gpu_snapshot(),
            "scope":"Repeated fixed inputs; GPU payload is not peak memory; engine integration remains required"}


def find_nomination(root,state,identifier):
    load_model(root,state,identifier)
    for path in sorted((Path(root)/"experiments").glob("cycle-*/report.json"),reverse=True):
        report=read_json(path)
        if report.get("nominee")==identifier and report.get("status")=="nomination_ready":
            if digest(Path(root)/"config.json")!=report["config_sha256"]:raise RuntimeError("Gate configuration changed; rerun evaluation")
            result=next(x for x in report["candidate_results"] if x["id"]==identifier)
            if result["sha256"]!=state["models"][identifier]["sha256"]:raise RuntimeError("Nomination model checksum mismatch")
            return report,result
    raise ValueError("model has no passing nomination")


def check_runtime(root,identifier,project,work,max_seconds=300):
    root=Path(root).resolve()
    with writer_lock(root):
        state=read_json(root/"registry.json");report,result=find_nomination(root,state,identifier)
        if state["accepted_lab"] not in (report["reference"],identifier):raise RuntimeError("Reference changed since evaluation")
        checks=result.setdefault("runtime_checks",[])
        out=root/"experiments"/f"cycle-{report['cycle']:06d}"/f"runtime-check-{len(checks)+1:03d}"
        evidence=gpu_evidence(Path(project).resolve(),Path(work).resolve(),out,
                              load_model(root,state,identifier),max_seconds,read_json(root/"config.json"))
        checks.append({"at":stamp(),"evidence":evidence})
        result["gpu"]=evidence;result["runtime_eligible"]=bool(evidence["pass"])
        atomic_json(root/"experiments"/f"cycle-{report['cycle']:06d}"/"report.json",report)
        state["events"].append({"at":stamp(),"action":"runtime_check","model":identifier,"passed":evidence["pass"]})
        atomic_json(root/"registry.json",state);write_dashboard(root)
    return evidence


def promote(root,identifier,scope="lab"):
    if scope not in ("lab","runtime"):raise ValueError("invalid promotion scope")
    root=Path(root).resolve()
    with writer_lock(root):
        state=read_json(root/"registry.json");report,result=find_nomination(root,state,identifier)
        permitted=(report["reference"],identifier) if scope=="runtime" else (report["reference"],)
        if state["accepted_lab"] not in permitted:raise RuntimeError("Reference changed since evaluation")
        if not result["lab_eligible"] or (scope=="runtime" and not result["runtime_eligible"]):
            raise RuntimeError("candidate has not passed all gates for this promotion scope")
        key="accepted_"+scope;previous="previous_"+scope
        if state[key]==identifier:raise ValueError("model already accepted for this scope")
        state[previous].append(state[key]);state[key]=identifier
        state["events"].append({"at":stamp(),"action":"promote","scope":scope,"model":identifier})
        atomic_json(root/"registry.json",state);write_dashboard(root)
    return state


def rollback(root,scope="lab"):
    if scope not in ("lab","runtime"):raise ValueError("invalid scope")
    root=Path(root).resolve()
    with writer_lock(root):
        state=read_json(root/"registry.json");key="accepted_"+scope;previous="previous_"+scope
        if not state[previous]:raise ValueError("no previous accepted version")
        identifier=state[previous][-1]
        if identifier is not None:load_model(root,state,identifier)
        state[previous].pop();state[key]=identifier
        state["events"].append({"at":stamp(),"action":"rollback","scope":scope,"model":identifier})
        atomic_json(root/"registry.json",state);write_dashboard(root)
    return state


def write_dashboard(root):
    import html
    root=Path(root);state=read_json(root/"registry.json")
    rows=[]
    for path in sorted((root/"experiments").glob("cycle-*/report.json")):
        r=read_json(path);nominee=r.get("nominee") or "none"
        rows.append(f'<tr><td>{r["cycle"]}</td><td>{html.escape(r["status"])}</td><td>{html.escape(nominee)}</td><td><a href="{path.relative_to(root).as_posix()}">Full report</a></td></tr>')
    content=f'''<!doctype html><meta charset="utf-8"><title>SI-GPU learning lab</title><style>body{{background:#101723;color:#eef;font:16px system-ui;max-width:1000px;margin:30px auto}}td,th{{padding:12px;border-bottom:1px solid #456}}a{{color:#8de}}</style><h1>SI-GPU / Learning laboratory</h1><p>Accepted offline model: {html.escape(state['accepted_lab'])}. Accepted runtime model: {html.escape(str(state['accepted_runtime']))}.</p><p>Original synthetic pairs; linear filter training. New candidates are evaluated against replay scenes, temporal regressions and a release audit. A nomination does not replace the accepted model.</p><table><tr><th>Cycle</th><th>Status</th><th>Nominee</th><th>Evidence</th></tr>{''.join(rows)}</table><p>GPU runtime promotion additionally requires measured monitor-size gates. Neural training, real capture and idle scheduling are future work.</p>'''
    (root/"index.html").write_text(content,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab",default="results/learning")
    sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("init");sub.add_parser("status")
    run=sub.add_parser("run");run.add_argument("--project",default=".");run.add_argument("--work",default="work/learning")
    run.add_argument("--cycles",type=int,default=1);run.add_argument("--max-seconds",type=int,default=300);run.add_argument("--gpu-gates",action="store_true")
    run.add_argument("--auto-promote-lab",action="store_true",help="Accept passing offline candidates between bounded cycles; never enables runtime promotion")
    promotion=sub.add_parser("promote");promotion.add_argument("model");promotion.add_argument("--scope",choices=("lab","runtime"),default="lab")
    check=sub.add_parser("check-runtime");check.add_argument("model");check.add_argument("--project",default=".");check.add_argument("--work",default="work/learning-runtime");check.add_argument("--max-seconds",type=int,default=300)
    back=sub.add_parser("rollback");back.add_argument("--scope",choices=("lab","runtime"),default="lab")
    args=parser.parse_args();root=Path(args.lab).resolve()
    if args.action=="init":state=initialize(root);write_dashboard(root);print("Initialized",root,state["accepted_lab"])
    elif args.action=="status":
        state=read_json(root/"registry.json");print(json.dumps({k:state[k] for k in ("cycle","accepted_lab","accepted_runtime","recent_seeds")},indent=2))
    elif args.action=="run":
        if not 1<=args.cycles<=10 or not 30<=args.max_seconds<=1800:parser.error("cycles 1..10 and max-seconds 30..1800 required")
        start=time.monotonic()
        for _ in range(args.cycles):
            remaining=args.max_seconds-(time.monotonic()-start)
            if remaining<30:print("Overall time budget reached; accepted model unchanged");break
            report=run_cycle(root,args.project,args.work,args.gpu_gates,remaining)
            print("Cycle",report["cycle"],report["status"],"nominee:",report["nominee"])
            if args.auto_promote_lab and report["nominee"]:
                promote(root,report["nominee"],"lab")
                print("Accepted offline model",report["nominee"],"after all lab gates")
    elif args.action=="promote":print("Accepted",promote(root,args.model,args.scope)["accepted_"+args.scope])
    elif args.action=="check-runtime":
        if not 30<=args.max_seconds<=1800:parser.error("max-seconds 30..1800 required")
        print(json.dumps(check_runtime(root,args.model,args.project,args.work,args.max_seconds),indent=2))
    elif args.action=="rollback":print("Restored",rollback(root,args.scope)["accepted_"+args.scope])


if __name__=="__main__":main()
