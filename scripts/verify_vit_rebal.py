#!/usr/bin/env python3
"""Independent numerical verifier for the matched frozen-encoder REBAL study.

No model code, runner metrics, sklearn metrics or pickle checkpoints are loaded.
The independently implemented NumPy confusion-matrix and inference utilities in
verify_nonimage.py are reused, with their source hash checked. Requires a full
30-cell production grid; incomplete studies are reported before scores are read.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import stdtr

spec = importlib.util.spec_from_file_location("vit_independent_math", Path(__file__).with_name("verify_nonimage.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
require, sha, fingerprint, close, read_json, hashes = v.require, v.sha, v.fingerprint, v.close, v.read_json, v.hashes
MATH_SOURCE = "cdc93b840851a0e7f71d4c6075750c730d023a10de24d60431e1fcc900fbd4a8"
ARCHIVE_SHA = "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821"
ENCODERS = ("resnet18", "deit_tiny")
DIMS = {"resnet18": 512, "deit_tiny": 192}
TRUNKS, HEADS = v.TRUNKS, v.HEADS
LEARNED = v.BASE_METHODS
BASE = LEARNED | {m+"_ncm" for m in TRUNKS}
METHODS = v.METHODS
ROW_FIELDS = ("classes", "train_files", "eval_files", "train_pixel_sha256", "eval_pixel_sha256", "archive_sha256", "ntrain", "n_eval", "original_manifest_sha256", "source_reader_sha256")
HELPERS = {
    "train_tiny_fullscale.py": "3d9de9336c7dc636f641286eddc0e9290f532048ea669d5eb934ed7ef123b61d",
    "train_nonimage.py": "be58b2706400b85340736c0ca0e3f07cbcb23895c4d07e92a6789de86d072b21",
    "benchmark_tiny_features.py": "315a900f5d4a4bb48db66e8508b3f19b902df0b906200c2091efada95d839309",
}
WEIGHTS = {
    "resnet18": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "deit_tiny": "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97",
}
STATES = {
    "resnet18": "de620ffcedf6c2818d8b6575ad3b3fc89a41351a1588ef8c0d904defc28ebf8e",
    "deit_tiny": "9a4e7cc392920096d1bf1be91854437b389ed01fb3f45ad9cb63ea60d616a530",
}
PARAMETERS = {"resnet18":11176512,"deit_tiny":5524416}
READER_SOURCE = "e48692697a091d2459e9659f1d9440f9830577df0418fab240921701be6a36ef"
ENCODER_FIELDS = {"encoder_id","weights_sha256","encoder_state_sha256","feature_dim","preprocessing","preprocessing_sha256",
                  "encoder_parameter_count","encoder_parameter_count_scope","extractor_source_sha256","source_reader_sha256","original_manifest_sha256"}


def key(encoder, rho):
    return encoder+"__rho"+format(float(rho), ".12g")


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def source_labels(archive):
    """Recover label/order truth from the source ZIP, without an extractor import."""
    prefix="tiny-imagenet-200/"
    with zipfile.ZipFile(archive) as z:
        classes=sorted(z.read(prefix+"wnids.txt").decode().split())
        mapping={name:i for i,name in enumerate(classes)}
        train=sorted(name for name in z.namelist() if name.startswith(prefix+"train/") and name.endswith(".JPEG"))
        evaluation=sorted(name for name in z.namelist() if name.startswith(prefix+"val/images/") and name.endswith(".JPEG"))
        annotations=dict(line.split("\t")[:2] for line in z.read(prefix+"val/val_annotations.txt").decode().splitlines())
        labels=np.asarray([mapping[name.split("/")[2]] for name in train]+[mapping[annotations[Path(name).name]] for name in evaluation],dtype=np.int64)
    return classes,train,evaluation,labels


def independent_split(y, manifest, seed, rho, holdout):
    """Rebuild duplicate exclusion and the declared three independent RNG streams."""
    ntrain = manifest["ntrain"]
    seen, evaluation = set(), set(manifest["eval_pixel_sha256"])
    retained = []
    for i, digest in enumerate(manifest["train_pixel_sha256"]):
        if digest in evaluation or digest in seen:
            continue
        seen.add(digest); retained.append(i)
    retained = np.asarray(retained, dtype=np.int64)
    classes = len(manifest["classes"])
    counts = np.bincount(y[retained], minlength=classes)
    maximum = int(counts.min()-holdout)
    require(maximum >= 2, "Insufficient clean training rows")
    calibration_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    rank_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    fit_rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    order = rank_rng.permutation(classes)
    rank = np.empty(classes, dtype=np.int64); rank[order] = np.arange(classes)
    requested = np.floor(maximum*float(rho)**(-rank/(classes-1))).astype(np.int64)
    fit, calibration = [], []
    for c in range(classes):
        candidate = calibration_rng.permutation(retained[y[retained] == c])
        calibration.extend(candidate[:holdout])
        available = fit_rng.permutation(candidate[holdout:])
        fit.extend(available[:requested[c]])
    fit, calibration = np.asarray(fit, dtype=np.int64), np.asarray(calibration, dtype=np.int64)
    return dict(train_indices=fit, holdout_indices=calibration,
                excluded_indices=np.setdiff1d(np.arange(ntrain), np.concatenate((fit, calibration))),
                hygiene_excluded_indices=np.setdiff1d(np.arange(ntrain), retained),
                class_order=order, class_rank=rank, train_counts=requested,
                max_train_per_class=np.asarray(maximum),
                evaluation_indices=np.arange(ntrain, len(y), dtype=np.int64))


def load_caches(directories, protocol, archive=None):
    caches = {}
    for directory in map(Path, directories):
        m = read_json(directory/"manifest.json"); encoder = m["encoder_id"]
        require(encoder in ENCODERS and encoder not in caches, "Unexpected/repeated encoder")
        require(protocol["cache_manifest_sha256"][encoder] == sha(directory/"manifest.json"), "Cache manifest changed")
        hashes(directory, m["artifacts"], {"features.npy", "labels.npy"})
        require(m["archive_sha256"] == protocol["source_archive_sha256"] == ARCHIVE_SHA, "Unexpected image archive")
        require(fingerprint(m["preprocessing"]) == m["preprocessing_sha256"], "Preprocessing fingerprint changed")
        require(m["weights_sha256"]==WEIGHTS[encoder] and m["encoder_state_sha256"]==STATES[encoder],"Not the selected official pretrained encoder")
        require(m["encoder_parameter_count"]==PARAMETERS[encoder] and m["encoder_training"] is False and m["evaluation_scores_computed"] is False,"Not the declared frozen extraction")
        require(m["source_reader_sha256"]==READER_SOURCE==sha(Path(__file__).with_name("extract_tiny_features.py")),"Frozen source-reader identity differs")
        require(m["extractor_source_sha256"]==sha(Path(__file__).with_name("extract_vit_comparison_features.py")),"Feature extractor source differs")
        t=m["preprocessing"]
        require(t["resize_shorter_side"]==256 and t["center_crop"]==[224,224] and t["interpolation"]=="PIL bicubic" and t["antialias"] is True
                and t["mean"]==[.485,.456,.406] and t["std"]==[.229,.224,.225] and t["augmentation"].startswith("none;"),"Not the matched224 transform")
        for field, value in protocol["encoders"][encoder].items():
            require(m[field] == value, "Frozen encoder manifest differs: "+field)
        y = np.load(directory/"labels.npy", allow_pickle=False)
        x = np.load(directory/"features.npy", mmap_mode="r", allow_pickle=False)
        require((m["ntrain"], m["n_eval"], len(m["classes"]), m["feature_dim"]) == (100000,10000,200,DIMS[encoder]), "Wrong production cache dimensions")
        require(y.shape == (110000,) and y.dtype == np.int64 and x.shape == (110000,DIMS[encoder]) and x.dtype == np.float32, "Invalid cache arrays")
        require(np.array_equal(np.bincount(y[:100000], minlength=200), np.full(200,500)) and np.array_equal(np.bincount(y[100000:], minlength=200), np.full(200,50)), "Official label support changed")
        require(m["classes"] == sorted(set(m["classes"])), "Invalid class inventory")
        for part, n in (("train",100000),("eval",10000)):
            require(len(m[part+"_files"]) == len(set(m[part+"_files"])) == len(m[part+"_pixel_sha256"]) == n, "Invalid source row inventory")
        for begin in range(0, len(x), 16384):
            require(np.isfinite(x[begin:begin+16384]).all(), "Nonfinite frozen feature")
        caches[encoder] = dict(manifest=m, labels=y, features=x, directory=directory)
    require(set(caches) == set(ENCODERS), "Both encoder caches required")
    a,b = (caches[e] for e in ENCODERS)
    require(all(a["manifest"][f] == b["manifest"][f] for f in ROW_FIELDS), "Encoder row identities differ")
    require(a["manifest"]["preprocessing"] == b["manifest"]["preprocessing"], "Encoder transforms differ")
    require(np.array_equal(a["labels"], b["labels"]), "Encoder label order differs")
    require(fingerprint({f:a["manifest"][f] for f in ROW_FIELDS}) == protocol["row_identity_sha256"], "Row identity hash changed")
    status = "source archive not supplied; frozen cache and manifest hashes verified"
    if archive is not None:
        require(sha(archive) == ARCHIVE_SHA, "Original image archive checksum differs")
        classes,train,evaluation,labels=source_labels(archive)
        require(classes==a["manifest"]["classes"] and train==a["manifest"]["train_files"] and evaluation==a["manifest"]["eval_files"]
                and np.array_equal(labels,a["labels"]),"Cache labels or ordering differ from independently parsed source ZIP")
        status = "original source archive, independently parsed labels/order, and frozen feature cache hashes verified"
    return caches, status


def expected_artifacts():
    return {"split.npz", "scaler.npz", "scaler.json", "predictions.npz"} | {
        f"{m}/{name}" for m in TRUNKS for name in
        ("checkpoint.pt", "history.json", "knn_graph.npy", "knn.json", "head_natural.pt", "head_ros.pt", "head_smote.pt")}


def check_training(cell, result, split, y, settings):
    counts = np.bincount(y[split["train_indices"]], minlength=len(split["class_rank"]))
    weights, beta = v.expected_cb(counts)
    steps = math.ceil(len(split["train_indices"])/settings["batch_size"])
    heads, adapters, priors = set(), set(), {}
    for method in TRUNKS:
        row = result["trunks"][method]
        require(row["completed_epochs"] == settings["epochs"] and row["optimizer_updates"] == steps*settings["epochs"], "Wrong adapter budget")
        require(math.isfinite(row["training_seconds"]) and row["training_seconds"] >= 0, "Invalid adapter duration")
        close(row["beta"], beta, "CB beta")
        close(row["class_weights"], weights.tolist(), "CB weights", 2e-8, 2e-7)
        close(row["expected_sample_weight"], float(np.dot(counts,weights)/counts.sum()), "Expected sample weight", 2e-8, 2e-7)
        prior = counts/counts.sum() if method == "ce" else counts*weights/np.sum(counts*weights)
        close(row["LA_prior"], prior.tolist(), "Native prior", 2e-8, 2e-7); priors[method] = prior
        adapters.add(row["initial_state_sha256"]); heads.add(row["initial_head_sha256"])
        history = read_json(cell/method/"history.json")
        require(len(history) == settings["epochs"], "Incomplete epoch history")
        for epoch, record in enumerate(history):
            require(record["epoch"] == epoch+1 and record["optimizer_updates"] == (epoch+1)*steps and record["batches"] == steps, "Epoch/update history differs")
            require(record["EQ_active"] == (method == "rebal" and epoch >= int(.8*settings["epochs"])), "EQ schedule differs")
            require(math.isfinite(record["loss"]) and math.isfinite(record["seconds"]) and record["seconds"] >= 0, "Invalid training history")
            order = np.random.default_rng(np.random.SeedSequence([result["identity"]["seed"],epoch,707])).permutation(split["train_indices"])
            eligible = np.zeros(len(counts),dtype=np.int64)
            for offset in range(0,len(order),settings["batch_size"]):
                eligible += np.bincount(y[order[offset:offset+settings["batch_size"]]],minlength=len(counts)) >= 2
            require(np.array_equal(eligible,record["EQ_eligible_batches_per_class"]), "EQ batch eligibility differs from fit-row order")
        close(sum(r["seconds"] for r in history), row["training_seconds"], "Adapter timing")
        graph = np.load(cell/method/"knn_graph.npy", allow_pickle=False)
        k = min(settings["smote_k"], int(counts.min())-1)
        require(graph.shape == (len(split["train_indices"]),k) and np.issubdtype(graph.dtype,np.integer), "Invalid neighbor shape")
        fit_labels = y[split["train_indices"]]
        needed = counts[fit_labels] < counts.max()
        require((graph[~needed] == -1).all(), "Maximal classes unexpectedly have synthetic neighbors")
        selected = graph[needed]
        require((selected >= 0).all() and (selected < len(fit_labels)).all(), "Neighbor outside fit set")
        require((fit_labels[selected] == fit_labels[needed,None]).all(), "Cross-class neighbor")
        require((selected != np.flatnonzero(needed)[:,None]).all(), "Self neighbor")
        require(all(len(set(row.tolist())) == k for row in selected), "Repeated neighbor")
        gm = read_json(cell/method/"knn.json")
        require(gm["graph_sha256"] == result["artifacts"][method+"/knn_graph.npy"] and gm["identity"]["cell"] == result["identity"] and gm["identity"]["trunk"] == method, "Graph provenance differs")
        for kind in HEADS:
            name = method+"_crt_"+kind; trained = result["metrics"][name]["training"]
            require(trained["optimizer_updates"] == settings["head_steps"], "Wrong classifier budget")
            require(trained["losses"] and trained["losses"][-1]["update"] == settings["head_steps"], "Incomplete classifier loss history")
            require(all(math.isfinite(r["loss"]) for r in trained["losses"]), "Invalid classifier history")
            heads.add(trained["initial_state_sha256"])
            prior = counts/counts.sum() if kind == "natural" else np.ones(len(counts))/len(counts)
            close(trained["training_prior"], prior.tolist(), "Retrained prior", 2e-8, 2e-7); priors[name] = prior
    require(len(adapters) == 1 and len(heads) == 1, "Initialization is not matched")
    return priors, next(iter(heads))


def reconstruct_la(name, base_cal, base_test, record, prior, calibration_y, rank):
    prior = np.asarray(prior,dtype=np.float32)
    require((prior>0).all() and np.isfinite(prior).all(), "Invalid class prior")
    close(record["prior"],prior.tolist(),name+" prior",2e-8,2e-7)
    uniform = np.ptp(prior) == 0
    log_prior = np.log(prior)
    curve = []
    for tau in v.TAUS:
        candidate = base_cal if uniform else base_cal-tau*log_prior
        curve.append(dict(tau=tau,macro_f1=v.metrics(calibration_y,candidate.argmax(1),rank)["macro_f1"]))
    close(record["curve"],curve,name+" calibration curve")
    # Use verified saved curve for its last-bit tie ordering, matching the fixed
    # smallest-tau rule without a false sklearn-vs-NumPy rounding disagreement.
    tau = 0. if uniform else max(record["curve"],key=lambda r:r["macro_f1"])["tau"]
    require(record["tau"] == tau, "LA choice is not from the declared calibration rule")
    return base_test if uniform else base_test-tau*log_prior


def verify_cell(cell, cache, split, protocol, encoder, rho, seed):
    result = read_json(cell/"metrics.json")
    identity = dict(protocol=protocol["fingerprint"],condition=key(encoder,rho),encoder=encoder,rho=float(rho),seed=seed,
                    split_sha256=sha(cell/"split.npz"),cache_manifest_sha256=protocol["cache_manifest_sha256"][encoder])
    require(result["identity"] == identity, "Cell identity differs")
    require(set(result["metrics"]) == METHODS and set(result["trunks"]) == set(TRUNKS), "Incomplete result method inventory")
    hashes(cell,result["artifacts"],expected_artifacts())
    stored_split = read_arrays(cell/"split.npz")
    require(set(stored_split) == set(split) and all(np.array_equal(stored_split[k],value) for k,value in split.items()), "Saved split differs from independent reconstruction")
    y, x = cache["labels"], cache["features"]
    fit, calibration, test = (split[k] for k in ("train_indices","holdout_indices","evaluation_indices"))
    require((result["fit_count"],result["calibration_count"],result["test_count"],result["input_dim"]) == (len(fit),len(calibration),len(test),x.shape[1]), "Cell counts differ")
    require(np.array_equal(result["fit_class_counts"],split["train_counts"]), "Fitted class supports differ")
    scaler = read_arrays(cell/"scaler.npz"); values=np.asarray(x[fit],dtype=np.float64)
    mean,scale=values.mean(0),values.std(0); scale[scale==0]=1.
    require(set(scaler)=={"mean","scale"} and np.array_equal(scaler["mean"],mean) and np.array_equal(scaler["scale"],scale), "Scaler is not fitted exclusively on declared fit features")
    del values
    sm = read_json(cell/"scaler.json")
    require(sm["identity"]==identity and sm["artifact_sha256"]==result["artifacts"]["scaler.npz"] and sm["fit_count"]==len(fit) and sm["feature_dim"]==x.shape[1], "Scaler provenance differs")
    priors, initial_head = check_training(cell,result,split,y,protocol["settings"])
    arrays = read_arrays(cell/"predictions.npz")
    expected={"test_labels","calibration_labels","class_rank"}|{m+"__"+part+"_logits" for m in BASE for part in ("test","calibration")}
    require(set(arrays)==expected and set(result["prediction_storage"])==METHODS, "Prediction inventory differs")
    require(np.array_equal(arrays["test_labels"],y[test]) and np.array_equal(arrays["calibration_labels"],y[calibration]) and np.array_equal(arrays["class_rank"],split["class_rank"]), "Saved label/rank order differs")
    rank=split["class_rank"]; c=len(rank); calculated={}
    for name in sorted(BASE):
        require(result["prediction_storage"][name]==dict(base=name,operation="stored"), "Base prediction recipe differs")
        for part, n in (("test",len(test)),("calibration",len(calibration))):
            logits=arrays[name+"__"+part+"_logits"]
            require(logits.shape==(n,c) and np.issubdtype(logits.dtype,np.floating) and np.isfinite(logits).all(), "Invalid logits")
        calculated[name]=v.metrics(y[test],arrays[name+"__test_logits"].argmax(1),rank)
        if name in LEARNED:
            la_name=name+"_la"
            require(result["prediction_storage"][la_name]==dict(base=name,operation="logit_adjustment",adjustment_in_metrics=True), "Adjusted prediction recipe differs")
            tl=reconstruct_la(name,arrays[name+"__calibration_logits"],arrays[name+"__test_logits"],result["metrics"][la_name]["adjustment"],priors[name],y[calibration],rank)
            calculated[la_name]=v.metrics(y[test],tl.argmax(1),rank)
    for name, computed in calculated.items():
        record=result["metrics"][name]
        extras={"adjustment"} if name.endswith("_la") else {"training"} if "_crt_" in name else set()
        require(set(record)==set(computed)|extras, "Metric fields differ: "+name)
        close({k:record[k] for k in computed},computed,name+" metrics")
    rows=[dict(condition=key(encoder,rho),encoder=encoder,rho=float(rho),seed=seed,method=name,**{k:value for k,value in metrics.items() if isinstance(value,(int,float))}) for name,metrics in calculated.items()]
    report=dict(condition=key(encoder,rho),encoder=encoder,rho=float(rho),seed=seed,methods_verified=len(calculated),calibration_curves_verified=len(LEARNED),metrics_sha256=sha(cell/"metrics.json"),predictions_sha256=result["artifacts"]["predictions.npz"],split_sha256=identity["split_sha256"],initial_head_sha256=initial_head,fit_count=len(fit),calibration_count=len(calibration),test_count=len(test))
    return rows,report


def paired(ea,ma,eb,mb,rho,seeds,rows):
    lookup={(r["encoder"],r["rho"],r["seed"],r["method"]):r["macro_f1"] for r in rows}
    d=np.array([lookup[ea,float(rho),seed,ma]-lookup[eb,float(rho),seed,mb] for seed in seeds])
    s=v.sample_summary(d); n=len(d)
    exact=sum(abs(float(np.dot(signs,d)/n))>=abs(s["mean"])-1e-12 for signs in itertools.product((-1.,1.),repeat=n))/(2**n)
    p=1. if np.all(d==0) else None if n<=1 or np.all(d==d[0]) else float(2*stdtr(n-1,-abs(s["mean"])/(s["sd"]/math.sqrt(n))))
    return dict(encoder_a=ea,method_a=ma,encoder_b=eb,method_b=mb,rho=float(rho),endpoint="macro_f1",seeds=seeds,differences=d.tolist(),mean=s["mean"],ci95=s["ci95"],signflip_p=exact,t_p=p)


def expected_aggregate(protocol,rows):
    s=protocol["settings"]; contrasts=[]; summary=[]
    for rho in s["rhos"]:
        candidates=[("deit_tiny","rebal_crt_smote","resnet18","rebal_crt_smote")]
        for encoder in ENCODERS:
            candidates += [(encoder,"rebal_crt_smote",encoder,"rebal_crt_ros"),(encoder,"rebal_crt_natural",encoder,"cb_crt_natural")]
        contrasts += [paired(*candidate,rho,s["seeds"],rows) for candidate in candidates]
    family=len(contrasts)
    for source,target in (("signflip_p","signflip_holm"),("t_p","t_holm")):
        order=sorted(range(family),key=lambda i:contrasts[i][source] if contrasts[i][source] is not None else 1.)
        running=0.
        for rank,i in enumerate(order):
            value=contrasts[i][source]; running=max(running,min(1.,(family-rank)*(value if value is not None else 1.)))
            contrasts[i][target]=running if value is not None else None
    for condition,method in sorted({(r["condition"],r["method"]) for r in rows}):
        group=[r for r in rows if r["condition"]==condition and r["method"]==method]
        summary.append(dict(condition=condition,method=method,n=len(group),metrics={m:v.sample_summary([r[m] for r in group]) for m in v.SUMMARY_METRICS}))
    return dict(rows=rows,summary=summary,paired_comparisons=contrasts,statistical_family=family)


def verify(feature_dirs,run,archive=None):
    run=Path(run); p=read_json(run/"protocol.json")
    require(p["execution"]=="full_study" and p["settings"]["pilot_steps"]==0,"Pilot is not a completed study")
    require(fingerprint({k:value for k,value in p.items() if k!="fingerprint"})==p["fingerprint"],"Protocol fingerprint changed")
    require(sha(Path(__file__).with_name("verify_nonimage.py"))==MATH_SOURCE,"Independent metric utility changed")
    require(sha(Path(__file__).with_name("benchmark_vit_rebal.py"))==p["source_sha256"],"Runner source changed")
    require(p["helper_sha256"]==HELPERS,"Incomplete or changed helper source inventory")
    require(set(p["cache_manifest_sha256"])==set(p["encoders"])==set(ENCODERS),"Incomplete encoder inventory")
    require(all(set(p["encoders"][e])==ENCODER_FIELDS for e in ENCODERS),"Incomplete encoder provenance fields")
    for name,digest in p["helper_sha256"].items():
        require(sha(v.relative_file(Path(__file__).parent,name))==digest,"Helper source changed")
    settings=p["settings"]
    require(settings["seeds"]==[0,1,2,3,4] and settings["rhos"]==[10.,50.,100.] and settings["epochs"]==50 and settings["head_steps"]==2000,"Not the declared production grid/budget")
    require(all(settings[k]==value for k,value in {"warmup_epochs":5,"batch_size":1024,"lr":.05,"head_batch_size":2048,"holdout_per_class":50,"smote_k":5}.items()),"Not the declared production optimization settings")
    grid=list(itertools.product(ENCODERS,settings["rhos"],settings["seeds"]))
    missing=[dict(condition=key(e,r),seed=s) for e,r,s in grid if not (run/key(e,r)/f"seed_{s}"/"metrics.json").is_file()]
    common=dict(protocol=p["fingerprint"],expected_cells=len(grid),verifier_sha256=sha(__file__),metric_utility_sha256=MATH_SOURCE,runner_source_sha256=p["source_sha256"],verified_utc=datetime.now(timezone.utc).isoformat())
    if missing:
        return dict(**common,status="incomplete",completed_cells=len(grid)-len(missing),missing_cells=missing,method_evaluations_verified=0,note="Scores not inspected; complete production grid required")
    caches,source_status=load_caches(feature_dirs,p,archive)
    rows=[]; completed=[]
    for encoder,rho,seed in grid:
        cache=caches[encoder]; split=independent_split(cache["labels"],cache["manifest"],seed,rho,settings["holdout_per_class"])
        cell=run/key(encoder,rho)/f"seed_{seed}"
        one,report=verify_cell(cell,cache,split,p,encoder,rho,seed); rows+=one; completed.append(report)
        shared=run/"shared_splits"/("rho"+format(float(rho),".12g"))/f"seed_{seed}"
        identity=read_json(shared/"identity.json")
        require(identity["identity"]==dict(protocol=p["fingerprint"],row_identity=p["row_identity_sha256"],seed=seed,rho=float(rho)) and identity["split_sha256"]==sha(shared/"split.npz")==report["split_sha256"],"Shared split provenance differs")
    for rho,seed in itertools.product(settings["rhos"],settings["seeds"]):
        pair=[r for r in completed if r["rho"]==rho and r["seed"]==seed]
        require(len({r["split_sha256"] for r in pair})==len({r["initial_head_sha256"] for r in pair})==1,"Cross-encoder pairing differs")
    expected=expected_aggregate(p,rows); stored=read_json(run/"aggregate.json")
    require(stored["status"]=="complete" and stored["protocol"]==p["fingerprint"] and stored["completed_cells"]==stored["expected_cells"]==30 and stored["evaluations"]==810 and stored["statistical_family"]==15,"Aggregate is incomplete or not the production grid")
    for name in ("rows","summary","paired_comparisons"):
        sort_key=(lambda r:(r["condition"],r["seed"],r["method"])) if name=="rows" else (lambda r:(r["condition"],r["method"])) if name=="summary" else (lambda r:(r["rho"],r["encoder_a"],r["method_a"],r["encoder_b"],r["method_b"]))
        close(sorted(stored[name],key=sort_key),sorted(expected[name],key=sort_key),"aggregate."+name)
    return dict(**common,status="verified_complete",completed_cells=30,missing_cells=[],method_evaluations_verified=len(rows),calibration_curves_verified=30*len(LEARNED),source_status=source_status,aggregate_sha256=sha(run/"aggregate.json"),aggregate_status="independently recomputed and verified",completed_cell_evidence=completed,recomputed_aggregate=expected,limits="Consistency verification; frozen encoder inference and pickle model contents are not re-executed; no superiority claim")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",type=Path,nargs=2,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=verify(args.features,args.run,args.archive); code=0 if result["status"]=="verified_complete" else 2
    except (ValueError,KeyError,OSError,TypeError) as error:
        # File-system exceptions can contain private absolute paths.
        result=dict(status="invalid",error="Evidence file could not be read" if isinstance(error,OSError) else str(error)); code=1
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","completed_cells","expected_cells","method_evaluations_verified","error") if k in result}))
    raise SystemExit(code)


if __name__=="__main__":
    main()
