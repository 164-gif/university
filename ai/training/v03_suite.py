"""Run the v0.3 experiments in order and resume from saved epochs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ai.training.v03_data import write_json


def run_command(module, options, log, resume=False):
    with log.open("a" if resume else "w", encoding="utf-8") as stream:
        print(f"{module} {options} -> {log}",flush=True)
        subprocess.run([sys.executable,"-m",module,*map(str,options)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(args):
    root = args.work
    root.mkdir(parents=True,exist_ok=True)
    common = ["--data",args.data,"--legacy",args.legacy,"--work",root]
    for name in ("small","base"):
        path = root/f"teacher-{name}.json"
        if not path.exists():
            run_command("ai.training.v03_experiments",
                        ["teacher",*common,"--teacher",name,"--output",root/"pilots"],
                        root/f"teacher-{name}.log")
    teacher = max(("small","base"),key=lambda name:read(root/f"teacher-{name}.json")
                  ["validation"]["group_mean"]["mrr_at5"])

    def experiment(name, kind, epochs, learning_rate, settings):
        out = root/name
        cross = kind == "c"
        main = "ai.training.v03_cross" if cross else "ai.training.v03_experiments"
        last = out/("cross-last.pt" if cross else "encoder-last.pt")
        path = out/"training-report.json"
        if path.exists() and len(read(path)["history"]) == epochs:
            report = read(path)
            recorded = report["config"]
            expected = {"teacher":teacher,"epochs":epochs,"lr":learning_rate}
            expected.update({key.replace("--","").replace("-","_"):value for key,value in zip(settings[::2],settings[1::2])})
            if any(recorded.get(key)!=value for key,value in expected.items()):
                raise ValueError(f"Existing run has different settings: {out}")
            return report
        options = ([] if cross else [kind]) + common + ["--output",out,"--teacher",teacher,
                   "--epochs",epochs,"--lr",learning_rate,*settings]
        if cross and args.local_cross_teacher:
            options += ["--local-cross-teacher",args.local_cross_teacher]
        if last.exists():
            options.append("--resume")
        run_command(main,options,root/f"{name}.log",resume=last.exists())
        return read(path)

    reports = []
    for ratio, weak, alpha, beta in ((20,.35,.4,.10),(40,.45,.6,.15),(50,.6,.8,.25)):
        settings = ["--typo-ratio",ratio/100,"--weak-weight",weak,"--alpha",alpha,"--beta",beta]
        reports.append(experiment(f"pilot-{ratio}","a",5,3e-5,settings))
    best = max(reports,key=lambda report:max(row["validation"]["group_mean"]["mrr_at5"] for row in report["history"]))
    cfg = best["config"]
    write_json(root/"selected-pilot.json",cfg)
    chosen = ["--typo-ratio",cfg["typo_ratio"],"--weak-weight",cfg["weak_weight"],
              "--alpha",cfg["alpha"],"--beta",cfg["beta"]]
    experiment("experiment-a","a",50,3e-5,chosen)
    experiment("experiment-b","b",50,3e-4,chosen)
    experiment("experiment-c","c",50,1e-4,chosen[:-2])

    compare = root/"final-comparison.json"
    if not compare.exists() or not read(compare).get("complete"):
        run_command("ai.training.v03_compare",["--data",args.data,"--legacy",args.legacy,
                    "--a",root/"experiment-a"/"encoder-best.pt","--b",root/"experiment-b"/"encoder-best.pt",
                    "--cross",root/"experiment-c"/"cross-best.pt","--output",compare],root/"compare.log")
    selected = read(compare)["selection"]["dense_export"]
    folder = root/f"experiment-{selected}" if selected in ("a","b") else Path("ai/weights/v0.2")
    export = root/"browser-export"
    run_command("ai.training.v03_export",["--data",args.data,"--checkpoint",folder/"encoder-best.pt",
                "--tokenizer",folder/"tokenizer.json","--output",export],root/"export.log")
    subprocess.run(["node","web/v03-browser-check.mjs",str(export/"browser")],check=True)
    run_command("ai.training.v03_benchmark",["--output",root/"benchmark-v03.json"],root/"benchmark-v03.log")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,default=Path("ai/generated/v03"))
    parser.add_argument("--data",type=Path,default=Path("ai/data/v03"))
    parser.add_argument("--legacy",type=Path,default=Path("ai/generated/full/dataset/test-examples.jsonl"))
    parser.add_argument("--local-cross-teacher",type=Path)
    run(parser.parse_args())
