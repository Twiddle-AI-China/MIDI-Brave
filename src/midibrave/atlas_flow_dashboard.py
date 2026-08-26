from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .atlas_flow_atlas import TimbreAtlas


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>midiBrave Atlas Flow 评估</title><style>
:root{color-scheme:dark;--bg:#080b13;--panel:#121827;--ink:#edf4ff;--muted:#92a2bd;--ok:#4fe3a4;--bad:#ff7a8a;--accent:#8d7cff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#202044 0,var(--bg) 40%);font:15px/1.55 system-ui;color:var(--ink)}
main{max-width:1240px;margin:auto;padding:28px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end}.hero h1{font-size:34px;margin:0}.hero p{color:var(--muted);max-width:760px}
.badge{padding:8px 14px;border-radius:999px;background:#2a2030;color:var(--bad);font-weight:700}.badge.ok{background:#123329;color:var(--ok)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px;margin-top:18px}.panel{background:rgba(18,24,39,.92);border:1px solid #27314a;border-radius:16px;padding:18px}.summary{grid-column:span 5}.atlas{grid-column:span 7}.samples{grid-column:1/-1}
h2{margin:0 0 12px;font-size:18px}.gates{display:grid;grid-template-columns:1fr 1fr;gap:8px}.gate{padding:9px 11px;background:#0c111e;border-radius:9px}.pass{color:var(--ok)}.fail{color:var(--bad)}
canvas{width:100%;height:300px;background:#090d17;border-radius:12px}.sample{display:grid;grid-template-columns:180px repeat(4,1fr);gap:10px;align-items:center;padding:13px 0;border-top:1px solid #27314a}.sample:first-of-type{border-top:0}.sample small{display:block;color:var(--muted)}audio{width:100%}
.legend{color:var(--muted);font-size:13px}.warning{margin-top:16px;padding:12px;border-left:3px solid #ffbf69;background:#1c1920;color:#ffd7a3}@media(max-width:900px){.summary,.atlas{grid-column:1/-1}.sample{grid-template-columns:1fr}.hero{display:block}}
</style></head><body><main><section class="hero"><div><h1>midiBrave · Atlas Flow</h1><p>Pad Top50 阶段评估：原音频、动态重建、静态轨迹反事实与 Flow 完整轨迹生成采用同一 MIDI 条件并排试听。</p></div><span id="badge" class="badge">等待评估</span></section>
<section class="grid"><article class="panel summary"><h2>硬门禁</h2><div id="gates" class="gates"></div><div class="warning">本页只展示完整 test 集结果，不以精选样例替代总体评估。四类别、正式听评与实时 soak 未完成前，不标记 ISMIR LBD Ready。</div></article>
<article class="panel atlas"><h2>Pad Atlas（PCA 前两维显示，实际控制为 8D）</h2><canvas id="atlas" width="760" height="300"></canvas><div class="legend">绿：train　黄：validation　紫：test；越界输入在后端投影到局部 k=4 合法区域。</div></article>
<article class="panel samples"><h2>盲听材料</h2><div id="samples"></div></article></section></main>
<script>
Promise.all([fetch('evaluation.json').then(r=>r.json()),fetch('atlas.json').then(r=>r.json())]).then(([report,atlas])=>{
 const badge=document.querySelector('#badge');badge.textContent=report.passed?'Pad 自动门禁通过':'Pad 自动门禁未通过';if(report.passed)badge.classList.add('ok');
 const gates=document.querySelector('#gates');Object.entries(report.gates).forEach(([k,v])=>{const d=document.createElement('div');d.className='gate '+(v?'pass':'fail');d.textContent=(v?'✓ ':'✕ ')+k;gates.appendChild(d)});
 const samples=document.querySelector('#samples');report.rows.forEach(row=>{const d=document.createElement('div');d.className='sample';d.innerHTML=`<div><b>${row.preset_id}</b><small>MIDI ${row.note} · dynamic +${(100*row.dynamic_improvement_fraction).toFixed(1)}%</small></div>`+['source','dynamic','static','flow'].map(name=>`<label><small>${name}</small><audio controls preload="none" src="${row.audio[name]}"></audio></label>`).join('');samples.appendChild(d)});
 const c=document.querySelector('#atlas'),x=c.getContext('2d'),pts=atlas.points;const xs=pts.map(p=>p.x),ys=pts.map(p=>p.y),pad=24,minx=Math.min(...xs),maxx=Math.max(...xs),miny=Math.min(...ys),maxy=Math.max(...ys);x.clearRect(0,0,c.width,c.height);pts.forEach(p=>{const px=pad+(p.x-minx)/(maxx-minx||1)*(c.width-2*pad),py=c.height-pad-(p.y-miny)/(maxy-miny||1)*(c.height-2*pad);x.beginPath();x.arc(px,py,p.split==='test'?6:4,0,Math.PI*2);x.fillStyle=p.split==='train'?'#4fe3a4':p.split==='validation'?'#ffd166':'#8d7cff';x.fill()})
}).catch(e=>{document.querySelector('#badge').textContent='结果尚未生成';console.error(e)});
</script></body></html>"""


def build_dashboard(evaluation_root: str | Path, atlas_path: str | Path, manifest: str | Path) -> Path:
    root = Path(evaluation_root)
    report_path = root / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint"] = Path(str(report.get("checkpoint", ""))).name
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_by_preset: dict[str, str] = {}
    with Path(manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            split_by_preset[str(value["preset_id"])] = str(value["split"])
    atlas = TimbreAtlas.load(atlas_path)
    points = [
        {"preset_id": preset, "x": float(atlas.coordinates[index, 0]),
         "y": float(atlas.coordinates[index, 1]), "split": split_by_preset[preset]}
        for index, preset in enumerate(atlas.preset_ids)
    ]
    (root / "atlas.json").write_text(json.dumps({"points": points}, indent=2) + "\n", encoding="utf-8")
    (root / "index.html").write_text(HTML, encoding="utf-8")
    return root / "index.html"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the static Atlas Flow evaluation dashboard.")
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(build_dashboard(args.evaluation_root, args.atlas, args.manifest))


if __name__ == "__main__":
    main()
