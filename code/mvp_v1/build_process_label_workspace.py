#!/usr/bin/env python3
"""Build an offline, browser-only workflow for fencing phrase annotation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


LABELS = [
    ("", "Unlabeled"),
    ("simultaneous_attack", "Simultaneous attack"),
    ("left_attack", "Valid attack from the left"),
    ("right_attack", "Valid attack from the right"),
    ("left_attack_no", "Left attack misses or is interrupted"),
    ("right_attack_no", "Right attack misses or is interrupted"),
    ("left_parry_riposte", "Left parry-riposte"),
    ("right_parry_riposte", "Right parry-riposte"),
    ("blade_exchange_unclear", "Blade exchange; phrase unclear"),
    ("camera_unclear", "Insufficient video evidence"),
]
WORKSPACE_SCHEMA = "sabervision.process.labels.v4"
LOCAL_STORAGE_KEY = "sabervision_phrase_labels_v4_blind"


def is_true(value: object) -> bool:
    return str(value or "").strip().casefold() in {"true", "1", "yes"}


def sanitize_prefill_row(row: dict[str, str]) -> None:
    """Only carry forward labels with complete blind-human provenance."""
    trusted = (
        row.get("workspace_schema") == WORKSPACE_SCHEMA
        and row.get("process_label_source") == "human_review"
        and is_true(row.get("outcome_hidden_during_label"))
        and is_true(row.get("outcome_header_redacted"))
    )
    if not trusted:
        for field in (
            "process_label", "process_confidence", "attack_initiator",
            "parry_riposte_side", "notes", "process_label_source",
            "outcome_hidden_during_label", "outcome_revealed",
        ):
            row[field] = ""
    row["workspace_schema"] = WORKSPACE_SCHEMA


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["image"] = Path(row["sequence_sheet"]).name
        sanitize_prefill_row(row)
    return rows


def add_motion_hints(rows: list[dict[str, str]], feature_paths: list[Path]) -> None:
    features: dict[str, dict[str, str]] = {}
    for path in feature_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                features[row["case_id"]] = row
    for row in rows:
        feature = features.get(row["case_id"])
        if not feature:
            row["motion_hint"] = "Pose motion cue unavailable"
            continue
        delta_035 = float(feature["left_speed_mean_035"]) - float(feature["right_speed_mean_035"])
        delta_075 = float(feature["left_speed_mean_075"]) - float(feature["right_speed_mean_075"])
        same_direction = delta_035 * delta_075 > 0
        if same_direction and min(abs(delta_035), abs(delta_075)) >= 1.70:
            side = "Left" if delta_035 > 0 else "Right"
            cue = f"{side} has a strong inward-speed advantage in both windows"
        elif same_direction and min(abs(delta_035), abs(delta_075)) >= 0.80:
            side = "Left" if delta_035 > 0 else "Right"
            cue = f"{side} has a candidate motion advantage in both windows"
        elif not same_direction:
            cue = "The two speed windows disagree in direction"
        else:
            cue = "Speeds are similar or the evidence is weak"
        row["motion_hint"] = f"{cue}（0.35s Δ={delta_035:+.2f}，0.75s Δ={delta_075:+.2f}）"


def redact_outcome_header_pixels(image):
    """Return image pixels with the baked-in result banner removed."""
    redacted = image.copy()
    # Existing review sheets are 1500 px tall and place the result banner in
    # roughly the first 70 px.  Use a proportional cap so other resolutions
    # receive the same treatment without covering the first timestamp row.
    header_height = max(1, min(redacted.shape[0], int(round(redacted.shape[0] * 0.052))))
    redacted[:header_height, :] = 0
    return redacted


def redact_outcome_header(source: Path, target: Path) -> None:
    """Remove the baked-in result banner from a review contact sheet."""
    # cv2.imread/imwrite may fail on non-ASCII Windows paths such as
    # Paths may contain non-ASCII characters. Decode and encode bytes explicitly so the workspace can be
    # rebuilt in its actual delivery directory.
    try:
        encoded_source = np.fromfile(str(source), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Unable to read contact sheet: {source}") from exc
    image = cv2.imdecode(encoded_source, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read contact sheet: {source}")
    encoded, buffer = cv2.imencode(
        target.suffix or ".jpg",
        redact_outcome_header_pixels(image),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    if not encoded:
        raise OSError(f"Unable to write redacted contact sheet: {target}")
    try:
        buffer.tofile(str(target))
    except OSError as exc:
        raise OSError(f"Unable to write redacted contact sheet: {target}") from exc


def bundle_assets(rows: list[dict[str, str]], output: Path, clips_dir: Path) -> None:
    image_dir = output.parent / "assets" / "images"
    clip_dir = output.parent / "assets" / "clips"
    image_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_source = Path(row["sequence_sheet"])
        image_target = image_dir / image_source.name
        redact_outcome_header(image_source, image_target)
        row["image"] = f"assets/images/{image_target.name}"
        row["sequence_sheet"] = row["image"]
        row["outcome_header_redacted"] = True
        clip_source = clips_dir / f"{row['case_id']}_lamp_window.mp4"
        if clip_source.exists():
            clip_target = clip_dir / clip_source.name
            shutil.copy2(clip_source, clip_target)
            row["clip"] = f"assets/clips/{clip_target.name}"


def render_workspace(rows: list[dict[str, str]]) -> str:
    dataset = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    workspace_schema = json.dumps(WORKSPACE_SCHEMA)
    storage_key = json.dumps(LOCAL_STORAGE_KEY)
    options = "".join(f'<option value="{value}">{label}</option>' for value, label in LABELS)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SaberVision Double-Light Phrase Annotation</title>
<style>
:root{{--bg:#101419;--panel:#1a222b;--line:#354453;--text:#edf3f7;--muted:#aab8c4;--cyan:#5bd9ff;--amber:#ffd36a}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:"Microsoft YaHei",sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#101419ee;border-bottom:1px solid var(--line);padding:14px 3vw;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
h1{{font-size:20px;margin:0 auto 0 0}}button,select,textarea{{font:inherit}}button,select{{background:#24313c;color:var(--text);border:1px solid #506272;border-radius:7px;padding:8px 11px}}button:hover{{border-color:var(--cyan)}}
main{{width:min(1500px,96vw);margin:20px auto 60px}}.notice{{color:var(--amber)}}.layout{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px}}
.viewer,.form{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}img,video{{width:100%;height:auto;display:block;border-radius:7px;background:#07090b}}img{{margin-top:14px}}
.meta{{color:var(--muted);margin:0 0 12px}}.official{{display:none;color:var(--amber)}}label{{display:block;margin:13px 0 6px;color:#cbd7df}}select,textarea{{width:100%}}textarea{{min-height:90px;background:#111820;color:var(--text);border:1px solid #506272;border-radius:7px;padding:9px;resize:vertical}}
.nav{{display:flex;gap:8px;margin-top:15px}}.nav button{{flex:1}}.progress{{color:var(--cyan);font-variant-numeric:tabular-nums}}.saved{{color:#80e59a}}.shortcut{{color:var(--muted);font-size:13px;line-height:1.6}}input[type=file]{{display:none}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}header{{position:static}}}}
</style></head><body>
<header><h1>SaberVision Double-Light Phrase Annotation</h1><span id="progress" class="progress"></span><span id="saved" class="saved">Autosaved</span>
<button id="exportCsv">Export CSV</button><button id="exportJson">Export backup</button><button id="importBtn">Import backup</button><input id="importFile" type="file" accept="application/json"></header>
<main><p class="notice">Label the fencing phrase before revealing the official outcome. The score banner has been removed from each contact sheet. Labels created or edited after revealing the outcome are automatically excluded from training.</p>
<div class="layout"><section class="viewer"><p id="meta" class="meta"></p><p id="hint" class="notice"></p><video id="clip" controls preload="metadata"></video>
<div class="nav"><button data-speed="0.25">0.25×</button><button data-speed="0.5">0.5×</button><button data-speed="1">1×</button></div>
<img id="sheet" alt="Action sequence around lamp onset">
<p id="official" class="official"></p><button id="reveal">Reveal official outcome</button></section>
<aside class="form"><label for="label">Fencing phrase</label><select id="label">{options}</select>
<label for="initiator">First effective initiator</label><select id="initiator"><option value="">Unconfirmed</option><option value="left">Left</option><option value="right">Right</option><option value="simultaneous">Simultaneous</option><option value="unclear">Unclear</option></select>
<label for="riposte">Parry-riposte side</label><select id="riposte"><option value="">Not applicable / unconfirmed</option><option value="left">Left</option><option value="right">Right</option><option value="none">None</option><option value="unclear">Unclear</option></select>
<label for="confidence">Annotation confidence</label><select id="confidence"><option value="">Unconfirmed</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select>
<label for="notes">Notes</label><textarea id="notes" placeholder="Example: left extends first; right retreats and ripostes; severe occlusion."></textarea>
<div class="nav"><button id="prev">Previous</button><button id="next">Save and next</button></div>
<p class="shortcut">Shortcuts: left/right arrow to navigate; 1 simultaneous attack; 2/3 valid left/right attack; 4/5 failed left/right attack; 6/7 left/right parry-riposte; 8 unclear phrase; 9 insufficient video.</p></aside></div></main>
<script>
const source={dataset}; const schema={workspace_schema}; const key={storage_key};
function loadLocal(){{try{{const data=JSON.parse(localStorage.getItem(key)||'null');return data&&data.schema===schema&&data.records&&typeof data.records==='object'?data.records:{{}}}}catch(_error){{return {{}}}}}}
let saved=loadLocal(); let index=0;
const ids=['label','initiator','riposte','confidence','notes']; const el=id=>document.getElementById(id);
function saveLocal(){{localStorage.setItem(key,JSON.stringify({{schema,records:saved}}))}}
function state(){{const row=source[index],old=saved[row.case_id]||{{}};return {{...row,...old}}}}
function persist(annotationEdited=false){{const row=source[index],prior=saved[row.case_id]||{{}},label=el('label').value,revealed=Boolean(prior.outcome_revealed??row.outcome_revealed);let hidden=prior.outcome_hidden_during_label??row.outcome_hidden_during_label??'';if(!label)hidden='';else if(annotationEdited)hidden=!revealed;saved[row.case_id]={{workspace_schema:schema,process_label:label,attack_initiator:el('initiator').value,parry_riposte_side:el('riposte').value,process_confidence:el('confidence').value,notes:el('notes').value,process_label_source:label?'human_review':'',outcome_revealed:revealed,outcome_hidden_during_label:hidden}};saveLocal();el('saved').textContent=label&&hidden===false?'Saved (outcome revealed; excluded from training)':'Autosaved';updateProgress()}}
function updateProgress(){{const done=source.filter(r=>(saved[r.case_id]||{{}}).process_label).length;el('progress').textContent=`Labeled ${{done}} / ${{source.length}}`}}
function show(){{const row=state();el('meta').textContent=`${{index+1}}/${{source.length}} · ${{row.case_id}} · venue ${{row.venue}} · lamp ${{row.lamp_time}}s`;el('hint').textContent=`Pose motion cue (not a decision): ${{row.motion_hint||'unavailable'}}`;el('sheet').src=row.image;el('clip').src=row.clip||'';el('clip').style.display=row.clip?'block':'none';el('clip').playbackRate=.5;el('official').textContent=`Official outcome: ${{row.official_outcome}} (result confidence: ${{row.result_confidence||'not specified'}})`;el('official').style.display='none';el('reveal').style.display='inline-block';el('reveal').textContent=row.outcome_revealed?'Reveal outcome again (later edits stay excluded)':'Reveal official outcome';el('label').value=row.process_label||'';el('initiator').value=row.attack_initiator||'';el('riposte').value=row.parry_riposte_side||'';el('confidence').value=row.process_confidence||'';el('notes').value=row.notes||'';el('saved').textContent=row.process_label&&row.outcome_hidden_during_label===false?'Saved (outcome revealed; excluded from training)':'Autosaved';updateProgress()}}
function move(delta){{persist(false);index=(index+delta+source.length)%source.length;show()}}
ids.forEach(id=>el(id).addEventListener('change',()=>persist(true)));el('notes').addEventListener('input',()=>{{el('saved').textContent='Saving...';persist(true)}});el('prev').onclick=()=>move(-1);el('next').onclick=()=>move(1);el('reveal').onclick=()=>{{const row=source[index],prior=saved[row.case_id]||{{}};saved[row.case_id]={{...prior,workspace_schema:schema,outcome_revealed:true}};saveLocal();el('official').style.display='block';el('reveal').style.display='none'}};
document.querySelectorAll('[data-speed]').forEach(button=>button.onclick=()=>{{el('clip').playbackRate=Number(button.dataset.speed);el('clip').play()}});
function merged(){{persist(false);return source.map(r=>({{...r,...(saved[r.case_id]||{{}})}}))}}
function download(name,text,type){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type}}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500)}}
el('exportJson').onclick=()=>download('sabervision_process_labels_v4_backup.json',JSON.stringify({{schema,records:saved}},null,2),'application/json');
el('exportCsv').onclick=()=>{{const rows=merged(),cols=['case_id','venue','lamp_time','workspace_schema','process_label','process_confidence','attack_initiator','parry_riposte_side','notes','process_label_source','outcome_hidden_during_label','outcome_revealed','outcome_header_redacted','official_outcome','result_confidence','sequence_sheet'];const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';download('sabervision_process_labels_v4.csv','\ufeff'+[cols.join(','),...rows.map(r=>cols.map(c=>q(r[c])).join(','))].join('\\r\\n'),'text/csv;charset=utf-8')}};
el('importBtn').onclick=()=>el('importFile').click();el('importFile').onchange=async e=>{{try{{const data=JSON.parse(await e.target.files[0].text());if(!data||data.schema!==schema||!data.records||typeof data.records!=='object')throw new Error('Backup schema mismatch');const allowed=new Set(source.map(r=>r.case_id));saved=Object.fromEntries(Object.entries(data.records).filter(([caseId,row])=>allowed.has(caseId)&&row&&typeof row==='object'&&row.workspace_schema===schema));saveLocal();show()}}catch(error){{alert(`Unable to import: ${{error.message}}. Legacy or unverified backups are not loaded into this training workspace.`)}}finally{{e.target.value=''}}}};
document.addEventListener('keydown',e=>{{if(document.activeElement===el('notes'))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);const map={{'1':'simultaneous_attack','2':'left_attack','3':'right_attack','4':'left_attack_no','5':'right_attack_no','6':'left_parry_riposte','7':'right_parry_riposte','8':'blade_exchange_unclear','9':'camera_unclear'}};if(map[e.key]){{el('label').value=map[e.key];persist(true)}}}});show();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--features", type=Path, nargs="*")
    parser.add_argument("--bundle-assets", action="store_true")
    args = parser.parse_args()
    rows = read_rows(args.manifest)
    add_motion_hints(rows, args.features or [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.bundle_assets:
        if args.clips_dir is None:
            raise ValueError("--bundle-assets requires --clips-dir")
        bundle_assets(rows, args.output, args.clips_dir)
    args.output.write_text(render_workspace(rows), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
