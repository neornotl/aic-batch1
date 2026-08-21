import json, os, re, subprocess, sys, tempfile
from urllib.request import Request, urlopen
from remote_zip import RemoteFile
import zipfile

BASE='https://aic-data.ledo.io.vn/'
VIDEO_ARCHIVES=['Videos_L21_a.zip','Videos_L22_a.zip','Videos_L23_a.zip','Videos_L24_a.zip','Videos_L25_a.zip','Videos_L26_a.zip','Videos_L26_b.zip','Videos_L26_c.zip','Videos_L26_d.zip','Videos_L26_e.zip','Videos_L27_a.zip','Videos_L28_a.zip','Videos_L29_a.zip','Videos_L30_a.zip']
KEYFRAME_ARCHIVES=['Keyframes_L21.zip','Keyframes_L22.zip','Keyframes_L23.zip','Keyframes_L24.zip','Keyframes_L25.zip','Keyframes_L26_a.zip','Keyframes_L26_b.zip','Keyframes_L26_c.zip','Keyframes_L26_d.zip','Keyframes_L26_e.zip','Keyframes_L27.zip','Keyframes_L28.zip','Keyframes_L29.zip','Keyframes_L30.zip']
LOGICAL_BATCHES={**{f'L{i}':[f'Videos_L{i}_a.zip',f'Keyframes_L{i}.zip'] for i in [21,22,23,24,25,27,28,29,30]}, **{f'L26_{p}':[f'Videos_L26_{p}.zip',f'Keyframes_L26_{p}.zip'] for p in 'abcde'}}
def ensure_sidecar(folder, vid, timelines):
    side=f'{folder}/{vid[:3]}/{vid}/timestamps.jsonl'; probe=subprocess.run(['rclone','lsjson',side],capture_output=True,text=True)
    try: exists=bool(json.loads(probe.stdout)) if probe.returncode==0 else False
    except json.JSONDecodeError: exists=False
    if exists: return
    fd, sp=tempfile.mkstemp(prefix=f'{vid}-timestamps-',suffix='.jsonl')
    try:
      with os.fdopen(fd,'w',encoding='utf8') as sf:
        for t in timelines.get(vid,[]): sf.write(json.dumps({k:t.get(k) for k in ('video_id','keyframe','keyframe_number','frame_id','timestamp_s','fps')},separators=(',',':'))+'\n')
      subprocess.run(['rclone','copyto',sp,side],check=True)
    finally:
      if os.path.exists(sp): os.unlink(sp)
def main(name):
    names=VIDEO_ARCHIVES+KEYFRAME_ARCHIVES if name=='all' else LOGICAL_BATCHES.get(name,[name])
    if any(x not in VIDEO_ARCHIVES+KEYFRAME_ARCHIVES for x in names): raise SystemExit('archive not in official allowlist')
    folder='drive:01_Videos_Original'; out=f'catalog_{name.replace(".zip","")}.jsonl'
    map_path=os.environ.get('TIMESTAMP_MAP')
    if not map_path:
      raise SystemExit('TIMESTAMP_MAP must point to the exact BTC timestamp map fetched by the workflow')
    timelines={}
    with open(map_path,encoding='utf8') as mapping:
      for line in mapping:
        try:
          row=json.loads(line); timelines.setdefault(row.get('video_id'),[]).append(row)
        except Exception: pass
    with open(out,'w',encoding='utf8') as cat:
      for archive in names:
        z=zipfile.ZipFile(RemoteFile(BASE+archive))
        for info in z.infolist():
          is_video=archive in VIDEO_ARCHIVES
          if info.is_dir() or (is_video and not info.filename.lower().endswith('.mp4')) or ((not is_video) and not info.filename.lower().endswith(('.jpg','.jpeg','.png'))): continue
          m=re.search(r'(L\d+_V\d+)',info.filename); vid=m.group(1) if m else None
          if not vid: continue
          leaf=os.path.basename(info.filename)
          dest=f'{folder}/{vid[:3]}/{vid}/{vid}.mp4' if is_video else f'{folder}/{vid[:3]}/{vid}/keyframes/{leaf}'
          probe=subprocess.run(['rclone','lsjson',dest],capture_output=True,text=True)
          try: existing=bool(json.loads(probe.stdout)) if probe.returncode==0 else False
          except json.JSONDecodeError: existing=False
          if existing:
            if is_video: ensure_sidecar(folder,vid,timelines)
            continue
          p=subprocess.Popen(['rclone','rcat',dest],stdin=subprocess.PIPE)
          with z.open(info) as src:
            while b:=src.read(1024*1024): p.stdin.write(b)
          p.stdin.close(); p.wait();
          if p.returncode: raise SystemExit(f'upload failed: {vid}')
          meta=subprocess.run(['rclone','lsjson',dest],capture_output=True,text=True,check=True)
          items=json.loads(meta.stdout); item=items[0] if items else {}
          file_id=item.get('ID') or item.get('Id')
          row={'video_id':vid,'archive':archive,'member':info.filename,'drive_path':dest,'drive_file_id':file_id,'drive_url':f'https://drive.google.com/file/d/{file_id}/view' if file_id else None,'timestamp_index':f'drive:{vid[:3]}/{vid}/timestamps.jsonl'}
          cat.write(json.dumps(row,ensure_ascii=False)+'\n'); cat.flush()
          if is_video:
            side=f'{folder}/{vid[:3]}/{vid}/timestamps.jsonl'
            probe=subprocess.run(['rclone','lsjson',side],capture_output=True,text=True)
            try: side_exists=bool(json.loads(probe.stdout)) if probe.returncode==0 else False
            except json.JSONDecodeError: side_exists=False
            if not side_exists:
              fd, sp=tempfile.mkstemp(prefix=f'{vid}-timestamps-',suffix='.jsonl')
              try:
                with os.fdopen(fd,'w',encoding='utf8') as sf:
                  for t in timelines.get(vid,[]):
                    sf.write(json.dumps({k:t.get(k) for k in ('video_id','keyframe','keyframe_number','frame_id','timestamp_s','fps')},separators=(',',':'))+'\n')
                subprocess.run(['rclone','copyto',sp,side],check=True)
              finally:
                if os.path.exists(sp): os.unlink(sp)
    subprocess.run(['rclone','copyto',out,f'drive:{out}'],check=True)
if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else 'all')
