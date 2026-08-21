import json, os, re, subprocess, sys
from urllib.request import Request, urlopen
from remote_zip import RemoteFile
import zipfile

BASE='https://aic-data.ledo.io.vn/'
ARCHIVES=['Videos_L21_a.zip','Videos_L22_a.zip','Videos_L23_a.zip','Videos_L24_a.zip','Videos_L25_a.zip','Videos_L26_a.zip','Videos_L26_b.zip','Videos_L26_c.zip','Videos_L26_d.zip','Videos_L26_e.zip','Videos_L27_a.zip','Videos_L28_a.zip','Videos_L29_a.zip','Videos_L30_a.zip']
def main(name):
    names=ARCHIVES if name=='all' else [name]
    if any(x not in ARCHIVES for x in names): raise SystemExit('archive not in official allowlist')
    folder=os.environ['DRIVE_FOLDER']; out=f'catalog_{name.replace(".zip","")}.jsonl'
    with open(out,'w',encoding='utf8') as cat:
      for archive in names:
        z=zipfile.ZipFile(RemoteFile(BASE+archive))
        for info in z.infolist():
          if info.is_dir() or not info.filename.lower().endswith('.mp4'): continue
          vid=os.path.splitext(os.path.basename(info.filename))[0]
          dest=f'drive:{vid}.mp4'
          probe=subprocess.run(['rclone','lsjson',dest],capture_output=True,text=True)
          try: existing=bool(json.loads(probe.stdout)) if probe.returncode==0 else False
          except json.JSONDecodeError: existing=False
          if existing: continue
          p=subprocess.Popen(['rclone','rcat',dest],stdin=subprocess.PIPE)
          with z.open(info) as src:
            while b:=src.read(1024*1024): p.stdin.write(b)
          p.stdin.close(); p.wait();
          if p.returncode: raise SystemExit(f'upload failed: {vid}')
          meta=subprocess.run(['rclone','lsjson',dest],capture_output=True,text=True,check=True)
          items=json.loads(meta.stdout); item=items[0] if items else {}
          file_id=item.get('ID') or item.get('Id')
          row={'video_id':vid,'archive':archive,'member':info.filename,'drive_path':dest,'drive_file_id':file_id,'drive_url':f'https://drive.google.com/file/d/{file_id}/view' if file_id else None,'timestamp_index':'02_Timelines/btc_keyframe_timestamps.jsonl'}
          cat.write(json.dumps(row,ensure_ascii=False)+'\n'); cat.flush()
    subprocess.run(['rclone','copyto',out,f'drive:{out}'],check=True)
if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else 'all')
