"""Read ZIP members over HTTP Range requests without downloading the archive."""
import argparse, json, os, re, sys, time
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead
from urllib.request import Request, urlopen
import zipfile

class RemoteFile:
    _attempts = 4
    def __init__(self, url):
        self.url=url; self.pos=0; self.size=None
        h=urlopen(Request(url, method='HEAD', headers={'User-Agent':'AIC2026-official-ingest/1.0'})); self.size=int(h.headers['Content-Length'])
    def seek(self, off, whence=0):
        self.pos = off if whence==0 else self.pos+off if whence==1 else self.size+off
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if self.pos >= self.size:
            return b''
        if n<0: n=self.size-self.pos
        if n<=0:return b''
        end=min(self.size,self.pos+n)-1
        req=Request(self.url,headers={'Range':f'bytes={self.pos}-{end}','User-Agent':'AIC2026-official-ingest/1.0'})
        expected=end-self.pos+1
        for attempt in range(self._attempts):
            try:
                with urlopen(req, timeout=60) as r: data=r.read()
                if len(data) != expected:
                    raise IOError(f'HTTP Range returned {len(data)} bytes, expected {expected}')
                break
            except (HTTPError, URLError, OSError, IncompleteRead, IOError) as exc:
                if attempt + 1 == self._attempts: raise IOError(f'failed HTTP Range {self.pos}-{end}') from exc
                time.sleep(0.5 * (2 ** attempt))
        self.pos += len(data); return data
    def close(self): pass
    def seekable(self): return True

def list_members(url):
    z=zipfile.ZipFile(RemoteFile(url)); return [{'name':i.filename,'size':i.file_size} for i in z.infolist() if not i.is_dir()]

def main():
    p=argparse.ArgumentParser(); p.add_argument('url'); p.add_argument('--list',action='store_true'); p.add_argument('--member'); p.add_argument('--output')
    a=p.parse_args(); z=zipfile.ZipFile(RemoteFile(a.url))
    if a.list: print(json.dumps([{'name':i.filename,'size':i.file_size} for i in z.infolist() if not i.is_dir()])); return
    if not a.member or not a.output: p.error('--member and --output required')
    with z.open(a.member) as src, open(a.output,'wb') as dst:
        while True:
            b=src.read(1024*1024)
            if not b: break
            dst.write(b)
if __name__=='__main__': main()
