#!/usr/bin/env python3
"""Local read-only cloud watcher: progress, periodic student copies (or final PPO candidates), final sync.

Uses a private AutoDL snapshot only for the existing SSH connection. Never
logs credentials, controls power, launches training, or commands hardware.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import time
import paramiko


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--connection-snapshot',type=Path,required=True)
    parser.add_argument('--pipeline',required=True)
    args=parser.parse_args()
    connection=json.loads(args.connection_snapshot.read_text())['data']
    root=Path(args.pipeline);root.mkdir(parents=True,exist_ok=True)
    remote_root='/root/code/dexscrew-repro/'
    started=time.monotonic();last_archive=0.;manifest={};latest={}

    def safe(path):
        p=PurePosixPath(path)
        assert not p.is_absolute() and '..' not in p.parts and p.parts[0]=='outputs',path
        return remote_root+str(p)

    def get_json(sftp,path):
        with sftp.open(safe(path),'rb') as f:return json.loads(f.read())

    def transfer(sftp,source,target=None):
        target=Path(source if target is None else target);target.parent.mkdir(parents=True,exist_ok=True)
        temp=target.with_name(target.name+'.download');digest=hashlib.sha256()
        with sftp.open(safe(source),'rb') as src,temp.open('wb') as dst:
            src.prefetch()
            while True:
                block=src.read(1024*1024)
                if not block:break
                dst.write(block);digest.update(block)
        assert hashlib.sha256(temp.read_bytes()).hexdigest()==digest.hexdigest()
        temp.replace(target);manifest[str(target)]=dict(remote=source,sha256=digest.hexdigest())

    def tree(sftp,path):
        for item in sftp.listdir_attr(safe(path)):
            child=path+'/'+item.filename
            if stat.S_ISDIR(item.st_mode):tree(sftp,child)
            elif stat.S_ISREG(item.st_mode):transfer(sftp,child)

    while time.monotonic()-started<10800:
        client=paramiko.SSHClient();client.load_system_host_keys()
        try:
            client.connect(connection['proxy_host'],port=connection['ssh_port'],username='root',
                           password=connection['root_password'],timeout=20)
            with client.open_sftp() as sftp:
                latest=get_json(sftp,args.pipeline+'/status.json')
                (root/'status.json').write_text(json.dumps(latest,indent=2)+'\n')
                for name in ('smoke_output','formal_output'):
                    if name not in latest:continue
                    try:
                        progress=get_json(sftp,latest[name]+'/progress.json')
                    except FileNotFoundError:
                        try:progress=get_json(sftp,latest[name]+'/stage1_nn/last.training_state.json')
                        except FileNotFoundError:continue
                    q=Path(latest[name]);q.mkdir(parents=True,exist_ok=True)
                    (q/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(json.dumps(dict(at=now(),phase=name,progress=progress)),flush=True)
                formal=latest.get('formal_output')
                if formal and time.monotonic()-last_archive>=300:
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    for label in ('model_last','model_best'):
                        try:transfer(sftp,formal+'/stage2_nn/'+label+'.ckpt',root/'periodic_student_copies'/stamp/(label+'.ckpt'))
                        except FileNotFoundError:pass
                    last_archive=time.monotonic()
                if not latest['status'].startswith('running:'):
                    tree(sftp,args.pipeline)
                    for name in ('smoke_output','formal_output'):
                        if name in latest:tree(sftp,latest[name])
                    for candidate in latest.get('candidate_checkpoints', {}).values():
                        transfer(sftp,candidate['path'])
                    receipt=dict(status='completed:automatic_local_artifact_sync',pipeline_status=latest['status'],
                                 completed_at=now(),files=manifest,visual_review_pending=True,
                                 hardware_deployment=False)
                    (root/'automatic_sync_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
                    with sftp.open(safe(args.pipeline+'/automatic_sync_receipt.json'),'w') as f:
                        f.write(json.dumps(receipt,indent=2)+'\n')
                    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
                    return
        except Exception as exc:
            print(json.dumps(dict(at=now(),status='retrying:artifact_monitor',error_type=type(exc).__name__,error=str(exc))),flush=True)
        finally:client.close()
        time.sleep(45)
    (root/'monitor_timeout.json').write_text(json.dumps(dict(status='monitor_time_limit',at=now(),last_pipeline_status=latest.get('status')))+'\n')


if __name__=='__main__':main()
