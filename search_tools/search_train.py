import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Train SCRFD search candidates')
    parser.add_argument('gpuid', type=int, help='visible GPU id for this worker')
    parser.add_argument('idx_from', type=int, help='candidate index start (inclusive)')
    parser.add_argument('idx_to', type=int, help='candidate index end (inclusive)')
    parser.add_argument('group', nargs='?', default='scrfdgen', help='config group under configs/')
    parser.add_argument('--prefix', default=None, help='config prefix, defaults to group')
    parser.add_argument('--config-root', default='configs', help='config root directory')
    parser.add_argument('--work-dir-root', default='work_dirs', help='root directory for checkpoints/logs')
    parser.add_argument('--launcher', choices=['single', 'dist'], default='single',
                        help='single uses tools/train.py directly; dist uses tools/dist_train.sh')
    parser.add_argument('--port-base', type=int, default=29100, help='base port for dist launcher mode')
    parser.add_argument('--no-validate', action='store_true', help='disable validation during training')
    parser.add_argument('--resume-from', default=None, help='resume checkpoint passed through to tools/train.py')
    parser.add_argument('--cfg-options', nargs='*', default=[],
                        help='config overrides forwarded to tools/train.py')
    return parser.parse_args()


def train(args, group, prefix, idx):
    assert idx >= 0
    config_path = os.path.join(args.config_root, group, '%s_%d.py' % (prefix, idx))
    work_dir = os.path.join(args.work_dir_root, '%s_%d' % (prefix, idx))

    if args.launcher == 'single':
        cmd = [
            sys.executable, '-u', 'tools/train.py', config_path,
            '--gpu-ids', '0',
            '--work-dir', work_dir,
        ]
    else:
        cmd = [
            'bash', './tools/dist_train.sh', config_path, '1',
            '--work-dir', work_dir,
        ]

    if args.no_validate:
        cmd.append('--no-validate')
    if args.resume_from:
        cmd.extend(['--resume-from', args.resume_from])
    if args.cfg_options:
        cmd.append('--cfg-options')
        cmd.extend(args.cfg_options)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    if args.launcher == 'dist':
        env['PORT'] = str(args.port_base + idx)

    print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    prefix = args.prefix or args.group
    for idx in range(args.idx_from, args.idx_to + 1):
        train(args, args.group, prefix, idx)


if __name__ == '__main__':
    main()

