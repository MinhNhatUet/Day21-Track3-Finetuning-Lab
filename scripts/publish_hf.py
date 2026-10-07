"""Publish only the correct adapter and its documented evaluation to Hugging Face."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd, get_token

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, help='username/repository')
    parser.add_argument('--publish', action='store_true', help='Create a PUBLIC model repo and upload')
    args = parser.parse_args()
    if len(args.repo.split('/')) != 2 or not all(args.repo.split('/')):
        parser.error('Expected username/repository')
    folder = ROOT / 'adapters/correct'
    names = ['adapter_config.json', 'adapter_model.safetensors', 'README.md',
             'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
    files = [(folder / name, name) for name in names]
    files += [(p, 'results/' + p.name) for p in sorted((ROOT/'results').glob('*.json'))]
    files += [(ROOT/'results/runs.csv', 'results/runs.csv'),
              (ROOT/'submission/REPORT.md', 'submission/REPORT.md')]
    for path, remote in files:
        if not path.is_file():
            raise SystemExit(f'Missing {path}')
        print(f'{remote}: {path.stat().st_size:,} bytes')
    if not args.publish:
        print('Prepared only; nothing uploaded. Use --publish after authenticating locally.')
        return
    token = get_token()
    if not token:
        raise SystemExit('No Hugging Face login. Run huggingface_hub.login() locally; never share tokens in chat.')
    api = HfApi(token=token)
    identity = api.whoami()
    allowed = {identity['name']} | {o['name'] for o in identity.get('orgs', [])}
    if args.repo.split('/')[0] not in allowed:
        raise SystemExit('Destination is not your account or one of your organizations.')
    api.create_repo(args.repo, repo_type='model', private=False, exist_ok=True)
    if api.model_info(args.repo).private:
        raise SystemExit('Existing repo is private; visibility was not changed.')
    commit = api.create_commit(repo_id=args.repo, repo_type='model',
        operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)) for path, remote in files],
        commit_message='Publish Lab 21 LoRA adapter and measured evaluation')
    # Check public visibility independently of credentials.
    info = HfApi(token=False).model_info(args.repo, files_metadata=True)
    found = {s.rfilename: s for s in info.siblings}
    for path, remote in files:
        assert remote in found, f'Missing remote file: {remote}'
        if found[remote].size is not None:
            assert found[remote].size == path.stat().st_size, f'Size mismatch: {remote}'
    url = f'https://huggingface.co/{args.repo}'
    (ROOT/'submission/HF_PUBLICATION.json').write_text(json.dumps(
        {'url': url, 'commit': commit.oid, 'public_verified': True}, indent=2)+'\n', encoding='utf-8')
    print(f'Published and verified: {url}')

if __name__ == '__main__':
    main()
