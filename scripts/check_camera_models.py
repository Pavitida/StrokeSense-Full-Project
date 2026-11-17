"""Check packaged CUDA inference with synthetic mean features; never fit a model."""
import hashlib
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import camera_case_worker as worker


def main():
    config = json.loads(worker.CONFIG.read_text(encoding='utf-8'))
    results = {}
    with tempfile.TemporaryDirectory(prefix='camera-model-check-') as directory:
        folder = Path(directory)
        for name, source in config['sources'].items():
            path = worker.ROOT / source['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256']
            with path.open('rb') as stream:
                package = pickle.load(stream)
            columns = package['columns' if name == 'face' else 'feature_columns']
            means = np.asarray(package['mean']).reshape(-1)
            features = dict(zip(columns, map(float, means)))
            (folder / 'request.json').write_text(json.dumps(
                {'assessment_id': 'synthetic-package-check', 'modality': name}))
            (folder / 'features.json').write_text(json.dumps(
                {'status': 'ok', 'modality': name, 'features': features}))
            worker.predict(folder)
            result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            assert np.isfinite(result['score']) and result['feature_count'] == len(columns)
            assert result['source_sha256'] == source['sha256']
            results[name] = result
            print(f'{name}: checksum and CUDA inference PASS ({len(columns)} features)')
        (folder / 'fusion_request.json').write_text(json.dumps(
            {'assessment_id': 'synthetic-package-check', 'results': results}))
        worker.fuse(folder)
        fused = json.loads((folder / 'fusion_result.json').read_text(encoding='utf-8'))
        assert fused['synthetic_demo'] is True and 0 <= fused['demo_score'] <= 1
        # A result from another source model must be rejected.
        results['speech']['source_sha256'] = '0' * 64
        (folder / 'fusion_request.json').write_text(json.dumps(
            {'assessment_id': 'synthetic-package-check', 'results': results}))
        try:
            worker.fuse(folder)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched model result was accepted')
        print('Synthetic fusion: CUDA inference and source mismatch rejection PASS')
    print('No fitting or clinical evaluation performed.')


if __name__ == '__main__':
    main()
