import io
import os
import sys
import time

import pytest
from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import compress


def make_pdf(pages=2):
    # Noisy RGB images so there is something to shrink; PIL writes one page per image.
    imgs = [Image.effect_noise((600, 600), 80).convert('RGB') for _ in range(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, 'PDF', save_all=True, append_images=imgs[1:], quality=100)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(compress, 'OUTPUT_DIR', str(tmp_path))
    return compress.app.test_client()


def post(client, data, level='high', name='doc.pdf'):
    return client.post('/compress', data={
        'file': (io.BytesIO(data), name), 'compression_level': level})


def run(client, data, **kw):
    """Submit a job and poll until it finishes; returns the final status body."""
    r = post(client, data, **kw)
    assert r.status_code == 202
    job_id = r.get_json()['job_id']
    for _ in range(100):
        body = client.get(f'/status/{job_id}').get_json()
        if body['state'] == 'done':
            return body
        time.sleep(0.1)
    raise AssertionError('job did not finish')


def test_compress_returns_smaller_valid_pdf(client):
    pdf = make_pdf()
    done = run(client, pdf)
    assert done['progress'] == 100
    body = done['result']
    assert body['success'] and body['compressed_size'] < len(pdf)
    out = client.get(body['download_url'])
    assert out.status_code == 200
    assert 'doc-compressed.pdf' in out.headers['Content-Disposition']
    assert len(PdfReader(io.BytesIO(out.data)).pages) == 2


def test_unknown_job_is_404(client):
    assert client.get('/status/nope').status_code == 404


def test_rejects_non_pdf(client):
    r = post(client, b'hello world', name='x.pdf')
    assert r.status_code == 400 and not r.get_json()['success']


def test_download_rejects_path_traversal(client):
    assert client.get('/download/..%2Fcompress.py').status_code == 404


def test_annotations_survive():
    from pypdf import PdfWriter
    from pypdf.annotations import Link
    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.add_annotation(0, Link(rect=(10, 10, 100, 100), url='https://example.com'))
    buf = io.BytesIO(); w.write(buf)
    out = io.BytesIO(); compress.lossless_optimize(buf.getvalue()).write(out)
    assert '/Annots' in PdfReader(io.BytesIO(out.getvalue())).pages[0]
