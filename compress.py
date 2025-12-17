import io
import os
import re
import tempfile
import time
import uuid
import logging

from flask import Flask, render_template, request, send_from_directory, jsonify
from pypdf import PdfReader, PdfWriter
from PIL import Image
from werkzeug.utils import secure_filename

log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Results live outside static/ so they are not served publicly, and are
# only reachable through /download/<random id>.
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'pdf-compression')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_TTL_SECONDS = 3600
FILE_ID_RE = re.compile(r'^[0-9a-f]{32}\.pdf$')

# level -> (JPEG quality, linear downscale factor) used for embedded images
IMAGE_SETTINGS = {
    'low': (75, 0.8),
    'medium': (50, 0.6),
    'high': (30, 0.4),
}
MIN_IMAGE_SIDE = 64  # don't touch icons / tiny images


def lossless_optimize(pdf_data):
    """Deflate content streams and merge duplicate objects. Keeps annotations,
    links, forms and metadata intact."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_data)))
    for page in writer.pages:
        page.compress_content_streams()
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    return writer


def recompress_images(writer, level):
    """Downscale + re-encode embedded images as JPEG, only where that shrinks them."""
    quality, scale = IMAGE_SETTINGS[level]
    seen = set()  # shared images must only be processed once

    for page in writer.pages:
        for img in page.images:
            ref = img.indirect_reference
            if ref is None or ref.idnum in seen:
                continue
            seen.add(ref.idnum)
            try:
                obj = ref.get_object()
                # Replacing the object would drop transparency / inverted decode arrays.
                if any(k in obj for k in ('/SMask', '/Mask', '/Decode')):
                    continue
                pil = img.image
                if pil is None or pil.mode not in ('RGB', 'L') or min(pil.size) < MIN_IMAGE_SIDE:
                    continue

                new_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
                small = pil.resize(new_size, Image.LANCZOS)

                probe = io.BytesIO()
                small.save(probe, 'JPEG', quality=quality, optimize=True)
                if probe.tell() >= len(obj._data):
                    continue
                img.replace(small, quality=quality, optimize=True)
            except Exception as e:
                log.warning("Skipping image %s: %s", ref.idnum, e)


def strategy_lossless(pdf_data, level):
    return lossless_optimize(pdf_data)


def strategy_images(pdf_data, level):
    writer = lossless_optimize(pdf_data)
    recompress_images(writer, level)
    return writer


STRATEGIES = {
    'Lossless': strategy_lossless,
    'Images + Lossless': strategy_images,
}


def run_strategy(name, pdf_data, level, page_count):
    """Returns (result_dict, compressed_bytes_or_None). Output is validated by
    reopening it, so a corrupt file can never be offered for download."""
    start = time.time()
    result = {
        'algorithm': name,
        'original_size': len(pdf_data),
        'compressed_size': len(pdf_data),
        'saved_percent': 0.0,
        'ok': False,
        'status': '',
        'download_url': None,
    }
    compressed = None
    try:
        out = io.BytesIO()
        STRATEGIES[name](pdf_data, level).write(out)
        data = out.getvalue()
        if len(PdfReader(io.BytesIO(data)).pages) != page_count:
            raise ValueError('page count changed')
        result['compressed_size'] = len(data)
        if len(data) < len(pdf_data):
            result['ok'] = True
            result['status'] = 'Success'
            result['saved_percent'] = round((1 - len(data) / len(pdf_data)) * 100, 1)
            compressed = data
        else:
            result['status'] = 'No reduction'
    except Exception as e:
        log.exception("Strategy %s failed", name)
        result['status'] = f'Failed: {e}'
    result['time_ms'] = int((time.time() - start) * 1000)
    return result, compressed


def compare_strategies(pdf_data, level, page_count, name='document'):
    results = []
    for strategy in STRATEGIES:
        result, compressed = run_strategy(strategy, pdf_data, level, page_count)
        if compressed is not None:
            file_id = f'{uuid.uuid4().hex}.pdf'
            with open(os.path.join(OUTPUT_DIR, file_id), 'wb') as f:
                f.write(compressed)
            result['download_url'] = f'/download/{file_id}?name={name}'
        results.append(result)
    return sorted(results, key=lambda r: r['compressed_size'])


def cleanup_old_outputs():
    cutoff = time.time() - OUTPUT_TTL_SECONDS
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def error(message, status=400, **extra):
    return jsonify({'success': False, 'error': message, **extra}), status


@app.errorhandler(413)
def too_large(_):
    return error('File too large (max 50 MB)', 413)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compress', methods=['POST'])
def compress():
    file = request.files.get('file')
    if file is None:
        return error('No file uploaded')
    if file.filename == '':
        return error('No file selected')

    level = request.form.get('compression_level', 'medium').lower()
    if level not in IMAGE_SETTINGS:
        level = 'medium'

    data = file.read()
    if b'%PDF-' not in data[:1024]:
        return error('Not a valid PDF file')

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return error('Encrypted PDFs are not supported')
        page_count = len(reader.pages)
    except Exception as e:  # PdfReadError and friends: truncated / corrupt files
        return error(f'Could not read PDF: {e}')

    name = secure_filename(os.path.splitext(file.filename)[0]) or 'document'
    start = time.time()
    cleanup_old_outputs()
    results = compare_strategies(data, level, page_count, name)
    total_time = int((time.time() - start) * 1000)

    best = next((r for r in results if r['ok']), None)
    if best is None:
        return error('No strategy reduced the file size', 200, comparison=results,
                     total_time=total_time)

    return jsonify({
        'success': True,
        'original_size': best['original_size'],
        'compressed_size': best['compressed_size'],
        'saved_percent': best['saved_percent'],
        'algorithm_used': best['algorithm'],
        'download_url': best['download_url'],
        'total_time': total_time,
        'comparison': results,
    })


@app.route('/download/<file_id>')
def download_file(file_id):
    if not FILE_ID_RE.match(file_id):
        return error('Not found', 404)
    name = secure_filename(request.args.get('name', '')) or 'document'
    return send_from_directory(OUTPUT_DIR, file_id, as_attachment=True,
                               download_name=f'{name}-compressed.pdf')


if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
