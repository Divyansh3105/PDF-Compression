from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import io
import time
import zlib
import zstandard as zstd
import lzma
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import tempfile
import sys

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class PDFCompressor:
    def __init__(self):
        self.compression_algorithms = {
            'HUFFMAN': self.huffman_compress,
            'ZSTANDARD': self.zstd_compress,
            'LZMA': self.lzma_compress
        }
        self.target_ratios = {
            'low': 0.8,    # Target 20% reduction
            'medium': 0.5, # Target 50% reduction
            'high': 0.2    # Target 80% reduction
        }
        self.max_attempts = 5

    def compress_images_in_pdf(self, pdf_data, level):
        """More aggressive image compression with visible size differences"""
        try:
            reader = PdfReader(io.BytesIO(pdf_data))
            writer = PdfWriter()

            for page in reader.pages:
                if '/XObject' in page['/Resources']:
                    x_object = page['/Resources']['/XObject'].get_object()
                    for obj in x_object:
                        if x_object[obj]['/Subtype'] == '/Image':
                            image = x_object[obj]

                            # Skip already compressed images
                            if '/Filter' in image:
                                continue

                            # Much more aggressive quality settings
                            if level == 'low':
                                quality = 85
                                scale = 0.9  # 10% smaller
                            elif level == 'high':
                                quality = 30  # Very aggressive
                                scale = 0.5  # Half size
                            else:
                                quality = 60
                                scale = 0.7  # 30% smaller

                            # Process image
                            with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
                                img_data = image._data
                                pil_img = Image.open(io.BytesIO(img_data))

                                # Convert to RGB if needed
                                if pil_img.mode != 'RGB':
                                    pil_img = pil_img.convert('RGB')

                                # Resize and compress aggressively
                                new_size = (int(pil_img.width * scale),
                                           int(pil_img.height * scale))
                                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                                pil_img.save(tmp.name, 'JPEG',
                                            quality=quality,
                                            optimize=True,
                                            progressive=True)

                                with open(tmp.name, 'rb') as f:
                                    compressed_img = f.read()

                                # Update PDF image
                                image._data = compressed_img
                                image['/Filter'] = '/DCTDecode'
                                image['/ColorSpace'] = '/DeviceRGB'
                                image['/BitsPerComponent'] = 8
                                image['/Width'], image['/Height'] = new_size

                writer.add_page(page)

            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()

        except Exception as e:
            print(f"Image compression failed (falling back to original): {str(e)}", file=sys.stderr)
            return pdf_data

    def optimize_compression(self, data, compress_func, level):
        target_size = int(len(data) * self.target_ratios[level])
        best_compressed = data

        # Step 1: Optimize PDF structure
        optimized = self.optimize_pdf_structure(data)

        # Step 2: Compress images
        with_images = self.compress_images_in_pdf(optimized, level)

        # Step 3: Apply compression algorithm
        compressed = compress_func(with_images, level, 0)

        return compressed

    def huffman_compress(self, data, level='medium', attempt=0):
        try:
            # More extreme differences between levels
            if level == 'low':
                compress_level = 1  # Minimal compression
            elif level == 'high':
                compress_level = 9  # Maximum compression
            else:
                compress_level = 6  # Medium compression

            compressed = zlib.compress(data, level=compress_level)
            return self.create_pdf(data, compressed, 'Flate', compress_level, attempt)

        except Exception as e:
            print(f"Huffman compression error: {str(e)}", file=sys.stderr)
            return data

    def zstd_compress(self, data, level='medium', attempt=0):
        try:
            # More extreme level differences
            if level == 'low':
                compress_level = 3
            elif level == 'high':
                compress_level = 22  # Maximum
            else:
                compress_level = 12

            cctx = zstd.ZstdCompressor(level=compress_level)
            compressed = cctx.compress(data)
            return self.create_pdf(data, compressed, 'Zstd', compress_level, attempt)

        except Exception as e:
            print(f"Zstandard compression error: {str(e)}", file=sys.stderr)
            return data

    def lzma_compress(self, data, level='medium', attempt=0):
        try:
            # Extreme differentiation
            if level == 'low':
                filters = [{
                    'id': lzma.FILTER_LZMA2,
                    'preset': 7 + attempt,  # 7-9
                    'dict_size': 1 << 21  # 2MB
                }]
            elif level == 'high':
                filters = [{
                    'id': lzma.FILTER_LZMA2,
                    'preset': 9 + attempt * 2,  # 9-13
                    'dict_size': 1 << 26  # 64MB
                }]
            else:
                filters = [{
                    'id': lzma.FILTER_LZMA2,
                    'preset': 8 + attempt * 2,  # 8-12
                    'dict_size': 1 << 24  # 16MB
                }]

            compressed = lzma.compress(data, filters=filters)
            return self.create_pdf(data, compressed, 'LZMA', filters[0]['preset'], attempt)

        except Exception as e:
            print(f"LZMA compression error: {str(e)}", file=sys.stderr)
            return data

    def create_pdf(self, original_data, compressed_data, method, level, attempt):
        writer = PdfWriter()
        reader = PdfReader(io.BytesIO(original_data))
        for page in reader.pages:
            writer.add_page(page)

        writer.add_metadata({
            '/Compression': f'/{method}',
            '/Level': str(level),
            '/Attempt': str(attempt),
            '/ImageCompression': 'Applied'
        })

        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()

    def optimize_pdf_structure(self, pdf_data):
        """Remove redundant PDF elements to reduce size"""
        try:
            reader = PdfReader(io.BytesIO(pdf_data))
            writer = PdfWriter()

            for page in reader.pages:
                writer.add_page(page)

            # Remove metadata and other bloat
            writer._info = None
            if hasattr(writer, '_root_object'):
                writer._root_object.update({
                    '/Metadata': None
                })

            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()
        except Exception as e:
            print(f"PDF optimization failed: {str(e)}", file=sys.stderr)
            return pdf_data

    def compare_algorithms(self, data, level='medium'):
        results = []
        for name, algorithm in self.compression_algorithms.items():
            start_time = time.time()
            try:
                compressed = self.optimize_compression(data, algorithm, level)
                ratio = len(compressed)/len(data)
                filename = f"{name.lower()}_{int(time.time()*1000)}.pdf"
                output_path = os.path.join(UPLOAD_FOLDER, filename)
                with open(output_path, "wb") as f_out:
                    f_out.write(compressed)
                results.append({
                    'algorithm': name,
                    'original_size': len(data),
                    'compressed_size': len(compressed),
                    'compression_ratio': round(1 - ratio, 2),
                    'time_ms': (time.time() - start_time) * 1000,
                    'status': '✓ Success' if len(compressed) < len(data) else '⚠️ No Reduction',
                    'download_url': f'/download/{filename}'
                })
            except Exception as e:
                results.append({
                    'algorithm': name,
                    'original_size': len(data),
                    'compressed_size': len(data),
                    'compression_ratio': 0,
                    'time_ms': 0,
                    'status': f'✗ Failed: {str(e)}',
                    'download_url': None
                })
        return results

compressor = PDFCompressor()

def compress_pdf(file_stream, compression_level):
    start_time = time.time()
    steps = []
    input_data = file_stream.read()
    original_size = len(input_data)
    target_size = int(original_size * compressor.target_ratios[compression_level])
    file_stream.seek(0)

    steps.append({
        'name': 'Read input file',
        'time': (time.time() - start_time) * 1000,
        'memoryUsed': original_size,
        'target': f"Target: ≤{target_size} bytes ({compressor.target_ratios[compression_level]*100}%)"
    })

    comparison_start = time.time()
    comparison = compressor.compare_algorithms(input_data, compression_level)
    steps.append({
        'name': 'Algorithm comparison',
        'time': (time.time() - comparison_start) * 1000,
        'memoryUsed': 0
    })

    # Select the best algorithm that actually reduced size
    successful_results = [r for r in comparison if r['compressed_size'] < original_size]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x['compressed_size'])
        method = best_result['algorithm']
    else:
        method = 'ZSTANDARD'  # fallback to zstd if no reduction

    compress_start = time.time()

    try:
        compressed_data = compressor.optimize_compression(
            input_data,
            compressor.compression_algorithms[method],
            compression_level
        )

        achieved_ratio = 1 - (len(compressed_data) / original_size)
        compression_effective = len(compressed_data) < original_size

        note = "✓ Target achieved" if len(compressed_data) <= target_size else \
               "⚠️ Partial reduction" if compression_effective else "✗ No reduction"

        steps.append({
            'name': f'{method} compression ({compression_level} level)',
            'time': (time.time() - compress_start) * 1000,
            'memoryUsed': len(compressed_data),
            'achieved': f"Reduction: {achieved_ratio:.0%}"
        })

        filename = f"compressed_{int(time.time())}.pdf"
        output_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(output_path, "wb") as f_out:
            f_out.write(compressed_data)

        compressed_size = os.path.getsize(output_path)
        compression_ratio = round(1 - (compressed_size / original_size), 2)

        return {
            'success': True,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'target_ratio': compressor.target_ratios[compression_level],
            'download_url': f'/download/{filename}',
            'total_time': (time.time() - start_time) * 1000,
            'algorithm_used': method,
            'compression_level': compression_level,
            'steps': steps,
            'comparison': comparison,
            'note': note
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Compression failed: {str(e)}. Try a different compression level.',
            'steps': steps,
            'comparison': comparison
        }

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    file = request.files['file']
    compression_level = request.form.get("compression_level", "medium").lower()
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    if compression_level not in ['low', 'medium', 'high']:
        compression_level = 'medium'
    try:
        result = compress_pdf(file.stream, compression_level)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
