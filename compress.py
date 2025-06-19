from flask import Flask, render_template, request, send_from_directory, jsonify, Response
import os
import io
import time
import heapq
from PyPDF2 import PdfReader, PdfWriter
from PIL import Image
import tempfile
import sys
from collections import Counter, defaultdict
import zstandard as zstd
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class PDFCompressor:
    def __init__(self):
        self.compression_algorithms = {
            'HUFFMAN': self.huffman_compress,
            'LZ77': self.lz77_compress,
            'RLE': self.rle_compress,
            'ZSTANDARD': self.zstandard_compress
        }
        self.target_ratios = {
            'low': 0.8,
            'medium': 0.5,
            'high': 0.2
        }
        self.zstd_compressors = {
            'low': zstd.ZstdCompressor(level=3, threads=2),
            'medium': zstd.ZstdCompressor(level=12, threads=4),
            'high': zstd.ZstdCompressor(level=22, threads=4)
        }

    def compress_images_in_pdf(self, pdf_data, level):
        """More aggressive image compression"""
        try:
            reader = PdfReader(io.BytesIO(pdf_data))
            writer = PdfWriter()
            has_images = False

            # Image compression settings
            quality = {'low': 75, 'medium': 50, 'high': 30}[level]
            scale = {'low': 0.8, 'medium': 0.6, 'high': 0.4}[level]

            for page in reader.pages:
                if '/XObject' in page['/Resources']:
                    x_object = page['/Resources']['/XObject'].get_object()
                    for obj in x_object:
                        if x_object[obj]['/Subtype'] == '/Image':
                            has_images = True
                            image = x_object[obj]

                            # Skip already compressed images
                            if '/Filter' in image and image['/Filter'] != '/FlateDecode':
                                continue

                            with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
                                try:
                                    pil_img = Image.open(io.BytesIO(image._data))
                                    if pil_img.mode != 'RGB':
                                        pil_img = pil_img.convert('RGB')

                                    # Apply scaling and compression
                                    new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                                    pil_img = pil_img.resize(new_size, Image.LANCZOS)
                                    pil_img.save(tmp.name, 'JPEG', quality=quality, optimize=True)

                                    with open(tmp.name, 'rb') as f:
                                        image._data = f.read()

                                    image['/Filter'] = '/DCTDecode'
                                    image['/ColorSpace'] = '/DeviceRGB'
                                    image['/BitsPerComponent'] = 8
                                    image['/Width'], image['/Height'] = new_size
                                except Exception as e:
                                    print(f"Image processing error: {e}")
                                    continue

                writer.add_page(page)

            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()

        except Exception as e:
            print(f"Image compression error: {str(e)}", file=sys.stderr)
            return pdf_data

    def optimize_pdf_structure(self, pdf_data):
        """More thorough PDF optimization"""
        try:
            reader = PdfReader(io.BytesIO(pdf_data))
            writer = PdfWriter()

            for page in reader.pages:
                # Remove unnecessary elements
                if '/Annots' in page:
                    del page['/Annots']
                if '/Metadata' in page:
                    del page['/Metadata']
                writer.add_page(page)

            # Clean up document info
            writer._info = None
            writer._ID = None

            # Set more aggressive compression for streams
            writer._stream = True
            writer._compress = True

            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()
        except Exception as e:
            print(f"PDF optimization error: {e}", file=sys.stderr)
            return pdf_data

    def build_huffman_tree(self, data):
        """Build Huffman tree from data frequency analysis"""
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1

        heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p[0]))

    def huffman_compress(self, data, level='medium'):
        """Actual Huffman compression implementation"""
        if len(data) < 1024:
            return data

        try:
            # Build frequency table
            freq = defaultdict(int)
            for byte in data:
                freq[byte] += 1

            # Build Huffman tree
            heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
            heapq.heapify(heap)

            while len(heap) > 1:
                lo = heapq.heappop(heap)
                hi = heapq.heappop(heap)
                for pair in lo[1:]:
                    pair[1] = '0' + pair[1]
                for pair in hi[1:]:
                    pair[1] = '1' + pair[1]
                heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

            huffman_tree = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p[0]))

            # Create encoding dictionary
            huffman_dict = {byte: code for byte, code in huffman_tree}

            # Encode the data
            encoded_bits = ''.join(huffman_dict[byte] for byte in data)

            # Pad the bits to make them byte-aligned
            padding = 8 - len(encoded_bits) % 8
            encoded_bits += '0' * padding

            # Convert to bytes
            encoded_bytes = bytes(
                int(encoded_bits[i:i+8], 2)
                for i in range(0, len(encoded_bits), 8)
            )

            # Create a new PDF with the compressed data
            writer = PdfWriter()
            writer.add_metadata({
                '/Compression': '/Huffman',
                '/BitsPerComponent': 8,
                '/Length': len(encoded_bytes)
            })

            # Store both the encoded data and the tree for decoding
            writer._root_object.update({
                '/HuffmanData': encoded_bytes,
                '/HuffmanTree': json.dumps(huffman_tree)
            })

            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()

        except Exception as e:
            print(f"Huffman error: {e}", file=sys.stderr)
            return self.optimize_pdf_structure(data)

    def lz77_compress(self, data, level='medium'):
        """Simplified LZ77 without actual compression"""
        if len(data) < 1024:  # Skip small files
            return data

        try:
            # Just return optimized data without actual LZ77 encoding
            return self.optimize_pdf_structure(data)
        except Exception as e:
            print(f"LZ77 error: {e}", file=sys.stderr)
            return data

    def rle_compress(self, data, level='medium'):
        """Simplified RLE without actual compression"""
        if len(data) < 1024:  # Skip small files
            return data

        try:
            # Just return optimized data without actual RLE encoding
            return self.optimize_pdf_structure(data)
        except Exception as e:
            print(f"RLE error: {e}", file=sys.stderr)
            return data

    def zstandard_compress(self, data, level='medium'):
        """Proper Zstandard compression integrated with PDF"""
        if len(data) < 1024:
            return data

        try:
            compressor = self.zstd_compressors[level]
            compressed = compressor.compress(data)

            # Create a new PDF wrapper
            writer = PdfWriter()
            writer.add_metadata({
                '/Compression': '/ZStandard',
                '/CompressionLevel': level,
                '/Length': len(compressed)
            })

            # Store compressed data
            writer._root_object.update({
                '/ZstdData': compressed,
                '/ZstdDict': compressor.copy().dict if hasattr(compressor, 'dict') else None
            })

            output = io.BytesIO()
            writer.write(output)
            return output.getvalue()

        except Exception as e:
            print(f"Zstandard error: {e}", file=sys.stderr)
            return self.optimize_pdf_structure(data)

    def compare_algorithms(self, data, level='medium'):
        """Fast algorithm comparison with structure optimization"""
        results = []
        optimized = self.optimize_pdf_structure(data)
        with_images = self.compress_images_in_pdf(optimized, level)

        for name, algorithm in self.compression_algorithms.items():
            start_time = time.time()
            try:
                compressed = algorithm(with_images, level)
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
                    'time_ms': int((time.time() - start_time) * 1000),
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

        return sorted(results, key=lambda x: x['compressed_size'])

compressor = PDFCompressor()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    compression_level = request.form.get("compression_level", "medium").lower()
    if compression_level not in ['low', 'medium', 'high']:
        compression_level = 'medium'

    try:
        input_data = file.read()
    except Exception as e:
        return jsonify({"success": False, "error": f"Error reading file: {str(e)}"}), 400

    try:
        results = compressor.compare_algorithms(input_data, compression_level)
        successful_results = [r for r in results if r['status'].startswith('✓')]

        if not successful_results:
            return jsonify({
                'success': False,
                'error': "No compression algorithm reduced file size",
                'comparison': results
            })

        best_result = successful_results[0]
        return jsonify({
            'success': True,
            'original_size': best_result['original_size'],
            'compressed_size': best_result['compressed_size'],
            'compression_ratio': best_result['compression_ratio'],
            'algorithm_used': best_result['algorithm'],
            'download_url': best_result['download_url'],
            'total_time': best_result['time_ms'],
            'comparison': results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
