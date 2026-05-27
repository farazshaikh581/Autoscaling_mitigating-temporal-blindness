import time
from flask import Flask, jsonify
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=16)

def factorize(n):
    """
    Simulate CPU-intensive work: busy-loop for 2-5 seconds
    This generates real CPU load that HPA can measure
    """
    start = time.time()
    # Busy loop for ~2 seconds of real CPU work
    while time.time() - start < 2.0:  # ✅ 2 seconds of work
        x = 1
        for i in range(1000):
            x += i
    return [n]

@app.route('/factor', methods=['GET'])
def get_factors():
    number_to_factor = 1000000016000000063
    start_time = time.time()
    future = executor.submit(factorize, number_to_factor)
    try:
        factors = future.result(timeout=30)
        duration = time.time() - start_time
        return jsonify({'number': number_to_factor, 'factors': factors, 'duration_seconds': duration, 'num_factors': len(factors)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status='healthy'), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
