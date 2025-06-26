import os
print("Current working directory:", os.getcwd())

from flask import Flask, render_template, request, redirect, url_for, session, Response
import cv2
import os
from datetime import datetime
import csv

app = Flask(__name__, template_folder='templates')
app.secret_key = 'sentron-secret-key'  # Change this in production

# Simulated user database
USER = {'admin': 'password123'}

# ========== VIDEO FEED ==========
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Stream frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ========== AUTH ==========
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if USER.get(username) == password:
            session['user'] = username
            return redirect(url_for('home'))
        return render_template('index.html', error="Invalid credentials.")
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ========== PROTECTED ROUTES ==========
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/threats')
def threats():
    if 'user' not in session:
        return redirect(url_for('login'))
    threats = []
    if os.path.exists('logs/threats.csv'):
        with open('logs/threats.csv') as f:
            reader = csv.DictReader(f)
            threats = list(reader)
    return render_template('threats.html', threats=threats)

@app.route('/snapshots')
def snapshots():
    if 'user' not in session:
        return redirect(url_for('login'))

    snaps = []
    path = 'static/snapshots'

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                try:
                    # Expecting filename like: threat_gun_20250625_150012.jpg
                    parts = filename.replace('.jpg', '').split('_')
                    threat = parts[1] if len(parts) > 2 else "Unknown"

                    # Parse date and time
                    date_str = parts[2]  # '20250625'
                    time_str = parts[3]  # '150012'

                    # Format date and time
                    formatted_date = f"{date_str[6:8]}/{date_str[4:6]}/{date_str[0:4]}"  # DD/MM/YYYY
                    formatted_time = f"{time_str[0:2]}:{time_str[2:4]}:{time_str[4:6]}"    # HH:MM:SS

                    snaps.append({
                        'filename': filename,
                        'threat': threat.capitalize(),
                        'date': formatted_date,
                        'time': formatted_time
                    })
                except Exception as e:
                    print(f"Error parsing filename '{filename}': {e}")
                    continue  # Skip invalid files

    return render_template('snapshots.html', snapshots=snaps)


@app.route('/capture_snapshot/<threat>')
def capture_snapshot(threat):
    if 'user' not in session:
        return redirect(url_for('login'))

    # Create snapshots folder if it doesn't exist
    path = 'static/snapshots'
    os.makedirs(path, exist_ok=True)

    # Capture a frame
    success, frame = camera.read()
    if success:
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')  # safe format
        filename = f"threat_{threat}_{timestamp}.jpg"
        filepath = os.path.join(path, filename)
        cv2.imwrite(filepath, frame)
        return f"Snapshot saved: {filename}", 200
    else:
        return "Failed to capture frame", 500

@app.route('/test')
def test():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


