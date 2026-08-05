import os
import time
import subprocess
import cv2
import mediapipe as mp
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceDetectorOptions, RunningMode

app = Flask(__name__)

# Carpetas de almacenamiento temporal
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# --- LÓGICA DE DETECCIÓN Y PIXELADO ---
def pixelate_region_adaptive(image, pixel_size=18):
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    num_blocks_x = max(2, w // pixel_size)
    num_blocks_y = max(2, h // pixel_size)
    temp = cv2.resize(image, (num_blocks_x, num_blocks_y), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_face_pixelation(frame, face_result, max_faces):
    h, w, _ = frame.shape
    if face_result.detections:
        detections = face_result.detections
        
        if max_faces != -1 and len(detections) > max_faces:
            detections = sorted(
                detections, 
                key=lambda d: d.categories[0].score, 
                reverse=True
            )[:max_faces]

        for detection in detections:
            bbox = detection.bounding_box
            padding_x = int(bbox.width * 0.20)
            padding_y = int(bbox.height * 0.20)

            x_start = max(0, int(bbox.origin_x) - padding_x)
            y_start = max(0, int(bbox.origin_y) - padding_y)
            x_end = min(w, int(bbox.origin_x + bbox.width) + padding_x)
            y_end = min(h, int(bbox.origin_y + bbox.height) + padding_y)

            face_roi = frame[y_start:y_end, x_start:x_end]
            if face_roi.size > 0:
                frame[y_start:y_end, x_start:x_end] = pixelate_region_adaptive(face_roi, pixel_size=18)

def finalize_video_with_audio_options(original_input, temp_video, final_output, audio_mode):
    """
    Procesa la salida final con FFmpeg:
    - Alta calidad visual (CRF 17)
    - Eliminación completa de metadatos (-map_metadata -1 y +bitexact)
    - Opciones de audio: 'keep' (mantener), 'mute' (sin audio), 'pitch' (distorsionar voz)
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,      # Vídeo pixelado (entrada 0)
    ]

    # Si se requiere audio (original o distorsionado), añadimos el vídeo original como entrada 1
    if audio_mode in ['keep', 'pitch']:
        cmd.extend(["-i", original_input])

    cmd.extend([
        "-c:v", "libx264",
        "-crf", "17",          # Máxima calidad visual (transparente)
        "-preset", "fast",
        "-map", "0:v:0",       # Tomar la pista de vídeo pixelada
    ])

    # Configuración de audio según la opción seleccionada
    if audio_mode == 'keep':
        cmd.extend([
            "-map", "1:a?",
            "-c:a", "aac"
        ])
    elif audio_mode == 'pitch':
        # Bajar el tono de la voz al 75% de frecuencia (voz grave de protección de identidad)
        # y reajustar el tiempo para que sincronice perfectamente
        audio_filter = "asetrate=44100*0.75,atempo=1/0.75"
        cmd.extend([
            "-map", "1:a?",
            "-af", audio_filter,
            "-c:a", "aac"
        ])
    elif audio_mode == 'mute':
        cmd.extend(["-an"])  # Desactivar audio por completo

    # Eliminación estricta de metadatos
    cmd.extend([
        "-map_metadata", "-1",
        "-fflags", "+bitexact",
        "-flags:v", "+bitexact",
        final_output
    ])
    
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        print(f"[!] Error procesando el vídeo final con FFmpeg: {e}")
        if os.path.exists(temp_video):
            os.replace(temp_video, final_output)

    # Restablecer la marca de tiempo a nivel de sistema operativo
    try:
        os.utime(final_output, (0, 0))
    except Exception:
        pass

def process_video_file(input_path, output_path, max_faces, audio_mode):
    base_options = python.BaseOptions(model_asset_path="blaze_face_full_range.tflite")
    options = FaceDetectorOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        min_detection_confidence=0.15,
        min_suppression_threshold=0.3
    )
    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0

    temp_video_path = output_path + "_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        timestamp_ms = int((frame_index / fps) * 1000)
        frame_index += 1

        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        apply_face_pixelation(frame, detection_result, max_faces)

        out.write(frame)

    cap.release()
    out.release()
    detector.close()

    # Ensamblar vídeo final con el ajuste de audio y metadatos
    finalize_video_with_audio_options(input_path, temp_video_path, output_path, audio_mode)

    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

# --- RUTAS DE FLASK ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)

        try:
            max_faces = int(request.form.get('max_faces', 1))
        except ValueError:
            max_faces = 1

        audio_mode = request.form.get('audio_mode', 'keep')

        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)

            base_name = os.path.splitext(filename)[0]
            output_filename = f"pixelated_{base_name}.mp4"
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

            process_video_file(input_path, output_path, max_faces, audio_mode)

            return send_file(output_path, as_attachment=True)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)