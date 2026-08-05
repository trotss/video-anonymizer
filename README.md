# Pixelador de Caras y Anonimizador de Audio para Vídeo

Aplicación web desarrollada con Flask, OpenCV y MediaPipe que permite cargar cualquier archivo de vídeo para pixelar automáticamente los rostros detectados, eliminar metadatos y aplicar tratamiento de privacidad al audio (incluyendo anonimización de voz).

---

## Características Principales

* Detección Dinámica de Rostros: Utiliza el detector BlazeFace Full Range de MediaPipe para identificar caras a corta y larga distancia.
* Pixelado Adaptativo: El tamaño de los bloques de censura se ajusta automáticamente según la escala del rostro en pantalla.
* Protección de Audio Integrada: Tres modos de audio (mantener original, silenciar por completo o distorsionar la voz bajando el tono).
* Eliminación Estricta de Metadatos: Utiliza FFmpeg para limpiar registros EXIF, marcas de software, contenedores de datos y restablecer fechas a nivel de sistema operativo.

---

## Requisitos del Sistema

* Python 3.9 o superior
* FFmpeg instalado en el sistema operativo

### Instalación de FFmpeg

* Linux (Ubuntu/Debian):
  sudo apt update && sudo apt install ffmpeg

* macOS (con Homebrew):
  brew install ffmpeg

* Windows:
  Descargar desde la página oficial de FFmpeg y añadir el directorio bin a las variables de entorno (PATH).

---

## Descarga del Modelo de Detección

La aplicación utiliza el modelo oficial BlazeFace (Full Range) de MediaPipe. Debes descargar el archivo del modelo y colocarlo en la raíz del proyecto.

1. Descarga el archivo .tflite directamente desde el repositorio oficial de MediaPipe:
   https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite
2. Guarda el archivo descargado en la raíz de la carpeta del proyecto con el nombre:
   blaze_face_full_range.tflite

Estructura de archivos esperada:

.
├── app.py
├── blaze_face_full_range.tflite
├── requirements.txt
├── templates/
│   └── index.html
└── README.md

---

## Instalación y Uso

1. Clonar el repositorio:
   git clone https://github.com/tu-usuario/nombre-repositorio.git
   cd nombre-repositorio

2. Crear y activar un entorno virtual:
   python3 -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate

3. Instalar dependencias de Python:
   pip install -r requirements.txt

4. Ejecutar el servidor Flask:
   python app.py

5. Abrir el navegador e ingresar a http://127.0.0.1:5000.

---

## Parámetros de Ajuste y Sensibilidad

Los principales parámetros de detección y procesamiento se pueden modificar directamente en app.py:

* min_detection_confidence (Valor por defecto: 0.15)
  Umbral mínimo de confianza para detectar un rostro. Un valor bajo (0.15) aumenta la sensibilidad para no perder la cara de perfil, mientras que un valor alto (0.50) reduce falsos positivos.

* min_suppression_threshold (Valor por defecto: 0.3)
  Umbral para la supresión no máxima (NMS). Evita solapamiento de múltiples cajas en el mismo rostro.

* pixel_size (Valor por defecto: 18)
  Tamaño en píxeles de cada bloque del mosaico en pixelate_region_adaptive. Cuanto mayor es el número, más grandes son los cuadros de censura.

* padding_x / padding_y (Valor por defecto: 0.20 / 20%)
  Margen adicional aplicado a la caja delimitadora para cubrir orejas, frente y barbilla.

* max_faces (Valor por defecto: 1 o -1)
  Ajustable desde la interfaz web. Controla cuántos rostros se pixelan por fotograma (-1 para pixelar todos).
