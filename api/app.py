"""Servicio de EPP con ONNX Runtime, sin Ultralytics/PyTorch.
YOLOv8n SH17: tensor de salida (1, 21, 8400), 17 clases.
"""
import io
import json
import os
import secrets
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException

BASE = Path(__file__).resolve().parent
TOKEN = os.environ.get("EPP_API_KEY")
if not TOKEN:
    raise RuntimeError("Falta configurar EPP_API_KEY")

REGLAS = json.loads((BASE / "reglas_epp_zonas.json").read_text(encoding="utf-8"))
CLASES = [
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask",
    "foot", "tool", "glasses", "gloves", "helmet", "hands", "head",
    "medical-suit", "shoes", "safety-suit", "safety-vest"
]
TAMANO_MODELO = 640
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELES = 25_000_000
MAX_LADO = 1920
CONF_MIN = 0.25
IOU_NMS = 0.70

opciones = ort.SessionOptions()
opciones.intra_op_num_threads = 1
opciones.inter_op_num_threads = 1
opciones.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sesion = ort.InferenceSession(
    str(BASE / "best.onnx"),
    sess_options=opciones,
    providers=["CPUExecutionProvider"],
)
NOMBRE_ENTRADA = sesion.get_inputs()[0].name
bloqueo_modelo = Lock()
app = FastAPI(title="Servicio EPP YOLOv8n ONNX")


def preparar_imagen(rgb: Image.Image):
    """Letterbox 640x640, RGB, float32 BCHW; conserva transformación."""
    ancho, alto = rgb.size
    escala = min(TAMANO_MODELO / ancho, TAMANO_MODELO / alto)
    nuevo_ancho = round(ancho * escala)
    nuevo_alto = round(alto * escala)
    pad_x = round((TAMANO_MODELO - nuevo_ancho) / 2 - 0.1)
    pad_y = round((TAMANO_MODELO - nuevo_alto) / 2 - 0.1)
    reducida = rgb.resize((nuevo_ancho, nuevo_alto), Image.Resampling.BILINEAR)
    lienzo = Image.new("RGB", (TAMANO_MODELO, TAMANO_MODELO), (114, 114, 114))
    lienzo.paste(reducida, (pad_x, pad_y))
    tensor = np.asarray(lienzo, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), escala, pad_x, pad_y


def iou_con_caja(caja, otras):
    if len(otras) == 0:
        return np.empty(0, dtype=np.float32)
    xx1 = np.maximum(caja[0], otras[:, 0])
    yy1 = np.maximum(caja[1], otras[:, 1])
    xx2 = np.minimum(caja[2], otras[:, 2])
    yy2 = np.minimum(caja[3], otras[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_a = np.maximum(0, caja[2] - caja[0]) * np.maximum(0, caja[3] - caja[1])
    area_b = np.maximum(0, otras[:, 2] - otras[:, 0]) * np.maximum(0, otras[:, 3] - otras[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def obtener_detecciones(salida, escala, pad_x, pad_y, ancho, alto):
    """Decodificación de (1,21,8400) y NMS por clase sin torch/cv2."""
    if salida.ndim != 3 or salida.shape[0] != 1 or salida.shape[1] != 4 + len(CLASES):
        raise RuntimeError(f"Salida ONNX inesperada: {salida.shape}")
    pred = salida[0].T
    probs = pred[:, 4:]
    clase_ids = probs.argmax(axis=1)
    confianza = probs[np.arange(len(probs)), clase_ids]
    candidatos = np.flatnonzero(confianza >= CONF_MIN)
    if not len(candidatos):
        return []
    xywh = pred[candidatos, :4]
    clases = clase_ids[candidatos]
    scores = confianza[candidatos]
    cajas = np.empty_like(xywh)
    cajas[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    cajas[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    cajas[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    cajas[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    elegidos = []
    for clase in np.unique(clases):
        indices = np.flatnonzero(clases == clase)
        orden = indices[np.argsort(scores[indices])[::-1]]
        while len(orden):
            actual = orden[0]
            elegidos.append(int(actual))
            orden = orden[1:]
            if len(orden):
                orden = orden[iou_con_caja(cajas[actual], cajas[orden]) <= IOU_NMS]
    elegidos.sort(key=lambda i: -float(scores[i]))
    detecciones = []
    for j in elegidos[:300]:
        x1, y1, x2, y2 = cajas[j]
        x1 = float(np.clip((x1 - pad_x) / escala, 0, ancho))
        y1 = float(np.clip((y1 - pad_y) / escala, 0, alto))
        x2 = float(np.clip((x2 - pad_x) / escala, 0, ancho))
        y2 = float(np.clip((y2 - pad_y) / escala, 0, alto))
        detecciones.append({
            "clase": CLASES[int(clases[j])],
            "confianza": round(float(scores[j]), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return detecciones


def proporcion_dentro(objeto, persona):
    x1 = max(objeto["x1"], persona["x1"])
    y1 = max(objeto["y1"], persona["y1"])
    x2 = min(objeto["x2"], persona["x2"])
    y2 = min(objeto["y2"], persona["y2"])
    interseccion = max(0, x2 - x1) * max(0, y2 - y1)
    area = (objeto["x2"] - objeto["x1"]) * (objeto["y2"] - objeto["y1"])
    return interseccion / area if area > 0 else 0.0


@app.get("/salud")
def salud():
    return {"estado": "activo", "modelo": "YOLOv8n_SH17_ONNX"}


@app.post("/analizar")
async def analizar(
    imagen: UploadFile = File(...),
    zona_id: str = Form(...),
    x_api_key: str | None = Header(default=None),
):
    if not secrets.compare_digest(x_api_key or "", TOKEN):
        raise HTTPException(401, "Acceso no autorizado")
    if zona_id not in REGLAS:
        raise HTTPException(400, "Zona no configurada")
    contenido = await imagen.read(MAX_BYTES + 1)
    if len(contenido) > MAX_BYTES:
        raise HTTPException(413, "Imagen demasiado grande")
    try:
        with Image.open(io.BytesIO(contenido)) as img:
            if img.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, "Formato no permitido")
            if img.width * img.height > MAX_PIXELES:
                raise HTTPException(413, "Resolución excesiva")
            rgb = ImageOps.exif_transpose(img)
            rgb.thumbnail((MAX_LADO, MAX_LADO), Image.Resampling.LANCZOS)
            rgb = rgb.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Imagen inválida")

    entrada, escala, pad_x, pad_y = preparar_imagen(rgb)
    with bloqueo_modelo:
        salida = sesion.run(None, {NOMBRE_ENTRADA: entrada})[0]
    detecciones = obtener_detecciones(salida, escala, pad_x, pad_y, rgb.width, rgb.height)
    personas = sorted(
        [d for d in detecciones if d["clase"] == "person" and d["confianza"] >= 0.50],
        key=lambda d: d["x1"],
    )
    regla = REGLAS[zona_id]
    requeridos = regla["epp_requeridos"]
    trabajadores = [
        {"persona_id": i + 1, "confianza_persona": p["confianza"],
         "epp_detectados": {}, "epp_no_confirmados": [], "estado": "revision_pendiente"}
        for i, p in enumerate(personas)
    ]
    epp_sin_asignar = 0
    for epp in detecciones:
        if epp["clase"] not in requeridos or epp["confianza"] < 0.40:
            continue
        coincidencias = [proporcion_dentro(epp, p) for p in personas]
        if not coincidencias or max(coincidencias) < 0.60:
            epp_sin_asignar += 1
            continue
        indice = int(np.argmax(coincidencias))
        categoria = epp["clase"]
        prev = trabajadores[indice]["epp_detectados"].get(categoria, 0.0)
        trabajadores[indice]["epp_detectados"][categoria] = max(prev, epp["confianza"])
    for trabajador in trabajadores:
        trabajador["epp_no_confirmados"] = [
            epp for epp in requeridos if epp not in trabajador["epp_detectados"]
        ]
    requieren_revision = sum(bool(t["epp_no_confirmados"]) for t in trabajadores)
    return {
        "zona_id": zona_id,
        "zona": regla["nombre"],
        "origen_reglas": regla["origen_regla_especifica"],
        "reglas_validadas": regla["validacion_profesional"],
        "epp_requeridos": requeridos,
        "modelo": "YOLOv8n_SH17_ONNX",
        "cantidad_trabajadores": len(trabajadores),
        "trabajadores": trabajadores,
        "resumen": {
            "requieren_revision": requieren_revision,
            "epp_sin_asignar": epp_sin_asignar,
            "estado": "revision_humana_requerida",
        },
        "advertencia": (
            "EPP no confirmado no equivale a EPP ausente. "
            "Las reglas de zona son académicas y requieren validación profesional. "
            "La decisión final corresponde al prevencionista."
        ),
    }
