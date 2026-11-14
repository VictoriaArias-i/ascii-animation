# 🌿 ascii-animation

Mi dibujo (hecho en Procreate) animado y convertido a arte ASCII a color, todo con Python.

<img src="victoria_ascii.gif" width="520" alt="Dibujo animado en ASCII">

## Qué hace
- **Mano** saludando, lenta y en bucle
- **Parpadeo** y **pelo** moviéndose (la cara queda quieta)
- **Fondo** de cuaderno con flores, dibujado con código
- Convierte cada frame en letras con los colores del dibujo

## Cómo funciona
| Archivo | Para qué sirve |
|---|---|
| `cutout.py` | Recorta a la chica del fondo (OpenCV) |
| `sketchbook.py` | Dibuja el fondo de cuaderno con flores (Pillow + numpy) |
| `animate.py` | Mueve mano, pelo y párpados deformando el dibujo |
| `ascii_girl.py` | Pasa cada frame a ASCII y exporta GIF y MP4 |
| `rot.png` | El dibujo original, ya girado |

## Usarlo
```bash
pip install -r requirements.txt
python ascii_girl.py              # crea girl_ascii.gif y girl_ascii.mp4
python ascii_girl.py --terminal   # lo ve en tu terminal (Ctrl+C para salir)
```
Para crear el GIF y el MP4 necesitas [ffmpeg](https://ffmpeg.org/) instalado.

## Con otro dibujo
Reemplaza `rot.png` por tu dibujo (de pie, cabeza arriba) y ajusta en `animate.py` las coordenadas de la muñeca, los ojos y la cara.

Dibujo y código: Victoria Arias
