# Rellenado guiado para el pensamiento en la detección cero-disparo de imágenes generadas por IA

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch 2.7.1](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![vLLM 0.10.1](https://img.shields.io/badge/vLLM-0.10.1-green.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.11031-b31b1b.svg)](https://arxiv.org/abs/2506.11031)

## 🤝 Cita

Si utiliza este código en su investigación, por favor cite nuestro artículo:

```bibtex
@misc{kachwala2025prefillguidedthinking,
      title={Prefill-Guided Thinking for zero-shot detection of AI-generated images}, 
      author={Zoher Kachwala and Danishjeet Singh and Danielle Yang and Filippo Menczer},
      year={2025},
      eprint={2506.11031},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.11031}, 
}
```

*Nota: Artículo presentado en ACL ARR.*

---

<p align="center">
  <img src="images/mosaic.png" alt="Imágenes de ejemplo de los conjuntos de datos D3, DF40 y GenImage" width="800"/>
</p>

> **¿Puedes determinar cuáles de las imágenes de arriba son reales en comparación con generadas por IA?** Responda en el pie de página¹

Este repositorio contiene el sistema de evaluación para nuestro artículo sobre el uso de **Prefill-Guided Thinking (PGT)** para detectar imágenes generadas por IA con modelos Visión-Lenguaje (VLM).

> 💡 **Para documentación técnica detallada, particularmente útil para agentes de código LLM**: Consulte [AGENTS.md](AGENTS.md) para obtener detalles completos de arquitectura, firmas de funciones y especificaciones de implementación.

**Descubrimiento clave:** Simplemente rellenar la respuesta de un VLM con la frase *"Let's examine the style and the synthesis artifacts"* (Examinemos el estilo y los artefactos de síntesis) mejora la detección en un **100% Macro F1** — sin ningún entrenamiento o afinamiento.

## 🎯 ¿Qué es Prefill-Guided Thinking?

<p align="center">
  <img src="images/strawberry_PGT.png" alt="PGT aplicado a una fresa generada por Midjourney" width="800"/>
</p>

En lugar de preguntar directamente a un VLM si una imagen es falsa, **rellenamos** su respuesta para guiar su razonamiento:

- **(a) Línea base**: Consulta directa → clasificación incorrecta (real)
- **(b) Cadena de pensamiento**: *"Let's think step by step"* (Pensemos paso a paso) → aún incorrecto
- **(c) S2 (nuestro método)**: *"Let's examine the style and the synthesis artifacts"* (Examinemos el estilo y los artefactos de síntesis) → correcto ✓

Esta técnica simple funciona en **3 VLM** y **16 diferentes generadores de imágenes** abarcando rostros, objetos y escenas naturales.

---

## 🚀 Primeros pasos

### Instalación

**Consulte [SETUP.md](SETUP.md)** para instrucciones completas de configuración del entorno (conda, PyTorch, vLLM, Flash-Attention).

### Uso

**Consulte [Ejemplos de uso](AGENTS.md#-usage-examples)** para ejemplos detallados en línea de comandos y todas las opciones disponibles.

---

## 📊 Conjuntos de datos

Evaluamos en tres líneas base diversas:

| Conjunto de datos | Contenido | Imágenes | Generadores |
|-----------------|-----------|----------|-------------|
| **D3** | Imágenes web diversas (objetos, escenas, arte) | 8.4k | 4 (variantes de Stable Diffusion, DeepFloyd) |
| **DF40** | Rostros humanos (deepfakes) | 10k | 6 (Midjourney, StyleCLIP, StarGAN, etc.) |
| **GenImage** | Objetos de ImageNet (animales, vehículos) | 10k | 8 (ADM, BigGAN, GLIDE, etc.) |

### Configuración de datos

**Consulte [Recopilación y configuración de datos](AGENTS.md#-data-collection--setup)** para instrucciones completas sobre la descarga y organización de todos los conjuntos de datos.

---

## 🧪 Modelos admitidos

- **Qwen2.5-VL-7B** — Transformador visual de resolución dinámica
- **LLaVA-OneVision-7B** — Modelo multimodal de seguimiento de instrucciones
- **Qwen3-VL-8B** — Actual modelo visión-lenguaje de Qwen, con configuraciones Instruct y Thinking

Todas las configuraciones de modelo se ejecutan a través de vLLM para una inferencia eficiente.

---

## 🎨 Tres métodos de evaluación

| Método | Descripción |
|--------|-------------|
| **Línea base** | Sin relleno, solo preguntar |
| **CoT** | Razonamiento de cadena de pensamiento |
| **S2** | (Nuestro método) Alineado con la tarea |

**Consulte [Ejemplos de uso](AGENTS.md#-usage-examples)** para ejemplos detallados en línea de comandos y todas las opciones disponibles.

---

## 📈 Resultados

<p align="center">
  <img src="images/macro_f1_bars.png" alt="Comparación de rendimiento Macro F1" width="900"/>
</p>

**Macro F1 de detección entre modelos, conjuntos de datos y variaciones de PGT.** Las barras muestran la mejora relativa de S2 sobre el siguiente mejor método.

Las figuras de recall por generador utilizadas en el artículo de COLM se generan en `results/figures/` mediante los scripts de graficación en `results/`.

### Interpretabilidad: Progreso de confianza

Para comprender cómo afectan los rellenos al razonamiento, seguimos la confianza en la respuesta en cinco intervalos de respuesta parcial (0–100% de las oraciones):

<p align="center">
  <img src="images/partial-responses.png" alt="Intervalos de respuesta parcial" width="700"/>
</p>

En cada intervalo, sondamos la respuesta del modelo y su confianza. Los resultes revelan un patrón impactante:

<p align="center">
  <img src="images/interval_progression_qwen25.png" alt="Progresión de confianza para Qwen" width="900"/>
</p>

**Evolución de la confianza en la respuesta y el Macro F1 a través de respuestas parciales para Qwen.** Las consultas de línea base activan una alta confianza inmediata a pesar de una mala detección — el modelo se compone de una respuesta antes de examinar la imagen. Los rellenos inducen una caída de confianza en medio de la respuesta, con una detección que mejora progresivamente a medida que avanza la respuesta.

---

## 🔬 Uso avanzado

- **Generación multi-respuesta (n>1)** - Generar múltiples respuestas con voto mayoritario → [Detalles](AGENTS.md#multi-response-evaluation-n1)
- **Modos de frase** - Probar prefill vs prompt vs instrucción de sistema → [Detalles](AGENTS.md#five-phrase-modes)
- **Modo debug** - Validación rápida con 5 ejemplos → [Detalles](AGENTS.md#debug-mode-testing)

---

## 📂 Estructura de salida

Los resultados se guardan en directorios jerárquicos con archivos JSON con marca de tiempo que contienen métricas y rastros completos de razonamiento.

**Consulte [Estructura de salida](AGENTS.md#-output-structure)** para organización detallada de archivos y esquemas JSON.

---

## 📊 Visualización y análisis

Generar gráficos listos para publicación (barras Macro F1, gráficos de radar, análisis de vocabulario, etc.)

**Consulte [Sistema de visualización y análisis](AGENTS.md#-plotting--visualization-system)** para obtener gráficos disponibles e instrucciones de uso.

---

## 📚 Documentación

- **[SETUP.md](SETUP.md)** - Instrucciones de configuración del entorno e instalación
- **[AGENTS.md](AGENTS.md)** - Referencia técnica completa (arquitectura, firmas de funciones, todos los detalles)
- **Artículo** - [arXiv:2506.11031](https://arxiv.org/abs/2506.11031)

---

## 👥 Autores

**Zoher Kachwala** · Danishjeet Singh · Danielle Yang · Filippo Menczer

Observatorio de Medios Sociales
Indiana University, Bloomington

---

<sub>¹ **Respuesta al cuestionario de imágenes:** Solo las imágenes 3, 10 y 11 en el mosaico son reales. Todas las demás son generadas por IA.</sub>
