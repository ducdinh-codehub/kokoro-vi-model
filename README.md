# kokoro-vi-model

Hosting repo for the **Vietnamese Kokoro text-to-speech ONNX models** used by the on-device Expo TTS app. The model files are published as **GitHub release assets** (not committed to git), so the app can download them directly on a phone.

The models are an ONNX export of [Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese) (`contextboxai/Kokoro-Vietnamese`). They are re-hosted here because Hugging Face moved that repo to the Xet CDN, which stalls the iOS on-device downloader. A GitHub release download is a plain redirect and works reliably on device.

In the app:

- **Kokoro** tab (**V1**): Vietnamese and English text-to-speech. Vietnamese runs through `onnxruntime-react-native`; English runs through `react-native-executorch`.
- **Kokoro Lite** tab (**V2**): the same screen, but Vietnamese runs an 8-bit quantized model. It is ~92 MB instead of 326 MB and ~2× faster (see [Kokoro V1 vs V2](#kokoro-v1-vs-v2)).

## Releases

| Version | Release tag | Asset | Size | sha256 |
|---|---|---|---|---|
| **V1** (Kokoro) | [`v1`](https://github.com/ducdinh-codehub/kokoro-vi-model/releases/tag/v1) | `kokoro_vi.onnx` | 325,731,953 bytes (~326 MB) | `da191277f58633649a9c0d2ae8012e80ef57ea8e2a56e30323c0f7df1ca29087` |
| **V2** (Kokoro Lite, int8) | [`v2-lite`](https://github.com/ducdinh-codehub/kokoro-vi-model/releases/tag/v2-lite) | `kokoro_vi_q8.onnx` | 91,779,346 bytes (~92 MB) | `6848197b218b032d490b0d9be0cef05367d15d5376aa19d9b0698821b512ae9b` |

Direct download URLs:

```
https://github.com/ducdinh-codehub/kokoro-vi-model/releases/download/v1/kokoro_vi.onnx
https://github.com/ducdinh-codehub/kokoro-vi-model/releases/download/v2-lite/kokoro_vi_q8.onnx
```

Verify a download:

```bash
shasum -a 256 kokoro_vi.onnx kokoro_vi_q8.onnx
```

## Kokoro V1 vs V2

V1 and V2 are **the same network** with the same inputs and outputs. They share the G2P, the voicepacks and the whole synthesis pipeline. The only difference is how the weights are stored.

| | **V1 (Kokoro tab)** | **V2 (Kokoro Lite tab)** |
|---|---|---|
| Weights | FP32 (original export) | 8-bit (dynamic `QUInt8` quantization) |
| Model size | 326 MB | **92 MB** (~3.5× smaller) |
| Synthesis speed* | 2.7 s for 10 s of audio | **1.3 s** for 10 s of audio |
| Audio quality | Reference | Very close (same timing and durations). Compare by ear. |
| On-device file | `<documents>/kokoro/kokoro_vi.onnx` | `<documents>/kokoro/kokoro_vi_q8.onnx` |
| Use when | You want the reference quality | You care about download size, storage or speed |

\*Measured on a Mac CPU (onnxruntime, 4 threads, 3 Vietnamese sentences). Not yet measured on a phone.

English is not affected by this choice. It always runs through `react-native-executorch`, and only Vietnamese has a lite build.

### Kokoro V1: the full model

- The original FP32 ONNX export of Kokoro-Vietnamese (StyleTTS2-based Kokoro architecture).
- This is the quality reference. V2 is measured against it.
- Its large size (326 MB) means a long first download and more memory on low-end phones.

### Kokoro V2: the lite model

V2 is V1 with every weight quantized to 8 bits. Activations stay in floating point and are quantized dynamically at runtime, so the model needs no calibration data.

Running `quantize_dynamic` directly on V1 only gets the file down to **~311 MB**. Most of Kokoro's weights live in Conv layers that use **weight normalization**. Their real weights are computed at runtime by a small subgraph (`g * v / ||v||`), so the quantizer never sees them as constant tensors and skips them.

The fix is to **constant-fold the graph first**. onnxruntime's basic graph optimizations evaluate those weight-norm subgraphs once and save the results as plain Conv weight initializers. After that, `quantize_dynamic` can quantize them, which brings the file to ~92 MB.

#### How to rebuild V2

```bash
python3 -m venv venv
./venv/bin/pip install onnx numpy onnxruntime==1.24.3
```

```python
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# 1. Constant-fold (bakes the weight-norm subgraphs into Conv weights)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath = "kokoro_vi_folded.onnx"
ort.InferenceSession("kokoro_vi.onnx", so, providers=["CPUExecutionProvider"])

# 2. Quantize all weights to 8-bit
quantize_dynamic("kokoro_vi_folded.onnx", "kokoro_vi_q8.onnx", weight_type=QuantType.QUInt8)
```

Before you publish, check that the result loads on the **same onnxruntime version the app ships**. That is `onnxruntime-react-native` 1.24.3, so test with `onnxruntime==1.24.3`.

## Model interface (V1 and V2)

| Name | Direction | Type / shape | Notes |
|---|---|---|---|
| `input_ids` | input | `int64 [1, N]` | Phoneme token ids wrapped with BOS/EOS `0`. `N ≤ 512`; split long text by sentence. |
| `ref_s` | input | `float32 [1, 256]` | Style vector. Row `min(max(phoneme_len, 1), 510) - 1` of the voicepack. |
| `speed` | input | `float32` scalar `[]` | `1.0` is normal speed. |
| `waveform` | output | `float32` | Mono audio at **24 kHz**. |
| `duration` | output | `int64` | Predicted frames per token. |

- **Phonemes:** Vietnamese text is converted to IPA with [`vig2p`](https://pypi.org/project/vig2p/), then each character is mapped to a token id with the Kokoro vocab.
- **Voicepacks:** one per voice, raw little-endian Float32 of shape `[510, 256]` (522,240 bytes). Voices: Diễm Trinh, Mai Linh, Mai Loan, Mỹ Yến, Ngọc Huyền, Thục Trinh (female); Hưng Thịnh, Mạnh Dũng, Phát Tài, Thành Đạt, Tuấn Ngọc, Đức An, Đức Duy (male).

### Minimal Python example

```python
import numpy as np
import onnxruntime as ort
import soundfile as sf

sess = ort.InferenceSession("kokoro_vi_q8.onnx")  # or kokoro_vi.onnx for V1

ids = [0, *token_ids, 0]          # token_ids: from vig2p phonemes + Kokoro vocab
voicepack = np.fromfile("diem_trinh.f32", dtype="<f4").reshape(510, 256)
ref_s = voicepack[min(max(phoneme_len, 1), 510) - 1][None, :]

waveform, duration = sess.run(
    ["waveform", "duration"],
    {
        "input_ids": np.array([ids], dtype=np.int64),
        "ref_s": ref_s.astype(np.float32),
        "speed": np.array(1.0, dtype=np.float32),
    },
)
sf.write("out.wav", waveform.squeeze(), 24000)
```

## How the app uses V1 and V2

- `ModelVariant = 'full' | 'lite'` in `src/tts/models.ts`. `MODELS.vi` is V1 and `VI_LITE_MODEL` is V2.
- Every function in `src/tts/kokoro.ts` takes an optional `variant` (default `'full'`). `TtsDemo` takes a `variant` prop: `src/app/index.tsx` renders V1 and `src/app/lite.tsx` renders V2.
- Each variant has its own ONNX session, downloaded file and cached output (`kokoro-out` vs `kokoro-out-lite`). Each tab downloads its model once, then works offline.
- Sessions run on CPU with 4 threads and full graph optimization. They prefer the XNNPACK execution provider and fall back to plain CPU when it isn't available (for example on iOS).

## Publishing a new model build

1. Upload it as a **new** release tag. Never replace an existing asset, because installed apps download by URL.
2. Record its sha256 in the release notes and in the [Releases](#releases) table above.
3. In the app, point `VI_LITE_MODEL.url` / `filename` (or `MODELS.vi`) at the new asset.

## Credits

- Vietnamese model and voices: [contextboxai/Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese)
- Base architecture: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) / StyleTTS2
- Vietnamese G2P: [`vig2p`](https://pypi.org/project/vig2p/)
