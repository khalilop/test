# Other open-source lightweight OCR (non-Paddle, non-VLM) for on-device phone use

Notes compiled 2026-11-19. Scope: classic engines, deep-learning toolkits and tiny research models other than PaddleOCR / PP-OCR, VLM-based OCR, and closed platform SDKs (those appear only as comparison baselines).

How to read the sourcing:
- "GitHub (fetched)" = number read directly from the project's GitHub page on 2026-11-19 (release assets, blob pages, docs in repo). File sizes are as displayed by GitHub.
- "(search snippet)" = number came from a web-search result summary of that page; the page itself could NOT be opened in this session (arxiv.org, huggingface.co and docs.swmansion.com were blocked by the session's network policy, and the shared web-search budget ran out part-way through). Treat these as less verified.
- Anything older than 2025 is labelled as older.

## 1. Which non-Paddle open-source OCR engines/models are realistically usable on phones in 2026, and what are their exact sizes, parameter counts and published CPU/ARM latency numbers?

### Takeaway
Only three non-Paddle options have an actual phone integration path today: Tesseract 5 (1-4 MB per language with tessdata_fast; maintained Android wrapper; WASM build for mobile web), EasyOCR's CRAFT+CRNN models (about 94 MB float; packaged by Qualcomm AI Hub and by older React Native ExecuTorch releases, with 1.3-2.1 GB RAM use), and DIY ONNX pipelines (OnnxTR's smallest pair is 9.5 MB int8 / 23 MB fp32; OpenOCR mobile is 35.8 MB; chineseocr_lite is 4.7 MB). Published phone/ARM latency numbers are sparse, mostly vendor-run, and missing entirely for docTR/OnnxTR, OpenOCR and Tesseract4Android.

### Cited Findings

Size / parameter inventory (smallest practical configuration of each):

| Engine / model | Components | Params | Size on disk | License | Source |
|---|---|---|---|---|---|
| Tesseract 5.5.x + tessdata_fast | 1 LSTM model per language (8-bit integer) | not published | eng 3.92 MB, ara 1.37 MB, chi_sim 2.35 MB, rus 3.68 MB, hin 1.07 MB, jpn 2.36 MB, kor 1.6 MB | Apache-2.0 | [tessdata_fast eng](https://github.com/tesseract-ocr/tessdata_fast/blob/main/eng.traineddata), [ara](https://github.com/tesseract-ocr/tessdata_fast/blob/main/ara.traineddata), [chi_sim](https://github.com/tesseract-ocr/tessdata_fast/blob/main/chi_sim.traineddata), [rus](https://github.com/tesseract-ocr/tessdata_fast/blob/main/rus.traineddata), [hin](https://github.com/tesseract-ocr/tessdata_fast/blob/main/hin.traineddata), [jpn](https://github.com/tesseract-ocr/tessdata_fast/blob/main/jpn.traineddata), [kor](https://github.com/tesseract-ocr/tessdata_fast/blob/main/kor.traineddata) |
| chineseocr_lite | DBNet det 1.8 MB + CRNN rec 2.5 MB + AngleNet 378 KB | not published | "total model only 4.7M" (4.7 MB) | GPL-2.0 | [chineseocr_lite README](https://github.com/DayBreak-u/chineseocr_lite) |
| OnnxTR / docTR smallest pair, 8-bit | db_mobilenet_v3_large (det) + crnn_mobilenet_v3_small (rec) | 4.2 M + 2.1 M | 4.11 MB + 5.4 MB (static 8-bit ONNX) | Apache-2.0 | [OnnxTR v0.1.2 assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.1.2), [docTR detection table](https://github.com/mindee/doctr/blob/main/docs/source/models/detection.md), [docTR recognition table](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md) |
| OnnxTR / docTR smallest pair, fp32 | same | same | 15.3 MB + 7.95 MB (fp32 ONNX) | Apache-2.0 | [OnnxTR v0.0.1 assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.0.1) |
| OpenOCR "mobile" | RepViT-DBNet det + RepSVTR rec | not stated | openocr_det_model.onnx 11.8 MB + openocr_rec_model.onnx 24 MB (uploaded 2025-03-24); PyTorch weights 12.1 MB + 62.3 MB | Apache-2.0 | [OpenOCR release assets](https://github.com/Topdu/OpenOCR/releases/expanded_assets/develop0.0.1) |
| EasyOCR (English) | CRAFT det + CRNN rec (gen2) | 20.8 M + 3.84 M | 79.2 MB + 14.7 MB float (Qualcomm export); upstream zips: craft_mlt_25k.zip 73.7 MB, english_g2.zip 13.4 MB | Apache-2.0 | [Qualcomm EasyOCR card (search snippet)](https://huggingface.co/qualcomm/EasyOCR), [EasyOCR pre-v1.1.6 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/pre-v1.1.6), [EasyOCR v1.3 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/v1.3) |
| CnOCR / CnSTD own models | densenet_lite_136-gru rec; db_mobilenet_v3_small or db_shufflenet_v2_small det | not published | rec 12 MB (digits-only model 2.7 MB); det 7.9 MB / 12 MB | Apache-2.0 | [CnOCR README](https://github.com/breezedeus/CnOCR) |
| OpenCV Zoo CRNN (recognizer only) | CRNN EN | not published | 32.3 MB fp32; 15.6 MB int8 | Apache-2.0 | [EN fp32 blob](https://github.com/opencv/opencv_zoo/blob/main/models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx), [EN int8 blob](https://github.com/opencv/opencv_zoo/blob/main/models/text_recognition_crnn/text_recognition_CRNN_EN_2022oct_int8.onnx) |
| ddddocr (captcha OCR, not general OCR) | single recognizer | not published | common_old.onnx 13 MB (default), common.onnx 51.6 MB (beta) | MIT | [common_old.onnx](https://github.com/sml2h3/ddddocr/blob/master/ddddocr/common_old.onnx), [common.onnx](https://github.com/sml2h3/ddddocr/blob/master/ddddocr/common.onnx), [ddddocr README](https://github.com/sml2h3/ddddocr) |
| EAST detector (frozen TensorFlow graph commonly used with OpenCV; older design) | detector only | not verified | frozen_east_text_detection.pb 92.2 MB | not verified | [frozen_east_text_detection.pb mirror](https://github.com/oyyd/frozen_east_text_detection.pb/blob/master/frozen_east_text_detection.pb) |
| TrOCR-small (recognizer only) | DeiT-small encoder + MiniLM decoder | 61.4 M (printed checkpoint); Qualcomm stage1 export: encoder 23.0 M + decoder 38.3 M | encoder 87.8 MB + decoder 146 MB (float) | not verified | [microsoft/trocr-small-printed (search snippet)](https://huggingface.co/microsoft/trocr-small-printed), [Qualcomm AI Hub TrOCR (search snippet)](https://aihub.qualcomm.com/mobile/models/trocr) |

Published latency / memory numbers that involve phones or ARM boards:
- EasyOCR on Snapdragon (vendor-run): Qualcomm's card lists on-device benchmarks with the detector on the NPU; the w8a8 (8-bit) detector runs at roughly 5-14 ms on recent Snapdragon chips, with peak memory from a few MB to about 240 MB depending on chipset — [Qualcomm EasyOCR card (search snippet)](https://huggingface.co/qualcomm/EasyOCR)
- EasyOCR-derived CRAFT + CRNN in React Native ExecuTorch (older docs, v0.4-0.7; forward pass only, excludes pre/post-processing): total 652 ms on iPhone 17 Pro, 600 ms on iPhone 16 Pro, 2855 ms on iPhone SE 3, 1092 ms on Galaxy S24, 1034 ms on OnePlus 12; CRAFT alone about 220 ms on newer iPhones and 490-520 ms on the Android phones; CRNN_512 about 38-45 ms; first runs "may be up to 2x longer" — [RN ExecuTorch inference-time docs (search snippet)](https://docs.swmansion.com/react-native-executorch/docs/next/benchmarks/inference-time)
- Same stack, memory: about 1400 MB on Android and 1320 MB on iOS for detector + recognizer (latest listing); older docs list 2100 MB Android / 1782 MB iOS for the multi-recognizer setup — [RN ExecuTorch memory docs (search snippet)](https://docs.swmansion.com/react-native-executorch/docs/0.7.x/benchmarks/memory-usage)
- Same stack, model files as ExecuTorch .pte (XNNPACK): CRAFT_800 83.1 MB; CRNN recognizers 547 MB (512 width), 277 MB (256), 142 MB (128); an older doc lists a quantized CRAFT_800 at 19.8 MB — [RN ExecuTorch model-size docs (search snippet)](https://docs.swmansion.com/react-native-executorch/docs/next/benchmarks/model-size); these recognizer sizes conflict with the 14.7 MB ONNX recognizer in [Qualcomm's card (search snippet)](https://huggingface.co/qualcomm/EasyOCR) — different export format / static shapes, not explained by either source
- TrOCR-small on Snapdragon 8 Gen 3 (vendor-run, stage1 checkpoint): float ONNX decoder 1.486 ms per step, float QNN encoder 8.017 ms — [Qualcomm AI Hub TrOCR (search snippet)](https://aihub.qualcomm.com/mobile/models/trocr)
- ViTSTR-Small on Raspberry Pi 4 (older, 2021): 0.464 s per word image (TorchScript JIT), 0.369 s quantized; desktop CPU JIT 0.028 s (Threadripper 3970X) — [ViTSTR repo](https://github.com/roatienza/deep-text-recognition-benchmark)
- Tesseract on Raspberry Pi Zero 2W (hobbyist test, older): individual images took from about 1 s to about 26 s — [element14 roadtest blog (search snippet)](https://community.element14.com/products/roadtest/b/blog/posts/ml-capability-of-raspberry-pi-zero-2w-text-recognition-using-google-stesseract-ocr-engine)
- Tesseract vs EasyOCR on Raspberry Pi 4 (academic, older): a Tesseract-based pipeline had the same recognition rate with execution time lower by 89% on average than EasyOCR — [IASC paper (search snippet)](https://techscience.com/iasc/v36n3/51941/html)

Desktop-CPU latency (no ARM numbers exist for these):
- docTR/OnnxTR smallest pair: about 0.17 s/page on FUNSD and 0.12 s/page on CORD, full precision, Intel i7-14700K — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- docTR model tables (hardware not stated on the page): db_mobilenet_v3_large 0.5 s/it at 1024x1024 batch 1; crnn_mobilenet_v3_small 0.05 s per batch of 64 crops; viptr_tiny 0.08 s; crnn_vgg16_bn 0.6 s; parseq 1.5 s — [detection table](https://github.com/mindee/doctr/blob/main/docs/source/models/detection.md), [recognition table](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md)
- OpenOCR mobile: only "quite similar inference speed [to PP-OCRv4] on NVIDIA 1080Ti GPU"; no latency or FPS numbers — [OpenOCR docs](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)

Projects that are not realistic phone options in 2026:
- MMOCR: latest release v1.0.1 dated Jul 4, 2023; no release since — [MMOCR tags](https://github.com/open-mmlab/mmocr/tags)
- keras-ocr (CRAFT + Keras CRNN, TensorFlow 2): latest tag v0.9.1 dated May 19, 2022; README has no mobile/TFLite mention — [keras-ocr tags](https://github.com/faustomorales/keras-ocr/tags), [keras-ocr README](https://github.com/faustomorales/keras-ocr)
- Kraken: actively released (7.1.1 on Sep 4, 2026) but release notes mention no mobile or ARM target; CoreML is kept only as a legacy weights format — [Kraken tags](https://github.com/mittagessen/kraken/tags), [Kraken releases](https://github.com/mittagessen/kraken/releases)
- ddddocr: an offline captcha-recognition SDK for Python (Windows/Linux x64, Linux ARM64, macOS); no Android/iOS mention — [ddddocr README](https://github.com/sml2h3/ddddocr)
- ocrs (Rust, Latin-only, "early preview. Expect more errors than commercial OCR engines"): targets many platforms including WebAssembly, no mobile mention; v0.14.0 released 2026-11-08 — [ocrs README](https://github.com/robertknight/ocrs), [ocrs CHANGELOG](https://github.com/robertknight/ocrs/blob/main/CHANGELOG.md)

### Inferences
- Summed pipeline sizes: OnnxTR smallest pair = 9.5 MB int8 (4.11 + 5.4) or 23.3 MB fp32 (15.3 + 7.95), 6.3 M params; OpenOCR mobile ONNX = 35.8 MB (11.8 + 24); EasyOCR English = about 94 MB float (79.2 + 14.7), 24.6 M params.
- OpenOCR does not publish mobile parameter counts; dividing fp32 ONNX size by 4 bytes suggests roughly 3 M (det) and 6 M (rec) parameters. This is an estimate, not a published figure.
- Ranking by on-disk size among non-Paddle options: Tesseract tessdata_fast (1-4 MB per language) < chineseocr_lite (4.7 MB) < OnnxTR int8 pair (9.5 MB) < OnnxTR fp32 pair (23 MB) < OpenOCR mobile (36 MB) < EasyOCR (about 94 MB).
- For context only (Paddle family is another researcher's scope): CnOCR's README lists ONNX sizes of the PP-OCR models it bundles — PP-OCRv6 tiny rec 4.3 MB + det 1.7 MB, PP-OCRv5 mobile rec 16 MB + det 4.6 MB, PP-OCRv4 mobile rec 10 MB + det 4.5 MB ([CnOCR README](https://github.com/breezedeus/CnOCR)). So no non-Paddle deep-learning pipeline with broad script coverage is smaller than PP-OCRv6 tiny (about 6 MB); only Tesseract and the unmaintained chineseocr_lite are.
- EasyOCR's phone numbers (0.6-1.1 s forward pass on 2023-2025 flagships, 2.9 s on iPhone SE 3, more than 1.3 GB RAM) make it unsuitable for old or low-RAM phones.

### Gaps
- No published ARM/phone latency or RAM numbers were found for Tesseract4Android, OnnxTR/docTR, OpenOCR mobile, chineseocr_lite, CnOCR or ocrs.
- Tesseract native library size per ABI (the .so / AAR overhead on top of traineddata) was not found.
- Parameter counts for Tesseract LSTM models, OpenOCR RepSVTR / RepViT-DB, chineseocr_lite and ocrs models are not published on the pages read.
- The Qualcomm AI Hub per-device tables (including older chipsets) could not be opened; only the search summary was available.

## 2. Tesseract: latest 5.x release, tessdata sizes, speed/accuracy on phones vs deep-learning OCR, maintained Android/iOS wrappers and minimum OS/ABI, and suitability for very old phones

### Takeaway
Tesseract 5.5.4 (Oct 28, 2026) is current; with tessdata_fast it is the smallest proven OCR (1-4 MB per language, CPU-only, NEON kernels, Android API 21+ including ARMv7 through Tesseract4Android) and is reasonable for clean printed documents on old Android phones. It is weak on photos, scene text and handwriting, its recognition models date from 2017, the Android wrapper lags upstream security fixes, and there is no maintained iOS wrapper.

### Cited Findings

Releases:
- 5.5.4 — Oct 28, 2026, marked Latest: "fixes 17 security advisories and several smaller issues"; also avoids a quadratic paragraph restart scan and redundant result-iterator restarts, and bounds seam search to prevent timeouts — [Tesseract releases](https://github.com/tesseract-ocr/tesseract/releases), [Tesseract tags](https://github.com/tesseract-ocr/tesseract/tags)
- Earlier 5.5 line: 5.5.3 — Jul 24, 2026 (fixes two security advisories, an LSTM integer overflow); 5.5.2 — Dec 26, 2025; 5.5.1 — May 25, 2025; 5.5.0 — Nov 10, 2024; 5.4.1 — Jun 11, 2024 — [Tesseract tags](https://github.com/tesseract-ocr/tesseract/tags), [Tesseract releases](https://github.com/tesseract-ocr/tesseract/releases)
- None of the latest three release notes mention ARM/NEON, Android or speed gains beyond the fixes above — [Tesseract releases](https://github.com/tesseract-ocr/tesseract/releases)
- Wikipedia's infobox still showed 5.5.3 (24 July 2026) as latest — [Wikipedia](https://en.wikipedia.org/wiki/Tesseract_(software)); contradicted by [GitHub tags](https://github.com/tesseract-ocr/tesseract/tags) (5.5.4)
- Engine: LSTM line recognizer (since v4) plus the legacy character-pattern engine (`--oem 0`); "more than 100 languages" out of the box; Apache-2.0; major version 5 started with 5.0.0 on November 30, 2021 — [Tesseract README](https://github.com/tesseract-ocr/tesseract)

Model sets:
- tessdata_fast: "Integerized LSTM of a smaller network", fastest, least accurate, not retrainable, LSTM only; tessdata_best: float LSTM, slowest, most accurate, the only retrainable set; tessdata: legacy models + integerized tessdata_best LSTM, the only set supporting the legacy engine — [tessdoc Data-Files](https://github.com/tesseract-ocr/tessdoc/blob/main/Data-Files.md)
- Official guidance: most users should start with tessdata_fast (what Linux distributions ship); tessdata_best is for people willing to "trade a lot of speed for slightly better accuracy" — [tessdoc Data-Files](https://github.com/tesseract-ocr/tessdoc/blob/main/Data-Files.md)
- tessdata_fast README: models are "8-bit integer", "a speed/accuracy compromise"; "For some languages, this is still best, but for most not"; fine-tuning is not possible; the legacy engine is not supported — [tessdata_fast README](https://github.com/tesseract-ocr/tessdata_fast)
- The tessdata_best and tessdata_fast models were released in September 2017 (updated data files dated September 15, 2017) — [tessdoc Data-Files](https://github.com/tesseract-ocr/tessdoc/blob/main/Data-Files.md)

File sizes as displayed on GitHub (fetched 2026-11-19):

| File | tessdata_fast | tessdata (legacy + int LSTM) | tessdata_best |
|---|---|---|---|
| eng | [3.92 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/eng.traineddata) | [22.4 MB](https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata) | [14.7 MB](https://github.com/tesseract-ocr/tessdata_best/blob/main/eng.traineddata) |
| ara | [1.37 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/ara.traineddata) | [2.38 MB](https://github.com/tesseract-ocr/tessdata/blob/main/ara.traineddata) | [12 MB](https://github.com/tesseract-ocr/tessdata_best/blob/main/ara.traineddata) |
| chi_sim | [2.35 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/chi_sim.traineddata) | [42.3 MB](https://github.com/tesseract-ocr/tessdata/blob/main/chi_sim.traineddata) | [12.5 MB](https://github.com/tesseract-ocr/tessdata_best/blob/main/chi_sim.traineddata) |
| hin | [1.07 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/hin.traineddata) | not fetched | [11.3 MB](https://github.com/tesseract-ocr/tessdata_best/blob/main/hin.traineddata) |
| rus | [3.68 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/rus.traineddata) | not fetched | not fetched |
| jpn | [2.36 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/jpn.traineddata) | not fetched | not fetched |
| kor | [1.6 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/kor.traineddata) | not fetched | not fetched |
| osd (orientation/script detection) | [10.1 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/osd.traineddata) | not fetched | n/a |
| script/Latin (all Latin-script languages) | [85.2 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/script/Latin.traineddata) | n/a | not fetched |
| script/Devanagari | [17.1 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/script/Devanagari.traineddata) | n/a | not fetched |
| script/Arabic | [8.88 MB](https://github.com/tesseract-ocr/tessdata_fast/blob/main/script/Arabic.traineddata) | n/a | not fetched |

Speed:
- Official FAQ: "If you are running Tesseract 4, you can use the 'fast' integer models"; "Tesseract 4 also uses up to four CPU threads while processing a page"; "If your computer has only two CPU cores, then running four threads will slow down things significantly"; limit with `OMP_THREAD_LIMIT` — [tessdoc FAQ](https://github.com/tesseract-ocr/tessdoc/blob/main/FAQ.md)
- SIMD matters (older, Tesseract 4.0-era, one Hindi page on an HP Z420 desktop; units not stated on the page): LSTM without OpenMP took 2.7 with AVX, 3.1 with SSE and 4.6 with no SIMD; legacy "Base Tess" took 2.9 — [tessdoc 4.0 Accuracy and Performance](https://github.com/tesseract-ocr/tessdoc/blob/main/tess4/4.0-Accuracy-and-Performance.md)
- The current source tree has ARM NEON kernels (dotproductneon.cpp, functionsneon.cpp, intsimdmatrixneon.cpp) next to the AVX/SSE ones, plus a RISC-V vector file — [tesseract src/arch](https://github.com/tesseract-ocr/tesseract/tree/main/src/arch)
- Desktop CPU, older versions (PyTesseract v0.3.10, Intel i7-14700K): about 0.50 s/page on FUNSD and 0.52 s/page on CORD, versus 0.17 / 0.12 s for docTR's smallest pair, 1.96 / 1.75 s for EasyOCR and 1.27 / 0.38 s for PaddleOCR v2.7.3 — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- One 800x600 invoice on an Apple M-series CPU (Codesota, March 2026): Tesseract 0.162 s, RapidOCR 0.212 s, EasyOCR 0.656 s, PaddleOCR 4.85 s — [Codesota (search snippet)](https://codesota.com/ocr/best-for-python)
- Informal developer figures: Tesseract about 0.1-0.3 s per image versus 1-5 s for the deep-learning libraries; install footprint about 36 MB versus roughly 800 MB for EasyOCR/PaddleOCR Python environments — [dev.to post (search snippet)](https://dev.to/felipe_anyformat/best-python-ocr-libraries-179)
- Phone-specific, methodology unverified: a 500-image test reported Tesseract 5 at about 220 ms average latency versus ML Kit at about 140 ms, with ML Kit adding about 18 MB — [Designveloper (search snippet)](https://www.designveloper.com/?p=47760)
- Low-end ARM proxies: 1 s to 26 s per image on a Raspberry Pi Zero 2W — [element14 (search snippet)](https://community.element14.com/products/roadtest/b/blog/posts/ml-capability-of-raspberry-pi-zero-2w-text-recognition-using-google-stesseract-ocr-engine); on a Raspberry Pi 4 a Tesseract pipeline matched EasyOCR's recognition rate with 89% lower execution time — [IASC paper (search snippet)](https://techscience.com/iasc/v36n3/51941/html)

Accuracy (documents vs photos vs handwriting):
- Input assumptions: "Tesseract works best on images which have a DPI of at least 300 dpi"; line segmentation quality "reduces significantly if a page is too skewed"; internal binarisation "can be suboptimal, particularly if the page background is of uneven darkness"; "By default Tesseract expects a page of text when it segments an image" — [tessdoc ImproveQuality](https://github.com/tesseract-ocr/tessdoc/blob/main/ImproveQuality.md)
- Handwriting: "You can, but it won't work very well, as Tesseract is designed for printed text" — [tessdoc FAQ](https://github.com/tesseract-ocr/tessdoc/blob/main/FAQ.md)
- Clean printed invoice (24 text items, single image, March 2026): PaddleOCR 100%, Tesseract 87.5%, RapidOCR 75%, EasyOCR 62.5% — [Codesota (search snippet)](https://codesota.com/ocr/best-for-python)
- Receipts (SROIE 2019, vendor-run, August 2026), character error rate: docTR 19.7%, PaddleOCR 20.5%, EasyOCR 28.3%, Tesseract 33.6%, Docling 58.4% — [ImageToTable.ai (search snippet)](https://imagetotable.ai/references/receipt-ocr-benchmark-results)
- Scene text on mobile (848 real word images from urban scenes): Apple Vision best, then ML Kit, then Tesseract with optimized preprocessing — [SIBGRAPI paper (search snippet)](https://sol.sbc.org.br/index.php/sibgrapi_estendido/article/view/38292)
- iOS, older (about 2019), 959 images: ML Kit "outperforms Tesseract OCR by a large margin, though it consumes more time on the CPU" — [Heartbeat / Fritz AI (search snippet)](https://heartbeat.comet.ml/comparing-on-device-text-recognition-ocr-sdks-24f8a0423461)
- Handwritten prescriptions (5,578 images, community benchmark, 2026): Tesseract 2.5% exact match at 49 ms/img, EasyOCR 2.6% at 14 ms/img, PP-OCRv5 21.4% at 103 ms/img, GLM-OCR 32.6% at 141 ms/img — [dev.to benchmark (search snippet)](https://dev.to/kaniel_outis/how-well-can-ocr-read-doctor-handwriting-in-2026-54hn)
- Practitioner view from an app that needed open-source Android OCR: "most open-source implementations were aging Tesseract ports with lackluster accuracy"; they ported PaddleOCR v5 to ONNX Runtime instead — [Ente blog (search snippet)](https://ente.com/blog/ocr/)
- Qualitative: even with the LSTM engine, Tesseract "still lagged" on occluded, blurry or curved text — [W&B report (search snippet)](https://wandb.ai/byyoung3/test_api/reports/Optical-Character-Recognition-Then-and-Now--VmlldzoxNDEzMzM4NQ)

Android wrappers:
- Tesseract4Android (adaptech-cz) 4.9.0 is the latest release: Tesseract 5.5.1, Leptonica 1.85.0, libjpeg v9f, libpng 1.6.48; requires "Android 5.0 (API 21) or higher"; Apache-2.0; distributed through JitPack as `cz.adaptech.tesseract4android:tesseract4android:4.9.0` (standard) or `tesseract4android-openmp:4.9.0` — [Tesseract4Android README](https://github.com/adaptech-cz/Tesseract4Android)
- Variants: Standard is "Single-threaded. Best for single-core processors or when using multiple Tesseract instances in parallel"; OpenMP is "Multi-threaded. Provides better performance on multi-core processors when using only single instance of Tesseract"; "TessBaseAPI is not thread-safe" — [Tesseract4Android README](https://github.com/adaptech-cz/Tesseract4Android)
- 4.8.0 (Dec 1, 2024) raised minSdk to 21 (a breaking change), moved to NDK r27c and added 16 KB page-size support for Android 15; 4.4.0 (Apr 3, 2023) had fixed a crash on Android 4.4, which implies releases before 4.8.0 still ran on pre-5.0 Android — [Tesseract4Android releases](https://github.com/adaptech-cz/Tesseract4Android/releases), [Tesseract4Android tags](https://github.com/adaptech-cz/Tesseract4Android/tags)
- Date conflict for 4.9.0: the releases page shows "11 Jun" with no year — [releases](https://github.com/adaptech-cz/Tesseract4Android/releases); the tags page shows Feb 5, 2026 — [tags](https://github.com/adaptech-cz/Tesseract4Android/tags). Not resolved.
- Build config: minSdk 21, compileSdk/targetSdk 35, ndkVersion 27.2.12479018, CMake flag `-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`; the `abiFilters` line is commented out (its commented list is 'x86', 'x86_64', 'armeabi-v7a', 'arm64-v8a'), i.e. the build does not restrict ABIs — [build.gradle](https://github.com/adaptech-cz/Tesseract4Android/blob/master/tesseract4android/build.gradle)
- A shipping app built on it (F-Droid "OCR (Tesseract)") lists ABIs arm64-v8a, armeabi-v7a, x86, x86_64 for its 2.5.0 build (March 2025), on Tesseract4Android 4.8.0 / Tesseract 5.5.0 — [IzzyOnDroid listing (search snippet)](https://apt.izzysoft.de/fdroid/index/apk/org.totschnig.ocr.tesseract?repo=main), [F-Droid listing (search snippet)](https://f-droid.org/en/packages/org.totschnig.ocr.tesseract/)
- The README tells users to use the v4.0.0 trained data files in a `tessdata` subdirectory, and suggests downloading them rather than bundling in the APK — [Tesseract4Android README](https://github.com/adaptech-cz/Tesseract4Android)
- React Native: react-native-tesseract-ocr wraps tess-two on Android only; iOS "has NOT been implemented yet"; MIT — [react-native-tesseract-ocr](https://github.com/jonathanpalma/react-native-tesseract-ocr)

iOS wrappers:
- SwiftyTesseract was "archived by the owner on Apr 8, 2022"; the project "is no longer maintained and will see no further updates"; the maintainer recommends Apple's Vision framework; it supported iOS 11+ through Swift Package Manager only — [SwiftyTesseract](https://github.com/SwiftyTesseract/SwiftyTesseract)
- Its last version is 4.0.1, dated 2021-03-07 — [Swift Package Registry (search snippet)](https://swiftpackageregistry.com/SwiftyTesseract/SwiftyTesseract)
- Flutter: flutter_tesseract_ocr's latest version is 0.4.31 and its iOS side uses SwiftyTesseract 4.0.1 through CocoaPods — [pub.dev versions (search snippet)](https://pub.dev/packages/flutter_tesseract_ocr/versions), [pub.dev docs (search snippet)](https://pub.dev/documentation/flutter_tesseract_ocr/latest/)
- Another Flutter plugin (tesseract_ocr 0.5.1) uses Apple Vision as its iOS engine and Tesseract only as the Android default — [pub.dev tesseract_ocr (search snippet)](https://pub.dev/documentation/tesseract_ocr/0.5.1/)

Mobile web (Tesseract.js):
- v7.0.0 — Dec 15, 2025: runtimes about 15-35% lower than v6 "depending on device and use-case" (largest gains on the latest Intel processors), new `relaxedsimd` WASM build, Node.js 14 dropped — [Tesseract.js releases](https://github.com/naptha/tesseract.js/releases), [Tesseract.js tags](https://github.com/naptha/tesseract.js/tags)
- v6.0.0 — Jan 7, 2025: fixed memory leaks where memory "would rise over time, eventually leading to a crash"; lower runtime and memory from new default output formats — [Tesseract.js releases](https://github.com/naptha/tesseract.js/releases)
- v5.0.0 — September 2023 (older): language files 54% smaller for English and 73% smaller for Chinese, about 50% lower first-run runtime, worker memory 311 MB to 164 MB in the web benchmark, "Compatible with iOS 17 (using default settings)" — [Tesseract.js releases](https://github.com/naptha/tesseract.js/releases)
- Limits: "does not support PDF files and does not modify the Tesseract recognition model to improve accuracy"; Apache-2.0 — [Tesseract.js README](https://github.com/naptha/tesseract.js)

### Inferences
- Tesseract with tessdata_fast is a sensible choice for very old Android phones when the input is clean, upright printed text in a known language: models are 1-4 MB, inference is CPU-only with NEON integer kernels, the standard (single-thread) build suits weak CPUs, and the wrapper supports API 21 and 32-bit ARM. Expect seconds per page on very old hardware (Raspberry Pi Zero 2W proxy: 1-26 s).
- It is not a sensible choice for camera photos of signs/receipts in poor light, curved or rotated text, or handwriting; every comparison found ranks it below deep-learning OCR there, and it needs app-side preprocessing (deskew, binarise, crop) to do acceptably.
- On dual-core or older quad-core phones the OpenMP build can be slower than the standard build, following the FAQ's warning about four threads on two cores.
- Security lag: Tesseract4Android 4.9.0 bundles Tesseract 5.5.1 (May 2025), so apps using it do not get the two advisories fixed in 5.5.3 or the 17 fixed in 5.5.4 unless they rebuild the native library themselves. This matters when OCRing untrusted images.
- iOS has no maintained open-source Tesseract wrapper; cross-platform plugins still depend on the archived SwiftyTesseract (2021 binary) or fall back to Apple Vision.
- Multi-language installs add up: the osd model alone is 10.1 MB and the all-Latin script model is 85.2 MB, so per-language files are the small option.
- The official model sets date from September 2017 and no newer retrained set is mentioned on the pages read; the 5.5.x releases are maintenance and security work, so accuracy should not be expected to have improved since then.

### Gaps
- No rigorous, recent phone benchmark of Tesseract 5 (ms per page on named Android devices, tessdata_fast vs best) was found; the only phone figure (220 ms vs ML Kit 140 ms) has unverified methodology.
- No fast-vs-best accuracy numbers per language were found on the pages read; the official docs only say "slightly" less accurate.
- Tesseract.js core WASM size, default traineddata variant and mobile-browser timings are not stated on the pages read.
- The per-engine accuracy and timing numbers of the SIBGRAPI mobile study were not retrievable (only the ranking).
- Maintenance status of gali8/Tesseract-OCR-iOS and of the Akylas fork / Maven Central "tesseract-android" artifact that surfaced in search was not checked.

## 3. OpenOCR (SVTRv2, RepSVTR, "OpenOCR mobile"): mobile-tier models, sizes, accuracy versus PP-OCR mobile, ONNX export and mobile deployment status

### Takeaway
OpenOCR ships one mobile tier (RepViT-DBNet detector + RepSVTR recognizer, Chinese + English) as 11.8 MB + 24 MB ONNX files that run with onnxruntime on CPU and score 67.22% versus PP-OCRv4's 62.77% on the project's own end-to-end leaderboard metric. There is no Android/iOS demo, no quantized model, no published CPU or ARM latency, and no published comparison against PP-OCRv5/v6 mobile.

### Cited Findings
- OpenOCR "provides server model and mobile model", "supports Chinese and English text detection and recognition", and claims it "outperforms PP-OCRv4 baseline by 4.5% on the OCR competition leaderboard in terms of accuracy, while preserving quite similar inference speed"; Apache-2.0 — [OpenOCR README](https://github.com/Topdu/OpenOCR)
- End-to-end metric table: PP-OCRv4 62.77%; SVTRv2 server recognizer 68.81%; RepSVTR mobile (rec `configs/rec/svtrv2/repsvtr_ch.yml`, det `configs/det/dbnet/repvit_db.yml`) 67.22%; evaluated on the Baidu AI Studio "OCR competition leaderboard" (competition 1131); speed only described as "quite similar inference speed on NVIDIA 1080Ti GPU" — [OpenOCR docs](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)
- Release assets (tag develop0.0.1): openocr_det_model.onnx 11.8 MB and openocr_rec_model.onnx 24 MB (both mobile, uploaded 2025-03-24); openocr_det_repvit_ch.pth 12.1 MB, openocr_repsvtr_ch.pth 62.3 MB, openocr_svtrv2_ch.pth 118 MB (uploaded 2024-11-23) — [OpenOCR release assets](https://github.com/Topdu/OpenOCR/releases/expanded_assets/develop0.0.1)
- A third-party NPU adaptation lists the same files at 23.97 MB (rec ONNX) and 11.85 MB (det ONNX) — [ai.gitcode mirror (search snippet)](https://ai.gitcode.com/gcw_AVRCax4T/OpenOCR)
- ONNX path: `OpenOCR(mode='mobile', backend='onnx')`; "When using ONNX for inference, there is no need to install Torch"; export with `tools/toonnx.py`; end-to-end run with `tools/infer_e2e.py --backend=onnx --device=cpu`; the page mentions no Android, iOS, NCNN, MNN or TFLite — [OpenOCR docs](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)
- Timeline: ONNX export released 2025.03.23; fine-tuning for det and rec released 2025.03.24; SVTRv2 accepted to ICCV 2025 (2025.07.10); openocr-python 0.1.3 (2026.02.06) and 0.1.5 (2026.02.12) — [OpenOCR README](https://github.com/Topdu/OpenOCR)
- SVTRv2 recognizer family (English, trained on Union14M-L-Filter; latency on one NVIDIA 1080Ti in PyTorch dynamic-graph mode): SVTRv2-T 5.13 M params, 5.0 ms, 94.78 common-benchmark average, 79.05 Union14M-Benchmark average; SVTRv2-S 11.25 M, 5.3 ms, 96.13, 83.70; SVTRv2-B 19.76 M, 7.0 ms, 96.57, 86.14 — [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md), [SVTRv2 config readme](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)
- SVTRv2-T per-dataset accuracy: IC13 98.6, SVT 96.6, IIIT5k 98.0, IC15 88.4, SVTP 90.5, CUTE80 96.5; long-text benchmark 47.83; occluded scene text 71.4 — [SVTRv2 config readme](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)
- Chinese benchmark (scene / web / document / handwriting / average): SVTRv2-T 77.8 / 78.8 / 99.3 / 62.0 / 79.45; SVTRv2-S 81.1 / 81.2 / 99.3 / 65.0 / 81.64; SVTRv2-B 83.5 / 83.3 / 99.5 / 67.0 / 83.31 — [SVTRv2 config readme](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)
- The README also links a PaddleOCR implementation of SVTRv2, so the recognizer is available inside the Paddle toolkit too — [OpenOCR README](https://github.com/Topdu/OpenOCR)
- Later OpenOCR models are not mobile-tier: UniRec-0.1B (2025.08.20, 0.1 B parameters, Chinese/English text and formulas), OpenDoc-0.1B (2025.12.25, 90.57% on OmniDocBench v1.5 with PP-DocLayoutV2 layout), ScriptMoE (2026.09.21, 45.85 M parameters, see question 6) — [OpenOCR README](https://github.com/Topdu/OpenOCR), [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)

### Inferences
- The "4.5% improvement" is 4.45 percentage points on OpenOCR's own leaderboard metric (67.22 vs 62.77); the metric and test set are not defined on the page, so it is not comparable with other benchmarks.
- At 35.8 MB the OpenOCR mobile ONNX pair is about 2.5x the size of PP-OCRv4 mobile (about 14.5 MB per CnOCR's listing) and about 6x PP-OCRv6 tiny (about 6 MB); it buys accuracy over PP-OCRv4, but the comparison baseline is two Paddle generations old.
- Mobile deployment is do-it-yourself: load the two ONNX files in ONNX Runtime Mobile (or convert them) and re-implement DB post-processing, cropping and CTC decoding natively. No evidence was found of anyone shipping it in a phone app.
- The RepSVTR .pth (62.3 MB) is much larger than its ONNX export (24 MB), which suggests the export is re-parameterized/stripped; the docs do not explain the difference.

### Gaps
- No parameter counts for RepSVTR or the RepViT-DB detector, no CPU/ARM latency, no quantized (int8) export, and no accuracy on public benchmarks for the mobile models.
- No comparison of OpenOCR mobile with PP-OCRv5 mobile or PP-OCRv6 tiny/small was found.
- The SVTRv2 paper (arXiv 2411.15858) could not be opened directly; numbers above come from the repository docs.

## 4. docTR / OnnxTR, EasyOCR, MMOCR: smallest detection + recognition combinations, sizes, quantized variants, speed, and whether anyone runs them on phones

### Takeaway
docTR/OnnxTR has the best small non-Paddle pipeline — db_mobilenet_v3_large + crnn_mobilenet_v3_small, 6.3 M parameters, 9.5 MB as 8-bit ONNX, close to the big models' accuracy on FUNSD/CORD — but it is Latin-only by default and has no mobile packaging. EasyOCR is the only one with real phone deployments, and those are heavy (about 94 MB, 1.3-2.1 GB RAM, 0.6-2.9 s per image); MMOCR has been frozen since July 2023.

### Cited Findings

docTR / OnnxTR:
- Versions: docTR v1.1.0 (Aug 2026), v1.0.1 (Feb 4, 2026), v1.0.0 (Jul 9, 2025), v0.12.0 (Jun 20, 2025) — [docTR tags](https://github.com/mindee/doctr/tags); OnnxTR v0.9.0 (Aug 24, 2026, "port of docTR v1.1.0") is the latest GitHub release — [OnnxTR tags](https://github.com/felixdittrich92/OnnxTR/tags), [OnnxTR releases](https://github.com/felixdittrich92/OnnxTR/releases); the README's PyPI badge shows 0.10.0 — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR) (conflict not resolved)
- v1.0.0 removed TensorFlow as a backend (PyTorch only); v0.12.0 added the lightweight `viptr_tiny` recognizer and was the last release with TensorFlow, pointing users to PyTorch or OnnxTR; v1.1.0 added layout (LW-DETR), table structure (TableCenterNet, 7.1 M params) and a CLI, and requires Python 3.11+ — [docTR releases](https://github.com/mindee/doctr/releases)
- Detection zoo (input 1024x1024; FUNSD recall/precision; CORD recall/precision; sec/it at batch 1, hardware not stated): db_mobilenet_v3_large 4.2 M params, 82.69/84.63, 94.51/70.28, 0.5 s; fast_tiny 13.5 M (8.5 M after re-parameterization), 84.90/85.04, 93.73/76.26, 0.7 s (0.4 s); fast_base (default) 16.3 M (10.6 M), 84.95/86.73, 94.39/85.36, 0.8 s (0.5 s); db_resnet50 25.4 M, 83.56/86.68, 92.61/86.39, 1.1 s — [docTR detection table](https://github.com/mindee/doctr/blob/main/docs/source/models/detection.md)
- Recognition zoo (input 32x128; exact-match word accuracy FUNSD / CORD; sec per batch of 64, hardware not stated): crnn_mobilenet_v3_small 2.1 M params, 87.25 / 93.91, 0.05 s; crnn_mobilenet_v3_large 4.5 M, 87.38 / 94.46, 0.08 s; viptr_tiny 3.2 M, 86.03 / 93.08, 0.08 s; crnn_vgg16_bn (default) 15.8 M, 88.21 / 95.47, 0.6 s; parseq 23.8 M, 88.53 / 95.56, 1.5 s; vitstr_small 21.4 M, 88.00 / 95.40, 1.4 s; master 58.7 M, 88.57 / 95.73, 2.7 s — [docTR recognition table](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md)
- End-to-end with db_resnet50 detection (FUNSD recall/precision, CORD recall/precision): + crnn_mobilenet_v3_small 73.06/75.79, 84.64/78.94; + crnn_vgg16_bn 73.37/76.11, 84.80/79.09; + parseq 73.52/76.27, 85.91/80.13; reference services: Google Vision document text detection 64.00/53.30, 68.90/61.10; AWS Textract 78.10/83.00, 87.50/66.00; Azure Form Recognizer v3.2 79.42/85.89, 89.62/88.93 — [docTR end-to-end table](https://github.com/mindee/doctr/blob/main/docs/source/models/end_to_end.md)
- ONNX file sizes, fp32: db_mobilenet_v3_large 15.3 MB; crnn_mobilenet_v3_small 7.95 MB; crnn_mobilenet_v3_large 17.2 MB; rep_fast_tiny 32.5 MB; rep_fast_base 40.4 MB; crnn_vgg16_bn 60.3 MB; vitstr_small 81.9 MB; parseq 92.1 MB; db_resnet50 96.2 MB; crop/page orientation classifiers 5.81 MB each — [OnnxTR v0.0.1 assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.0.1)
- ONNX file sizes, 8-bit: db_mobilenet_v3_large 4.11 MB; crnn_mobilenet_v3_small 5.4 MB; crnn_mobilenet_v3_large 8.89 MB; crnn_vgg16_bn 18.1 MB; vitstr_small (dynamic) 21.1 MB; parseq (dynamic) 32.9 MB; db_resnet50 24.3 MB; orientation classifiers 1.64 MB each — [OnnxTR v0.1.2 assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.1.2)
- 8-bit models are loaded with `load_in_8_bit=True` ("8-Bit quantized models for faster inference on CPU"); they "are not available for FAST, MASTER, VIPTR, LW-DETR, TableCenterNet and FormulaNet models" — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- OnnxTR has "no PyTorch or TensorFlow dependencies"; execution providers are CPU, CUDA, OpenVINO and CoreML (Apple Silicon Macs); the README does not mention mobile, Android, iOS, ARM or Raspberry Pi — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- CPU benchmark (Intel i7-14700K; older library versions, seconds per page FUNSD / CORD): docTR v0.8.1 1.29 / 0.60; OnnxTR v0.6.0 fp32 0.57 / 0.25; OnnxTR 8-bit 0.38 / 0.14; OnnxTR OpenVINO 0.15 / 0.14 (these with fast_base or db_resnet50 + crnn_vgg16_bn); the smallest pair in fp32 0.17 / 0.12 — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- docTR's own guidance: "For the smallest footprint on CPU, export models to ONNX and run them with OnnxTR"; its deployment docs do not mention mobile, edge, TFLite, Core ML, ARM or browsers — [docTR performance docs](https://github.com/mindee/doctr/blob/main/docs/source/deployment/performance.md), [docTR ONNX docs](https://github.com/mindee/doctr/blob/main/docs/source/deployment/onnx.md)
- Language scope: pretrained recognition models are trained on the `french` vocabulary (Latin letters with Western European accents, digits, punctuation, currency symbols); for other languages the docs suggest a whitelist, a community model such as multilingual PARSeq, or training on one of the 100+ vocabularies — [docTR recognition docs](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md), [docTR vocabs docs](https://github.com/mindee/doctr/blob/main/docs/source/models/vocabs.md)

EasyOCR:
- Latest release is v1.7.2 (24 September 2024, "Fix several compatibilities"); before it v1.7.1 (4 September 2023) and v1.7.0 (25 May 2023) — [EasyOCR releases](https://github.com/JaidedAI/EasyOCR/releases)
- Architecture: CRAFT detector by default, optional DBNet (`dbnet18`), CRNN recognizer (ResNet/VGG features + LSTM + CTC); "80+ supported languages"; handwriting is listed under "What's coming next"; Apache-2.0 — [EasyOCR README](https://github.com/JaidedAI/EasyOCR)
- Download sizes: craft_mlt_25k.zip 73.7 MB; first-generation recognizers 191-203 MB each as zips (arabic 191, devanagari 191, thai 191, latin 191, cyrillic 191, korean 193, japanese 196, chinese 200, chinese_sim 203) — [pre-v1.1.6 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/pre-v1.1.6)
- Second-generation recognizers: english_g2.zip 13.4 MB, latin_g2.zip 13.6 MB, korean_g2.zip 14.2 MB, japanese_g2.zip 15.3 MB, zh_sim_g2.zip 19.3 MB, cyrillic_g2.zip 13.5 MB — [v1.3 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/v1.3), [v1.6.1 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/v1.6.1)
- DBNet alternatives: pretrained_ic15_res18.zip 49.6 MB, pretrained_ic15_res50.zip 103 MB — [v1.6.0 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/v1.6.0)
- Qualcomm export: detector 20.8 M params / 79.2 MB float; recognizer 3.84 M params / 14.7 MB; w8a8 detector about 5-14 ms on recent Snapdragon NPUs — [Qualcomm EasyOCR card (search snippet)](https://huggingface.co/qualcomm/EasyOCR)
- A community ONNX port lists CRAFT at 79.3 MB and English/Latin recognizers at 14.4-14.7 MB while claiming a total reduction from 108.4 MB to 17.0 MB after JPQD quantization; the two statements do not agree — [asmud/EasyOCR-onnx (search snippet)](https://hface.pages.dev/asmud/EasyOCR-onnx/blame/main/README.md)
- Phone deployment evidence: React Native ExecuTorch shipped `useOCR` with CRAFT + CRNN (numbers in question 1) — [RN ExecuTorch useOCR docs (search snippet)](https://docs.swmansion.com/react-native-executorch/docs/computer-vision/useOCR)
- The same project's current benchmark page (v0.11.0-dev, "main shortly after v0.10.0") lists only PP-OCRv6 for OCR ("a single fused model with two execution methods", int8 DBNet detector + fp32 SVTR recognizer) and no CRAFT/CRNN; its test devices include Galaxy S20+ and iPhone SE (3rd gen) — [RN ExecuTorch benchmarks source](https://github.com/software-mansion/react-native-executorch/blob/main/docs/docs/04-benchmarks/01-benchmarks.mdx)
- Desktop CPU speed (older versions): EasyOCR v1.7.1 about 1.96 s/page on FUNSD and 1.75 s/page on CORD, the slowest detector+recognizer pipeline in that table after Surya — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)

MMOCR and keras-ocr:
- MMOCR's latest release is v1.0.1 (Jul 4, 2023), preceded by v1.0.0 (Apr 6, 2023) — [MMOCR tags](https://github.com/open-mmlab/mmocr/tags)
- keras-ocr's latest tag is v0.9.1 (May 19, 2022); on the first 1,000 COCO-Text validation images it reported 417 ms latency (Tesla P4 GPU, scale=2) with precision 0.53 / recall 0.54, versus Google Cloud Vision 388 ms, 0.53 / 0.58 and AWS Rekognition 719 ms, 0.45 / 0.48; the author calls the numbers "a very rough first draft"; MIT — [keras-ocr tags](https://github.com/faustomorales/keras-ocr/tags), [keras-ocr README](https://github.com/faustomorales/keras-ocr)

### Inferences
- The smallest docTR pair gives up very little accuracy: the MobileNetV3-small recognizer is 1.0 point (FUNSD) and 1.6 points (CORD) below the default VGG CRNN at one-twelfth the per-batch time, and the MobileNetV3 detector is close to db_resnet50 on recall but loses 16 points of precision on CORD.
- With TensorFlow removed in July 2025, docTR's old TFLite / TensorFlow.js route is gone; ONNX Runtime Mobile is the only practical path to phones, and all pre/post-processing would have to be ported. No evidence was found of anyone running docTR or OnnxTR on Android or iOS.
- EasyOCR is not competitive for old phones: stale since September 2024, the detector alone is about 79 MB / 20.8 M parameters, and the React Native ExecuTorch project appears to have replaced it with PP-OCRv6 in its current benchmarks.
- EasyOCR's small footprint applies only to scripts with a second-generation recognizer (see question 8); Arabic, Devanagari, Thai, Bengali and Tamil still use the first-generation downloads of about 191 MB each.
- MMOCR and keras-ocr should be treated as unmaintained research/legacy code, not mobile options.

### Gaps
- No phone/ARM latency for the docTR MobileNetV3 pair; no int8 accuracy numbers (the README only warns 8-bit "can in general lead to poorer accuracy").
- docTR latency-table hardware is not stated on the current docs pages.
- viptr_tiny ONNX file size and MMOCR model-zoo sizes were not found.
- The exact contents of React Native ExecuTorch's current OCR benchmark numbers (stored in JSON) could not be read.

## 5. Smallest text recognizers in the literature / GitHub / Hugging Face (under about 5 MB or 5 M parameters), their benchmark accuracy, and ultra-light detectors

### Takeaway
The smallest credible recognizers are CRNN-MobileNetV3-small (2.1 M params, 5.4-8 MB), VIPTR-tiny (3.2 M), EasyOCR's gen2 CRNN (3.84 M), GlanceOCR-tiny (4.35 M, unbenchmarked), SVTRv2-T (5.13 M; 94.8% on the six common benchmarks and 79.1% on Union14M when trained on real data), ViTSTR-Tiny (5.4 M) and SVTR-T (6.03 M). Among these SVTRv2-T has by far the best published accuracy for its size; the lightest detectors are DBNet variants with MobileNetV3/ShuffleNet backbones (1.8-15 MB), with FAST-T (8.5 M params) the fastest published design.

### Cited Findings

Recognizers (sorted by parameter count):

| Model | Params | File size | Accuracy | Speed | Source |
|---|---|---|---|---|---|
| STRIDE (Samsung, 2021; older) | about 0.88 M per network | not found | not captured | on-device CNN-LSTM-CTC | [arXiv 2105.07795 (search snippet)](https://arxiv.org/abs/2105.07795v1) |
| crnn_mobilenet_v3_small (docTR) | 2.1 M | 7.95 MB fp32 ONNX, 5.4 MB 8-bit | FUNSD 87.25, CORD 93.91 (exact-match words) | 0.05 s per 64 crops (CPU, hardware not stated) | [docTR recognition table](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md), [OnnxTR assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.1.2) |
| chineseocr_lite CRNN | not published | 2.5 MB | not published | not published | [chineseocr_lite README](https://github.com/DayBreak-u/chineseocr_lite) |
| CnOCR number-densenet_lite_136-fc (digits only) | not published | 2.7 MB | not published | not published | [CnOCR README](https://github.com/breezedeus/CnOCR) |
| viptr_tiny (docTR, added June 2025) | 3.2 M | not found | FUNSD 86.03, CORD 93.08 | 0.08 s per 64 crops | [docTR recognition table](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md) |
| EasyOCR gen2 CRNN | 3.84 M | 14.7 MB | not published | not published | [Qualcomm EasyOCR card (search snippet)](https://huggingface.co/qualcomm/EasyOCR) |
| GlanceOCR v2.2 tiny (2026) | 4.35 M | not stated (fp32 + quantized ONNX provided) | none published | none published | [GlanceOCR HF card (search snippet)](https://huggingface.co/afirelily/GlanceOCR-v2_2-tiny), [GlanceOCR repo](https://github.com/afirelily/GlanceOCR) |
| crnn_mobilenet_v3_large (docTR) | 4.5 M | 17.2 MB fp32, 8.89 MB 8-bit | FUNSD 87.38, CORD 94.46 | 0.08 s per 64 crops | [docTR recognition table](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md), [OnnxTR assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.1.2) |
| SVTRv2-T (ICCV 2025) | 5.13 M | not published | common-6 average 94.78 (IIIT5k 98.0, SVT 96.6, IC13 98.6, IC15 88.4, SVTP 90.5, CUTE80 96.5); Union14M-Benchmark 79.05 | 5.0 ms (1080Ti) | [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md), [SVTRv2 readme](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md) |
| ViTSTR-Tiny (2021; older) | 5.4 M | not found | 80.3% (82.1% with augmentation) average under the older MJ+ST protocol; TRBA baseline 84.3% | 9.3 ms/image (GPU), 1.3 GFLOPs | [arXiv 2105.08582 (search snippet)](https://arxiv.org/pdf/2105.08582v1.pdf), [ViTSTR repo](https://github.com/roatienza/deep-text-recognition-benchmark) |
| PARSeq-Ti (2022; older) | not captured (d_model 192, DeiT-Ti-like; "more similar to CRNN in terms of parameter count and FLOPS") | parseq_tiny checkpoint 23 MB | not captured | not captured | [PARSeq ECCV supplement (search snippet)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880177-supp.pdf), [PARSeq release assets](https://github.com/baudm/parseq/releases/expanded_assets/v1.0.0) |
| SVTR-T (2022; older) | 6.03 M | not found | not captured | 4.5 ms per text image (1080Ti) | [arXiv 2205.00159 (search snippet)](https://arxiv.org/pdf/2205.00159) |
| AutoSTR (2020; older) | 6.04 M | not found | common-6 92.19; Union14M 70.09 | 12.1 ms (1080Ti) | [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md) |
| SVTRv2-S | 11.25 M | not published | common-6 96.13; Union14M 83.70 | 5.3 ms | [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md) |
| CnOCR densenet_lite_136-gru (zh + en) | not published | 12 MB | not published | not published | [CnOCR README](https://github.com/breezedeus/CnOCR) |
| CRNN (ResNet31 + LSTM, retrained) | 16.20 M | not published | common-6 90.21; Union14M 49.24 | 5.8 ms | [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md) |
| OpenCV Zoo CRNN EN (36 chars) | not published | 32.3 MB fp32, 15.6 MB int8 | ICDAR03 81.66, IIIT5k 74.33, CUTE80 52.78 (int8: 81.75 / 75.33 / 52.43) | not published | [OpenCV Zoo CRNN](https://github.com/opencv/opencv_zoo/tree/main/models/text_recognition_crnn) |
| PARSeq (small, default) | 23.83 M, 3.255 GFLOPs | checkpoint 91 MB; OnnxTR 92.1 MB fp32 / 32.9 MB 8-bit | common-6 96.40; Union14M 84.26 | 19.0 ms (1080Ti) | [PARSeq repo](https://github.com/baudm/parseq), [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md) |
| TrOCR-small | 61.4 M | 87.8 MB + 146 MB (Qualcomm float export) | not captured | 8.0 ms encoder + 1.5 ms per decoder step (Snapdragon 8 Gen 3) | [trocr-small-printed (search snippet)](https://huggingface.co/microsoft/trocr-small-printed), [Qualcomm TrOCR (search snippet)](https://aihub.qualcomm.com/mobile/models/trocr) |

- In OpenOCR's unified re-training (24 methods on Union14M-L-Filter), only AutoSTR (6.04 M), SVTRv2-T (5.13 M) and SVTRv2-S (11.25 M) are under 15 M parameters; SVTR-B is 18.09 M (94.58 / 71.17, 6.2 ms), ABINet 36.86 M (95.83 / 75.72, 13.7 ms), MAERec 35.69 M (96.36 / 85.17, 58.4 ms) — [OpenOCR SVTRv2 docs](https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md)
- Training data matters more than architecture for the small models: SVTRv2-B trained on synthetic MJ+ST scores 62.83 on Union14M-Benchmark versus 86.14 when trained on Union14M-L-Filter — [SVTRv2 config readme](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)
- Tiny CTC models collapse on multi-script data: on the 10-script TextMuSS-Bench a CRNN scores 55.55% average (Arabic 42.77, Russian 30.93, Tibetan 48.60) versus SVTRv2 77.59 and PARSeq 76.55 — [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)

Detectors (sorted by size):

| Model | Params | File size | Accuracy | Speed | Source |
|---|---|---|---|---|---|
| chineseocr_lite DBNet | not published | 1.8 MB | not published | not published | [chineseocr_lite README](https://github.com/DayBreak-u/chineseocr_lite) |
| db_mobilenet_v3_large (docTR) | 4.2 M | 15.3 MB fp32, 4.11 MB 8-bit | FUNSD recall 82.69 / precision 84.63; CORD 94.51 / 70.28 | 0.5 s at 1024x1024 (CPU, hardware not stated) | [docTR detection table](https://github.com/mindee/doctr/blob/main/docs/source/models/detection.md), [OnnxTR assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.1.2) |
| CnSTD db_mobilenet_v3_small / db_shufflenet_v2_small | not published | 7.9 MB / 12 MB | not published | not published | [CnOCR README](https://github.com/breezedeus/CnOCR) |
| OpenOCR RepViT-DBNet (mobile) | not published | 11.8 MB ONNX | not published separately | not published | [OpenOCR release assets](https://github.com/Topdu/OpenOCR/releases/expanded_assets/develop0.0.1) |
| FAST-T (repo's latest news item is dated Nov 28, 2024) | 8.5 M (FAST-S 9.7 M, FAST-B 10.6 M) | 32.5 MB (docTR rep_fast_tiny ONNX) | Total-Text F 81.6 at 448 px, 83.5 at 512, 84.2 at 640; ICDAR 2015 F 81.7 at 736; CTW1500 F 81.5 at 512 | 152.8 FPS at 448, 131.1 at 512, 95.5 at 640, 60.9 at 736 (GPU; hardware not named on the page) | [FAST repo](https://github.com/czczup/FAST), [OnnxTR assets](https://github.com/felixdittrich92/OnnxTR/releases/expanded_assets/v0.0.1) |
| CRAFT (EasyOCR default; older design) | 20.8 M | 79.2 MB | not captured | about 220 ms (recent iPhones) to 490-520 ms (Galaxy S24 / OnePlus 12) per forward pass | [Qualcomm EasyOCR card (search snippet)](https://huggingface.co/qualcomm/EasyOCR), [RN ExecuTorch docs (search snippet)](https://docs.swmansion.com/react-native-executorch/docs/next/benchmarks/inference-time) |
| EasyOCR DBNet-ResNet18 | not published | 49.6 MB zip | not published | not published | [EasyOCR v1.6.0 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/v1.6.0) |
| EAST (older design, frozen graph) | not verified | 92.2 MB | not captured | not captured | [frozen_east_text_detection.pb mirror](https://github.com/oyyd/frozen_east_text_detection.pb/blob/master/frozen_east_text_detection.pb) |

### Inferences
- A 2.1 M-parameter CRNN reaching 87-94% word accuracy on document crops (FUNSD/CORD) shows that tiny recognizers are adequate for printed Latin documents; the gap opens on scene text, where a retrained ResNet31 CRNN gets only 49% on Union14M while the 5.13 M SVTRv2-T gets 79%.
- SVTRv2-T is the best published accuracy-per-parameter recognizer outside the Paddle family, but it is published as PyTorch weights for English only; nobody has published a mobile-format (TFLite/Core ML/int8 ONNX) build or CPU timing for it.
- Parameter counts for PARSeq-Ti (23 MB fp32 checkpoint implies about 6 M parameters) are an estimate from file size.
- ViTSTR-Tiny's 80.3% and SVTRv2-T's 94.78% are not comparable: different training data (synthetic vs real), test protocol and year.
- For detection, an 8-bit MobileNetV3 DBNet (4.11 MB) is the smallest verifiable non-Paddle detector with published accuracy; CRAFT and EAST are 20-25x larger and are legacy choices.

### Gaps
- PARSeq-Ti and SVTR-T accuracy tables, and STRIDE's accuracy, could not be retrieved (arXiv / ECVA pages not reachable).
- No ICDAR13/15 or Union14M numbers exist for the document-oriented tiny models (docTR CRNNs, VIPTR-tiny as trained by docTR, EasyOCR CRNN, GlanceOCR, chineseocr_lite); they are evaluated on different data or not at all.
- No CPU/ARM latency for SVTRv2-T, PARSeq-Ti, SVTR-T or FAST-T was found; FAST's FPS hardware is not named on the repository page.
- Papers with Code leaderboards were not consulted (site not reachable in this session).

## 6. New lightweight (non-VLM) OCR models or toolkits from 2025-2026, and whether any target mobile/edge

### Takeaway
Very little new non-Paddle, non-VLM work targets phones: the notable 2025-2026 items are SVTRv2 (ICCV 2025) with OpenOCR's ONNX mobile export, docTR's VIPTR-tiny and TensorFlow removal, incremental Tesseract / Tesseract.js / ocrs releases, niche CPU-first tools (NDLOCR-Lite, meikiocr, GlanceOCR) and the 46 M-parameter multilingual ScriptMoE. Several non-Paddle toolkits instead adopted PP-OCRv6 in 2026, and the older deep-learning toolkits (EasyOCR, MMOCR, keras-ocr, chineseocr_lite) have stopped releasing.

### Cited Findings

New or updated in 2025-2026:
- SVTRv2 ("CTC Beats Encoder-Decoder Models in Scene Text Recognition") accepted to ICCV 2025 (2025.07.10); OpenOCR ONNX export released 2025.03.23 — [OpenOCR README](https://github.com/Topdu/OpenOCR)
- ScriptMoE released 2026.09.21 (arXiv 2609.24058, preprint): one recognizer for 10 scripts and 229 languages, SVTRv2 encoder + sparse mixture-of-experts decoder, 45.85 M parameters (41.13 M activated), 19,684-character set; on a V100 at batch 256: 541.07 ms latency, 473.1 img/s, 2091 MB memory — [OpenOCR README](https://github.com/Topdu/OpenOCR), [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)
- ScriptMoE accuracy: 82.06% average on TextMuSS-Bench (10,899 images) versus SVTRv2-AR 80.75, SVTRv2 77.59, PARSeq 76.55, CRNN 55.55, PP-OCRv5 multilingual (zero-shot) 63.59; with the PP-OCRv5 detector it reaches 80.89 F1 on CC-OCR multilingual versus 65.71 for PP-OCRv5's own recognizer — [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)
- An `openocr-mlt` package (archive dated 2026-11-01) bundles ScriptMoE recognition with PP-OCRv6 medium detection as ONNX models — [openocr-mlt HF card (search snippet)](https://huggingface.co/ericleyu/openocr-mlt-models)
- UniRec-0.1B (2025.08.20; 0.1 B parameters; Chinese/English text and formula recognition) and OpenDoc-0.1B (2025.12.25; document parsing) are encoder-decoder parsers, not mobile-tier OCR — [OpenOCR README](https://github.com/Topdu/OpenOCR)
- docTR: v0.12.0 (Jun 20, 2025) added `viptr_tiny`; v1.0.0 (Jul 9, 2025) removed TensorFlow; v1.1.0 (Aug 2026) added layout, table and formula models — [docTR releases](https://github.com/mindee/doctr/releases), [docTR tags](https://github.com/mindee/doctr/tags)
- OnnxTR: v0.7.1 (Jun 20, 2025) added VIPTR; v0.8.0 (Aug 13, 2025); v0.8.1 (Feb 4, 2026); v0.9.0 (Aug 24, 2026) — [OnnxTR tags](https://github.com/felixdittrich92/OnnxTR/tags), [OnnxTR releases](https://github.com/felixdittrich92/OnnxTR/releases)
- Tesseract 5.5.1 to 5.5.4 (May 2025 to Oct 2026) are maintenance and security releases; Tesseract.js v6 (Jan 2025) and v7 (Dec 15, 2025) improved memory and speed (15-35%) — [Tesseract tags](https://github.com/tesseract-ocr/tesseract/tags), [Tesseract.js releases](https://github.com/naptha/tesseract.js/releases)
- ocrs (Rust; detection is a pixel-segmentation model, recognition a CRNN, trained on HierText): 0.13.0 (2026-08-29) improved recognition speed 15-20% and added a C API; 0.13.1 (2026-09-13) made ONNX-format models the default; 0.14.0 (2026-11-08) fixed bugs "that blocked quantized text recognition models"; Latin alphabet only — [ocrs CHANGELOG](https://github.com/robertknight/ocrs/blob/main/CHANGELOG.md), [ocrs-models](https://github.com/robertknight/ocrs-models), [ocrs README](https://github.com/robertknight/ocrs)
- Kraken 7.0 (Apr 2, 2026): batched recognition with about 80% CPU speedups, safetensors default; 7.1 (Aug 4, 2026): "a new recognition architecture based on PP-OCRv6" with tiny, small and medium multilingual base models for historical material — [Kraken releases](https://github.com/mittagessen/kraken/releases), [Kraken tags](https://github.com/mittagessen/kraken/tags)
- NDLOCR-Lite (National Diet Library of Japan): GPU-free OCR for Japanese books and magazines, DEIMv2 layout (distilled DINOv3 ViT-Tiny backbone) + PARSeq recognition + reading order, all ONNX, CPU by default; desktop builds for Windows 11, macOS and Ubuntu; v1.2 update dated 2026/04/24 improved handwriting (average CER 0.268 on 1,065 JaWildText images); CC BY 4.0 — [ndlocr-lite repo](https://github.com/ndl-lab/ndlocr-lite)
- NDLOCR-Lite Web, a browser port launched at the end of February 2026, runs four ONNX models in-browser with an initial model download of about 146 MB — [issoh.co.jp article (search snippet)](https://www.issoh.co.jp/tech/details/11281/)
- GlanceOCR: "lightweight, high-performance OCR text recognition model" with a synthetic-data to quantization-aware-training pipeline; `tiny` checkpoint v2.2 has 4.35 M parameters; positioned as a drop-in replacement for the PP-OCR ONNX recognizer (swap `keys.txt` and `rec.onnx`); Apache-2.0; no benchmarks published; repository has 9 commits and 4 stars — [GlanceOCR repo](https://github.com/afirelily/GlanceOCR), [GlanceOCR HF card (search snippet)](https://huggingface.co/afirelily/GlanceOCR-v2_2-tiny)
- meikiocr: "high-speed, high-accuracy, local ocr for japanese video games"; two ONNX models (detection + recognition) on ONNX Runtime; Japanese only; Apache-2.0; limits of 64 text boxes and 48 characters per line; no numeric benchmarks in text — [meikiocr repo](https://github.com/rtr46/meikiocr)
- CnOCR V2.3.3 (2026.07.04): its main change is PP-OCRv6 multilingual model support — [CnOCR README](https://github.com/breezedeus/CnOCR)
- React Native ExecuTorch's current benchmarks list PP-OCRv6 as the OCR model — [RN ExecuTorch benchmarks source](https://github.com/software-mansion/react-native-executorch/blob/main/docs/docs/04-benchmarks/01-benchmarks.mdx)
- Baseline for context (Paddle family): PP-OCRv6 tiers are Tiny 1.5 M, Small 7.7 M and Medium 34.5 M parameters with 50 languages — [ai-tldr PP-OCRv6 note (search snippet)](https://ai-tldr.dev/releases/paddlepaddle-pp-ocrv6/)
- Edge-oriented evaluation (September 2025): E-ARMOR compared five large vision-language models with two traditional OCR systems (Sprinklr-Edge-OCR and SuryaOCR) and concluded "the most optimal OCR systems for edge deployment are the traditional ones even in the era of LLMs" — [arXiv 2509.03615 (search snippet)](https://arxiv.org/html/2509.03615v1)

Stalled projects:
- EasyOCR: no release since v1.7.2 (24 September 2024) — [EasyOCR releases](https://github.com/JaidedAI/EasyOCR/releases)
- MMOCR: no release since v1.0.1 (Jul 4, 2023) — [MMOCR tags](https://github.com/open-mmlab/mmocr/tags)
- keras-ocr: no tag since v0.9.1 (May 19, 2022) — [keras-ocr tags](https://github.com/faustomorales/keras-ocr/tags)
- chineseocr_lite: most recent commits on its `onnx` branch are dated May 20, 2023 — [chineseocr_lite commits](https://github.com/DayBreak-u/chineseocr_lite/commits/onnx/)
- SwiftyTesseract: archived Apr 8, 2022 — [SwiftyTesseract](https://github.com/SwiftyTesseract/SwiftyTesseract)

### Inferences
- The lightweight non-VLM frontier in 2025-2026 is being set by the Paddle family: three independent non-Paddle projects (CnOCR, Kraken, React Native ExecuTorch) adopted PP-OCRv6 models or architecture in 2026, and GlanceOCR defines itself as a PP-OCR drop-in.
- None of the new non-Paddle releases ships an Android/iOS package or phone benchmark; "CPU-only" in this group means desktop CPU (NDLOCR-Lite, meikiocr, ocrs).
- ScriptMoE is the most interesting new recognizer for non-Latin scripts, but at 45.85 M parameters (roughly 180 MB in fp32) with an autoregressive decoder it is server-class, not old-phone-class.

### Gaps
- Release dates and benchmarks for GlanceOCR and meikiocr are not published; neither could be assessed for accuracy.
- A systematic scan of 2025-2026 arXiv scene-text papers for mobile-targeted models was cut short when the search budget ran out; candidates that surfaced but were not examined include "Efficient and Accurate Scene Text Recognition with Cascaded-Transformers" ([arXiv 2503.18883 (search snippet)](https://arxiv.org/html/2503.18883v1)) and a training-free lightweight segmentation/recognition network ([arXiv 2503.15639 (search snippet)](https://ar5iv.labs.arxiv.org/html/2503.15639)).
- MDiff4STR's venue is listed inconsistently in the OpenOCR README (AAAI 2026 in the news, "AAAI 2025 Oral" in a table) — [OpenOCR README](https://github.com/Topdu/OpenOCR).

## 7. Independent head-to-head comparisons (accuracy, speed, size) and what they conclude

### Takeaway
No rigorous, independent phone benchmark covers these engines together; the available comparisons are desktop-CPU, vendor-run or small-sample. They agree on the pattern: Tesseract is the lightest and often the fastest on CPU but trails on receipts, scene text and handwriting; docTR is fast and accurate on Latin documents; EasyOCR is slow on CPU and mid-to-low on accuracy; PaddleOCR-family models usually lead on accuracy.

### Cited Findings
- Maintainer-run CPU benchmark (Intel i7-14700K; library versions from about 2024; seconds per page on FUNSD 199 pages / CORD 900 pages): docTR v0.8.1 1.29 / 0.60; OnnxTR v0.6.0 0.57 / 0.25; OnnxTR 8-bit 0.38 / 0.14; OnnxTR OpenVINO 0.15 / 0.14; EasyOCR v1.7.1 1.96 / 1.75; PyTesseract v0.3.10 0.50 / 0.52; Surya v0.4.4 48.76 / 35.49; PaddleOCR v2.7.3 (no cls) 1.27 / 0.38 — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- Same source, RTX 4080 GPU: docTR 0.07 / 0.05; OnnxTR 0.06 / 0.04; EasyOCR 0.31 / 0.19; PaddleOCR 0.08 / 0.03; Surya float16 3.70 / 2.81 — [OnnxTR README](https://github.com/felixdittrich92/OnnxTR)
- Receipts, vendor-run (ImageToTable.ai, August 2026): CER on SROIE 2019 — docTR 19.7%, PaddleOCR 20.5%, EasyOCR 28.3%, Tesseract 33.6%, Docling 58.4%; throughput on an RTX 4090 — docTR 256.0, PaddleOCR 157.8, EasyOCR 77.4, Tesseract 54.4, Docling 33.2 pages per minute; field-extraction F1 — PaddleOCR 32.5% (30.5-34.5% CI), Tesseract 23.2%, Docling 22.3%, EasyOCR 14.8%, docTR 7.7% — [ImageToTable.ai receipt benchmark (search snippet)](https://imagetotable.ai/references/receipt-ocr-benchmark-results)
- Same vendor: with an LLM post-processor added, docTR scored 0.6171 (best of 8 engines) and EasyOCR 0.3717 (worst of 8) — [ImageToTable.ai EasyOCR vs docTR (search snippet)](https://imagetotable.ai/references/easyocr-vs-doctr-receipt-benchmark)
- One clean invoice, Apple M-series CPU (Codesota, March 2026): accuracy PaddleOCR 100%, Tesseract 87.5%, RapidOCR 75%, EasyOCR 62.5%; time Tesseract 0.162 s, RapidOCR 0.212 s, EasyOCR 0.656 s, PaddleOCR 4.85 s; its guidance: PaddleOCR for accuracy, Tesseract for bulk speed, RapidOCR when install size matters — [Codesota (search snippet)](https://codesota.com/ocr/best-for-python)
- The same site withdrew an earlier three-way PaddleOCR / Tesseract table because it lacked linked inputs, ground truth and run artifacts, stating that no shared controlled benchmark is claimed — [Codesota PaddleOCR vs Tesseract (search snippet)](https://www.codesota.com/ocr/paddleocr-vs-tesseract)
- Handwriting, community benchmark on 5,578 prescriptions (2026): exact match Tesseract 2.5% (49 ms/img), EasyOCR 2.6% (14 ms/img), PP-OCRv5 21.4% (103 ms/img), GLM-OCR 32.6% (141 ms/img) — [dev.to benchmark (search snippet)](https://dev.to/kaniel_outis/how-well-can-ocr-read-doctor-handwriting-in-2026-54hn)
- Mobile, academic (SIBGRAPI, Brazil): on 848 real word images from urban scenes, Apple Vision performed best, then ML Kit, then Tesseract with optimized preprocessing; the paper also scores privacy and usability — [SIBGRAPI paper (search snippet)](https://sol.sbc.org.br/index.php/sibgrapi_estendido/article/view/38292)
- Mobile, older (about 2019), 959 images: ML Kit outperformed Tesseract "by a large margin" on accuracy while using more CPU time — [Heartbeat / Fritz AI (search snippet)](https://heartbeat.comet.ml/comparing-on-device-text-recognition-ocr-sdks-24f8a0423461)
- Mobile, unverified methodology: Tesseract 5 about 220 ms versus ML Kit about 140 ms average over 500 images — [Designveloper (search snippet)](https://www.designveloper.com/?p=47760)
- Edge board: on a Raspberry Pi 4, a Tesseract pipeline matched EasyOCR's recognition rate with 89% lower average execution time — [IASC paper (search snippet)](https://techscience.com/iasc/v36n3/51941/html)
- Multi-script recognizers (authors' own benchmark, 2026): ScriptMoE 82.06 vs PP-OCRv5 multilingual zero-shot 63.59 vs CRNN 55.55 on TextMuSS-Bench — [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)
- OpenOCR's own leaderboard metric: OpenOCR mobile 67.22% vs PP-OCRv4 62.77% — [OpenOCR docs](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)
- docTR vs cloud APIs on FUNSD / CORD end-to-end: docTR (db_resnet50 + crnn_vgg16_bn) recall 73.37 / 84.80 versus Google Vision document text detection 64.00 / 68.90, AWS Textract 78.10 / 87.50, Azure Form Recognizer 79.42 / 89.62 — [docTR end-to-end table](https://github.com/mindee/doctr/blob/main/docs/source/models/end_to_end.md)
- A 2026 practitioner guide notes VLMs run 5-10x slower than traditional engines and sometimes hallucinate plausible text, and advises testing on your own documents — [slavadubrov OCR guide, 2026-03-04 (search snippet)](https://slavadubrov.github.io/blog/2026/03/04/ocr-guide/)

### Inferences
- Rankings flip with the metric: on the same receipts docTR has the best CER but the worst field F1, so "works well" has to be defined per use case before picking an engine.
- Speed rankings flip with hardware: Tesseract is among the fastest on CPU (0.16-0.5 s/page) but falls to second-last in the RTX 4090 throughput test, where the deep-learning pipelines benefit from the GPU; phones without an NPU resemble the CPU case.
- Every comparison that includes a Paddle-family model puts it at or near the top on accuracy; none of the non-Paddle deep-learning toolkits beats it while also being smaller.
- The evidence base for old or low-end phones specifically is essentially absent; Raspberry Pi results are the closest proxy.

### Gaps
- No independent benchmark on named low-end Android phones (e.g., 2-3 GB RAM, Cortex-A53 class) comparing Tesseract, docTR/OnnxTR, OpenOCR, EasyOCR and PP-OCR was found.
- The TildAlice "PaddleOCR vs EasyOCR vs docTR memory & latency test" surfaced in search but its numbers could not be retrieved.
- Several sources above are vendor blogs or single-image tests; none is peer-reviewed except the SIBGRAPI and IASC papers, whose detailed tables were not accessible.

## 8. Script coverage for non-Latin scripts (Arabic, CJK, Cyrillic, Indic) and handwriting in these small models

### Takeaway
Tesseract has the broadest coverage per megabyte (100+ languages, 1-4 MB each, including Arabic, CJK, Cyrillic and Indic), while the small deep-learning options are narrow: docTR's pretrained recognizers are Latin-only, OpenOCR mobile and chineseocr_lite are Chinese + English, and EasyOCR's small models cover only Latin, Cyrillic, Simplified Chinese, Japanese, Korean, Telugu and Kannada. Handwriting is poor or unsupported across all of the small non-Paddle options.

### Cited Findings
- Tesseract recognizes "more than 100 languages" out of the box; tessdata_fast also has per-script models (e.g., Latin, Devanagari covering hin/san/mar/nep/eng) — [Tesseract README](https://github.com/tesseract-ocr/tesseract), [tessdata_fast README](https://github.com/tesseract-ocr/tessdata_fast)
- Tesseract tessdata_fast sizes for non-Latin languages: ara 1.37 MB, chi_sim 2.35 MB, jpn 2.36 MB, kor 1.6 MB, rus 3.68 MB, hin 1.07 MB; script models Arabic 8.88 MB, Devanagari 17.1 MB — [ara](https://github.com/tesseract-ocr/tessdata_fast/blob/main/ara.traineddata), [chi_sim](https://github.com/tesseract-ocr/tessdata_fast/blob/main/chi_sim.traineddata), [jpn](https://github.com/tesseract-ocr/tessdata_fast/blob/main/jpn.traineddata), [kor](https://github.com/tesseract-ocr/tessdata_fast/blob/main/kor.traineddata), [rus](https://github.com/tesseract-ocr/tessdata_fast/blob/main/rus.traineddata), [hin](https://github.com/tesseract-ocr/tessdata_fast/blob/main/hin.traineddata), [script/Arabic](https://github.com/tesseract-ocr/tessdata_fast/blob/main/script/Arabic.traineddata), [script/Devanagari](https://github.com/tesseract-ocr/tessdata_fast/blob/main/script/Devanagari.traineddata)
- Tesseract LSTM vs legacy on Hindi (older, 4.0-era Google test): character errors 7.6 vs 13.9, word recall errors 20.9 vs 30; the doc notes "It is the complex scripts that will run faster relative to base Tesseract" — [tessdoc 4.0 Accuracy and Performance](https://github.com/tesseract-ocr/tessdoc/blob/main/tess4/4.0-Accuracy-and-Performance.md)
- Tesseract handwriting: "it won't work very well, as Tesseract is designed for printed text" — [tessdoc FAQ](https://github.com/tesseract-ocr/tessdoc/blob/main/FAQ.md)
- EasyOCR advertises "80+ supported languages" across Latin, Chinese, Arabic, Devanagari and Cyrillic scripts, with handwriting still under "What's coming next" — [EasyOCR README](https://github.com/JaidedAI/EasyOCR)
- EasyOCR second-generation (small) recognizers exist only for english, latin, zh_sim, japanese, korean, telugu, kannada and cyrillic; Arabic, Devanagari, Bengali, Thai, Tamil and Traditional Chinese have only first-generation models — [EasyOCR config.py](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py)
- EasyOCR first-generation downloads are 191 MB (arabic, devanagari, thai) to 203 MB (chinese_sim) per script — [EasyOCR pre-v1.1.6 assets](https://github.com/JaidedAI/EasyOCR/releases/expanded_assets/pre-v1.1.6)
- docTR pretrained recognition models use the `french` vocabulary (Latin + Western European accents); 100+ vocabularies exist for training; recent releases added Hebrew, Russian and Croatian (v0.12.0), Gujarati (v0.11.0), Vietnamese and Ukrainian (v0.10.0) vocabs — [docTR recognition docs](https://github.com/mindee/doctr/blob/main/docs/source/models/recognition.md), [docTR releases](https://github.com/mindee/doctr/releases)
- OpenOCR's mobile and server models are Chinese + English only — [OpenOCR docs](https://github.com/Topdu/OpenOCR/blob/main/docs/openocr.md)
- SVTRv2 on Chinese data: document text 99.3-99.5% but handwriting only 62.0% (T), 65.0% (S), 67.0% (B); scene 77.8-83.5% — [SVTRv2 config readme](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/svtrv2/readme.md)
- ScriptMoE (45.85 M params) per-script word accuracy on TextMuSS-Bench: Arabic 78.09, Bangla 88.30, Chinese 95.38, Hindi 86.77, Japanese 71.21, Korean 87.19, Latin 91.67, Russian 61.20, Thai 72.00, Tibetan 88.76 — [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)
- Same benchmark, small/classic models: CRNN Arabic 42.77, Bangla 55.47, Chinese 64.00, Hindi 63.87, Japanese 44.28, Korean 70.10, Latin 82.38, Russian 30.93, Thai 53.13, Tibetan 48.60; PP-OCRv5 multilingual zero-shot Arabic 68.30, Chinese 80.00, Hindi 60.05, Japanese 57.41, Korean 75.85, Latin 83.43, Russian 47.82, Thai 35.87 (Bangla and Tibetan unsupported) — [ScriptMoE readme](https://github.com/Topdu/OpenOCR/tree/main/configs/rec/scriptmoe)
- chineseocr_lite: Chinese-focused (DBNet + CRNN + AngleNet); CnOCR's own models cover Simplified Chinese, English and digits and do not support vertical text — [chineseocr_lite README](https://github.com/DayBreak-u/chineseocr_lite), [CnOCR README](https://github.com/breezedeus/CnOCR)
- OpenCV Zoo CRNN character sets: EN 36 characters (digits + lowercase letters), CH 94 characters, CN 3,944 characters including Chinese — [OpenCV Zoo CRNN](https://github.com/opencv/opencv_zoo/tree/main/models/text_recognition_crnn)
- ocrs supports the Latin alphabet only; additional languages are planned — [ocrs README](https://github.com/robertknight/ocrs)
- ddddocr recognizes Chinese, English letters, digits and some special characters, but for captchas — [ddddocr README](https://github.com/sml2h3/ddddocr)
- Japanese: NDLOCR-Lite handles vertical and horizontal print and reports handwriting CER 0.268 (vertical 0.279, horizontal 0.264) on JaWildText; meikiocr is Japanese-only with vertical text in beta — [ndlocr-lite repo](https://github.com/ndl-lab/ndlocr-lite), [meikiocr repo](https://github.com/rtr46/meikiocr)
- Historical and handwritten material: Kraken 7.1 added tiny/small/medium multilingual base models "for historical material" built on a PP-OCRv6-based architecture — [Kraken releases](https://github.com/mittagessen/kraken/releases)
- Handwriting in practice: on 5,578 handwritten prescriptions Tesseract and EasyOCR reached 2.5% and 2.6% exact match versus PP-OCRv5 21.4% — [dev.to benchmark (search snippet)](https://dev.to/kaniel_outis/how-well-can-ocr-read-doctor-handwriting-in-2026-54hn)

### Inferences
- For a small on-device footprint with Arabic, Indic or mixed scripts, the only non-Paddle option that is both tiny and maintained is Tesseract (printed text only); EasyOCR covers those scripts but at about 190 MB per script plus a 79 MB detector.
- The CRNN row in the ScriptMoE table suggests that classic tiny CTC recognizers lose 20-30 points on Arabic, Cyrillic and Indic scene text relative to modern recognizers, so "tiny + multi-script + scene text" is not solved outside larger models.
- Handwriting should be treated as unsupported for every small non-Paddle engine reviewed here; the usable handwriting numbers found are for Japanese desktop OCR (NDLOCR-Lite, 26.8% CER) and Chinese SVTRv2 (62-67% line accuracy), neither phone-ready.
- Cyrillic is the best-served non-Latin script among small models (Tesseract rus 3.68 MB; EasyOCR cyrillic_g2 13.5 MB), followed by Simplified Chinese / Japanese / Korean; Arabic and Indic scripts are the worst served.

### Gaps
- No per-language accuracy numbers were found for Tesseract tessdata_fast on Arabic, CJK, Cyrillic or Indic printed text from the last several years.
- No evaluation of EasyOCR's per-script accuracy, or of docTR community multilingual models, was retrieved.
- TrOCR-small handwritten-checkpoint accuracy and any phone deployment of it were not verified.
- Whether the right-to-left and vertical-text handling of these small pipelines works acceptably on-device was not addressed by any source found.
