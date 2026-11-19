# PaddleOCR mobile ("PP-OCR mobile") model family and its on-device ports — state as of 2026-11-19

> **Method caveat (read first).** In this session direct page fetches were blocked (`www.paddleocr.ai` and `arxiv.org` did not resolve from the sandbox; GitHub API access was limited to an unrelated repo), and the shared web-search budget ran out after ~75 queries. Every figure below therefore comes from **search-result extracts of the cited pages**, not from reading the full pages. Where an extract was ambiguous, single-sourced, or conflicting, it is flagged inline ("single extract", "conflict", "attribution ambiguous"). Anything not confirmed is listed under *Gaps* rather than estimated. "MB" figures are copied as printed by each source (official docs call the column "Model Storage Size (MB)"; third parties sometimes use MiB or decimal MB).

---

## 1. What is the newest PP-OCR generation as of November 2026, and what did each release (v3, v4, v5, later) change in the mobile-tier models? (release dates, PaddleOCR 2.x vs 3.x)

### Takeaway
The newest generation is **PP-OCRv6, released 2026-06-11 with PaddleOCR 3.7.0**; it retires the "mobile/server" naming in favour of **tiny (1.5M params) / small (7.7M) / medium (34.5M)** tiers, where *small* is the mobile successor to PP-OCRv5_mobile and *tiny* is a new, much smaller edge class. I found no newer generation or PaddleOCR release (3.8+) in search indexes up to November 2026. Earlier mobile generations: PP-OCRv3 (PaddleOCR 2.5, 2022-05-09), PP-OCRv4 (2.7, 2023-08-07), PP-OCRv5 (3.0.0, 2025-05-20).

### Cited Findings

#### PP-OCRv6 (current)
- PP-OCRv6 was released on 2026-06-11 as part of PaddleOCR 3.7.0 and is described as the newest version of the PP-OCR series. — [PaddleOCR wiki](https://github.com/PaddlePaddle/PaddleOCR/wiki); [PaddleOCR docs home](https://www.paddleocr.ai/latest/index.html); [PP-OCRv6 docs (v3.7.0, Jun 2026)](https://www.paddleocr.ai/v3.7.0/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- v6 has no "mobile" label: it ships three open-weight tiers — tiny (1.5M params), small (7.7M), medium (34.5M) — "for edge, mobile, and server deployment", all on a PPLCNetV4 backbone. The docs describe small as suited to mobile deployment and tiny to edge/IoT; the default model in PaddleOCR 3.7 is `PP-OCRv6_medium`. — [PyPI paddleocr](https://pypi.org/project/PaddleOCR/); [PP-OCRv6 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [HF blog "PP-OCRv6 on Hugging Face" (Jun 2026)](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6)
- Recognition architecture is PPLCNetV4 + LightSVTR + CTC/NRTR multi-head; the tiny recogniser uses "a direct reshape projection instead of an encoder neck" and is trained with knowledge distillation from the medium model. — [Text recognition module docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md); [HF Transformers docs, PP-OCRv6_tiny_rec](https://huggingface.co/docs/transformers/en/model_doc/pp_ocrv6_tiny_rec); [Moonlight paper review](https://www.themoonlight.io/tw/review/pp-ocrv6-from-15m-to-345m-parameters-surpassing-billion-scale-vlms-on-ocr-tasks)
- Headline claims vs v5 (medium vs v5 server): +4.6% detection and +5.1% recognition accuracy, and a 5.2x end-to-end CPU speed-up (PP-OCRv6_medium 1.40 s vs PP-OCRv5_server 7.30 s on Intel Xeon + OpenVINO). — [PaddleOCR docs home (main)](https://www.paddleocr.ai/main/index.html)
- Paper: "PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks", arXiv 2606.13108 (June 2026); abstract states the tiny tier is 3.9x faster than PP-OCRv5_mobile on an Intel Xeon CPU "while maintaining comparable accuracy". — [arXiv 2606.13108](https://arxiv.org/abs/2606.13108)
- v6 weights are published on Hugging Face in Paddle, safetensors and ONNX repos (e.g. `PP-OCRv6_tiny_det`, `PP-OCRv6_tiny_det_safetensors`, `PP-OCRv6_tiny_det_onnx`) and plug into PaddleOCR 3.7's unified inference engine. — [HF PP-OCRv6 collection](https://huggingface.co/collections/PaddlePaddle/pp-ocrv6); [HF PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx); [hyper.ai write-up](https://hyper.ai/en/stories/d3ee689571cc134a197942095f45b44d)
- Nothing newer was found: searches for PaddleOCR 3.8/3.9 and for Oct–Nov 2026 release notes returned no results; a third-party tracker's history covers "3.0–3.7"; the latest README news item surfaced is HPD-Parsing (2026-07-22, a document-parsing model, not a pip release). — [paddleocr.dev release history (third party)](https://paddleocr.dev/releases); [PaddleOCR README (jsDelivr mirror)](https://cdn.jsdelivr.net/gh/PaddlePaddle/PaddleOCR@main/README.md)

#### Release timeline (mobile-relevant items)

| Date | PaddleOCR version | Generation / change relevant to mobile tier | Source |
|---|---|---|---|
| Sep 2020 (paper) | 1.x / 2.0 era | PP-OCR (v1): "ultra lightweight" system, 3.5M total for 6,622 Chinese chars, 2.8M for 63 alphanumerics | [arXiv 2009.09941](https://arxiv.org/abs/2009.09941) |
| Sep 2021 (paper) | 2.x | PP-OCRv2 "Bag of Tricks"; system Hmean 57.6, 11.6M, 330 ms CPU, 111 ms T4 (as tabulated in the v3 paper) | [arXiv 2109.03144](https://arxiv.org/abs/2109.03144); [arXiv 2206.03001](https://arxiv.org/abs/2206.03001) |
| 2022-05-09 | 2.5 | **PP-OCRv3**: Chinese accuracy +5% vs PP-OCRv2 at comparable speed, English +11%. Paper: system Hmean 62.9 vs 57.6, size 15.6M vs 11.6M, CPU 331 ms vs 330 ms, T4 87 ms vs 111 ms | [PaddleOCR update log](https://paddlepaddle.github.io/PaddleOCR/main/en/update/update.html); [arXiv 2206.03001](https://arxiv.org/abs/2206.03001) |
| 2023-08-07 | 2.7 | **PP-OCRv4** (mobile + server): mobile improves Chinese scenes +4.5%, English +10%, the 80-language multilingual models +8% vs PP-OCRv3 at comparable speed; detection student backbone switched MobileNetV3 -> PP-LCNetV3 (hmean 78.24% -> 79.08%) | [update log (release/2.7)](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/update_en.md); [PaddlePaddle on X, Oct 2023](https://x.com/PaddlePaddle/status/1715401001652662307); [PP-OCRv4 introduction (release/2.7)](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/PP-OCRv4_introduction.md) |
| 2025-05-20 | 3.0.0 | **PP-OCRv5** (mobile + server): one model for five text types plus complex handwriting; +13 percentage points end-to-end over PP-OCRv4 on the internal multi-scenario set | [PaddleOCR docs home v3.0.0](https://www.paddleocr.ai/v3.0.0/en/index.html); [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html) |
| 2025-06-05 | 3.0.1 | Python defaults changed from mobile to **server** det/rec models (mobile must be selected explicitly) | [update log v3.0.1](https://paddlepaddle.github.io/PaddleOCR/v3.0.1/en/update/update.html) |
| 2025-06-29 | 3.1.0 | PP-OCRv5 multilingual recognisers for 37 languages (French, Spanish, Portuguese, Russian, Korean, ...), average accuracy +30% | [update log](https://paddlepaddle.github.io/PaddleOCR/main/en/update/update.html) |
| 2025-08-21 | 3.2.0 | PP-OCRv5 English, Thai, Greek recognisers (Thai 82.68%, Greek 89.28%; English model +11% over the main v5 model on English) | [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases); [README](https://github.com/PaddlePaddle/PaddleOCR) |
| 2025-10-16 | 3.3.0 | PaddleOCR-VL; PP-OCRv5 multilingual coverage raised to 109 languages (adds Cyrillic, Arabic, Devanagari, Telugu, Tamil) | [update log](https://paddlepaddle.github.io/PaddleOCR/main/en/update/update.html) |
| 2026-01-29 | 3.4.0 | PaddleOCR-VL-1.5 | [PaddleOCR docs home](https://www.paddleocr.ai/latest/en/index.html) |
| 2026-04-21 (PyPI; PaddleX lists Apr 17) | 3.5.0 | Switchable inference backends (Paddle static graph / dynamic graph / Transformers); official browser SDK **PaddleOCR.js** running PP-OCRv5 | [PyPI paddleocr 3.5.0](https://pypi.org/project/paddleocr/3.5.0/); [PaddleOCR wiki](https://github.com/PaddlePaddle/PaddleOCR/wiki) |
| 2026-05-28 | 3.6.0 | PaddleOCR-VL-1.6; async-API SDKs; multi-page TIFF | [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases) |
| 2026-06-11 | 3.7.0 | **PP-OCRv6** tiny/small/medium; official ONNX Runtime Android demo/SDK | [PaddleOCR wiki](https://github.com/PaddlePaddle/PaddleOCR/wiki); [Android deployment docs v3.7.0](https://www.paddleocr.ai/v3.7.0/en/version3.x/inference_deployment/cross_platform/android_deployment.html) |

- PP-OCRv5 paper: "PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter Vision-Language Models on OCR Tasks" (arXiv 2603.24373, March 2026; CVPR 2026). "PP-OCRv5" in the paper means PP-OCRv5_mobile (5M parameters); PP-OCRv5_server is the higher-capacity variant. — [arXiv 2603.24373](https://arxiv.org/abs/2603.24373); [CVPR 2026 open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Cui_PP-OCRv5_A_Specialized_5M-Parameter_Model_Rivaling_Billion-Parameter_Vision-Language_Models_on_CVPR_2026_paper.pdf)
- PP-OCRv5's recogniser uses a larger dictionary than v4, which makes it slower than PP-OCRv4 (official note). — [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)

### Inferences
- `PP-OCRv6_small_*` is the direct successor of `PP-OCRv5_mobile_*`; `PP-OCRv6_tiny_*` is a size class that did not exist in v3–v5 except as the INT8 "slim" variants of v2/v3.
- Because 3.0.1 switched the Python defaults to server models and 3.7 defaults to medium, a phone developer must name the mobile/small/tiny models explicitly; "default PaddleOCR" numbers quoted in blogs are usually not mobile-tier numbers.
- The 2.x -> 3.x boundary (May 2025) matters for mobile: 2.x-era models (v2/v3/v4) have documented Paddle Lite `.nb` files and slim variants; 3.x-era models (v5/v6) are distributed mainly as Paddle/ONNX/safetensors and Baidu's own v6 phone demos use ONNX Runtime.

### Gaps
- I could not open the GitHub releases page, so "no 3.7.x / 3.8 after June 2026" rests on search results only (no hits for 3.8/3.9; latest item found was July 2026).
- The exact list of PP-OCRv5's five text types (docs list Simplified Chinese, Traditional Chinese, English, Japanese and Pinyin) and the component names of v3/v4 (LK-PAN, DML, SVTR-LCNet, GTC-NRTR, etc.) were not re-confirmed from extracts in this session.
- PP-OCRv2's PaddleOCR version number and exact release date were not confirmed.

---

## 2. What are the exact sizes (MB) and parameter counts of the mobile detection, recognition and orientation-classifier models per generation and format, and what is the minimal det+rec(+cls) pipeline size?

### Takeaway
As of Nov 2026 the smallest current-generation pipeline is **PP-OCRv6 tiny: 1.9 MB det + 4.4 MB rec = about 6.3 MB (1.5M params, FP32)**; PP-OCRv6 small is about 30 MB (7.7M params), PP-OCRv5 mobile about 20.7 MB (4.7 + 16), PP-OCRv4 mobile about 15.2 MB (4.7 + 10.5). The smallest officially published pipeline of any generation remains the INT8 "slim" Paddle Lite builds: **PP-OCRv3 slim 5.9M** and **PP-OCRv2 slim 4.6M** (det+cls+rec), with the 2020 PP-OCR v1 at 3.5M.

### Cited Findings

#### PP-OCRv6 (PaddleOCR 3.7, June 2026) — official "Model Storage Size"

| Model | Params | Size (MB) | Accuracy on v6 in-house set | Source |
|---|---|---|---|---|
| PP-OCRv6_tiny_det | 0.43M | 1.9 (single extract) | Hmean 80.6 | [Text detection module docs](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html); [HF PP-OCRv6_tiny_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det) |
| PP-OCRv6_small_det | 2.48M | 9.6 (single extract) | Hmean 84.1 | [PaddleX OCR pipeline docs](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html); [HF PP-OCRv6_small_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det) |
| PP-OCRv6_tiny_rec | 1.1M | 4.4 (single extract) | Rec avg 73.5 | [Text recognition docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md); [HF Transformers docs](https://huggingface.co/docs/transformers/en/model_doc/pp_ocrv6_tiny_rec) |
| PP-OCRv6_small_rec | 5.2M | 20.4 (single extract) | Rec avg 81.3 | [Text recognition docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md); [HF PP-OCRv6_small_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec) |
| PP-OCRv6 medium (server class, for reference) | 34.5M total | not captured | Hmean 86.2 / rec 83.2 | [HF blog](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6) |

- Tier totals: tiny 1.5M params, small 7.7M, medium 34.5M (det + rec). — [arXiv 2606.13108](https://arxiv.org/abs/2606.13108); [PyPI](https://pypi.org/project/PaddleOCR/)
- A third-party test site measured downloaded model folders at 6.6, 31.6, 139.2, 22.1 and 173.6 MB (decimal) and separately quotes PP-OCRv5 mobile at 20.7 MB, describing v6 tiny as "roughly a third of the download" of v5 mobile. — [paddleocr.dev/models (third party)](https://paddleocr.dev/models)
- Community ncnn conversions of v5 and v6 exist on Hugging Face. — [HF developerabu/pp-OCRv5-6-ncnn](https://huggingface.co/developerabu/pp-OCRv5-6-ncnn)

#### PP-OCRv5 mobile (PaddleOCR 3.0, May 2025)

| Model | Size | Accuracy / notes | Source |
|---|---|---|---|
| PP-OCRv5_mobile_det (Paddle inference) | 4.7 MB | Hmean 79.0 (v5-era eval set) | [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html); [Text detection docs](https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/text_detection.html) |
| PP-OCRv5_mobile_rec (Paddle inference) | 16 MB | Chinese avg 81.29; English 66.00; Traditional Chinese 83.55; Japanese 54.65 | [Text recognition docs v3.1.1](https://www.paddleocr.ai/v3.1.1/en/version3.x/module_usage/text_recognition.html) |
| PP-OCRv5_mobile_det (ONNX) | 4.6 MiB (older listing: 4.8 MB) | community ONNX export | [oar-ocr docs/models.md (~Jun 2026)](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md); [crates.io oar-ocr 0.2.1](https://crates.io/crates/oar-ocr/0.2.1) |
| PP-OCRv5_mobile_rec (ONNX) | 15.8 MiB (older listing: 16.5 MB) | community ONNX export | [oar-ocr docs/models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md); [crates.io oar-ocr 0.2.1](https://crates.io/crates/oar-ocr/0.2.1) |
| Whole PP-OCRv5_mobile system | "5M parameters" | paper title/abstract | [arXiv 2603.24373](https://arxiv.org/abs/2603.24373) |

- For scale, the server variants are far larger: PP-OCRv5 server det 84.0 MiB, server rec 80.6 MiB (ONNX). — [oar-ocr docs/models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md)

#### PP-OCRv4 mobile (PaddleOCR 2.7, Aug 2023)

| Model | Size | Accuracy / notes | Source |
|---|---|---|---|
| PP-OCRv4_mobile_det | 4.7 MB | Hmean 77.79 in older PaddleX docs; 63.8 on the newer v5-era eval set | [PaddleX 3.0-beta2 OCR docs](https://paddlepaddle.github.io/PaddleX/3.0-beta2/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html); [PaddleX 3.3 model list](https://paddlepaddle.github.io/PaddleX/3.3/en/support_list/models_list.html) |
| PP-OCRv4_mobile_rec | 10.6 M (older docs) / 10.5 MB (newer docs) | avg accuracy 78.20 (older) / 78.74 (newer) | [PaddleX 3.0-beta2 OCR docs](https://paddlepaddle.github.io/PaddleX/3.0-beta2/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html); [Text recognition docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md) |
| ONNX exports | det 4.6 MiB / 4.75 MB; rec 10.4 MiB / ~10.9 MB | community | [oar-ocr docs/models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md); [HF Desperado-JT/CH-PP-OCRv4](https://huggingface.co/Desperado-JT/CH-PP-OCRv4/tree/main) |
| en_PP-OCRv4_mobile_rec | 7.5 MB | accuracy 70.39 | [Text recognition docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md) |

#### PP-OCRv3 / PP-OCRv2 / PP-OCR v1 (legacy 2.x; includes the slim INT8 variants)

| Item | Size | Notes | Source |
|---|---|---|---|
| PP-OCRv3 system (paper) | 15.6M | Hmean 62.9 | [arXiv 2206.03001](https://arxiv.org/abs/2206.03001) |
| PP-OCRv3_mobile_rec (2.x list) | 12.4M | 3.x docs list the same model at 10.3 MB (different measurement) | [2.x model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md); [Text recognition docs](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_recognition.html) |
| PP-OCRv3_mobile_rec_slim | 4.9M | "Slim quantization with distillation lightweight model", Chinese + English | [2.x model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md) |
| PP-OCRv3 det slim (`en_`/`ml_PP-OCRv3_det_slim` entries, built from `ch_PP-OCRv3_det_cml.yml`) | 1.1M | slim quantization + distillation detector | [2.x model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md); [docs v2.10 model list](https://www.paddleocr.ai/v2.10.0/en/ppocr/model_list.html) |
| PP-OCRv3 ONNX rec | 10.2 MiB | community | [oar-ocr docs/models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md) |
| **Paddle Lite `.nb` pipelines (det+cls+rec)** | PP-OCRv3 16.2M; **PP-OCRv3(slim) 5.9M**; PP-OCRv2 11M; **PP-OCRv2(slim) 4.6M** (v2.9 docs say 4.9M) | all built for the Paddle-Lite v2.10 branch; slim files are `ch_PP-OCRv3_det_slim_opt.nb`, `ch_PP-OCRv3_rec_slim_opt.nb` | [deploy/lite/readme.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/lite/readme.md); [legacy lite docs](https://paddlepaddle.github.io/PaddleOCR/main/en/version2.x/legacy/lite.html) |
| ch_PP-OCRv2_det_slim | 3.0M | old bcebos link `.../PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar` (not tested) | [docs v2.10 model list](https://www.paddleocr.ai/v2.10.0/en/ppocr/model_list.html); [release/2.4 model list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_ch/models_list.md) |
| PP-OCR v1 (2020) | 3.5M total: detector 1.4M (2.6M before pruning), direction classifier 500KB, recogniser 1.6M (900KB alphanumeric-only) | pruned (FPGM) + quantised (PACT) | [arXiv 2009.09941](https://arxiv.org/abs/2009.09941); [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2009.09941) |
| "PP-OCR mobile" in legacy benchmark page | 8.1 total (det 2.6, cls 0.9, rec 4.6), F-score 0.503 | later benchmark, not the paper table | [legacy benchmark docs](https://www.paddleocr.ai/main/en/version2.x/legacy/benchmark.html) |

#### Orientation classifiers

| Model | Size | Accuracy | Source |
|---|---|---|---|
| PP-LCNet_x0_25_textline_ori (text line 0/180 degrees) | 0.96 MB | 98.85% | [HF PP-LCNet_x0_25_textline_ori_onnx](https://huggingface.co/PaddlePaddle/PP-LCNet_x0_25_textline_ori_onnx) |
| PP-LCNet_x1_0_textline_ori | 6.5 MB | 99.42% (the HF README for this model shows 98.85% / 0.96 MB, apparently copying the x0_25 row — conflict) | [PaddleX textline orientation docs](https://paddlepaddle.github.io/PaddleX/3.0/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html); [HF PP-LCNet_x1_0_textline_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori) |
| PP-LCNet_x1_0_doc_ori (whole image 0/90/180/270) | 7 MB | 99.06% | [HF PP-LCNet_x1_0_doc_ori](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori) |
| ch_ppocr_mobile_v2.0_cls (legacy) | 1.38M; "slim quantized" entry listed at 2.1M | used as `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` in the Paddle Lite demos | [release/2.1 model list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/models_list_en.md); [On-device deployment docs v3.6.0](http://www.paddleocr.ai/v3.6.0/en/version3.x/inference_deployment/cross_platform/on_device_deployment.html) |

### Inferences
- Minimal pipeline sizes computed from the rows above (FP32 unless noted): **v6 tiny 6.3 MB** (1.9 + 4.4; third-party folder measurement 6.6 MB); **v6 small 30.0 MB** (9.6 + 20.4; folder 31.6 MB); **v5 mobile 20.7 MB** (4.7 + 16), 21.7 MB with the 0.96 MB text-line classifier; **v4 mobile 15.2 MB** (4.7 + 10.5); **v3 mobile 16.2 MB / v3 slim 5.9 MB** (Paddle Lite, incl. classifier); **v2 slim 4.6 MB**; **v1 3.5 MB**. The mapping of paddleocr.dev's five folder sizes to tiny/small/medium/v5-mobile/v5-server is my reading, supported by the sums.
- 1.5M params x 4 bytes is about 6 MB, so the published v6 tiny size is plain FP32. An INT8 build would in principle be about a quarter of that, but no official INT8/slim v6 (or v5/v4) artefact was found — see section 6.
- The classifier is optional for upright text; the smallest practical deployment is det + rec only.

### Gaps
- Exact `.onnx` file sizes for PP-OCRv6 tiny/small on Hugging Face were not captured (only Paddle "storage size" figures).
- No published `.nb` (Paddle Lite) file sizes for PP-OCRv4/v5 were found; the 3.x on-device doc fetches them by script.
- Per-model parameter counts for v3/v4/v5 mobile det and rec were not found (only "5M" for the v5 mobile system).
- The stand-alone PP-OCRv3 detector size (commonly quoted as about 3.6–3.8 MB) and the Chinese `ch_PP-OCRv3_det_slim` naming were not confirmed in extracts; only the 1.1M slim figure and the 16.2M/5.9M pipeline totals were.
- Whether the old bcebos download links for v1/v2/v3 slim models still resolve in Nov 2026 was not tested.

---

## 3. What accuracy and latency numbers are officially published for the mobile models, on what hardware, and are there any phone / ARM CPU numbers?

### Takeaway
Official numbers are almost entirely server/desktop: on Baidu's benchmarks **PP-OCRv6 tiny is the fastest PP-OCR ever shipped** (0.20 s/image end-to-end on Xeon+OpenVINO vs 0.78 s for PP-OCRv5_mobile; 0.96 s vs 5.82 s on Apple M4 with PaddlePaddle) with recognition accuracy level with v5 mobile (73.5 vs 73.7), while **v6 small** is about as fast as v5 mobile but far more accurate (81.3 vs 73.7). **I found no official phone-SoC benchmark for PP-OCRv3/v4/v5/v6**; the only phone numbers located are the 2020 PP-OCR v1 Snapdragon 855 figures and one third-party Pixel 8a GPU measurement for PP-OCRv5.

### Cited Findings

#### Accuracy — PP-OCRv6 in-house multi-scenario benchmark (same set for all rows)
- Detection Hmean: tiny 80.6, small 84.1, medium 86.2. — [HF blog](https://huggingface.co/blog/PaddlePaddle/pp-ocrv6); [PP-OCRv6 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- The small detector's 84.12% Hmean is stated to be +6.28 over the PP-OCRv5_mobile_det baseline (implying about 77.8 for v5 mobile on this set); tiny "clearly outperform[s] PP-OCRv5_mobile" on detection. — [arXiv 2606.13108 HTML](https://arxiv.org/html/2606.13108v1)
- Recognition weighted average: PP-OCRv5_mobile 73.7, PP-OCRv6_tiny 73.5, PP-OCRv6_small 81.3, PP-OCRv6_medium 83.2. — [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [paddleocr.dev/models](https://paddleocr.dev/models)
- Per-scenario (v5 mobile / v6 tiny / v6 small): Chinese handwriting 41.7 / 40.1 / 57.6; English handwriting 50.9 / 39.3 / 61.1; printed English 86.0 / 88.4 / 93.3. Tiny "falls far behind on handwriting". — [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [paddleocr.dev/models](https://paddleocr.dev/models)
- Other category deltas quoted for small vs v5 mobile: Screen 79.7 vs 57.6, Card 86.9 vs 81.7 (extract warns these came from different tables). — [HF PP-OCRv6_medium_rec card](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec)
- Official warning: v6 metrics use an internal multi-scenario set, v5/v4 metrics a "general evaluation set"; the two are not directly comparable. — [Text recognition docs](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html)

#### Accuracy — PP-OCRv5 / v4 / v3 (their own eval sets)
- PP-OCRv5_mobile_det Hmean 79.0 vs PP-OCRv4_mobile_det 63.8 at the same 4.7 MB size and speed (v5-era set). — [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- PP-OCRv5_mobile_rec: Chinese avg 81.29, English 66.00, Traditional Chinese 83.55, Japanese 54.65; PP-OCRv4_mobile_rec avg 78.74. — [Text recognition docs v3.1.1](https://www.paddleocr.ai/v3.1.1/en/version3.x/module_usage/text_recognition.html); [PaddleX 3.3 model list](https://paddlepaddle.github.io/PaddleX/3.3/en/support_list/models_list.html)
- PP-OCRv5 improves end-to-end by 13 percentage points over PP-OCRv4 on the internal multi-scenario set. — [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- PP-OCRv3 system Hmean 62.9 vs PP-OCRv2 57.6 (paper, 2022). — [arXiv 2206.03001](https://arxiv.org/abs/2206.03001)

#### Latency — official end-to-end pipeline (seconds per image; 200 images; includes image loading and pre/post-processing)

| Platform / engine | v6 tiny | v6 small | v6 medium | v5 mobile | v5 server | v4 mobile | Source |
|---|---|---|---|---|---|---|---|
| NVIDIA A100, PaddlePaddle | 0.13 | 0.25 | 0.29 | 0.25 | 0.32 | 0.14 | [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [Bilibili repost of the table](https://www.bilibili.com/read/cv50806852) |
| Intel Xeon, OpenVINO | 0.20 | 0.59 | 1.40 | 0.78 | 7.30 | not captured | [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [Bilibili repost](https://www.bilibili.com/read/cv50806852) |
| Apple M4 (ARM, laptop class), PaddlePaddle | 0.96 | 3.07 | 8.82 | 5.82 | >10 | not captured | [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [PaddlePaddle on X, "Tech Deep Dive Ep.4"](https://x.com/PaddlePaddle/status/2070492274404777991) |

- Official ratios: tiny is 6.1x faster than PP-OCRv5_mobile on Apple M4/PaddlePaddle and 3.9x on Xeon/OpenVINO; small "matches PP-OCRv5_mobile in latency on most platforms with higher accuracy" and is 1.9x faster on M4. — [PP-OCRv6 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html)
- A model-card/benchmark extract gives PP-OCRv6 tiny with ONNX Runtime as 0.29 (NVIDIA V100), 0.22 (Intel Xeon 8350C), 0.35 (Apple M4), averaged over 30 iterations after 15 warm-ups; units were not shown in the extract (most plausibly seconds per image). — [HF PP-OCRv6_tiny_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- Paper module-level latencies (server with Xeon Gold 6271C + Tesla V100; as I read the extract these are at 2048-px input): tiny 317.06 ms CPU-ONNX (lowest of all compared models); small 654.25 ms CPU-ONNX (slightly faster than PP-OCRv5_mobile) and 42.70 ms GPU vs 37.36 ms for PP-OCRv5_mobile; medium 2327.23 ms vs 3034.93 ms for PP-OCRv5_server. — [arXiv 2606.13108 HTML](https://arxiv.org/html/2606.13108v1)

#### Latency — PP-OCRv5 / v4 / v3 official
- PP-OCRv5 pipeline on Tesla V100 + Xeon Gold 6271C, PaddlePaddle 3.0.0, 200 images: CPU 1.75 s/image for v5_mobile (about 372 chars/s) vs 1.37 s for v4_mobile (about 444 chars/s) vs 4.34 s for v5_server (about 150 chars/s); GPU 0.62 s (v5_mobile), 0.29 s (v4_mobile), 0.74 s (v5_server). — [PP-OCRv5 docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md)
- Module level, "normal / high-performance mode" (model inference only, no pre/post): PP-OCRv5_mobile_det GPU 8.79 / 3.13 ms, CPU 51.00 / 28.58 ms (another docs version prints 57.77 / 28.15 ms); PP-OCRv5_mobile_rec GPU 5.43 / 1.46 ms, CPU 21.20 / 5.32 ms. — [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html); [Text recognition docs v3.1.1](https://www.paddleocr.ai/v3.1.1/en/version3.x/module_usage/text_recognition.html)
- PP-OCRv4_mobile_det GPU 9.87 / 4.17 ms, CPU 56.60 / 20.79 ms; PP-OCRv4_mobile_rec GPU 5.26 / 1.12 ms, CPU 17.48 / 3.61 ms. — [PaddleX 3.3 model list](https://paddlepaddle.github.io/PaddleX/3.3/en/support_list/models_list.html)
- PaddleX ONNX Runtime module timings (hardware not shown in extract; not phone numbers): PP-OCRv5_mobile_rec 2.05 ms inference / 4.91 ms end-to-end; PP-OCRv5_mobile_det 5.70 ms inference / 17.90 ms end-to-end. — [PaddleX 3.2 text recognition docs](https://paddlepaddle.github.io/PaddleX/3.2/en/module_usage/tutorials/ocr_modules/text_recognition.html)
- PP-OCRv3 system: 331 ms CPU, 87 ms T4 (vs PP-OCRv2 330 ms / 111 ms) — server CPU/GPU, 2022. — [arXiv 2206.03001](https://arxiv.org/abs/2206.03001)

#### Third-party desktop/server CPU measurements (not phones)
- On a 4-vCPU Xeon without GPU using ONNX Runtime: PP-OCRv6 small averaged about 0.9 s/page and medium about 2.9 s/page; tiny about 0.26 s with ONNX Runtime vs 0.39 s with PaddlePaddle 3.2.0 on a 4-core CPU (another extract of the same site gives 0.73 s ORT vs 0.90 s Paddle for one configuration). — [paddleocr.dev (third party)](https://paddleocr.dev/models)
- Same site: PP-OCRv5 *server* took 35.7 s/page under ONNX Runtime vs 5.2 s under PaddlePaddle — a warning that engine choice can dominate. — [paddleocr.dev](https://paddleocr.dev/)
- ncnn: PP-OCRv6 tiny on a VPS, 67.45 ms for a simple image and 420.28 ms for a complex one (x86 VPS, not ARM). — [ncnn-ocr on PyPI](https://pypi.org/project/ncnn-ocr/)
- ncnn C++ port README (hardware not shown in extract; snapshot roughly Aug 2025): PP-OCRv5 mobile about 92 ms / 106 MB peak memory on a simple image and about 1,063 ms / 292 MB on a dense one; PP-OCRv5 server about 3,949 ms / 1.545 GB and out-of-memory on the dense image. An older snapshot lists PP-OCRv3 at 387.80 ms / 101.3 MB and 5,213.68 ms / 273.6 MB. — [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP); [ecosyste.ms snapshot](https://awesome.ecosyste.ms/projects/github.com%2Favafly%2Fpaddleocr-ncnn-cpp)
- OpenVINO gives 2–4x over native PyTorch for PP-OCRv6 on a Ryzen 7 8845HS; a pure ONNX Runtime v6 implementation averages 282 ms per full page on Apple M4. — [Sekinal/ppocrv6-fast-cpu](https://github.com/Sekinal/ppocrv6-fast-cpu); [AIwork4me/ppocrv6_onnx](https://github.com/AIwork4me/ppocrv6_onnx)

#### Phone / ARM-SoC numbers actually found

| What | Device | Numbers | Reliability | Source |
|---|---|---|---|---|
| PP-OCR v1 detector (2020) | Snapdragon 855, batch 1, timing includes pre/post | 2.6M / 164 ms unpruned -> 1.4M / 133 ms after FPGM pruning | primary (paper), old | [arXiv 2009.09941](https://arxiv.org/pdf/2009.09941) |
| PP-OCR v1 classifier and recogniser (2020) | Snapdragon 855 | classifier PACT-quantised 0.85M -> 0.46M, about 25.9% faster; recogniser PACT: size -67.39%, inference +8.3% faster | paper via secondary summary | [ar5iv 2009.09941](https://ar5iv.labs.arxiv.org/html/2009.09941); [alphaXiv summary](https://www.alphaxiv.org/abs/2009.09941) |
| PP-OCRv5 det + rec via LiteRT, fully on GPU | Pixel 8a (Tensor G3) | about 9 ms per model on the OpenCL delegate (det 777/777 and rec 827/827 ops on GPU); 3 text lines end-to-end in 252 ms | third-party PR author's own measurement; v5 variant not stated in extract | [google-ai-edge/litert-samples PR #165](https://github.com/google-ai-edge/litert-samples/pull/165); [john-rocky/LiteRT-Models](https://github.com/john-rocky/LiteRT-Models) |
| PP-OCRv5 vs v4 | "Snapdragon 865", 736x1280 input | 17.5 ms vs 18.7 ms | **unreliable**: no framework/model/method given, and contradicts official docs where v5_mobile is slower than v4_mobile | [Baidu Cloud article](https://cloud.baidu.com/article/3693488) |
| RapidOCR on phones | unspecified | "15–30 FPS on modern phones" | **unsourced claim** | [brightcoding blog](https://prompts.brightcoding.dev/blog/rapidocr-the-lightning-fast-ocr-every-developer-needs) |

- FP16 caveat with phone relevance: an fp16-quantised PP-OCRv5 recogniser ran 21x slower than its fp32 parent on the XNNPACK CPU path of LiteRT.js/WASM. — [google-ai-edge/LiteRT issue #9663](https://github.com/google-ai-edge/LiteRT/issues/9663)

### Inferences
- Relative ordering for CPU-only phones should follow the official CPU tables: v6 tiny (fastest) < v4 mobile < v6 small ≈ v5 mobile; v6 tiny is roughly 4x faster than v5 mobile on CPU and v4 mobile is about 20–25% faster than v5 mobile. Absolute phone latency is not established by any source found.
- The Apple M4 + PaddlePaddle column (0.96–8.82 s) is not representative of ARM potential: ONNX Runtime on the same M4 is reported at about 0.35 (tiny) and 282 ms per page in a community port. On phones, the engine (ONNX Runtime/ncnn/LiteRT-GPU) will matter as much as the model.
- Rough plausibility only: a 2019-flagship Snapdragon 855 ran the v1 detector in about 0.13–0.16 s including pre/post; v6 tiny's detector has only 0.43M params, so sub-second full-pipeline latency on old mid-range phones is plausible for sparse images but **unmeasured**. Latency scales with the number of text lines (recogniser runs per line), as the ncnn 92 ms vs 1,063 ms pair shows.
- Peak memory of roughly 100–300 MB (ncnn, v5 mobile/v3) suggests the mobile tiers fit phones with 2–3 GB RAM; the server tiers (1.5 GB, OOM) do not.

### Gaps
- No official Snapdragon/MediaTek/Apple-A-series latency or memory table exists in the extracts for PP-OCRv3, v4, v5 or v6; the official Android/iOS deployment pages surfaced no benchmark numbers.
- No measured numbers on genuinely old/low-end phones (e.g. Cortex-A53/A55-class, 2–3 GB RAM) were found for any generation.
- The Japanese category row is unresolved: one extract attributed 89.8 to v6 tiny, but tiny officially does not support Japanese (see section 4), so that figure probably belongs to another tier.
- The paper's full latency table (including the truncated PP-OCRv5_mobile CPU-ONNX value) could not be read.
- Hardware for the Avafly ncnn timings and the PaddleX ONNX Runtime module timings was not visible.

---

## 4. Which languages/scripts do the mobile recognisers cover, are multilingual recognisers separate per-script models, and how big is each?

### Takeaway
PP-OCRv6 small covers 50 languages in one model (Simplified/Traditional Chinese, English, Japanese + 46 Latin-script languages); **tiny covers 49 (no Japanese)**; v6 does **not** cover Korean, Cyrillic, Arabic, Indic, Thai or Greek. For those scripts you still use the **separate per-script PP-OCRv5 mobile recognisers (about 7.5–14 MB each)**, which share the same script-agnostic detector.

### Cited Findings
- PP-OCRv6 medium/small support 50 languages with a single unified model: Simplified Chinese, Traditional Chinese, English, Japanese and 46 Latin-script languages (about 200 diacritical characters were added to the dictionary). — [PP-OCRv6 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [arXiv 2606.13108 HTML](https://arxiv.org/html/2606.13108v1)
- PP-OCRv6 tiny supports 49 languages, "excluding Japanese"; the OCR pipeline docs note `PP-OCRv6_tiny` "does not support japan". — [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [OCR pipeline docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md)
- PP-OCRv6 can only output characters from its built-in 18,708-character set; Cyrillic, Korean (Hangul) and Arabic are not covered, and the PP-OCRv5 language models (`cyrillic_`, `korean_`, `arabic_PP-OCRv5_mobile_rec`) are recommended for them. — [Ultralytics OCR guide](https://docs.ultralytics.com/guides/ocr)
- Downstream confirmation: Korean "is not actually supported in PP-OCRv6" even though an alias exists in RapidOCR (table dated 2026-07-28). — [Docling OCR engines page](https://docling-project.github.io/docling/concepts/OCR_native/)
- Conflict: one news write-up claims the 50 languages include Cyrillic, Korean and Arabic; this contradicts the official docs and the Ultralytics guide. — [artificialintelligenceherald.com](https://artificialintelligenceherald.com/posts/pp-ocrv6-hugging-face-50-language-ocr-2026)
- The main PP-OCRv5_mobile_rec handles Simplified Chinese, Traditional Chinese, English and Japanese plus handwriting in one model (per-category scores in section 2). — [PP-OCRv5 docs](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- PP-OCRv5 multilingual coverage is delivered as separate small per-script/per-language recognisers; release notes count 109 languages as of 3.3.0 (the multilingual docs page says 106). — [update log](https://paddlepaddle.github.io/PaddleOCR/main/en/update/update.html); [PP-OCRv5 multilingual docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)

#### Per-script PP-OCRv5 mobile recognisers

| Model | Added in | Paddle size (docs) | ONNX size (community) | Accuracy (own test set) | Source |
|---|---|---|---|---|---|
| latin_PP-OCRv5_mobile_rec | 3.1.0 | 14 MB | 7.7 MiB (older 7.9–8.0 MB) | 84.7 | [OCR pipeline docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md); [oar-ocr models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md) |
| korean_PP-OCRv5_mobile_rec | 3.1.0 | 14 MB | 12.8 MiB (older 13.4 MB) | 88.0 (PaddleX page: 90.45) | [OCR pipeline docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md); [PaddleX text recognition docs](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html) |
| eslav_PP-OCRv5_mobile_rec (East Slavic) | 3.1.0 | 14 MB | 7.9 MB | 85.8 | [PP-OCRv5 multilingual docs v3.1.1](http://www.paddleocr.ai/v3.1.1/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html); [crates.io oar-ocr 0.2.1](https://crates.io/crates/oar-ocr/0.2.1) |
| en_PP-OCRv5_mobile_rec | 3.2.0 | not captured | 7.5 MiB | +11% over main v5 model on English | [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases); [oar-ocr models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md) |
| th_PP-OCRv5_mobile_rec (Thai) | 3.2.0 | not captured | not captured | 82.68% | [README](https://github.com/PaddlePaddle/PaddleOCR) |
| el_PP-OCRv5_mobile_rec (Greek) | 3.2.0 | not captured | 7.5 MiB | 89.28% | [README](https://github.com/PaddlePaddle/PaddleOCR); [oar-ocr models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md) |
| cyrillic_PP-OCRv5_mobile_rec | 3.3.0 | 7.7 MB | 7.7 MiB | 80.27 | [OCR pipeline docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md) |
| arabic_PP-OCRv5_mobile_rec | 3.3.0 | not captured | 7.7 MiB | not captured | [oar-ocr models.md](https://github.com/GreatV/oar-ocr/blob/main/docs/models.md) |
| devanagari_PP-OCRv5_mobile_rec | 3.3.0 | 7.5 MB | 7.6 MiB | 84.96 | [OCR pipeline docs](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md) |
| ta_ / te_PP-OCRv5_mobile_rec (Tamil, Telugu) | 3.3.0 | not captured | not captured | not captured | [HF ta_PP-OCRv5_mobile_rec](https://huggingface.co/PaddlePaddle/ta_PP-OCRv5_mobile_rec/commit/848e822bf21cff1fce686f0b977214796d1cd9f3) |

- Older per-language recognisers (PP-OCRv3 era, still listed): korean_PP-OCRv3_mobile_rec 60.21 accuracy / 9.6 MB (an older page says 11.0M); en_PP-OCRv3_mobile_rec 70.69 / 17.3 MB as printed in the extract; en_PP-OCRv4_mobile_rec 70.39 / 7.5 MB. Korean improved from 60.21 (v3) to 88.0 (v5). — [Text recognition docs (GitHub main)](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md)
- The east-Slavic recogniser's output layer has 519 classes (useful sanity check when converting). — [i30mb1/PPOCRv5-Android](https://github.com/i30mb1/PPOCRv5-Android)

### Inferences
- A multi-script phone app pays about 7.5–14 MB per extra script on top of one shared detector; an "English/Latin-only" app can use v6 tiny (4.4 MB rec, Latin diacritics included) or `en_`/`latin_PP-OCRv5_mobile_rec` (7.5–7.7 MiB ONNX).
- The docs' "14 MB" for the Latin and East-Slavic models vs 7.7–7.9 MB ONNX files is a format/packaging discrepancy I could not resolve; treat ONNX figures as the on-device download size.
- No v6-generation Korean/Cyrillic/Arabic/Indic recognisers were found as of Nov 2026, so "v6 detector + v5 per-script recogniser" (or v5 det + v5 rec) is the practical route for those scripts; compatibility of mixing a v6 detector with v5 recognisers is expected (detectors only output boxes) but was not verified by any source.

### Gaps
- The explicit list of the 46 Latin-script languages and the full 106/109-language list were not captured.
- Sizes for the Thai, Tamil, Telugu and English (Paddle format) models and accuracy for Arabic were not found.
- Whether Baidu released additional v6 language packs after June 2026 is unknown (none surfaced).

---

## 5. What are the practical deployment routes on phones (Android and iOS) and how healthy is each as of 2026?

### Takeaway
The centre of gravity has moved from Paddle Lite to **ONNX Runtime**: PaddleOCR 3.7 ships an official ONNX Runtime **Android SDK + demo (minSdk 26)** and an official **iOS SwiftUI demo (iOS 16+, CPU/XNNPACK/Core ML providers)** for PP-OCRv6. **Paddle Lite is stagnant** (last release v2.14-rc, July 2024) though it remains the documented route for PP-OCRv3/v4/v5 `.nb` models and the only official one with 32-bit ARMv7 instructions. Community ports are active: RapidOCR (v6 default since v3.9.x, July 2026), ncnn (nihui and others, v3–v6), MNN, LiteRT (GPU), Flutter and React Native wrappers.

### Cited Findings

#### Official ONNX Runtime route (PP-OCRv6; current)
- The Android deployment doc (v3.7.0) describes an example "implementing mobile OCR inference using ONNX Runtime" with an SDK/demo split (`ppocr-sdk` library module + app module titled "PP-OCRv6 Demo"), supporting the PP-OCRv6 ONNX models stored at `models/det/inference.onnx` and `models/rec/inference.onnx`. — [Android deployment docs (latest)](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/cross_platform/android_deployment.html)
- Stated requirements: minSdk 26 (Android 8.0), ONNX Runtime 1.21.1, OpenCV 4.5.3; call `OpenCVUtils.init(context)` before `PaddleOCR.create(context)` and `release()` when done. — [Android deployment docs v3.7.0](https://www.paddleocr.ai/v3.7.0/en/version3.x/inference_deployment/cross_platform/android_deployment.html)
- The Android demo ships the small and tiny models. — [paddleocr.dev (third party)](https://paddleocr.dev/)
- Official iOS route: a sample app under `deploy/ios_demo`; SwiftUI UI calling the ONNX Runtime Objective-C API for det and rec; requires macOS with Xcode 16.0+ and a deployment target of iOS 16+; includes an execution-provider selector (CPU / XNNPACK / Core ML). — [iOS deployment docs (main)](http://www.paddleocr.ai/main/en/version3.x/inference_deployment/cross_platform/ios_deployment.html)
- ONNX Runtime itself supports Android API level 21+ (tested at API 28) and its Android build targets arm64-v8a (default) or armeabi-v7a. — [ONNX Runtime compatibility](https://onnxruntime.ai/docs/reference/compatibility.html); [ONNX Runtime Android build docs](https://onnxruntime.ai/docs/build/android.html)

#### Paddle Lite route (legacy; PP-OCRv2–v5)
- Paddle-Lite's latest GitHub release is v2.14-rc dated 2024-07-19; on PyPI the newest pre-release is 2.14rc0 (2024-08-15) and the "latest" stable label is still 2.12 (2022-11-22). — [Paddle-Lite releases](https://github.com/PaddlePaddle/Paddle-Lite/releases); [PyPI paddlelite](https://pypi.org/project/paddlelite/)
- A third-party snapshot checked 2026-11-13 lists the repo as not archived with the last push on 2026-04-27. — [GitHot snapshot](https://githot.net/projects/PaddlePaddle/Paddle-Lite)
- The PaddleOCR 3.x "On-Device Deployment" doc (present through at least the v3.6.0 docs) uses an Android shell demo from the `feature/paddle-x` branch of Paddle-Lite-Demo, defaults to PP-OCRv5_mobile, and also lists PP-OCRv4_mobile and PP-OCRv3_mobile det/rec plus a text-line orientation model; models are converted to `.nb` with the Paddle Lite `opt` tool (2.14rc build shown). — [On-device deployment docs v3.6.0](http://www.paddleocr.ai/v3.6.0/en/version3.x/inference_deployment/cross_platform/on_device_deployment.html); [PaddleX on-device deployment guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/on_device_deployment.html)
- That guide requires Android NDK >= r17c (r20b or later recommended; r20b is the tested version) and gives library-replacement steps for both **armeabi-v7a** and **arm64-v8a**. — [On-device deployment docs v3.6.0](http://www.paddleocr.ai/v3.6.0/en/version3.x/inference_deployment/cross_platform/on_device_deployment.html)
- The PaddleOCR 3.0 technical report (July 2025) lists on-device support via the Paddle-Lite runtime. — [arXiv 2507.05595](https://arxiv.org/abs/2507.05595)
- The older `deploy/lite` readme targets the Paddle-Lite 2.10 branch and provides ready `.nb` files for PP-OCRv3, PP-OCRv3(slim), PP-OCRv2 and PP-OCRv2(slim); its FAQ covers the frequent "kernel for 'io_copy' is not supported" error caused by an `opt` tool / runtime version mismatch. — [deploy/lite/readme.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/lite/readme.md)
- Known problems: PP-OCRv4 converted to `.nb` crashing with SIGSEGV in the Android demo on Paddle-Lite 2.13rc0; OpenCL deployment aborting with SIGABRT. — [Paddle-Lite issue #10467](https://github.com/PaddlePaddle/Paddle-Lite/issues/10467); [Paddle-Lite issue #9436](https://github.com/PaddlePaddle/Paddle-Lite/issues/9436)
- A third-party Flutter wrapper built on Paddle Lite v2.10 (v0.0.2) states that upstream "hasn't published mobile-optimized .nb files" for PP-OCRv5 and that its prebuilt Paddle Lite archive lacks 32-bit libraries (arm64-v8a only). — [pub.dev flutter_paddle_ocr](https://pub.dev/packages/flutter_paddle_ocr)
- An open-source Android app has open issues to replace its bundled Paddle Lite OCR runtime in order to qualify for F-Droid. — [MaxDac/fpink issue #27](https://github.com/MaxDac/fpink/issues/27); [issue #28](https://github.com/MaxDac/fpink/issues/28)

#### RapidOCR (ONNX-first re-packaging of PP-OCR)
- RapidOCR v3.9.0 made PP-OCRv6 small det + rec the default models (wheel grew from 15 MB to 29 MB) with ONNX Runtime and OpenVINO only; v3.9.1 added PaddlePaddle, PyTorch and MNN back-ends for v6; v3.9.2 (2026-07-21) is the latest 3.x tag found and adds TensorRT accuracy verification. — [RapidOCR v3.9.0 notes](https://newreleases.io/project/github/RapidAI/RapidOCR/release/v3.9.0); [v3.9.1](https://newreleases.io/project/github/RapidAI/RapidOCR/release/v3.9.1); [v3.9.2](https://newreleases.io/project/github/RapidAI/RapidOCR/release/v3.9.2)
- The RapidOCR model list covers PP-OCRv6 tiny/small/medium det and rec for ONNX Runtime, OpenVINO, MNN, Paddle and PyTorch. — [RapidOCR model list](https://rapidai.github.io/RapidOCRDocs/main/model_list/)
- RapidAI also maintains RapidOCRLite, a C++ PP-OCRv6 engine ("for smaller, faster"), and a .NET port (RapidOcrNet) supports v4/v5/v6 ONNX models. — [RapidAI/RapidOCRLite](https://github.com/RapidAI/RapidOCRLite); [BobLd/RapidOcrNet](https://github.com/BobLd/RapidOcrNet)
- Integration note (2025): the PP-OCRv5 mobile detector converted with paddle2onnx ran in RapidOCR with unchanged pre/post-processing from v4; the author recommends v4 for plain Chinese/English and v5 for mixed Chinese/English/Japanese/handwriting. — [Juejin write-up](https://juejin.cn/post/7509702916456251442)

#### ncnn
- `nihui/ncnn-android-ppocrv5`: a C++ Android app by the ncnn author (created May 2025, last pushed Oct 2025, about 127 stars), using the mobile det/rec by default; models are converted from ONNX with pnnx; fp16 must be disabled for the server recogniser. — [nihui/ncnn-android-ppocrv5](https://github.com/nihui/ncnn-android-ppocrv5); [ecosyste.ms listing](https://awesome.ecosyste.ms/projects/github.com%2Fnihui%2Fncnn-android-ppocrv5)
- `Avafly/PaddleOCR-ncnn-CPP`: C++ implementations of PP-OCRv3/4/5/6 on ncnn; notes that enabling FP16 "may be faster but can cause NaN results on some devices". — [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP)
- `ncnn-ocr` (PyPI) ships `v6_tiny`, `v6_small`, `v6_medium` profiles on the same engine. — [ncnn-ocr on PyPI](https://pypi.org/project/ncnn-ocr/)

#### MNN
- The Rust crate `ocr-rs` ships `PP-OCRv5_mobile_det.mnn` / `PP-OCRv5_mobile_rec.mnn` (plus FP16 variants described as faster and lighter, without figures); RapidOCR added an MNN back-end for v6 in v3.9.1. — [ocr-rs README](https://docs.rs/crate/ocr-rs/2.2.1/source/README.md); [RapidOCR v3.9.1](https://newreleases.io/project/github/RapidAI/RapidOCR/release/v3.9.1)

#### LiteRT / TFLite
- `iFleey/PPOCRv5-Android`: real-time Android OCR with PP-OCRv5 on LiteRT, FP16 models, OpenCL GPU with automatic CPU fallback; det input 1x640x640x3, rec 1x48xWx3; conversion needs static shapes (onnxsim), HardSigmoid decomposed into primitive ops and Resize switched from half_pixel to asymmetric for GPU-delegate compatibility. — [iFleey/PPOCRv5-Android](https://github.com/iFleey/PPOCRv5-Android)
- `i30mb1/PPOCRv5-Android`: recipe paddle2onnx -> simplify -> onnx2tf (FP16 TFLite), fixed 1x3x48x320 recogniser input. — [i30mb1/PPOCRv5-Android](https://github.com/i30mb1/PPOCRv5-Android)
- Google's `litert-samples` has a PP-OCRv5 on-device sample PR (fully GPU; Pixel 8a numbers in section 3) and a web demo PR; converted models are hosted by the author and the litert-community hub. — [litert-samples PR #165](https://github.com/google-ai-edge/litert-samples/pull/165); [litert-samples PR #417](https://github.com/google-ai-edge/litert-samples/pull/417)
- Conversion pitfalls: onnx2tf requires version pinning against TensorFlow and has open numerical-mismatch reports; FP16 is very slow on the XNNPACK CPU path. — [onnx2tf on PyPI](https://pypi.org/project/onnx2tf/); [onnx2tf issue #327](https://github.com/PINTO0309/onnx2tf/issues/327); [LiteRT issue #9663](https://github.com/google-ai-edge/LiteRT/issues/9663)

#### PP-OCRv5 export / conversion issues (ONNX)
- PP-OCRv5 ships in the new format without a `.pdmodel` file (uses `inference.json`), which broke older OpenVINO/paddle2onnx workflows; related threads cover converting v5 rec to ONNX, a mobile-export bug report, and multilingual models exported with PaddlePaddle 3.1 failing under 3.0. — [PaddleOCR discussion #16100](https://github.com/PaddlePaddle/PaddleOCR/discussions/16100); [issue #15300](https://github.com/PaddlePaddle/PaddleOCR/issues/15300); [issue #16503](https://github.com/PaddlePaddle/PaddleOCR/issues/16503); [issue #15908](https://github.com/PaddlePaddle/PaddleOCR/issues/15908)

#### Flutter, React Native, JS and other wrappers
- `flespark/paddle_ocr_native` (pub.dev `paddle_ocr_native` 0.1.0): offline PP-OCRv6 for Flutter on Android and iOS via ONNX Runtime + OpenCV; bundles only the v6 **small** Chinese/English models; iOS 16 minimum, arm64 devices, CocoaPods static linkage (no SPM); toolchain Flutter 3.44+, Java 17, Android SDK 36, Xcode 16+; explicitly not an official PaddlePaddle package. — [flespark/paddle_ocr_native](https://github.com/flespark/paddle_ocr_native); [pub.dev 0.1.0](https://pub.dev/packages/paddle_ocr_native/versions/0.1.0)
- `mencomao/flutter-paddle-ocr`: on-device OCR for Flutter using PP-OCR models, ONNX Runtime and paddleocr-js. One of these two Flutter plugins lists Android API 26 / arm64-v8a with ONNX Runtime 1.21.1, the other uses ONNX Runtime Android 1.27.0 (attribution ambiguous in the extracts). — [mencomao/flutter-paddle-ocr](https://github.com/mencomao/flutter-paddle-ocr)
- React Native: `react-native-executorch` (Software Mansion) documents a fused PP-OCRv6 pipeline with a `createPaddleOcr()` factory and has deprecated its earlier CRAFT-based detector; `@mirulkhanall/rn-paddle-ocr` runs PaddleOCR models through `onnxruntime-react-native` (RN >= 0.70, iOS + Android, Expo-compatible). — [react-native-executorch OCR docs](https://docs.swmansion.com/react-native-executorch/docs/extensions/optical-character-recognition); [pkgstats rn-paddle-ocr](https://www.pkgstats.com/pkg:@mirulkhanall/rn-paddle-ocr)
- Browser/JS: the official PaddleOCR.js SDK (since 3.5.0) runs PP-OCR in the browser, and the docs position v6 tiny for browser use; `ppu-paddle-ocr` is a community JS/TS package that also hosts pre-quantised INT8 `.ort` models. — [PaddleOCR wiki](https://github.com/PaddlePaddle/PaddleOCR/wiki); [PaddleOCR docs home](https://www.paddleocr.ai/latest/index.html); [JSR @snowfluke/ppu-paddle-ocr](https://jsr.io/@snowfluke/ppu-paddle-ocr)
- Rust: `oar-ocr` (ONNX) and `paddleocr_rs_onnx` (lists iOS 15.1). — [GreatV/oar-ocr](https://github.com/GreatV/oar-ocr); [lib.rs paddleocr_rs_onnx](https://lib.rs/crates/paddleocr_rs_onnx)
- Ente (photo app) published a write-up on porting PaddleOCR v5 to ONNX Runtime on Android, citing RapidOCR as proof that ONNX conversion works. — [Ente blog "The best open-source Android OCR"](https://community.ente.io/blog/ocr)

#### Licences
- PP-OCRv5 mobile det/rec models published on Hugging Face ship under Apache-2.0 (as reported in a search extract citing a community mirror of the v5 models); the PaddleOCR-VL card is also apache-2.0. — [HF fastforks/ppocrv5 (community mirror)](https://huggingface.co/fastforks/ppocrv5); [PaddleOCR-VL card mirror](https://cnb.cool/ai-models/PaddlePaddle/PaddleOCR-VL)

#### Route health summary (my synthesis of the cited items above)

| Route | PP-OCR generations | Health as of Nov 2026 | Min OS / ABI where stated | Key sources |
|---|---|---|---|---|
| Official ONNX Runtime Android SDK + demo | v6 (tiny, small) | new (June 2026), actively documented | minSdk 26; ABIs not stated | [Android docs](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/cross_platform/android_deployment.html) |
| Official iOS ONNX Runtime demo | v6 | new, documented | iOS 16+, Xcode 16+ | [iOS docs](http://www.paddleocr.ai/main/en/version3.x/inference_deployment/cross_platform/ios_deployment.html) |
| Paddle Lite (`.nb`) | v2, v3 (incl. slim), v4, v5 mobile | runtime stagnant since v2.14-rc (Jul 2024); doc still present in 3.6 docs | NDK >= r17c; armeabi-v7a + arm64-v8a | [On-device docs](http://www.paddleocr.ai/v3.6.0/en/version3.x/inference_deployment/cross_platform/on_device_deployment.html); [Paddle-Lite releases](https://github.com/PaddlePaddle/Paddle-Lite/releases) |
| RapidOCR (ORT / OpenVINO / MNN / Paddle / Torch) | v4, v5, v6 | active (v3.9.2, Jul 2026) | Python-first; mobile status not verified | [RapidOCR releases](https://github.com/RapidAI/RapidOCR/releases) |
| ncnn | v3–v6 | active community (nihui app last push Oct 2025) | not stated | [nihui repo](https://github.com/nihui/ncnn-android-ppocrv5); [Avafly repo](https://github.com/Avafly/PaddleOCR-ncnn-CPP) |
| MNN | v5, v6 | community + RapidOCR back-end | not stated | [ocr-rs](https://docs.rs/crate/ocr-rs/2.2.1/source/README.md) |
| LiteRT / TFLite (GPU) | v5 | community, conversion is fiddly | not stated | [iFleey repo](https://github.com/iFleey/PPOCRv5-Android); [litert-samples PR #165](https://github.com/google-ai-edge/litert-samples/pull/165) |
| Flutter / React Native wrappers | v5, v6 | young (0.x versions) | iOS 16 / arm64; Android API 26 | [paddle_ocr_native](https://pub.dev/packages/paddle_ocr_native/versions/0.1.0); [react-native-executorch](https://docs.swmansion.com/react-native-executorch/docs/extensions/optical-character-recognition) |

### Inferences
- Baidu's own phone demos for the newest generation use ONNX Runtime rather than Paddle Lite, and no evidence of PP-OCRv6 `.nb` models was found; Paddle Lite should be regarded as a maintenance-mode route useful mainly for v3/v4/v5 and for 32-bit ARMv7 devices.
- For **old phones** the official v6 Android SDK's minSdk 26 excludes Android 7 and earlier, and the Flutter plugins are arm64-only with iOS 16+. To reach older devices (Android 5–7, armeabi-v7a), the realistic options are: ONNX Runtime used directly (API 21+, armeabi-v7a build exists), ncnn, or Paddle Lite with v3/v5 `.nb` models.
- FP16 is a recurring hazard across ports (NaNs on some devices in ncnn, a 21x slowdown on XNNPACK CPU, disabled for some models by nihui); on old CPU-only phones FP32 (or true INT8) is the safer default.
- ONNX is the de facto interchange format for every non-Paddle route (ORT, ncnn via pnnx, MNN, LiteRT via onnx2tf, OpenVINO), and since v6 Baidu publishes official ONNX files, which removes the paddle2onnx step that caused many v5 issues.

### Gaps
- ABI list (armeabi-v7a?) and any benchmark for the official PP-OCRv6 Android SDK were not visible; whether the Paddle Lite on-device page survives in the 3.7 docs was not confirmed.
- Paddle-Lite-Demo repository activity (last commit on `feature/paddle-x`) was not checked; whether official iOS Paddle Lite OCR demos still build is unknown.
- Status of RapidOCR's Android/iOS projects (e.g. RapidOcrAndroidOnnx) and details of RapidOCRLite (platforms, binary size) were not retrieved before the search budget ran out.
- Licences of the runtimes (Paddle Lite, ONNX Runtime, ncnn, MNN) and of the PP-OCRv6 model cards were not confirmed in-session (PaddleOCR and its earlier model cards are Apache-2.0; the v6 arXiv paper text is CC BY 4.0).
- No size/latency figures were found in the Ente write-up extract.

---

## 6. How do slim / quantised (INT8) / distilled variants trade accuracy for size and speed, and what is the smallest viable configuration?

### Takeaway
Official INT8 "slim" models exist only for the 2.x generations (v1, v2, v3): **PP-OCRv3 slim is 5.9M for det+cls+rec (vs 16.2M)** and PP-OCRv2 slim 4.6M, usable only through Paddle Lite 2.10-era tooling. From v4 onward Baidu shipped no slim builds; with v6 it instead ships a **distilled FP32 tiny tier (about 6.3 MB)** that matches v5-mobile recognition accuracy, which makes it the smallest *current, maintained* configuration.

### Cited Findings
- PP-OCRv3 slim = quantisation + distillation: recogniser 12.4M -> 4.9M, detector slim 1.1M; Paddle Lite pipeline 16.2M -> 5.9M. PP-OCRv2 pipeline 11M -> 4.6M slim. — [2.x model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.en.md); [deploy/lite/readme.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/lite/readme.md)
- PP-OCR v1 (2020) reached 3.5M by FPGM pruning of the detector (size -46.2%, inference -18.9% on Snapdragon 855, "HMean slightly dropped") and PACT quantisation of the classifier (size -45.9%) and recogniser (size -67.39%, +8.3% speed; LSTM layers left unquantised). — [arXiv 2009.09941](https://arxiv.org/pdf/2009.09941); [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2009.09941)
- A secondary summary reports the v1 recogniser's accuracy moving from 0.6581 to 0.674 after PACT quantisation (i.e. no loss). — [alphaXiv summary](https://www.alphaxiv.org/abs/2009.09941)
- The slim toolchain is fragile today: a Dec 2024 discussion titled "Training Paddle Slim for mobile but it is too bad" reports poor results after PaddleSlim training plus `paddle_lite_opt` 2.14rc conversion; maintainers pointed to path and version-compatibility checks. — [PaddleOCR discussion #14350](https://github.com/PaddlePaddle/PaddleOCR/discussions/14350)
- PP-OCRv6 tiny's small size comes from architecture and distillation rather than quantisation: 0.43M-param detector, 1.1M-param recogniser with no neck, distilled from the medium model. — [HF Transformers docs](https://huggingface.co/docs/transformers/en/model_doc/pp_ocrv6_tiny_rec); [Text detection docs](http://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html)
- v6 tiny's trade-off vs v6 small: 6.3 MB vs 30 MB and roughly 3x faster on CPU (0.20 s vs 0.59 s Xeon/OpenVINO), but recognition 73.5 vs 81.3, detection 80.6 vs 84.1, much weaker handwriting (40.1 / 39.3 vs 57.6 / 61.1) and no Japanese. — [PP-OCRv6 docs](http://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html); [paddleocr.dev/models](https://paddleocr.dev/models)
- Community INT8 for v5: `ppu-paddle-ocr` dynamically quantises the PP-OCRv5 mobile recogniser's MatMul ops and reports unchanged accuracy (99.22% before and after, self-reported on its own test) with 20–50% speed-up on x86-64 VNNI CPUs and WebAssembly, but INT8 was *slower* than FP32 on Apple M-series, so it recommends FP32 on macOS ARM64; it publishes a conversion script and INT8 `.ort` files. — [JSR @snowfluke/ppu-paddle-ocr](https://jsr.io/@snowfluke/ppu-paddle-ocr); [HF snowfluke/ppu-paddle-ocr-models](https://huggingface.co/snowfluke/ppu-paddle-ocr-models)
- FP16 variants are common in community ports (LiteRT FP16 via onnx2tf; MNN FP16 in ocr-rs) but carry the CPU and stability caveats listed in section 5. — [i30mb1/PPOCRv5-Android](https://github.com/i30mb1/PPOCRv5-Android); [ocr-rs README](https://docs.rs/crate/ocr-rs/2.2.1/source/README.md); [LiteRT issue #9663](https://github.com/google-ai-edge/LiteRT/issues/9663)

### Inferences
- **Smallest/fastest shortlist for phones (my synthesis):**
  1. **PP-OCRv6 tiny** — about 6.3 MB, 1.5M params, about 4x faster than v5 mobile on CPU, accuracy about equal to v5 mobile on print; weak on handwriting; Chinese/English/Latin only (no Japanese, Korean, Cyrillic, Arabic). Runs on ONNX Runtime (official Android/iOS demos), ncnn, MNN. Best default for old/low-end phones when the script coverage fits.
  2. **PP-OCRv6 small** — about 30 MB, v5-mobile-class latency, clearly best accuracy among phone-sized models (81.3 rec avg); the choice when quality matters more than size.
  3. **PP-OCRv5 mobile + per-script recogniser** — 4.7 MB det + 7.5–16 MB rec; required for Korean, Cyrillic, Arabic, Devanagari, Thai, Greek, Tamil, Telugu.
  4. **PP-OCRv4 mobile** — about 15 MB, about 20–25% faster than v5 mobile on CPU, Chinese/English only, lower accuracy; now dominated by v6 tiny on both size and speed.
  5. **PP-OCRv3 slim (INT8, Paddle Lite 2.10)** — 5.9 MB including the classifier; the only official true-INT8 pipeline with ARMv7 instructions; 2022-level accuracy and a stagnant runtime. Marginally smaller than v6 tiny but older and less accurate, so only attractive where Paddle Lite is already integrated.
- A hypothetical INT8 v6 tiny (about 1.6–2 MB) would be the smallest possible configuration, but no published artefact or accuracy measurement exists; static INT8 on such a small model would need validation (the one community data point shows INT8 can even be slower on modern ARM cores).
- "Still downloadable and usable?": the v2/v3 slim `.nb` links are still printed in the current `deploy/lite/readme.md` on the main branch, which suggests they remain published, but I did not test the downloads.

### Gaps
- No accuracy figures for PP-OCRv3 slim vs non-slim (Hmean / rec accuracy drop) were found in extracts.
- No official slim/INT8 variants for PP-OCRv4, v5 or v6 were found, and no quantisation-accuracy study for v6.
- Download availability of legacy v1/v2/v3 slim models on Baidu's bcebos CDN in Nov 2026 is untested.
- No phone-measured speed-up for INT8 vs FP32 on ARM for any v3–v6 model was found.

---

## 7. Briefly: what is PaddleOCR-VL and how big is it? (pointer only)

### Takeaway
PaddleOCR-VL is Baidu's 0.9B-parameter vision-language document-parsing model line (not part of the PP-OCR det+rec family); at roughly gigabyte scale it is two to three orders of magnitude larger than the 6–30 MB PP-OCR mobile pipelines and is not a low-end-phone option.

### Cited Findings
- PaddleOCR-VL was announced on 2025-10-16 in PaddleOCR 3.3.0: a 0.9B-parameter model pairing a NaViT-style dynamic-resolution visual encoder with the ERNIE-4.5-0.3B language model via a light MLP projector; Apache-2.0; 109 languages. — [howaiworks.ai announcement summary](https://howaiworks.ai/blog/paddleocr-vl-announcement); [emergentmind topic page](https://www.emergentmind.com/topics/paddleocr-vl-0-9b); [PaddleOCR-VL card mirror](https://cnb.cool/ai-models/PaddlePaddle/PaddleOCR-VL)
- Successors: PaddleOCR-VL-1.5 (2026-01-29, PaddleOCR 3.4.0) and PaddleOCR-VL-1.6 (2026-05-28, PaddleOCR 3.6.0; same architecture as 1.5, drop-in replacement, 96.3% claimed on OmniDocBench v1.6). — [PaddleOCR docs home](https://www.paddleocr.ai/latest/en/index.html); [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases)
- One blog gives disk sizes of about 3.6 GB (FP16) and about 1.8 GB (INT8); no official figure was found. — [howaiworks.ai](https://howaiworks.ai/blog/paddleocr-vl-announcement)

### Inferences
- 0.9B parameters at 2 bytes each is about 1.8 GB, so the blog's "3.6 GB FP16" looks like an FP32 figure mislabelled; either way the model is roughly 60–600x the download of PP-OCRv6 small/tiny.

### Gaps
- Official on-disk size and any phone deployment data for PaddleOCR-VL were not retrieved (covered by another researcher).
