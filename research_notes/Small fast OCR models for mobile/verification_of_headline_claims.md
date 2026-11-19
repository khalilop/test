# Verification of headline claims: PP-OCRv6 and its phone deployment

All sources were read on 2026-11-19. Markdown, Gradle, JSON and source files were downloaded from raw.githubusercontent.com and read as exact text. github.com HTML pages (releases, tags, repository front pages, directory listings) were read through WebFetch, which returns a model-rendered reading of the page, so quotes from those pages are as that tool returned them. Not reachable in this session: the model-download hosts `paddle-model-ecology.bj.bcebos.com` and `paddleocr.bj.bcebos.com` (proxy 403), `api.github.com` (403), raw files of `microsoft/onnxruntime` (404), and the hosts excluded in the brief (arxiv.org, huggingface.co, www.paddleocr.ai).

## Verification table (all ten claims)

### Takeaway
Claims 1 to 7 hold as written, and Claim 10 holds except that the 2020 PP-OCR download now listed is a 3.3M set rather than 3.5M. Two claims need correction: RapidOCR's latest release is v3.10.0 (8 Oct 2026), not v3.9.2, and the react-native-executorch OCR timings were measured on a synthetic image that contains no text, so they are not a text-reading workload. Two open questions were also settled: the official Android demo declares no ABI filter and both pinned native libraries ship `armeabi-v7a` as well as `arm64-v8a`; the model tier behind the 420 ms "GM1900" sample is not stated anywhere.

### Cited Findings

In this summary table, rows taken from source tables are shown with their column separators replaced by commas or slashes; the exact rows are reproduced in the per-claim sections below.

| # | Claim (short) | Verdict | Main URL read | Supporting quote or row |
|---|---|---|---|---|
| 1 | PP-OCRv6 released 2026-06-11 with PaddleOCR 3.7.0; tiny 1.5M / small 7.7M / medium 34.5M params; PP-LCNetV4 backbone; "small" follows PP-OCRv5_mobile | CONFIRMED. Latest release today is still v3.7.0 (2026-06-11); nothing newer than PP-OCRv6. The docs spell the backbone "PPLCNetV4"; "small follows v5 mobile" is positioning, not a literal sentence | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/README.md and https://github.com/PaddlePaddle/PaddleOCR/tags | "2026.06.11: Release of PaddleOCR 3.7.0" ... "tiny (1.5M) / small (7.7M) / medium (34.5M) for edge, mobile, and server deployment."; tags page top entry "v3.7.0: Jun 11, 2026" |
| 2 | Sizes: v6 tiny 1.9 + 4.4 MB; v6 small 9.6 + 20.4 MB; v5 mobile 4.7 + 16 MB; v4 mobile 4.7 + 10.5 MB; text-line orientation 0.96 MB | CONFIRMED (all nine numbers). File format is not named by the docs; the size column sits beside the Paddle "Inference Model" download | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_detection.en.md and .../text_recognition.en.md | Detection column "Model Size (MB)": PP-OCRv6_small_det 9.6, PP-OCRv6_tiny_det 1.9, PP-OCRv5_mobile_det 4.7, PP-OCRv4_mobile_det 4.7. Recognition column "Model Storage Size (MB)": PP-OCRv6_small_rec 20.4, PP-OCRv6_tiny_rec 4.4, PP-OCRv5_mobile_rec 16, PP-OCRv4_mobile_rec 10.5 |
| 3 | Accuracy on the v6 set: rec tiny 73.5, small 81.3, v5 mobile 73.7; det Hmean tiny 80.6, small 84.1; handwriting about 40 (tiny) vs 58-61 (small) | CONFIRMED (handwriting: tiny 40.1 / 39.3, small 57.6 / 61.1) | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md | Recognition rows (W-Avg, HW-CN, HW-EN): PP-OCRv6_small 81.3, 57.6, 61.1; PP-OCRv6_tiny 73.5, 40.1, 39.3; PP-OCRv5_mobile 73.7, 41.7, 50.9 |
| 4 | Xeon + OpenVINO s/image: tiny 0.20, small 0.59, v5 mobile 0.78; Apple M4 figures; phone figures | CONFIRMED. Apple M4 rows exist; the table has no phone or Android row | same PP-OCRv6.en.md | Row Intel Xeon 8350C, OpenVINO: 1.40, 0.59, 0.20, 7.30, 0.78, 0.60 (columns medium, small, tiny, v5 server, v5 mobile, v4 mobile); row Apple M4, ONNX Runtime: 5.55, 1.29, 0.35, 7.20, 1.10, 1.02 |
| 5 | v6 small = 50 languages, tiny = 49 (no Japanese); no Korean, Cyrillic, Arabic, Indic, Thai, Greek; those need PP-OCRv5 per-script models of 7.5-14 MB | CONFIRMED | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/pipeline_usage/OCR.en.md | "`PP-OCRv6_tiny` does not support `japan`; set `PP-OCRv6_tiny_det`/`PP-OCRv6_tiny_rec` explicitly."; the PP-OCRv6 `lang` list holds ch, chinese_cht, en, japan and 46 Latin-script codes only |
| 6 | Android SDK: ONNX Runtime 1.21.1, minSdk 26; iOS 16+, CPU / XNNPACK / Core ML; 420 ms for 5 lines on "GM1900" | CONFIRMED for versions, OS floors, providers and timings. ABIs: not arm64-only, no ABI filter is declared and the pinned libraries ship arm64-v8a, armeabi-v7a, x86, x86_64. Model tier of the 420 ms sample: COULD NOT VERIFY (not stated) | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md | Rows minSdk, 26 (Android 8.0) and ONNX Runtime, 1.21.1; sample block: Device: GM1900 / OS: Android 9 / Lines: 5; Cold load: 158ms / Warmup: 3 / Measured: 10; Total pipeline 420.40; Detection (total) 348.70 |
| 7 | Latest Paddle-Lite release v2.14-rc (2024-07-19); repo not archived; no doc says Paddle Lite supports PP-OCRv6 | CONFIRMED | https://github.com/PaddlePaddle/Paddle-Lite/releases | "released this 19 Jul 07:52", labelled "Latest", assets stamped "2024-07-22T11:23:21Z"; only OCR line in the notes: "PaddleOCR：PP-OCRv4-mobile-rec、PP-OCRv4-mobile-det、picodet_layout_1x" |
| 8 | PP-OCRv6 is RapidOCR's default since v3.9.x; latest tag v3.9.2 (2026-07-21) | CORRECTED: default since v3.9.0 (23 Jun 2026); latest release today is v3.10.0 (8 Oct 2026), with alpha tags v5.0.0a2 to v5.0.0a5 (13 Oct 2026) above it | https://github.com/RapidAI/RapidOCR/releases and https://github.com/RapidAI/RapidOCR/tags | "This release sets the PP-OCRv6 det small detection model and rec small recognition model as the default models." (v3.9.0); tags page "v3.10.0: Oct 8, 2026" |
| 9 | RN ExecuTorch: PP-OCRv6-small 640x640, S20+ 425 ms CPU / 360 ms Vulkan, 556-888 MB peak; iPhone SE 3 167 ms CPU / 9.5 ms Core ML | Numbers CONFIRMED. Scope CORRECTED: the figure is the whole pipeline entry point, but the input image is a text-free synthetic scene, so it is not a text-recognition workload | https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/04-benchmarks/imported.json and .../apps/benchmarks/BENCHMARK_SPEC.md | Row: model "paddle-ppocrv6-small", backend "xnnpack", device "s20plus", "medianMs": 425.47, "peakMb": 801; spec: OCR input is "A fixed 512x512 or 640x640 scene: a vertical gradient, three solid ellipses, and a seeded dither" |
| 10 | Legacy INT8 slim pipelines: PP-OCRv3 slim 5.9 MB, PP-OCRv2 slim 4.6 MB, PP-OCR v1 3.5 MB; links still listed? | CONFIRMED for v3 slim 5.9M and v2 slim 4.6M, links still listed. PP-OCR v1: the 3.5M figure is in the changelog, but the download table lists "V2.0(slim)" at 3.3M, not a 3.5M set | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/lite/readme.md | Rows PP-OCRv3(slim), extra-lightweight chinese OCR optimized model, 5.9M, three download links, v2.10; and PP-OCRv2(slim), extra-lightweight chinese OCR optimized model, 4.6M, three download links, v2.10 |

### Inferences
- The earlier notes were accurate on almost every figure; the errors are of staleness (RapidOCR version) and of interpretation (what the ExecuTorch timing covers), not of transcription.
- For a report about old or low-end phones, the two findings with the most consequence are the 32-bit ARM availability of the official Android stack and the fact that no primary source gives a real-text end-to-end timing for PP-OCRv6 on a named phone with a named model tier.

### Gaps
- Link liveness of every model download could not be tested: both bcebos.com hosts are blocked in this session. "Links are listed" means only that the documentation still prints them.
- Dates on github.com release pages are shown without a year; years were taken from the tags pages or from asset timestamps where available.

## Claim 1: PP-OCRv6 release, tiers and backbone; latest PaddleOCR release today

### Takeaway
Confirmed. PaddleOCR 3.7.0 of 2026-06-11 introduced PP-OCRv6 in three tiers, and as of 2026-11-19 v3.7.0 is still the newest tag and the release marked "Latest"; nothing newer than PP-OCRv6 exists in the PP-OCR line.

### Cited Findings
- README, recent-updates block: "2026.06.11: Release of PaddleOCR 3.7.0" and under it "**Three tiers for all scenarios**: tiny (1.5M) / small (7.7M) / medium (34.5M) for edge, mobile, and server deployment." — [PaddleOCR README](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/README.md)
- "**PP-OCRv6** is the latest generation of the PP-OCR universal text recognition solution. Built on the newly designed PPLCNetV4 unified backbone, it offers tiny, small, and medium tiers targeting edge/IoT, mobile/desktop, and server scenarios respectively." and "A three-tier OCR model family spanning 1.5M to 34.5M parameters." — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- Relation to PP-OCRv5_mobile, as worded by the source: "PP-OCRv6_small matches PP-OCRv5_mobile in latency on most platforms with higher accuracy; 1.9× faster on Apple M4 PaddlePaddle (3.07s vs 5.82s)." — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- Tier descriptions in the detection table: small is "PP-OCRv6 small text detection model, balancing accuracy and efficiency, suitable for mobile deployment"; tiny is "PP-OCRv6 ultra-lightweight text detection model (0.43M params), suitable for edge/IoT scenarios". — [text_detection.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_detection.en.md)
- The pipeline default is the medium tier, not a mobile tier: "This pipeline supports PP-OCRv3, PP-OCRv4, PP-OCRv5, and PP-OCRv6 models, with the default model being **PP-OCRv6_medium** released with PaddleOCR 3.7." — [OCR.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/pipeline_usage/OCR.en.md)
- Releases page, first entry: v3.7.0, heading "2026.6.11 v3.7.0 released", shown as "11 Jun 12:09", labelled "Latest"; next entries v3.6.0 ("2026.5.28 v3.6.0 released") and v3.5.0 ("2026.4.21 v3.5.0 released"). — [PaddleOCR releases](https://github.com/PaddlePaddle/PaddleOCR/releases)
- Tags page, top to bottom: "v3.7.0: Jun 11, 2026", "api_sdk/go/v0.2.0: Jun 11, 2026", "v3.6.0: May 28, 2026"; no tag above v3.7.0. — [PaddleOCR tags](https://github.com/PaddlePaddle/PaddleOCR/tags)
- The only README news item newer than 3.7.0 is not an OCR model generation: "2026.07.22: HPD-Parsing is now available", described as "a lightweight vision-language model designed for high-throughput document parsing". — [PaddleOCR README](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/README.md)
- The v3.7.0 release notes do not mention an Android or iOS demo; their mobile wording is limited to the tiers line. — [v3.7.0 release](https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0)

### Inferences
- "PP-OCRv6_small succeeds PP-OCRv5_mobile" is a fair reading (same deployment tier, latency described as matching) but the documents never use the word successor; a report should say "the mobile-tier counterpart".
- The spelling in the primary docs is "PPLCNetV4" (also "LCNetV4"); "PP-LCNetV4" is the earlier notes' rendering.

### Gaps
- `docs/update/update.en.md` in the repository stops at "2025.08.21: Release of PaddleOCR 3.2.0", so it cannot be used to date later releases. — [update.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/update/update.en.md)
- The arXiv paper and the Hugging Face model cards cited in the earlier notes were not re-read (hosts excluded).

## Claim 2: model sizes and the file format they refer to

### Takeaway
All nine sizes are confirmed from the module tables. The docs never name the file format: the figures appear under "Model Size (MB)" or "Model Storage Size (MB)" next to the Paddle "Inference Model" download, in tables benchmarked with the `paddle_static` engine at FP32.

### Cited Findings

| Model | Size in the doc | Column header | Source |
|---|---|---|---|
| PP-OCRv6_tiny_det | 1.9 | Model Size (MB) | [text_detection.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_detection.en.md) |
| PP-OCRv6_small_det | 9.6 | Model Size (MB) | same |
| PP-OCRv6_medium_det | 59.4 | Model Size (MB) | same |
| PP-OCRv5_mobile_det | 4.7 | Model Size (MB) | same |
| PP-OCRv4_mobile_det | 4.7 | Model Size (MB) | same |
| PP-OCRv6_tiny_rec | 4.4 | Model Storage Size (MB) | [text_recognition.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_recognition.en.md) |
| PP-OCRv6_small_rec | 20.4 | Model Storage Size (MB) | same |
| PP-OCRv6_medium_rec | 73.3 | Model Storage Size (MB) | same |
| PP-OCRv5_mobile_rec | 16 | Model Storage Size (MB) | same |
| PP-OCRv4_mobile_rec | 10.5 | Model Storage Size (MB) | same |
| PP-LCNet_x0_25_textline_ori | 0.96 | Model Storage Size (MB) | [textline_orientation_classification.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/textline_orientation_classification.en.md) |
| PP-LCNet_x1_0_textline_ori | 6.5 | Model Storage Size (MB) | same |

- The same rows appear in the pipeline page. — [OCR.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/pipeline_usage/OCR.en.md)
- Each size cell follows a download cell of the form "Inference Model/Training Model", whose first link is a Paddle archive such as `.../official_inference_model/paddle3.0.0/PP-OCRv6_tiny_det_infer.tar`. — [text_detection.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_detection.en.md)
- The tables' timing note says: "The \"Standard Mode\" values correspond to the local `paddle_static` inference engine", and the mode table lists "FP32 precision / 8 threads" with "PaddleInference". — [text_detection.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_detection.en.md)
- ONNX builds are separate downloads with no size given, for example `PP-OCRv6_tiny_det_onnx_infer.tar`, to be unpacked into `inference.onnx` (plus `inference.yml` for recognition). — [android_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md)
- PP-OCRv6 rows carry no per-module timings: the GPU and CPU cells read "- / -". — [text_recognition.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_recognition.en.md)

### Inferences
- Pipeline totals by addition: v6 tiny 6.3 MB, v6 small 30.0 MB, v5 mobile 20.7 MB, v4 mobile 15.2 MB, v6 medium 132.7 MB.
- The sizes are consistent with FP32 weights: 1.5M, 7.7M and 34.5M parameters at 4 bytes each give 6.0, 30.8 and 138 MB against listed 6.3, 30.0 and 132.7 MB. They are therefore the unquantised Paddle inference models; an ONNX export of the same FP32 weights should be about the same size, but the docs do not print ONNX sizes.

### Gaps
- No sentence in the docs states the format, and the archive sizes could not be checked directly (bcebos.com blocked).
- Whether "MB" means 10^6 bytes or MiB is not stated.

## Claim 3: accuracy on the PP-OCRv6 evaluation set

### Takeaway
Confirmed exactly. Tiny equals PP-OCRv5_mobile on the weighted recognition average (73.5 vs 73.7) but is clearly weaker on handwriting; small is 7.6 points above v5 mobile.

### Cited Findings
- Recognition table header and rows, "Text recognition accuracy (%) on our in-house multi-scenario benchmark (15 categories)": — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)

```
| Model | W-Avg | HW-CN | HW-EN | Print-CN | Print-EN | TC | Anc. | JP | Conf. | Spec. | Gen. | Pin. | Art. | Indus. | Screen | Card |
| **PP-OCRv6_small** | **81.3** | 57.6 | 61.1 | 90.5 | 93.3 | 77.0 | 71.1 | 88.2 | 64.1 | 60.2 | 85.7 | 75.9 | 68.4 | 76.4 | 79.7 | 86.9 |
| **PP-OCRv6_tiny** | 73.5 | 40.1 | 39.3 | 86.7 | 88.4 | 65.0 | 68.4 | 89.8 | 52.3 | 57.1 | 78.0 | 65.4 | 54.7 | 62.1 | 71.2 | 80.5 |
| PP-OCRv5_mobile | 73.7 | 41.7 | 50.9 | 86.0 | 86.0 | 72.0 | 57.8 | 75.8 | 55.7 | 54.8 | 80.7 | 72.5 | 54.0 | 59.3 | 57.6 | 81.7 |
```

- Detection table, "Text detection Hmean (%) on our in-house multi-scenario benchmark (16 categories)", AVG column: PP-OCRv6_medium 86.2, PP-OCRv6_small 84.1, PP-OCRv6_tiny 80.6, PP-OCRv5_server 81.6, PP-OCRv5_mobile 75.2. — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- The module tables warn that scores are not comparable across generations: "PP-OCRv6 metrics are evaluated on an internal multi-scenario evaluation set, while PP-OCRv5/v4 metrics are based on a general evaluation set. As the evaluation sets differ, the metrics are not directly comparable." There PP-OCRv5_mobile_rec is listed at 81.29 and PP-OCRv5_mobile_det at 79.0. — [text_recognition.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_recognition.en.md)

### Inferences
- "58-61 for small" in the earlier notes is 57.6 (Chinese handwriting) and 61.1 (English handwriting) rounded; "about 40 for tiny" is 40.1 and 39.3.
- Tiny is also below v5 mobile on Traditional Chinese (65.0 vs 72.0) and pinyin (65.4 vs 72.5), and above it on screen text (71.2 vs 57.6) and ancient text (68.4 vs 57.8).

### Gaps
- The source is internally odd on Japanese: tiny is said to exclude Japanese, yet its "JP" recognition cell is 89.8, above small's 88.2. The page does not explain this.
- The evaluation set is in-house and unpublished; no third-party replication was read.

## Claim 4: CPU latency

### Takeaway
Confirmed: 0.20 / 0.59 / 0.78 s per image for tiny / small / PP-OCRv5_mobile on a Xeon 8350C with OpenVINO. Apple M4 figures are published; no phone or Android figure appears in this table.

### Cited Findings
- Section "3.3 End-to-End Inference Speed (s/image)": "Tested on 200 images (general + document scenes), including image I/O, pre/post-processing, and model inference." — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)

```
| Hardware | Backend | PP-OCRv6_medium | PP-OCRv6_small | PP-OCRv6_tiny | PP-OCRv5_server | PP-OCRv5_mobile | PP-OCRv4_mobile |
| Intel Xeon 8350C | PaddlePaddle | 2.05 | 0.79 | 0.32 | 2.04 | 0.80 | 0.62 |
| Intel Xeon 8350C | OpenVINO | 1.40 | 0.59 | 0.20 | 7.30 | 0.78 | 0.60 |
| Intel Xeon 8350C | ONNX Runtime | 3.31 | 0.61 | 0.22 | 6.36 | 0.61 | 0.49 |
| Apple M4 | PaddlePaddle | 8.82 | 3.07 | 0.96 | >10 | 5.82 | 5.65 |
| Apple M4 | ONNX Runtime | 5.55 | 1.29 | 0.35 | 7.20 | 1.10 | 1.02 |
```

- Source's own summary: "PP-OCRv6_tiny is the fastest model across all platforms: 6.1× over PP-OCRv5_mobile on Apple M4 PaddlePaddle (0.96s vs 5.82s), 3.9× on Intel Xeon OpenVINO (0.20s vs 0.78s), reaching 0.13s on A100." — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- The only phone timing in the official PaddleOCR docs is the Android sample output covered under Claim 6. The iOS page describes a benchmark pipeline but prints no results. — [android_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md); [ios_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/ios_deployment.en.md)

### Inferences
- With ONNX Runtime, the engine used by the official phone demos, small is not faster than PP-OCRv5_mobile: equal on the Xeon (0.61 vs 0.61) and about 17% slower on Apple M4 (1.29 vs 1.10). Tiny is about 2.8x (Xeon) to 3.1x (M4) faster than v5 mobile on that engine. The headline "3.9x" and "6.1x" ratios belong to OpenVINO and PaddlePaddle respectively.
- PP-OCRv4_mobile is the fastest of the older mobile models on ONNX Runtime in this table (0.49 s Xeon, 1.02 s M4).

### Gaps
- Thread counts, image sizes and the M4 device model are not given in the table.
- No ARM phone CPU figure exists in the PP-OCRv6 introduction or module pages.

## Claim 5: language coverage

### Takeaway
Confirmed. PP-OCRv6 small and medium cover Simplified and Traditional Chinese, English, Japanese and 46 Latin-script languages; tiny drops Japanese. Korean, Cyrillic, Arabic, Devanagari, Tamil, Telugu, Thai and Greek are reachable only through PP-OCRv5 per-script recognisers of 7.5 to 14 MB.

### Cited Findings
- "the medium/small tiers support 50 languages with a single unified model, including Simplified Chinese, Traditional Chinese, English, Japanese, and 46 Latin-script languages (tiny supports 49, excluding Japanese)." — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)
- Appendix table "Correspondence Between OCR Model Versions and Supported Languages", PP-OCRv6 row: `ch`, `chinese_cht`, `en`, `japan`, then 46 Latin-script codes (`af`, `az`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `es`, `et`, `eu`, `fi`, `fr`, `ga`, `gl`, `hr`, `hu`, `id`, `is`, `it`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`, `nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`, `sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`) plus the aliases `french` and `german`, ending "`PP-OCRv6_tiny` does not support `japan`; set `PP-OCRv6_tiny_det`/`PP-OCRv6_tiny_rec` explicitly." — [OCR.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/pipeline_usage/OCR.en.md)
- In the same table `korean`, `ru`, `be`, `uk`, `bg`, `ar`, `fa`, `ur`, `hi`, `mr`, `ne`, `ta`, `te`, `th` and `el` are absent from the PP-OCRv6 row and present in the PP-OCRv5 row. The parameter text adds: "not every `ocr_version` supports all `lang` options." — [OCR.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/pipeline_usage/OCR.en.md)
- PP-OCRv5 per-script recognisers and their "Model Storage Size (MB)": korean_PP-OCRv5_mobile_rec 14; latin_PP-OCRv5_mobile_rec 14; eslav_PP-OCRv5_mobile_rec 14; cyrillic_PP-OCRv5_mobile_rec 7.7; arabic_PP-OCRv5_mobile_rec 7.6; devanagari_PP-OCRv5_mobile_rec 7.5; te_PP-OCRv5_mobile_rec 7.5; ta_PP-OCRv5_mobile_rec 7.5; th_PP-OCRv5_mobile_rec 7.5; el_PP-OCRv5_mobile_rec 7.5; en_PP-OCRv5_mobile_rec 7.5. — [text_recognition.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/module_usage/text_recognition.en.md)
- Script-to-language mapping rows, for example `| eslav_PP-OCRv5_mobile_rec | Russian, Belarusian, Ukrainian, English |` and `| arabic_PP-OCRv5_mobile_rec | Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English |`; PP-OCRv5 as a whole is described as "covering 106 languages". — [PP-OCRv5_multi_languages.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md)

### Inferences
- An app that needs, say, Chinese plus Russian must ship a PP-OCRv6 recogniser and a second PP-OCRv5 recogniser (7.7 MB cyrillic or 14 MB eslav) and choose between them per image or per line; the detector can be shared.

### Gaps
- None of the per-script PP-OCRv5 models has a PP-OCRv6 equivalent announced in the pages read.
- Whether the official Android or iOS demo can load the per-script PP-OCRv5 models was not checked; both list only PP-OCRv6_small, PP-OCRv6_tiny and PP-OCRv5_mobile.

## Claim 6: official on-device deployment (Android and iOS)

### Takeaway
Confirmed: Android uses ONNX Runtime 1.21.1 with minSdk 26, iOS needs iOS 16+ and offers CPU, XNNPACK and Core ML providers. The Android build is not arm64-only: neither Gradle file restricts ABIs and both pinned native libraries include `armeabi-v7a`. The only phone timing is one "Sample output" block (420.40 ms, 5 lines, "GM1900", Android 9) whose model tier, thread count and image size are not stated.

### Cited Findings
- Requirements table rows: `| JDK | 17 |`, `| Kotlin | 2.1.0 |`, `| minSdk | 26 (Android 8.0) |`, `| ONNX Runtime | 1.21.1 |`, `| OpenCV | 4.5.3 |`. Supported models: PP-OCRv6_small, PP-OCRv6_tiny, PP-OCRv5_mobile, as ONNX files placed in `models/det/` and `models/rec/`. — [android_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md)
- Build files agree: `onnxruntime = "1.21.1"`, `opencv = "4.5.3"` with libraries `com.microsoft.onnxruntime:onnxruntime-android` and `com.quickbirdstudios:opencv`. — [libs.versions.toml](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/ppocr-android/gradle/libs.versions.toml)
- SDK module: `compileSdk = 35`, `minSdk = 26`; the whole file contains no `ndk`, `abiFilters` or `externalNativeBuild` block. — [ppocr-sdk/build.gradle.kts](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/ppocr-android/ppocr-sdk/build.gradle.kts)
- App module: `minSdk = 26`, `targetSdk = 35`; again no ABI filter. — [app/build.gradle.kts](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/ppocr-android/app/build.gradle.kts)
- The pinned ONNX Runtime package contains `jni/arm64-v8a/libonnxruntime.so` (17,908,496 bytes), `jni/armeabi-v7a/libonnxruntime.so` (13,039,136 bytes), `jni/x86/...` and `jni/x86_64/...`; the archive is 27,944,395 bytes, last modified 21 Apr 2025. Read from the archive's file index. — [onnxruntime-android-1.21.1.aar on Maven Central](https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.21.1/onnxruntime-android-1.21.1.aar)
- The pinned OpenCV package contains `libopencv_java4.so` and `libc++_shared.so` for the same four ABIs: arm64-v8a 17,109,960 and 6,238,016 bytes, armeabi-v7a 10,442,416 and 4,173,552 bytes; the archive is 54,063,091 bytes, last modified 24 Aug 2021. — [opencv-4.5.3.aar on Maven Central](https://repo1.maven.org/maven2/com/quickbirdstudios/opencv/4.5.3/opencv-4.5.3.aar)
- Sample benchmark block, introduced by "# Run benchmark (10 tests, 3 warmup)" and "# Sample output": — [android_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md)

```
║  Device: GM1900  |  OS: Android 9  |  Lines: 5                          ║
║  Cold load: 158ms  |  Warmup: 3  |  Measured: 10                        ║
| Stage                       |  Mean ms |    Stdev |      P90 |    Min ms|
| Total pipeline              |   420.40 |     6.37 |      427 |      413 |
|   Detection (total)         |   348.70 |     4.67 |      356 |      343 |
|     Preprocess              |    33.30 |     2.90 |       36 |       28 |
|     Inference               |   311.00 |     2.93 |      315 |      304 |
|     Postprocess             |     4.40 |     0.49 |        5 |        4 |
|   Recognition (total)       |    66.20 |     3.16 |       68 |       64 |
|     Preprocess              |     3.00 |     0.89 |        4 |        2 |
|     Inference               |    60.60 |     3.14 |       63 |       58 |
|     Postprocess             |     2.60 |     0.92 |        4 |        1 |
|   Pipeline overhead         |     5.50 |     0.50 |        6 |        5 |
```

- The benchmark test prints `Build.MODEL` as the device name and loads whatever model sits at the default asset paths, with default engine settings: `ocr = PaddleOCR.create(context, config, EngineConfig())`. It names no model tier. — [OCRBenchmarkTest.kt](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/ppocr-android/ppocr-sdk/src/androidTest/java/com/paddle/ocr/benchmark/OCRBenchmarkTest.kt)
- The wrapper script's header says "defaults: 10 iterations, 3 warmup" but its code sets `ITERATIONS="${1:-50}"` and `WARMUP="${2:-30}"`. — [run_benchmark.sh](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/ppocr-android/run_benchmark.sh)
- No model files are committed under the SDK's `src/main`; the listing shows only `java/com/paddle/ocr` and `AndroidManifest.xml`. — [ppocr-sdk/src/main listing](https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/ppocr-android/ppocr-sdk/src/main)
- iOS: "**macOS** with **Xcode 16.0+** (deployment target **iOS 16+**)"; presets "**`PP-OCRv6_small`** (default)", "**`PP-OCRv6_tiny`**", "**`PP-OCRv5_mobile`**"; "The **CPU** / **XNNPACK** / **Core ML** segmented control chooses which ONNX Runtime EP is preferred first when creating the session." — [ios_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/ios_deployment.en.md)
- iOS dependencies: `platform :ios, '16.0'`, `pod 'onnxruntime-objc', '~> 1.24'` with the comment "includes Core ML EP + XNNPACK EP pre-built", `pod 'OpenCV', '~> 4.3.0'`. The iOS demo therefore uses a newer ONNX Runtime line than the Android SDK. — [ios_demo/Podfile](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/ios_demo/Podfile)
- The iOS page can convert models to ORT format and has a benchmark script reporting latency and memory, but it prints no measured numbers. — [ios_deployment.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/inference_deployment/cross_platform/ios_deployment.en.md)

### Inferences
- Because no ABI filter is set, a default build packages all four ABIs carried by its dependencies, so the official SDK can be built for 32-bit ARM phones (API 26 and up). This is deduced from the build files and package contents; the PaddleOCR docs do not state it and no 32-bit run is documented.
- Native runtime payload per ABI, summed from the two packages: arm64-v8a about 14.6 MB compressed (41.4 MB unpacked), armeabi-v7a about 11.9 MB compressed (27.7 MB unpacked), before models. The runtime libraries are thus larger than the tiny models (6.3 MB) and comparable to the small models (30 MB).
- In the sample, detection is 82.9% of the pipeline (model inference alone 74.0%) and recognition costs about 13 ms per line (66.20 ms for 5 lines).

### Gaps
- Model tier, thread count, input resolution and the commercial identity of "GM1900" are not stated in the doc or the test code. The earlier notes' "Snapdragon-855-class phone" is not in the source and was not verified here.
- The default value of `numThreads` in `EngineConfig` was not read.
- The ONNX Runtime repository could not be opened, so the official minimum Android API of ONNX Runtime 1.21.1 itself remains unverified; only the package contents were checked.

## Claim 7: Paddle Lite status and PP-OCRv6

### Takeaway
Confirmed. The newest Paddle-Lite release is v2.14-rc of 19 July 2024, the repository is not archived but its last commit on `develop` is dated 22 May 2025, and no page read claims Paddle Lite support for PP-OCRv6. PaddleOCR removed its Paddle Lite on-device guide in v3.7.0, the release that introduced PP-OCRv6.

### Cited Findings
- Releases page, first entry v2.14-rc: "released this 19 Jul 07:52", labelled "Latest"; assets stamped "2024-07-22T11:23:21Z", source archives "2024-07-16T07:27:00Z". Its only OCR line: "PaddleOCR：PP-OCRv4-mobile-rec、PP-OCRv4-mobile-det、picodet_layout_1x". PP-OCRv5 and PP-OCRv6 are not mentioned on the page. — [Paddle-Lite releases](https://github.com/PaddlePaddle/Paddle-Lite/releases)
- Repository front page: no archive banner, status label "Public", default branch `develop`, "7,526 Commits"; the README does not mention PaddleOCR or any PP-OCR version. — [Paddle-Lite repository](https://github.com/PaddlePaddle/Paddle-Lite)
- Newest commits on `develop`: "Commits on May 22, 2025", "Commits on May 16, 2025", "Commits on Apr 24, 2025" ("[OHOS] Support for harmony os next (#10654)"). — [Paddle-Lite commits](https://github.com/PaddlePaddle/Paddle-Lite/commits/develop)
- At tag v3.6.0 PaddleOCR still had a Paddle Lite guide: "The following OCR models are supported in this guide: PP-OCRv3_mobile (cpu), PP-OCRv4_mobile (cpu), PP-OCRv5_mobile (cpu)" and "This guide has only validated the `PP-OCRv3_mobile`, `PP-OCRv4_mobile`, and `PP-OCRv5_mobile` models. Other models may not be compatible." It used the `feature/paddle-x` branch of Paddle-Lite-Demo and gave library paths for both `armeabi-v7a` and `arm64-v8a`. — [on_device_deployment.en.md at v3.6.0](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.6.0/docs/version3.x/inference_deployment/cross_platform/on_device_deployment.en.md)
- The same path returns 404 at tag v3.7.0 and on `main`. The current navigation lists under cross-platform deployment only "Android 部署", "iOS 部署" and "浏览器端部署"; the Paddle Lite page appears only in the version2.x block as "端侧部署: version2.x/legacy/lite.md". — [mkdocs.yml](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/mkdocs.yml)
- The legacy Lite readme covers only PP-OCRv3 and PP-OCRv2 models, built for "Paddle-Lite branch" v2.10. — [deploy/lite/readme.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/lite/readme.md)
- The PP-OCRv6 introduction's deployment section lists Windows, Linux and Mac, several server hardware targets, the high-performance inference plugin and serving; Paddle Lite is not named. — [PP-OCRv6.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)

### Inferences
- The newest generation with a documented Paddle Lite path is PP-OCRv5_mobile (FP32 `.nb`, guide present up to v3.6.0); PP-OCRv4 mobile is the newest named in Paddle Lite's own release notes. For PP-OCRv6 the vendor's on-device route is ONNX Runtime.
- "Not archived" should not be read as "maintained": about 18 months passed between the last commit seen (22 May 2025) and the read date.

### Gaps
- Absence of a statement was checked only in the pages listed here, not across all Paddle-Lite issues or third-party conversions.
- Whether PP-OCRv6's operators convert with the Paddle Lite `opt` tool was not tested and no source read addresses it.

## Claim 8: RapidOCR default model, latest version, Android and iOS demos

### Takeaway
Corrected. PP-OCRv6 small (detector and recogniser) became RapidOCR's default in v3.9.0 on 23 June 2026; the latest release today is v3.10.0 of 8 October 2026, and alpha tags v5.0.0a2 to v5.0.0a5 dated 13 October 2026 sit above it. RapidOCR has no current mobile demo: the main repository is Python-only, the Android demo repositories last changed their changelogs in 2022-2023 and predate PP-OCRv5, and the iOS folder is a call for a contributor.

### Cited Findings
- Releases page, top to bottom: v3.10.0 "SWHL released this 08 Oct 15:21" labelled "Latest"; v3.9.2 "21 Jul 10:53"; v3.9.1 "02 Jul 13:52"; v3.9.0 "23 Jun 02:29". The v3.9.0 notes say: "This release sets the PP-OCRv6 det small detection model and rec small recognition model as the default models." — [RapidOCR releases](https://github.com/RapidAI/RapidOCR/releases)
- Tags page with years: "v5.0.0a5: Oct 13, 2026", "v5.0.0a4: Oct 13, 2026", "v5.0.0a3: Oct 13, 2026", "v5.0.0a2: Oct 13, 2026", "v3.10.0: Oct 8, 2026", "v3.9.2: Jul 21, 2026", "v3.9.1: Jul 2, 2026", "v3.9.0: Jun 23, 2026". — [RapidOCR tags](https://github.com/RapidAI/RapidOCR/tags)
- v3.10.0 notes contain nothing about PP-OCRv6 defaults, Android or iOS; items include "add multilingual model routing and licensing metadata.", "lazily initialize OCR models" and "make OpenCV headless by default with optional GUI support". — [RapidOCR v3.10.0](https://github.com/RapidAI/RapidOCR/releases/tag/v3.10.0)
- Later 3.9.x notes: v3.9.1 "add PP-OCRv6 **PaddlePaddle** backend support", "support PP-OCRv6 **PyTorch** as the inference engine", "add PP-OCrv6 **MNN** backend support."; v3.9.2 "align the performance of the torch version of the PP-OCRv6 Rec tiny model". — [RapidOCR releases](https://github.com/RapidAI/RapidOCR/releases)
- Default configuration on `main`: under `Det:` and `Rec:` the values are `engine_type: "onnxruntime"`, `model_type: "small"`, `ocr_version: "PP-OCRv6"`; under `Cls:` they are `model_type: "mobile"`, `ocr_version: "PP-OCRv4"`. — [rapidocr/config.yaml](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/python/rapidocr/config.yaml)
- "This repository contains the **Python** component of RapidOCR. Components for other languages have been migrated to separate repositories." — [RapidOCR README](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/README.md)
- The `android` folder holds one line: "See [RapidOcrAndroidOnnx](https://github.com/RapidAI/RapidOcrAndroidOnnx) for details." — [RapidOCR android/README.md](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/android/README.md)
- The `ios` folder holds only: "需要志愿者捐赠，请直接进群联系，qq群号：887298230" and "A contributor is wanted." — [RapidOCR ios/README.md](https://raw.githubusercontent.com/RapidAI/RapidOCR/main/ios/README.md)
- RapidOcrAndroidOnnx: last changelog entry "2023-02-16 update v1.3.0" with "onnxruntime 1.14.0"; model note "det模型更换为效果更好的ch_PP-OCRv2_det_infer" (2021-09-30); "opencv 3.4.15(特别说明：4.5.3也可以支持，如果换成cv4，minSdkVersion必须改为24)"; pointer "新版：https://github.com/RapidAI/RapidOcrAndroidOnnxCompose". PP-OCRv5 and PP-OCRv6 are not mentioned. — [RapidOcrAndroidOnnx README](https://raw.githubusercontent.com/RapidAI/RapidOcrAndroidOnnx/main/README.md)
- RapidOcrAndroidOnnx front page: not archived, "61 Commits". — [RapidOcrAndroidOnnx repository](https://github.com/RapidAI/RapidOcrAndroidOnnx)
- RapidOcrAndroidOnnxCompose, the "new version": not archived, "40 Commits", most recent changelog entry "2022-11-12 update 0.1.0" with "onnxruntime 1.13.1"; no mention of PP-OCRv5 or PP-OCRv6. — [RapidOcrAndroidOnnxCompose repository](https://github.com/RapidAI/RapidOcrAndroidOnnxCompose)

### Inferences
- Release-page dates carry no year; 2026 follows from the tags page and from v3.9.0 depending on PP-OCRv6, which was released on 2026-06-11.
- RapidOCR today is a desktop and server Python wrapper for PP-OCRv6; for phones it offers model files (ONNX, MNN) rather than a maintained app or SDK.

### Gaps
- Last commit dates of the two Android demo repositories were not shown on the pages read; "stale" rests on their changelogs.
- Other RapidAI mobile-related repositories (for example a Lite or ncnn variant) were not checked.
- The v5.0.0 alpha tags were not opened, so their content and default model are unknown.

## Claim 9: react-native-executorch OCR benchmarks

### Takeaway
The numbers are confirmed, but they must not be presented as time to read a page of text. Each figure is the median of the whole OCR entry point on a 640x640 synthetic frame made of a gradient and three ellipses with no text, so the recogniser runs only on whatever regions the detector reports on that scene, and that count is not published. The model is a fused ExecuTorch `.pte` bundle of PP-OCRv6 small, measured in September 2026 on React Native ExecuTorch 0.11.0-dev.

### Cited Findings
- Version and date: "All numbers come from release builds of React Native ExecuTorch 0.11.0-dev (`main` shortly after v0.10.0), measured in September 2026, with supplemental quantized LLM, depth estimation, ColBERT, and inpainting runs in October 2026." The device table notes the Samsung Galaxy S20+ as "Exynos 990; historical results plus supplemental quantized LLM, depth, ColBERT, and inpainting measurements" and the iPhone SE (3rd gen) as "A15 Bionic with 4 GB RAM; historical standard-task measurements". — [01-benchmarks.mdx](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/04-benchmarks/01-benchmarks.mdx)
- OCR block: "**Input:** 640x640 RGBA frame." and "Powered by PP-OCRv6 inside a single fused model with two execution methods. First, DBNet text detection (`forward_detect`) letterboxes the 640x640 frame and detects oriented quadrilaterals ... Second, each quad is rectified to the text recognizer canvas and evaluated by SVTR (`forward_recognize`), followed by greedy CTC decoding to text strings." and "The model uses mixed precision: an int8 DBNet detector paired with an fp32 SVTR recognizer (kept in fp32 because int8 quantization degrades accuracy on the SVTR attention stack)." — [01-benchmarks.mdx](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/04-benchmarks/01-benchmarks.mdx)
- All 15 OCR rows use model `paddle-ppocrv6-small`. Rows for the two devices in the claim (median ms / mean ms / ExecuTorch ms / share % / peak MB / load ms / size MB): — [imported.json](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/04-benchmarks/imported.json)

| Device | Backend, precision | medianMs | meanMs | etMs | share | peakMb | loadMs | sizeMb |
|---|---|---|---|---|---|---|---|---|
| s20plus | xnnpack, default | 425.47 | 423.78 | 405.68 | 96 | 801 | 102 | 23 |
| s20plus | xnnpack, fp32 | 534.72 | 546.2 | 527.23 | 97 | 888 | 67 | 30 |
| s20plus | vulkan, default | 359.66 | 373.7 | 349.51 | 94 | 556 | 116 | 25 |
| iphonese | xnnpack, default | 167.33 | 166.23 | 161.5 | 97 | 372 | 38 | 23 |
| iphonese | xnnpack, fp32 | 198.25 | 196.57 | 191.57 | 97 | 457 | 30 | 30 |
| iphonese | coreml, default | 9.54 | 9.63 | 5.86 | 61 | 282 | 159 | 8 |

- The remaining nine rows (Pixel 10, Galaxy S26 Ultra, iPhone 17) match the table in the earlier notes value for value. — [imported.json](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/04-benchmarks/imported.json)
- Metric definitions: `meanMs` is "Mean end-to-end pipeline execution time (pre-processing, inference, and post-processing)."; `etMs` is "Time spent exclusively executing the model forward pass within the ExecuTorch runtime."; `peakMb` is "Sampled process footprint during inference, including the app baseline: total PSS on Android and phys_footprint on iOS. GPU memory may be under-counted; loading peaks and brief peaks are not captured reliably."; `sizeMb` is "Binary download file size of the compiled model bundle." `imported.json` holds "published rows that have no raw runs behind them". — [docs/src/data/benchmarks.ts](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/src/data/benchmarks.ts)
- The input for OCR is not a text image. Input table row: "Vision (classification, depth estimation, detection, segmentation, style transfer, OCR, keypoints, image embeddings): A fixed 512x512 or 640x640 scene: a vertical gradient, three solid ellipses, and a seeded dither". Also: "**20 timed iterations** per variant, after 3 untimed warmups. The reported figure is their median". — [BENCHMARK_SPEC.md](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/apps/benchmarks/BENCHMARK_SPEC.md)
- The generator confirms it: "Builds an RGBA/HWC image buffer holding a fixed synthetic scene: a vertical gradient, three solid ellipses, and a light deterministic dither", with `export const IMAGE_640 = syntheticImage(640, 640);`. — [apps/benchmarks/src/inputs.ts](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/apps/benchmarks/src/inputs.ts)
- "`pipeline.median`: The task's entry point end to end: preprocessing, execute, post-processing". The harness treats the number of OCR regions as workload: "If a pipeline decoded 14 tokens in one run and 19 in the other, or OCR recognized a different number of regions, it did different work". — [apps/benchmarks/README.md](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/apps/benchmarks/README.md)
- Pipeline code: one `model.execute('detect', [tDetInput], [tDetProbas]);` per image, then inside `for (const quad of quads)` one `model.execute('recognize', [tRecInput], [tRecProbas]);` per quad (more for very wide lines). — [paddleOcr.ts](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/packages/react-native-executorch/src/extensions/cv/tasks/paddleOcr.ts)
- Model files: "mixed-precision fused PP-OCRv6 models from the Software Mansion HuggingFace OCR Collection", family "PP-OCRv6 Small", "7.9 MB – 25.0 MB", backends "XNNPACK (CPU), Core ML (Apple), Vulkan (Android)", loaded as `models.ocr.PADDLE.PPOCRV6_SMALL.DEFAULT`; custom models are a `.pte` file plus a `charset.json`. Only the small tier is offered. — [OCR extension page](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/02-extensions/computer-vision/05-optical-character-recognition.md)
- `measured.json`, the file built from committed raw runs, contains no OCR rows (its tasks are inpainting, depth estimation, ColBERT embeddings and LLM). — [measured.json](https://raw.githubusercontent.com/software-mansion/react-native-executorch/main/docs/docs/04-benchmarks/measured.json)

### Inferences
- On a text-free scene the detector should return few or no regions, so these figures are best read as roughly one detector pass on a 640x640 frame plus pre- and post-processing, a lower bound for real documents. A real page adds one recogniser pass per text line. This is a deduction from the input definition; the region count for the published rows is not available.
- The same caution applies to the 9.5 ms Core ML figure on the iPhone SE: it shows the detector graph runs fast on Apple's accelerator, not that a page is read in 10 ms.
- The earlier notes' phrase "timings cover the end-to-end pipeline: detection, quad sorting, rectification, recognition, CTC decoding" is true of the code path measured but overstates the work done on this input.
- Peak memory of 556-888 MB on the S20+ includes the React Native app baseline, so it is not the model's own footprint.

### Gaps
- Exact file names of the `.pte` bundles and their Hugging Face revision could not be read (huggingface.co excluded; the generated API reference is not in the repository).
- Why the Core ML bundle is 8 MB against 23-30 MB for the others is not explained in the pages read.
- No raw run files exist for the OCR rows, so the number of detected regions, thermal state, exact build and the harness version that produced them cannot be audited.
- The published benchmarks page describes the OCR input only as "640x640 RGBA frame"; that the frame is the text-free gradient-and-ellipses scene comes from the benchmark app's specification and input generator in the same repository (the page does say so explicitly for object detection at the same size).
- The extension page's size range (7.9-25.0 MB) omits the 30 MB fp32 variant listed in the benchmark data; the discrepancy is unexplained.

## Claim 10: legacy INT8 "slim" pipelines and whether their links are still listed

### Takeaway
PP-OCRv3 slim at 5.9M and PP-OCRv2 slim at 4.6M are confirmed, and their Paddle Lite `.nb` download links are still printed in the main branch. For the 2020 PP-OCR, the 3.5M figure survives only in the changelog; the download table that remains lists a "V2.0(slim)" set at 3.3M.

### Cited Findings
- Model table in the Lite readme (link targets elided here as `…`; each is a paddleocr.bj.bcebos.com URL): — [deploy/lite/readme.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/lite/readme.md)

```
|Version|Introduction|Model size|Detection model|Text Direction model|Recognition model|Paddle-Lite branch|
|PP-OCRv3|extra-lightweight chinese OCR optimized model|16.2M|[download link](…)|[download link](…)|[download link](…)|v2.10|
|PP-OCRv3(slim)|extra-lightweight chinese OCR optimized model|5.9M|[download link](…)|[download link](…)|[download link](…)|v2.10|
|PP-OCRv2|extra-lightweight chinese OCR optimized model|11M|[download link](…)|[download link](…)|[download link](…)|v2.10|
|PP-OCRv2(slim)|extra-lightweight chinese OCR optimized model|4.6M|[download link](…)|[download link](…)|[download link](…)|v2.10|
```

- The slim rows point to `ch_PP-OCRv3_det_slim_infer.nb`, `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` and `ch_PP-OCRv3_rec_slim_infer.nb` (PP-OCRv3), and to `ch_PP-OCRv2_det_slim_opt.nb`, `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` and `ch_PP-OCRv2_rec_slim_opt.nb` (PP-OCRv2). — [deploy/lite/readme.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/lite/readme.md)
- The documentation-site copy, which is in the navigation under version2.x, prints only the two PP-OCRv3 rows (16.2M and 5.9M). — [docs/version2.x/legacy/lite.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version2.x/legacy/lite.en.md)
- INT8 is how the slim files are run: `./ocr_db_crnn system ch_PP-OCRv3_det_slim_opt.nb  ch_PP-OCRv3_rec_slim_opt.nb  ch_ppocr_mobile_v2.0_cls_slim_opt.nb  arm8 INT8 10 1  ./11.jpg  config.txt  ppocr_keys_v1.txt  True` and "# precision can be INT8 for quantitative model or FP32 for normal model." — [deploy/lite/readme.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/lite/readme.md)
- Legacy model list, section "4. Paddle-Lite Model": "This chapter lists OCR nb models with PP-OCRv2 or earlier versions." Its rows, in the same column layout as above, give PP-OCRv2 11.0M (branch v2.10), PP-OCRv2(slim) 4.6M (v2.10), PP-OCRv2 11.0M (v2.9), PP-OCRv2(slim) 4.9M (v2.9), "V2.0" with the description "ppocr_v2.0 extra-lightweight chinese OCR optimized model" 7.8M (v2.9) and "V2.0(slim)" 3.3M (v2.9), each with three download links. It also states: "It's suggested to optimize the quantization model using Paddle-Lite because the `INT8` format is used for the model storage and inference." — [docs/version2.x/ppocr/model_list.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version2.x/ppocr/model_list.en.md)
- Individual slim components in the same file: PP-OCRv3_mobile_rec_slim 4.9M; en_PP-OCRv3_det_slim 1.1M; ch_PP-OCRv2_rec_slim 9.0M; ch_ppocr_mobile_slim_v2.0_rec 6.0M; ch_ppocr_mobile_slim_v2.0_cls 2.1M. Each row offers "inference model", "trained model" and, for most, "nb model" links; the file does not say which of those formats the "model size" column measures. — [docs/version2.x/ppocr/model_list.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version2.x/ppocr/model_list.en.md)
- The 3.5M figure: "2020.9.19 Update the ultra lightweight compressed ppocr_mobile_slim series models, the overall model size is 3.5M, suitable for mobile deployment". — [update.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/update/update.en.md)
- The 2.x model list that the site navigation points to is a list of names only, without sizes or links (it includes `ch_PP-OCRv3_det_slim`, `ch_PP-OCRv2_det_slim`, `ch_ppocr_mobile_slim_v2.0_det`). — [docs/version2.x/legacy/model_list_2.x.en.md](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/version2.x/legacy/model_list_2.x.en.md)

### Inferences
- The smallest pipeline still downloadable from current documentation is "V2.0(slim)" at 3.3M, then PP-OCRv2 slim at 4.6M and PP-OCRv3 slim at 5.9M; all are Chinese-and-English `.nb` files tied to Paddle Lite 2.9 or 2.10 libraries from 2021-2022.
- The earlier notes' "PP-OCR v1 at 3.5M" matches the 2020 changelog; the 3.3M "V2.0(slim)" row is the PaddleOCR 2.0 repackaging of that system, so a report should not promise a 3.5M download.
- `docs/version2.x/ppocr/model_list.en.md` exists on `main` but is not in the site navigation, so the PP-OCRv2 and V2.0 slim links are easiest to cite from the GitHub files rather than from the documentation site.

### Gaps
- Whether any of these `.nb` links still download could not be tested (bcebos.com blocked).
- The PP-OCR 2020 paper's own size table (arXiv 2009.09941) was not re-read (host excluded).
- No accuracy or phone timing for the slim pipelines was re-verified in this pass; only sizes and link listings.
