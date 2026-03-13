# Annotated Bibliography — FinDocOCR Paper

**Last updated**: 2026-03-11 (rev 2 — added Poddar 2021, Hase 2003, Tan 2025; confirmed Dual LoRA via full PDF OCR)
**Coverage**: OCR/VLM models, document benchmarks, financial document AI, LoRA variants, table recognition

---

## Summary Table

| Key | Title | Venue/Year | One-Sentence Summary | Relevance |
|---|---|---|---|---|
| hunyuanocr | HunyuanOCR Technical Report | arXiv 2511.19575, 2025 | Tencent's 1B-parameter end-to-end VLM (ViT + 0.5B LLM) achieving SOTA on document parsing, text spotting, and info extraction. | **Baseline model** for our work; we fine-tune it with Dual LoRA |
| gotocr2 | General OCR Theory: Towards OCR-2.0 | arXiv 2409.01704, 2024 | 580M unified end-to-end OCR model supporting plain text, math, tables, charts, and formatted Markdown/LaTeX output. | Comparison system; demonstrates VLM trend in OCR |
| olmocr | olmOCR: Unlocking Trillions of Tokens | arXiv 2502.18443, 2025 | Allen AI's pipeline for linearizing PDF documents at scale for LLM training, handling complex layouts. | Related VLM-OCR pipeline; outputs plain markdown, ignores formatting attributes |
| olmocr2 | olmOCR 2: Unit Test Rewards for Document OCR | arXiv 2510.19817, 2025 | Qwen2.5-VL-7B fine-tuned with RLVR (unit-test rewards) achieving SOTA on olmOCR-Bench for math, tables, and multi-column layouts. | Related work on RLVR for OCR; does not address strikethrough/underline/color |
| mistralOCR | Mistral OCR (commercial) | Mistral AI, March 2025 | Proprietary API for high-fidelity document parsing (tables, equations, handwriting) claiming SOTA on commercial benchmarks. | Comparison system; proprietary, no arXiv, does not output text formatting attributes |
| paddleocrVL | PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model | arXiv:2510.14528, 2025 | 0.9B VLM for multilingual document parsing covering text, tables, formulas, and complex layouts, from Baidu PaddlePaddle. | Comparison system; covers general document parsing but does not address strikethrough, underline, or colored text |
| omnidocbench | OmniDocBench: Benchmarking Diverse PDF Document Parsing | arXiv 2412.07626, CVPR 2025 | 1,355-page benchmark with 15 block-level and 4 span-level annotation types for PDF parsing evaluation. | Key related benchmark; **confirmed does not annotate** strikethrough, underline, or text color |
| li2020docbank | DocBank: A Benchmark Dataset for Document Layout Analysis | COLING 2020 | 500K document pages with token-level layout annotations across 12 semantic categories derived from LaTeX source. | Related layout benchmark; scientific documents; no formatting attribute annotations |
| pfitzmann2022doclaynet | DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation | KDD 2022 | 80K human-annotated pages across 11 document categories (including financial reports) with 11 layout classes. | Related layout benchmark; includes financial documents; no strikethrough/underline/color annotations |
| synfintabs | SynFinTabs: Synthetic Financial Tables for Information and Table Extraction | arXiv 2412.04262, 2024 | 100,000 synthetic financial table images with HTML/JSON/CSV annotations and bounding boxes. | We use 5,200 samples (200 eval + 5,000 replay) from this dataset; related work on financial table recognition |
| fintabnet | FinTabNet: Global Table Extractor (GTE) | WACV 2021 | Financial table dataset from Fortune 500 earnings with cell-structure annotations; companion to the GTE detection framework. | Direct predecessor to FinTabNet.c; establishes financial table recognition as a distinct challenge |
| fintabnetc | FinTabNet.c: Aligning Benchmark Datasets for Table Structure Recognition | ICDAR 2023 | Curated and corrected version of FinTabNet with 2,064 verifiable annotation pairs. | We use FinTabNet.c test set for baseline TEDS evaluation (Task 1.1 of project plan) |
| pubtables1m | PubTables-1M: Towards Comprehensive Table Extraction | arXiv 2110.00061, CVPR 2022 | ~1M scientific table images with detection, structure recognition, and functional analysis annotations; introduces GriTS metric. | Establishes large-scale table recognition; scientific domain (not financial); structural comparison to our EDGAR pipeline |
| teds | Image-based Table Recognition: Data, Model, and Evaluation | arXiv 1911.10683, ECCV 2020 | Introduces TEDS metric (Tree Edit Distance Similarity) and PubTabNet dataset; proposes EDD encoder-dual-decoder architecture. | **TEDS is our primary table evaluation metric**; PubTabNet is a related dataset |
| lora | LoRA: Low-Rank Adaptation of Large Language Models | arXiv 2106.09685, ICLR 2022 | Freezes pre-trained weights and injects trainable rank-decomposition matrices, reducing trainable parameters by ~10,000x vs full fine-tuning. | **Foundation of our training method**; standard LoRA is our ablation baseline |
| duallora | Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates | arXiv 2512.03402, 2025/2026 | Decomposes LoRA weight updates into explicit magnitude (ReLU) and direction (Sign) groups, improving NLU and commonsense reasoning tasks. | **Our proposed training method**; we are first to apply Dual LoRA to VLM document OCR |
| qlora | QLoRA: Efficient Finetuning of Quantized LLMs | arXiv 2305.14314, NeurIPS 2023 | Enables fine-tuning of 65B models on a single 48GB GPU via 4-bit NF4 quantization combined with LoRA adapters. | Related PEFT method; cited for context on the LoRA variant landscape |
| dora | DoRA: Weight-Decomposed Low-Rank Adaptation | arXiv 2402.09353, ICML 2024 (Oral) | Decomposes pre-trained weights into magnitude and direction for fine-tuning, improving over LoRA on LLaMA, LLaVA, and VL-BART. | Closest prior work to Dual LoRA (also magnitude/direction); key distinction: DoRA decomposes the **full pre-trained weight** $W_0$; Dual LoRA decomposes the **update** $\Delta W$ |
| strikethrough2014 | An Approach of Strike-Through Text Identification from Handwritten Documents | IEEE ICFHR 2014, pp. 643–648 | Binary detection of struck-out words in unconstrained offline **handwritten** document images to prevent garbage OCR output. | Only prior work on strikethrough detection; out of scope on all dimensions: handwritten, binary task, no financial domain, classical methods |
| poddar2021 | Detection and Localisation of Struck-Out-Strokes in Handwritten Manuscripts | ICDAR 2021 Workshops, LNCS 12917, pp. 98–112 | Stroke-level localization of struck-out content in handwritten historical manuscripts using image analysis. | Second strikethrough-adjacent paper; handwritten manuscripts only; not applicable to printed financial documents |
| hase2003 | Color segmentation for text extraction | IJDAR vol. 6, pp. 271–284, 2003 | Uses color channel segmentation to extract character regions from color documents for improved OCR accuracy. | Most relevant prior work on color in OCR; but targets character extraction quality, not semantic color as output |
| tan2025 | Fine-Tuning VLMs for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports | arXiv:2508.05669, 2025 | Qwen2.5-VL-7B fine-tuned with LoRA on 2,152 pairs for financial table Markdown extraction; achieves 96.53% TEDS. | Closest prior work to our table contribution; single-feature (tables only); no strikethrough/underline/color |
| finqa | FinQA: A Dataset of Numerical Reasoning over Financial Reports | EMNLP 2021 | QA dataset requiring multi-step numerical reasoning over financial documents (earnings reports); 8,281 QA pairs. | Cited for motivation: financial documents contain structured data that current NLP systems struggle with; QA not OCR |
| tatqa | TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance | ACL 2021 | 16,552 QA pairs over financial reports requiring reasoning over tables and text together. | Cited for motivation; establishes financial table understanding as challenging; QA not OCR |

---

## Detailed Entries

### OCR/VLM Models

#### hunyuanocr — HunyuanOCR
- **Title**: HunyuanOCR Technical Report
- **Authors**: Tencent Hunyuan Vision Team
- **Venue/Year**: arXiv:2511.19575, November 2025
- **Architecture**: Native-resolution ViT (0.4B, adaptive patching) + MLP connector + Hunyuan-0.5B LLM; end-to-end trained with RL optimization
- **Key results**: SOTA on text spotting, document parsing, and information extraction across multiple benchmarks
- **Relevance**: This is the model we fine-tune. We use its VLM architecture (ViT + LLM) and apply Dual LoRA to its LLM layers (and optionally ViT).
- **Features supported**: Tables (HTML), formulas (LaTeX), general text (Markdown). **Does not output** strikethrough, underline, or text color.
```bibtex
@article{hunyuanocr2025,
  title   = {{HunyuanOCR} Technical Report},
  author  = {{Tencent Hunyuan Vision Team}},
  journal = {arXiv preprint arXiv:2511.19575},
  year    = {2025},
  note    = {arXiv:2511.19575}
}
```

#### gotocr2 — GOT-OCR 2.0
- **Title**: General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model
- **Authors**: Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang, Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao, Jianjian Sun, Yuang Peng, Chunrui Han, Xiangyu Zhang
- **Venue/Year**: arXiv:2409.01704, September 2024
- **Architecture**: 580M model with high-compression encoder + long-context decoder; unified across scene/document OCR
- **Key results**: Handles plain texts, math/molecular formulas, tables, charts, sheet music, geometric shapes; outputs Markdown/LaTeX
- **Relevance**: Comparison system; demonstrates the trend toward unified OCR-as-generation models. Does not address strikethrough/underline/color.
```bibtex
@article{wei2024gotocr2,
  title   = {General {OCR} Theory: Towards {OCR}-2.0 via a Unified End-to-end Model},
  author  = {Wei, Haoran and Liu, Chenglong and Chen, Jinyue and Wang, Jia and Kong, Lingyu and Xu, Yanming and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Peng, Yuang and Han, Chunrui and Zhang, Xiangyu},
  journal = {arXiv preprint arXiv:2409.01704},
  year    = {2024},
  note    = {arXiv:2409.01704}
}
```

#### olmocr — olmOCR
- **Title**: olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models
- **Authors**: Jake Poznanski, Jon Borchardt, Jason Dunkelberger, Regan Huff, Daniel Lin, Aman Rangapur, Christopher Wilhelm, Kyle Lo, Luca Soldaini
- **Venue/Year**: arXiv:2502.18443, February 2025
- **Relevance**: Demonstrates large-scale PDF linearization for LLM training; output is plain Markdown without formatting attributes. No strikethrough/underline/color.
```bibtex
@article{poznanski2025olmocr,
  title   = {{olmOCR}: Unlocking Trillions of Tokens in {PDFs} with Vision Language Models},
  author  = {Poznanski, Jake and Borchardt, Jon and Dunkelberger, Jason and Huff, Regan and Lin, Daniel and Rangapur, Aman and Wilhelm, Christopher and Lo, Kyle and Soldaini, Luca},
  journal = {arXiv preprint arXiv:2502.18443},
  year    = {2025},
  note    = {arXiv:2502.18443}
}
```

#### olmocr2 — olmOCR 2
- **Title**: olmOCR 2: Unit Test Rewards for Document OCR
- **Authors**: Jake Poznanski, Luca Soldaini, et al. (Allen Institute for AI)
- **Venue/Year**: arXiv:2510.19817, October 2025
- **Architecture**: Qwen2.5-VL-7B fine-tuned with RLVR (binary unit-test rewards)
- **Key results**: 82.4 on olmOCR-Bench; largest gains in math formulas, table parsing, multi-column
- **Relevance**: Most related VLM-OCR training approach; uses RL rather than LoRA; does not target formatting attributes
```bibtex
@article{poznanski2025olmocr2,
  title   = {{olmOCR} 2: Unit Test Rewards for Document {OCR}},
  author  = {Poznanski, Jake and Soldaini, Luca and others},
  journal = {arXiv preprint arXiv:2510.19817},
  year    = {2025},
  note    = {arXiv:2510.19817}
}
```

---

### Document Benchmarks

#### omnidocbench — OmniDocBench
- **Title**: OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations
- **Authors**: Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, Jin Shi, Fan Wu, Pei Chu, Minghao Liu, Zhenxiang Li, Chao Xu, Bo Zhang, Botian Shi, Zhongying Tu, Conghui He
- **Venue/Year**: arXiv:2412.07626, CVPR 2025
- **Dataset**: 1,355 PDF pages; 9 document types; 4 layout types; 3 language types
- **Annotations**: 15 block-level types (title, text_block, table, figure, equation, header, footer, etc.); 4 span-level types (text_span, equation_inline, equation_ignore, footnote_mark)
- **Critical fact**: Does NOT annotate strikethrough, underline, or text color (verified from official README)
- **Relevance**: Primary motivation for our work — the most comprehensive existing benchmark misses all four of our target features
```bibtex
@inproceedings{ouyang2025omnidocbench,
  title     = {{OmniDocBench}: Benchmarking Diverse {PDF} Document Parsing with Comprehensive Annotations},
  author    = {Ouyang, Linke and Qu, Yuan and Zhou, Hongbin and Zhu, Jiawei and Zhang, Rui and Lin, Qunshu and Wang, Bin and Zhao, Zhiyuan and Jiang, Man and Zhao, Xiaomeng and Shi, Jin and Wu, Fan and Chu, Pei and Liu, Minghao and Li, Zhenxiang and Xu, Chao and Zhang, Bo and Shi, Botian and Tu, Zhongying and He, Conghui},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  note      = {arXiv:2412.07626}
}
```

#### li2020docbank — DocBank
- **Title**: DocBank: A Benchmark Dataset for Document Layout Analysis
- **Authors**: Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhenghao Liu, Ming Zhou
- **Venue/Year**: COLING 2020
- **One-sentence summary**: 500K document page images with token-level layout annotations derived from LaTeX source, covering 12 semantic categories including text, title, figure, table, and equation.
- **Relevance**: Related document layout analysis benchmark; scientific documents only; does not annotate strikethrough, underline, or text color
```bibtex
@inproceedings{li2020docbank,
  title     = {{DocBank}: A Benchmark Dataset for Document Layout Analysis},
  author    = {Li, Minghao and Xu, Yiheng and Cui, Lei and Huang, Shaohan and Wei, Furu and Liu, Zhenghao and Zhou, Ming},
  booktitle = {Proceedings of the 28th International Conference on Computational Linguistics (COLING)},
  year      = {2020}
}
```

#### pfitzmann2022doclaynet — DocLayNet
- **Title**: DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation
- **Authors**: Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, Peter Staar
- **Venue/Year**: KDD 2022
- **One-sentence summary**: 80K human-annotated document pages across 11 document categories (financial reports, patents, manuals, etc.) with 11 layout classes for bounding-box-level layout segmentation.
- **Relevance**: Related layout segmentation benchmark; includes financial reports as a document category; does not annotate text formatting attributes (strikethrough, underline, color)
```bibtex
@inproceedings{pfitzmann2022doclaynet,
  title     = {{DocLayNet}: A Large Human-Annotated Dataset for Document-Layout Segmentation},
  author    = {Pfitzmann, Birgit and Auer, Christoph and Dolfi, Michele and Nassar, Ahmed S. and Staar, Peter},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2022}
}
```

---

### Comparison Systems

#### mistralOCR — Mistral OCR
- **Title**: Mistral OCR
- **Authors**: Mistral AI
- **Venue/Year**: Mistral AI, March 2025 (proprietary product, no arXiv)
- **Key capabilities**: High-fidelity document parsing API covering tables, equations, handwriting, and complex layouts
- **Relevance**: Comparison system; proprietary, no public model weights, does not output strikethrough/underline/color formatting attributes
```bibtex
@misc{mistral2025ocr,
  title        = {{Mistral OCR}},
  author       = {{Mistral AI}},
  year         = {2025},
  howpublished = {\url{https://mistral.ai/news/mistral-ocr}},
  note         = {Accessed 2026-03-11}
}
```

#### paddleocrVL — PaddleOCR-VL
- **Title**: PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model
- **Authors**: Cheng Cui, Ting Sun, Suyin Liang, Tingquan Gao, Zelun Zhang, Jiaxuan Liu, Xueqing Wang, Changda Zhou, Hongen Liu, Manhui Lin, Yue Zhang, Yubo Zhang, Handong Zheng, Jing Zhang, Jun Zhang, Yi Liu, Dianhai Yu, Yanjun Ma
- **Venue/Year**: arXiv:2510.14528, October 2025 (Baidu PaddlePaddle)
- **Architecture**: 0.9B VLM for multilingual document parsing; covers text, tables, formulas, complex layouts
- **Relevance**: Comparison system; compact VLM approach to document OCR; does not address strikethrough, underline, or colored text as output features
```bibtex
@article{cui2025paddleocrvl,
  title   = {{PaddleOCR-VL}: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
  author  = {Cui, Cheng and Sun, Ting and Liang, Suyin and Gao, Tingquan and Zhang, Zelun and Liu, Jiaxuan and Wang, Xueqing and Zhou, Changda and Liu, Hongen and Lin, Manhui and Zhang, Yue and Zhang, Yubo and Zheng, Handong and Zhang, Jing and Zhang, Jun and Liu, Yi and Yu, Dianhai and Ma, Yanjun},
  journal = {arXiv preprint arXiv:2510.14528},
  year    = {2025},
  note    = {arXiv:2510.14528}
}
```

---

### Table Recognition

#### teds — TEDS Metric / PubTabNet
- **Title**: Image-based Table Recognition: Data, Model, and Evaluation
- **Authors**: Xu Zhong, Elaheh ShafieiBavani, Antonio Jimeno Yepes
- **Venue/Year**: arXiv:1911.10683, ECCV 2020
- **Key contribution**: Proposes TEDS (Tree-Edit-Distance-based Similarity) metric; introduces PubTabNet (568K tables); proposes EDD architecture (TEDS 88.3%)
- **Relevance**: TEDS is our primary table recognition metric
```bibtex
@inproceedings{zhong2020teds,
  title     = {Image-based Table Recognition: Data, Model, and Evaluation},
  author    = {Zhong, Xu and ShafieiBavani, Elaheh and Jimeno Yepes, Antonio},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020},
  note      = {arXiv:1911.10683}
}
```

#### pubtables1m — PubTables-1M
- **Title**: PubTables-1M: Towards Comprehensive Table Extraction from Unstructured Documents
- **Authors**: Brandon Smock, Rohith Pesala, Robin Abraham
- **Venue/Year**: arXiv:2110.00061, CVPR 2022
- **Dataset**: ~1M scientific table images; tasks: table detection, structure recognition, functional analysis
- **Relevance**: Large-scale benchmark establishing table extraction as a key document AI task; scientific domain only
```bibtex
@inproceedings{smock2022pubtables,
  title     = {{PubTables}-1M: Towards Comprehensive Table Extraction from Unstructured Documents},
  author    = {Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  note      = {arXiv:2110.00061}
}
```

#### fintabnet — FinTabNet
- **Title**: Global Table Extractor (GTE): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context
- **Authors**: Xinyi Zheng, Douglas Burdick, Lucian Popa, Xu Zhong, Nancy Xin Ru Wang
- **Venue/Year**: WACV 2021
- **Dataset**: Financial table images from Fortune 500 earnings reports with cell-structure annotations
- **Relevance**: Establishes financial table recognition as distinct from general tables; our EDGAR pipeline extends this with real SEC filings
```bibtex
@inproceedings{zheng2021fintabnet,
  title     = {Global Table Extractor ({GTE}): A Framework for Joint Table Identification and Cell Structure Recognition Using Visual Context},
  author    = {Zheng, Xinyi and Burdick, Douglas and Popa, Lucian and Zhong, Xu and Wang, Nancy Xin Ru},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2021}
}
```

#### fintabnetc — FinTabNet.c
- **Title**: Aligning Benchmark Datasets for Table Structure Recognition
- **Authors**: Brandon Smock, Rohith Pesala, Robin Abraham
- **Venue/Year**: ICDAR 2023
- **Dataset**: 2,064 curated table annotation pairs from FinTabNet
- **Relevance**: We use FinTabNet.c test set for baseline TEDS evaluation of vanilla HunyuanOCR
```bibtex
@inproceedings{smock2023fintabnetc,
  title     = {Aligning Benchmark Datasets for Table Structure Recognition},
  author    = {Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2023}
}
```

#### synfintabs — SynFinTabs
- **Title**: SynFinTabs: A Dataset of Synthetic Financial Tables for Information and Table Extraction
- **Authors**: Ethan Bradley et al. (Queen's University Belfast)
- **Venue/Year**: arXiv:2412.04262, December 2024
- **Dataset**: 100,000 synthetic financial table images with HTML/JSON/CSV annotations and bounding boxes
- **Relevance**: We use 5,200 samples for training (replay) and eval; related work on synthetic financial table generation
```bibtex
@article{bradley2024synfintabs,
  title   = {{SynFinTabs}: A Dataset of Synthetic Financial Tables for Information and Table Extraction},
  author  = {Bradley, Ethan and others},
  journal = {arXiv preprint arXiv:2412.04262},
  year    = {2024},
  note    = {arXiv:2412.04262}
}
```

---

### LoRA Variants

#### lora — LoRA
- **Title**: LoRA: Low-Rank Adaptation of Large Language Models
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Venue/Year**: arXiv:2106.09685, ICLR 2022
- **Key contribution**: Freezes pre-trained weights; injects trainable $\Delta W = BA$ (rank $r$); reduces trainable params by ~10,000x vs GPT-3 full fine-tuning
- **Relevance**: Foundation method; standard LoRA is our ablation baseline (Section 3.2 of project plan)
```bibtex
@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  note      = {arXiv:2106.09685}
}
```

#### duallora — Dual LoRA
- **Title**: Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates
- **Authors**: Yixing Xu, Chao Li, Xuanwu Yin, Spandan Tiwari, Dong Li, Ashish Sirasao, Emad Barsoum
- **Venue/Year**: arXiv:2512.03402, 2025
- **Key equations**: $\Delta W = \frac{\alpha}{\sqrt{r_1 r_2}} \operatorname{ReLU}(BA) \odot \operatorname{Sign}(DC)$; STE for Sign gradient
- **Key results**: NLU/commonsense tasks only; RoBERTa, DeBERTa, LLaMA families; no VLM results
- **Relevance**: Our proposed method; see `sources/dual_lora_notes.md` for full details
```bibtex
@article{xu2025duallora,
  title   = {Dual {LoRA}: Enhancing {LoRA} with Magnitude and Direction Updates},
  author  = {Xu, Yixing and Li, Chao and Yin, Xuanwu and Tiwari, Spandan and Li, Dong and Sirasao, Ashish and Barsoum, Emad},
  journal = {arXiv preprint arXiv:2512.03402},
  year    = {2025},
  note    = {arXiv:2512.03402}
}
```

#### dora — DoRA
- **Title**: DoRA: Weight-Decomposed Low-Rank Adaptation
- **Authors**: Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen
- **Venue/Year**: arXiv:2402.09353, ICML 2024 (Oral)
- **Key contribution**: Decomposes pre-trained weight $W_0$ into magnitude (column norms) and direction (unit vectors) for fine-tuning; uses LoRA for directional updates
- **Key distinction from Dual LoRA**: DoRA decomposes the full pre-trained weight $W_0$; Dual LoRA decomposes the **update** $\Delta W$. Both use magnitude/direction intuition but at different levels.
- **Relevance**: Closest prior work to Dual LoRA; important to distinguish in related work section
```bibtex
@inproceedings{liu2024dora,
  title     = {{DoRA}: Weight-Decomposed Low-Rank Adaptation},
  author    = {Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and Molchanov, Pavlo and Wang, Yu-Chiang Frank and Cheng, Kwang-Ting and Chen, Min-Hung},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024},
  note      = {arXiv:2402.09353}
}
```

#### qlora — QLoRA
- **Title**: QLoRA: Efficient Finetuning of Quantized LLMs
- **Authors**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer
- **Venue/Year**: arXiv:2305.14314, NeurIPS 2023
- **Key contribution**: 4-bit NF4 quantization + LoRA; enables 65B model fine-tuning on a single 48GB GPU
- **Relevance**: Cited for completeness of LoRA variant landscape; not directly used in our method
```bibtex
@inproceedings{dettmers2023qlora,
  title     = {{QLoRA}: Efficient Finetuning of Quantized {LLMs}},
  author    = {Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  note      = {arXiv:2305.14314}
}
```

---

### Financial Document AI (Motivation)

#### strikethrough2014 — Strikethrough Identification (IEEE 2014)
- **Title**: An Approach of Strike-Through Text Identification from Handwritten Documents
- **Authors**: Chandranath Adak, Bidyut Baran Chaudhuri
- **Venue/Year**: IEEE ICFHR 2014, pp. 643–648
- **Relevance**: Only prior strikethrough work; handwritten documents only; binary detection (not OCR); classical methods; explicitly out of scope for our problem

```bibtex
@inproceedings{adak2014strikethrough,
  title     = {An Approach of Strike-Through Text Identification from Handwritten Documents},
  author    = {Adak, Chandranath and Chaudhuri, Bidyut Baran},
  booktitle = {14th International Conference on Frontiers in Handwriting Recognition (ICFHR)},
  pages     = {643--648},
  year      = {2014},
  organization = {IEEE}
}
```

#### poddar2021 — Struck-Out-Strokes Detection
- **Title**: Detection and Localisation of Struck-Out-Strokes in Handwritten Manuscripts
- **Authors**: Arnab Poddar, Akash Chakraborty, Jayanta Mukhopadhyay, Prabir Kumar Biswas
- **Venue/Year**: ICDAR 2021 Workshops, LNCS vol. 12917, pp. 98–112, Springer. DOI: 10.1007/978-3-030-86159-9\_7
- **Relevance**: Second strikethrough-adjacent paper; handwritten manuscripts only; out of scope for our printed financial document setting
```bibtex
@inproceedings{poddar2021struck,
  title     = {Detection and Localisation of Struck-Out-Strokes in Handwritten Manuscripts},
  author    = {Poddar, Arnab and Chakraborty, Akash and Mukhopadhyay, Jayanta and Biswas, Prabir Kumar},
  booktitle = {Document Analysis and Recognition -- ICDAR 2021 Workshops},
  series    = {Lecture Notes in Computer Science},
  volume    = {12917},
  pages     = {98--112},
  publisher = {Springer},
  year      = {2021},
  doi       = {10.1007/978-3-030-86159-9_7}
}
```

#### hase2003 — Color Segmentation for Text Extraction
- **Title**: Color segmentation for text extraction
- **Authors**: H. Hase, M. Yoneda, S. Tokai, J. Kato, C. Y. Suen
- **Venue/Year**: International Journal on Document Analysis and Recognition (IJDAR), vol. 6, pp. 271–284, Springer, 2003. DOI: 10.1007/s10032-003-0119-7
- **Relevance**: Most relevant prior color-in-OCR work; targets character extraction quality improvement, not semantic color as output; pre-deep-learning
```bibtex
@article{hase2003color,
  title   = {Color segmentation for text extraction},
  author  = {Hase, H. and Yoneda, M. and Tokai, S. and Kato, J. and Suen, C. Y.},
  journal = {International Journal on Document Analysis and Recognition (IJDAR)},
  volume  = {6},
  pages   = {271--284},
  year    = {2003},
  doi     = {10.1007/s10032-003-0119-7}
}
```

#### tan2025 — VLM Fine-Tuning for Financial Tables
- **Title**: Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports
- **Authors**: Jin Khye Tan et al.
- **Venue/Year**: arXiv:2508.05669, August 2025
- **Key results**: Qwen2.5-VL-7B + LoRA; 2,152 image-text pairs; 96.53% TEDS; outperforms GPT-4o and Gemini 2.5 Flash
- **Relevance**: Closest prior work to our table contribution; single-feature (financial tables → Markdown); no strikethrough/underline/color; uses standard LoRA not Dual LoRA
```bibtex
@article{tan2025vlmtable,
  title   = {Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in {Malaysian} Audited Financial Reports},
  author  = {Tan, Jin Khye and others},
  journal = {arXiv preprint arXiv:2508.05669},
  year    = {2025},
  note    = {arXiv:2508.05669}
}
```

#### finqa — FinQA
- **Title**: FinQA: A Dataset of Numerical Reasoning over Financial Reports
- **Authors**: Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, William Yang Wang
- **Venue/Year**: EMNLP 2021
- **Relevance**: Motivation — establishes that financial tables and text require specialized reasoning; QA dataset, not OCR
```bibtex
@inproceedings{chen2021finqa,
  title     = {{FinQA}: A Dataset of Numerical Reasoning over Financial Reports},
  author    = {Chen, Zhiyu and Chen, Wenhu and Smiley, Charese and Shah, Sameena and Borova, Iana and Langdon, Dylan and Moussa, Reema and Beane, Matt and Huang, Ting-Hao and Routledge, Bryan and Wang, William Yang},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2021}
}
```

#### tatqa — TAT-QA
- **Title**: TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance
- **Authors**: Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, Tat-Seng Chua
- **Venue/Year**: ACL 2021
- **Relevance**: Motivation — establishes the complexity of financial tabular-textual reasoning; QA dataset, not OCR
```bibtex
@inproceedings{zhu2021tatqa,
  title     = {{TAT-QA}: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance},
  author    = {Zhu, Fengbin and Lei, Wenqiang and Huang, Youcheng and Wang, Chao and Zhang, Shuo and Lv, Jiancheng and Feng, Fuli and Chua, Tat-Seng},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2021}
}
```
