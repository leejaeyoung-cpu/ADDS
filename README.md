<div align="center">

<img src="https://img.shields.io/badge/ADDS-v3.5.0-blueviolet?style=for-the-badge&logo=python" alt="ADDS Version"/>

# ADDS â AI-Driven Drug Synergy & Diagnostic System

**ì ë° ì¢ìíì ìí ë©í°ëª¨ë¬ AI íë«í¼**  
*Multimodal AI Platform for Precision Oncology*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_GPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Cellpose](https://img.shields.io/badge/Cellpose-cyto3-00C49F)](https://cellpose.readthedocs.io/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-FF6B35)](https://github.com/MIC-DKFZ/nnUNet)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-Inha_University_Hospital-003DA5)](https://www.inha.com/)

<br/>

> **ADDS**ë CT ë°©ì¬ì í, ì¸í¬ ííê³ì¸¡í, ì½ëí ëª¨ë¸ë§, ê¸°ê³íìµì íëì íµí© íë«í¼ì¼ë¡ ìµí©íì¬  
> ëì¥ì(CRC) íìë¥¼ ìí ê°ì¸í í­ì ì½ë¬¼ ì¹µíì¼ì ì¶ì²íë ì ë° ì¢ìí AI ìì¤íìëë¤.

</div>

---

## ð ëª©ì°¨ / Table of Contents

- [ìì¤í ê°ì](#-ìì¤í-ê°ì--system-overview)
- [ì ì²´ ìí¤íì²](#-ì ì²´-ìí¤íì²--architecture)
- [íµì¬ ëª¨ë](#-íµì¬-ëª¨ë--core-modules)
  - [CT ë¶ì íì´íë¼ì¸](#1-ct-ë¶ì-íì´íë¼ì¸)
  - [Cellpose íë¯¸ê²½ ë¶ì](#2-cellpose-íë¯¸ê²½-ë¶ì)
  - [KRAS-PrPc ì½ë¬¼ ìëì§](#3-kras-prpc-ì½ë¬¼-ìëì§)
  - [ì½ëí (PK/PD) ëª¨ë¸ë§](#4-ì½ëí-pkpd-ëª¨ë¸ë§)
  - [ìì ìì¬ê²°ì  ì§ì](#5-ìì-ìì¬ê²°ì -ì§ì-cds)
  - [íì ê´ë¦¬ ìì¤í](#6-íµí©-íì-ê´ë¦¬-ìì¤í)
- [ì±ë¥ ì§í](#-ì±ë¥-ì§í--performance-metrics)
- [14ì°¨ì í¹ì§ ë²¡í°](#-14ì°¨ì-ë©í°ëª¨ë¬-í¹ì§-ë²¡í°)
- [ì¤ì¹ ë° ì¤í](#-ì¤ì¹-ë°-ì¤í--installation)
- [API ì°¸ì¡°](#-api-ì°¸ì¡°--api-reference)
- [ë°ì´í° êµ¬ì¡°](#-ë°ì´í°-êµ¬ì¡°--data-structure)
- [ì°êµ¬ ë°°ê²½](#-ì°êµ¬-ë°°ê²½--research-background)
- [ì¸ì©](#-ì¸ì©--citation)

---

## ð¬ ìì¤í ê°ì / System Overview

ADDS (AI-Driven Drug Synergy) ë ì´íëíêµë³ìê³¼ì ê³µë ì°êµ¬ë¥¼ íµí´ ê°ë°ë **ì ë° ì¢ìí AI ìíê³**ìëë¤.

### íµì¬ íì  í¬ì¸í¸

| íì  | ì¤ëª |
|------|------|
| **ë©í°ëª¨ë¬ ë°ì´í° ìµí©** | CT ë°©ì¬ì í + ì¸í¬ ë³ë¦¬í + ìì ë©íë°ì´í°ë¥¼ ë¨ì¼ 14ì°¨ì í¹ì§ ë²¡í°ë¡ íµí© |
| **ì´ì¤ ì¶ë¡  ìì§** | ADDS ê²½ë¡ ê¸°ë° ìì§ + OpenAI GPT-4 ëì ì¤í ë° êµì°¨ ê²ì¦ |
| **RAG ê¸°ë° ê·¼ê±° ìì±** | ìì¬ ìê²¬ìë¥¼ 1ìì íë¡¬íí¸ë¡ íì©íë ê²ì ì¦ê° ìì±(RAG) ìì¤í |
| **PrPc ë°ì´ì¤ë§ì»¤ ë°ê²¬** | TCGA ë°ì´í°(n=2,285)ìì KRAS-RPSA ìê·¸ëë¡ì ê¸°ë° ì ê· ë°ì´ì¤ë§ì»¤ ë°ê²¬ |
| **ì¤ìê° ìì ì ì©** | 15.67ì´ ë´ ìë-í¬-ìë ë¶ì ìë£ (530Ã751Ã750 ë³¼ë¥¨ ê¸°ì¤) |

---

## ðï¸ ì ì²´ ìí¤íì² / Architecture

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â                    ADDS Precision Oncology Platform v3.5             â
â                      Inha University Hospital                        â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
                                    â
          âââââââââââââââââââââââââââ¼ââââââââââââââââââââââââââ
          â¼                         â¼                         â¼
  âââââââââââââââââ       âââââââââââââââââââ       ââââââââââââââââââ
  â  Streamlit UI â       â  FastAPI Backend â       â  Data Layer    â
  â  (Port 8505)  ââââââââºâ  (Port 8000)    ââââââââºâ  SQLite / NFS  â
  â               â       â                 â       â                â
  â â¢ íì ê´ë¦¬   â       â /api/v1/        â       â patients.db    â
  â â¢ AI ë¶ì     â       â  ââ patients    â       â ct_data/       â
  â â¢ ì½ë¬¼ ì¶ì²   â       â  ââ ct          â       â microscopy/    â
  â â¢ ë³´ê³ ì ìì± â       â  ââ cellpose    â       â literature/    â
  âââââââââââââââââ       â  ââ pharmacoki  â       ââââââââââââââââââ
                          â  ââ adds        â
                          â  ââ openai      â
                          âââââââââââââââââââ
                                    â
         ââââââââââââââââââââââââââââ¼âââââââââââââââââââââââââââ
         â¼                          â¼                          â¼
ââââââââââââââââââ        âââââââââââââââââââ        ââââââââââââââââââ
â  CT Pipeline   â        â Cellpose Pipelineâ        â  Drug Synergy  â
â  (6 Stages)    â        â                 â        â  Engine        â
â                â        â cyto3 Model     â        â                â
â S1: DICOMâNIfTIâ        â â Segmentation  â        â KRAS-PrPc      â
â S2: Organ Seg  â        â â Ki-67 Index   â        â Signalosome    â
â S3: Tumor Det  â        â â Morphology    â        â                â
â S4: Radiomics  â        â â Heterogeneity â        â Pritamab       â
â S5: Staging    â        â                 â        â Prediction     â
â S6: ADDS Integ â        â n=43,190 cells  â        â                â
â                â        â analyzed        â        â PK/PD Modeling â
â Acc: 98.65%    â        â                 â        â                â
ââââââââââââââââââ        âââââââââââââââââââ        ââââââââââââââââââ
         â                          â                          â
         ââââââââââââââââââââââââââââ¼âââââââââââââââââââââââââââ
                                    â¼
                    âââââââââââââââââââââââââââââââââ
                    â    14D Multimodal Feature      â
                    â    Vector Fusion               â
                    â                                â
                    â  CT Radiomics (7D):            â
                    â  Sphericity, Entropy,          â
                    â  Contrast, Size, Circularity,  â
                    â  Mean HU, Confidence           â
                    â                                â
                    â  Cell Culture (7D):            â
                    â  Density, Drug Resistance,     â
                    â  Proliferation, Complexity,    â
                    â  Circularity, Clark-Evans,     â
                    â  Viability                     â
                    âââââââââââââââââââââââââââââââââ
                                    â
                    âââââââââââââââââ´ââââââââââââââââ
                    â¼                               â¼
         âââââââââââââââââââ             ââââââââââââââââââââ
         â  ADDS Engine    â             â  OpenAI Engine   â
         â  (Pathway-Based)â             â  (GPT-4 Medical) â
         â                 â             â                  â
         â KRAS/RAF/MEK/   â             â Clinical Summary â
         â ERK Signaling   ââââ Cross âââºâ Treatment Plan   â
         â Synergy Scoring â  Validate   â MDT Consensus    â
         âââââââââââââââââââ             ââââââââââââââââââââ
                    â                               â
                    âââââââââââââââââ¬ââââââââââââââââ
                                    â¼
                    âââââââââââââââââââââââââââââââââ
                    â   Final Drug Cocktail          â
                    â   Recommendation               â
                    â                                â
                    â  FOLFOX + Bevacizumab          â
                    â  + PK-Optimized Dosing         â
                    â  + Outcome Simulation          â
                    â   (ORR / PFS / OS)             â
                    âââââââââââââââââââââââââââââââââ
```

---

## âï¸ íµì¬ ëª¨ë / Core Modules

### 1. CT ë¶ì íì´íë¼ì¸

**6ë¨ê³ 3D CT ì¢ì ê²ì¶ ë° ë°©ì¬ì í ë¶ì íì´íë¼ì¸**

```
Stage 1: 3D Volume Reconstruction
    DICOM Series â 1mmÂ³ Isotropic NIfTI Volume
    (SimpleITK, scipy ê¸°ë° ë¦¬ìíë§)

Stage 2: Anatomical Organ Segmentation
    nnU-Net v2 â Colon / Liver / Lymph Node Parsing

Stage 3: Tumor Detection  â VerifiedCTDetector (98.65% Accuracy)
    HU Thresholding: 60â120 HU (Arterial Phase)
    2D Slice-by-Slice Morphological Filtering
    Min Size: 30 px (noise), 50 mmÂ³ (clinical threshold)

Stage 4: Radiomics Extraction
    PyRadiomics â 100+ Phenotypic Features
    (Sphericity, Entropy, GLCM Contrast, Surface Area...)

Stage 5: Biomarker Prediction
    Malignancy Score / TNM Staging / MSI / KRAS Status

Stage 6: ADDS Integration
    Radiomics â PK Sensitivity Model â Drug Recommendation
```

**ì£¼ì ì±ë¥ ì§í (ì´íëíêµë³ì ì½í¸í¸)**

| ì§í | ê° |
|------|-----|
| ê²ì¶ ì íë | **98.65%** (74ê° ì¬ë¼ì´ì¤ ì¤ 73ê°) |
| ì²ë¦¬ ìê° | **15.67ì´** (530Ã751Ã750 ë³¼ë¥¨) |
| ì²ë¦¬ë | **33.8 ì¬ë¼ì´ì¤/ì´** |
| HU íì§ ë²ì | 60â120 HU (ëë§¥ê¸°) |
| ìµì ë³ë³ í¬ê¸° | 50 mmÂ³ |

ê´ë ¨ ì¤í¬ë¦½í¸:
```bash
python ct_pipeline_v4.py                    # CT íì´íë¼ì¸ ë©ì¸
python detect_tumors_inha_corrected.py      # ê²ì¦ë ê²ì¶ê¸° (98.65%)
python ct_crc_detection_pipeline.py         # CRC í¹í íì´íë¼ì¸
python batch_tumor_detection_dcm.py         # ë°°ì¹ ì²ë¦¬
```

---

### 2. Cellpose íë¯¸ê²½ ë¶ì

**HUVEC ì¸í¬ ííê³ì¸¡í ìëí ë¶ì (Cellpose cyto3 ëª¨ë¸ ê¸°ë°)**

```
Raw Microscopy Image
       â
       â¼
CLAHE + Denoising (Preprocessing)
       â
       â¼
Cellpose cyto3 Segmentation
       â
       âââ Cell Count & Density
       âââ Elongation Ratio (ì¥ì¶/ë¨ì¶)
       âââ Circularity Score
       âââ Clark-Evans Index (êµ°ì§ ë¶í¬)
       âââ Ki-67 Proliferation Index Estimation
       âââ Tumor Heterogeneity Score
```

**ë¶ì ê²°ê³¼ (HUVEC Serum ì¤í, n = 43,190 cells)**

| ì¡°ê±´ | ì¸í¬ ì | ì¥ì¶ë¹ | ì¸í¬ë©´ì  | í´ì |
|------|---------|--------|---------|------|
| Control | 11,717 | 1.831 | 696 pxÂ² | ì ì§ ìí |
| Healthy Serum | 6,538 | 1.865 | 618 pxÂ² | ì ì íì±í |
| HGPS Serum | 13,676 | 1.902 | 756 pxÂ² | ë³ë¦¬ì  íì±í |
| **HGPS + MT-Exo** | **11,259** | **1.992** | **775 pxÂ²** | **ìµë ë´í¼ íì±í** |

> MT-Exo ì²ë¦¬êµ°ìì ì¸í¬ ì¥ì¶ë¹ ì ìë¯¸í ì¦ê° (p < 0.001) â ë´í¼ì¸í¬ ì´ë ë¥ë ¥ ì¦ê° ìì¬

ê´ë ¨ ì¤í¬ë¦½í¸:
```bash
python analysis/huvec/01_preprocess.py     # ì´ë¯¸ì§ ì ì²ë¦¬
python analysis/huvec/02_cellpose_run.py   # Cellpose ì¸ë¶í
python analysis/huvec/07_ppt_figures.py    # ë¼ë¬¸ì© Figure ìì±
python verify_cellpose_pipeline.py          # íì´íë¼ì¸ ê²ì¦
```

---

### 3. KRAS-PrPc ì½ë¬¼ ìëì§

**ê¸°ì  ê¸°ë° ì½ë¬¼ ìëì§ ìì¸¡ ìì§**

#### PrPc ì¡°ì§-íì²­ í¨ë¬ëì¤ í´ê²°

| ì¸¡ì  | CRC ì¡°ì§ | íì²­ | ê¸°ì  |
|------|---------|------|------|
| PRNP mRNA | â ë®ì | â | ì¢ì ìµì  |
| PrPc ë¨ë°±ì§ | â | ââ ëì | **ADAM10/17 ìë©** |

> ADAM10/17 í¨ìê° ì¸í¬ë§ GPI-ìµì»¤ PrPcë¥¼ ì ë¨ â íë¥ë¡ ë°©ì¶  
> TCGA ì¤ë°ì´í° ê²ì¦: n = 2,285 (BRCA, STAD, COAD, PAAD, READ)

#### KRAS-RPSA ìê·¸ëë¡ì ê²½ë¡

```
KRAS Mutation (G12D/G12V)
       â
       â¼
RAF â MEK â ERK Activation
       â
       âââ PrPc-RPSA Complex Formation
       â         â
       â         âââ Laminin Binding (ì¸í¬ ì¹¨ì¤ ì´ì§)
       â
       âââ Downstream Survival Pathways
                 â
                 âââ mTOR Axis
                 âââ PI3K/AKT
                 âââ WNT/Î²-catenin
```

#### ì½ë¬¼ ì§ì ë² ì´ì¤

| ì§í | ê° |
|------|-----|
| ì´ ë¼ë¬¸ ì | 311í¸ (Nature/Cell/Science ë± Tier-1) |
| ë°ì´í° ìí | 2,348 ìì ìí |
| ë±ë¡ ì½ë¬¼ | 113ì¢ |
| ìì© ê¸°ì  | 90ê° |
| ë°ì´ì¤ë§ì»¤ | 69ê° |
| ìëì§ ì¡°í© | 59ê° |

---

### 4. ì½ëí (PK/PD) ëª¨ë¸ë§

**íì ë§ì¶¤í í­ìì  ì©ë ìµì í 1-êµ¬í ëª¨ë¸**

$$C_{max} = \frac{D}{V_d} \cdot e^{-k_e \cdot t}$$

| íë¼ë¯¸í° | ê³µì | ë¨ì |
|---------|------|------|
| **ì²­ìì¨ (Cl)** | $120.0 \times \max(0.7, 1.0 - \frac{V_{tumor}}{500})$ | mL/min |
| **ë¶í¬ì©ì  (Vd)** | $45.0 + (V_{tumor} \times 0.5)$ | L |
| **ë°ê°ê¸° (tÂ½)** | $0.693 \times \frac{V_d}{Cl \times 0.06}$ | hours |
| **ìµì  ì©ë (D)** | $200.0 \times (1.0 + \frac{Ki67}{200})$ | mg/mÂ² |

**ìì  ì ì½ ì¡°ê±´:**
- í¬ì¬ ê°ê²©: 6h â 24h (íë í´ë¨í)
- ìµë ë°ìë¥ : 95% (ìì íì¤ì± ì ì§)
- ì ì¥/ê° ê¸°ë¥ ëë¦¬ ì§í: `cl_factor` (ì¢ì ë¶ë´ ê¸°ë°)

---

### 5. ìì ìì¬ê²°ì  ì§ì (CDS)

**ì´ì¤ ì¶ë¡  ìì§ ê¸°ë° êµì°¨ ê²ì¦ ìì¤í**

```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â            6-Step Dynamic Inference Pipeline             â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

Step 0: RAG Analysis
    ìì¬ ìê²¬ì â ìë¯¸ë¡ ì  ìì ì»¨íì¤í¸ ì¶ì¶
    (ì¦ì, ë³ë ¥, íì ì í¸ë)

Step 1: CT Analysis (Live API)
    DICOM ìë¡ë â /api/v1/ct/analyze
    ê²°ê³¼: ë°©ì¬ì í JSON + ìê°í ì´ë¯¸ì§ ì¤í¸ë¦¼

Step 2: Cell Analysis (ì¡°ê±´ë¶)
    Cellpose ì¸ë¶í â Ki-67 ì ëí
    (íë¯¸ê²½ ì´ë¯¸ì§ ìì¼ë©´ ê±´ëë)

Step 3: Pharmacokinetics
    CT + Cellpose ê²°ê³¼ â PK ìµì í íë¼ë¯¸í°

Step 4: ADDS Inference
    ê²½ë¡ ê¸°ë° ê¸°ì  ì¶ì²
    (RAG ì»¨íì¤í¸ + ë©í°ëª¨ë¬ ë°ì´í°)

Step 5: OpenAI Inference
    GPT-4 ìì íµí© (ìì¬ ìê²¬ì 1ìì íë¡¬íí¸)

Step 6: Cross-Validation
    ìê²¬ì â CT ê²°ê³¼ â ë³ë¦¬ ê²°ê³¼ ìë ì¼ì¹ì± ê²ì¦
```

**ìµì¢ ì¶ì² ìì±:**
- ð¯ í­ìì  ì¹µíì¼ (ì: FOLFOX + Bevacizumab)
- ð ìµì íë í¬ì¬ë ë° ê²½ë¡
- ð ìí ìë®¬ë ì´ì (ORR / PFS / OS)
- ð ì´ì¤ ë³´ê³ ì (ìë£ì§ ê¸°ì  ë³´ê³ ì + íì ê°ì´ë)

---

### 6. íµí© íì ê´ë¦¬ ìì¤í

**ìí°íë¼ì´ì¦ê¸ ìì ë°ì´í° ê´ë¦¬ (IPMS)**

```python
# íì ID íì
Patient ID: P-2026-001

# íµì¬ ìì ë©íë°ì´í°
{
  "tnm_stage": "T4N0M0",
  "msi_status": "MSS",
  "kras_mutation": "G12D",
  "ecog_score": 1,
  "ki67_index": 45.2,
  "tumor_location": "Sigmoid Colon"
}
```

| ê¸°ë¥ | ì¤ëª |
|------|------|
| **íì CRUD** | P-YYYY-NNN íì ìêµ¬ ë ì½ë |
| **ì¢ë¨ ì¶ì ** | ì¹ë£ ê²½ê³¼ì ë°ë¥¸ ë°ì´í° ì´ë ¥ ê´ë¦¬ |
| **ë©í°ëª¨ë¬ ìë¡ë** | CT DICOM + íë¯¸ê²½ ì´ë¯¸ì§ + ìê²¬ì íµí© |
| **ì¤ìê° ì§í** | ë¶ì ë¨ê³ë³ ì¤ìê° ìí ì¶ì  |
| **PDF ë³´ê³ ì** | ìë ìì± (ìë£ì§ì© / íìì©) |

---

## ð ì±ë¥ ì§í / Performance Metrics

### CT ë¶ì ì±ë¥
```
âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
â  CT Detection Performance (Inha University Hospital) â
â  âââââââââââââââââââââââââââââââââââââââââââââââââââ â
â  Accuracy:      ââââââââââââââââââââ 98.65%         â
â  Speed:         15.67s / patient (E2E)               â
â  Throughput:    33.8 slices/sec                      â
â  Volume Size:   530 Ã 751 Ã 750 voxels               â
â  HU Range:      60 â 120 HU (arterial phase)         â
â  Min Lesion:    50 mmÂ³                               â
âââââââââââââââââââââââââââââââââââââââââââââââââââââââ
```

### ìì¤í ë²¤ì¹ë§í¬
| êµ¬ì± | ì²ë¦¬ ìê° |
|------|---------|
| CT E2E ë¶ì (íì¤) | ~45.2ì´ |
| CT E2E ë¶ì (ìµì í) | **15.67ì´** |
| Cellpose (GPU, 1ì¥) | ~3.2ì´ |
| ì½ë¬¼ ì¶ì² ìì± | ~2.1ì´ |
| ì ì²´ íì´íë¼ì¸ | **< 90ì´** |

### ì°êµ¬ ë°ì´í° ê·ëª¨

| ë°ì´í° ì í | ê·ëª¨ |
|------------|------|
| HUVEC ë¶ì ì¸í¬ ì | **43,190ê°** |
| TCGA PrPc ì¤ì  ìí | **2,285ê°** |
| ë¼ë¬¸ ì§ì ë² ì´ì¤ | **311í¸** |
| ì´í CT ì½í¸í¸ ë³¼ë¥¨ | 530Ã751Ã750 |
| ìì ìí (ì ì²´) | **2,348ê°** |

---

## ð§¬ 14ì°¨ì ë©í°ëª¨ë¬ í¹ì§ ë²¡í°

```python
feature_vector = {
    # CT Radiomics (7D) â ê±°ìì  ìì í¹ì§
    "sphericity":          float,  # ì¢ì êµ¬íë
    "energy":              float,  # GLCM íì¤ì² ìëì§
    "contrast":            float,  # ìì ëë¹ë
    "tumor_size_mm2":      float,  # ì¢ì í¬ê¸° (mmÂ²)
    "circularity":         float,  # ìíë
    "mean_hu":             float,  # íê·  íì´ì¤íë ë¨ì
    "detection_confidence":float,  # ê²ì¶ ì ë¢°ë

    # Cell Culture (7D) â ë¯¸ìì  ì¸í¬ í¹ì§
    "cell_density":        float,  # ì¸í¬ ë°ë (cells/mmÂ²)
    "drug_resistance":     float,  # ì½ë¬¼ ì í­ ì ì
    "proliferation_score": float,  # Ki-67 ê¸°ë° ì¦ì ì§ì
    "microenv_complexity": float,  # ë¯¸ì¸íê²½ ë³µì¡ë
    "mean_circularity":    float,  # íê·  ì¸í¬ ìíë
    "clark_evans_index":   float,  # ê³µê°ì  êµ°ì§ ì§ì
    "estimated_viability": float,  # ìì ì¸í¬ ìì¡´ì¨
}
```

---

## ð ì¤ì¹ ë° ì¤í / Installation

### ìì¤í ìêµ¬ì¬í­

| í­ëª© | ìµì | ê¶ì¥ |
|------|------|------|
| Python | 3.11 | 3.11+ |
| GPU | CUDA 11.x | CUDA 12.8 (RTX 50-series) |
| RAM | 16 GB | 32 GB |
| VRAM | 8 GB | 16 GB |
| ì ì¥ê³µê° | 50 GB | 200 GB |

### ë¹ ë¥¸ ì¤ì¹

```bash
# 1. ë í¬ì§í ë¦¬ í´ë¡ 
git clone https://github.com/leejaeyoung-cpu/ADDS.git
cd ADDS

# 2. ê°ìíê²½ ìì±
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. ìì¡´ì± ì¤ì¹
pip install -r requirements.txt

# 4. íê²½ ë³ì ì¤ì 
cp .env.example .env
# .env íì¼ìì OPENAI_API_KEY, DB_PATH ë± ì¤ì 

# 5. ë°ì´í°ë² ì´ì¤ ì´ê¸°í
cd backend
python -c "from database_init import init_database; init_database()"
cd ..
```

### ìì¤í ì¤í

```bash
# â ë°©ë² 1: íµí© ì¤í (ê¶ì¥)
START_ALL.bat           # ë°±ìë(8000) + Streamlit UI(8505) ëì ì¤í

# â ë°©ë² 2: ìë ì¤í
# í°ë¯¸ë 1 â ë°±ìë
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# í°ë¯¸ë 2 â Streamlit UI
python -m streamlit run src/ui/app.py --server.port 8505
```

> **ì ê·¼ ì£¼ì:**
> - ð¥ï¸ ìì UI: `http://localhost:8505`
> - ð¡ API ìë²: `http://localhost:8000`
> - ð API ë¬¸ì: `http://localhost:8000/docs`

### GPU ì¤ì  (RTX 50-series / Blackwell)

```bash
# PyTorch Nightly (cu128 ì§ì)
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# GPU ìí íì¸
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## ð ë°ì´í° êµ¬ì¡° / Data Structure

```
ADDS/
âââ ð src/                         â íµì¬ ìì¤ ëª¨ë
â   âââ adds/                       â ADDS ì¶ë¡  ìì§
â   âââ medical_imaging/            â CT íì´íë¼ì¸
â   â   âââ detection/              â ì¢ì ê²ì¶ (SimpleHUDetector)
â   â   âââ preprocessing/          â DICOM ì ì²ë¦¬
â   â   âââ radiomics/              â ë°©ì¬ì í í¹ì§ ì¶ì¶
â   â   âââ segmentation/           â ì¥ê¸° ë¶í 
â   âââ pathology/                  â Cellpose íë¯¸ê²½ ë¶ì
â   âââ clinical/                   â ìì ë°ì´í° ê´ë¦¬
â   âââ ml/                         â ë¨¸ì ë¬ë ëª¨ë¸
â   â   âââ fusion/                 â ë©í°ëª¨ë¬ ìµí©
â   â   âââ survival/               â PFS/OS ìì¸¡
â   âââ protein/                    â PrPc ë¨ë°±ì§ ë¶ì
â   âââ recommendation/             â ì½ë¬¼ ì¶ì² ìì§
â   âââ knowledge/                  â ì§ì ë² ì´ì¤ (311í¸ ë¼ë¬¸)
â   âââ knowledge_base/             â êµ¬ì¡°íë ì½ë¬¼ DB
â   âââ reporting/                  â PDF ë³´ê³ ì ìì±
â   âââ visualization/              â ë°ì´í° ìê°í
â   âââ xai/                        â ì¤ëª ê°ë¥ AI (XAI)
â   âââ ui/                         â Streamlit UI ì»´í¬ëí¸
â
âââ ð backend/                     â FastAPI ë°±ìë
â   âââ main.py                     â ì± ì§ìì 
â   âââ api/                        â REST API ë¼ì°í°
â   â   âââ ct_analysis.py          â  /api/v1/ct
â   â   âââ patients.py             â  /api/v1/patients
â   â   âââ pharmacokinetics.py     â  /api/v1/pharmacokinetics
â   â   âââ adds_inference.py       â  /api/v1/adds
â   â   âââ openai_inference.py     â  /api/v1/openai
â   âââ services/                   â ë¹ì¦ëì¤ ë¡ì§ ìë¹ì¤
â   â   âââ ct_pipeline_service.py
â   â   âââ cell_culture_service.py
â   â   âââ adds_service.py
â   â   âââ openai_service.py
â   âââ models/                     â SQLAlchemy ORM ëª¨ë¸
â   âââ schemas/                    â Pydantic ì¤í¤ë§
â
âââ ð analysis/                    â ì°êµ¬ ë¶ì ì¤í¬ë¦½í¸
â   âââ huvec/                      â HUVEC ì¸í¬ ë¶ì
â   âââ ct/                         â CT ë¶ì íì´íë¼ì¸
â   âââ pritamab/                   â Pritamab ì½ë¬¼ ìëì§
â
âââ ð figures/                     â ë¼ë¬¸ì© Figure (300 DPI)
âââ ð docs/                        â ìì¤í ë¬¸ì
âââ ð configs/                     â ì¤ì  íì¼
âââ ð tests/                       â ì ë íì¤í¸
âââ ð notebooks/                   â Jupyter ë¶ì ë¸í¸ë¶
âââ ð data/samples/                â ìµëªíë ìí ë°ì´í°
â
âââ ð³ Dockerfile                   â ì»¨íì´ë ì´ë¯¸ì§
âââ ð³ docker-compose.yml           â ìë¹ì¤ ì¤ì¼ì¤í¸ë ì´ì
âââ ð requirements.txt             â Python ìì¡´ì±
âââ ð pyproject.toml               â íë¡ì í¸ ì¤ì 
âââ ð .env.example                 â íê²½ë³ì ííë¦¿
```

---

## ð¡ API ì°¸ì¡° / API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### íµì¬ ìëí¬ì¸í¸

| Method | Endpoint | ì¤ëª |
|--------|---------|------|
| `GET` | `/health` | ìì¤í ìí íì¸ |
| `GET` | `/patients` | íì ëª©ë¡ ì¡°í |
| `POST` | `/patients` | ì ê· íì ë±ë¡ |
| `GET` | `/patients/{id}` | íì ìì¸ ì¡°í |
| `POST` | `/ct/analyze` | CT DICOM ë¶ì ì¤í |
| `GET` | `/ct/health` | CT íì´íë¼ì¸ ìí |
| `GET` | `/ct/models/status` | nnU-Net ëª¨ë¸ ìí |
| `POST` | `/pharmacokinetics/analyze` | PK íë¼ë¯¸í° ê³ì° |
| `POST` | `/adds/infer` | ADDS ê²½ë¡ ê¸°ë° ì¶ë¡  |
| `POST` | `/openai/infer` | GPT-4 ìì ì¶ë¡  |

### CT ë¶ì ìì²­ ìì

```python
import requests

# DICOM íì¼ ìë¡ë ë° ë¶ì
with open("tumor_series.dcm", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/ct/analyze",
        files={"dicom_file": f},
        data={"patient_id": "P-2026-001"}
    )

result = response.json()
print(f"ì¢ì ê²ì¶: {result['tumors_detected']}ê°")
print(f"ì ë¢°ë: {result['confidence']:.2%}")
print(f"TNM ì¶ì : {result['tnm_stage']}")
```

### PK ìµì í ìì²­ ìì

```python
pk_response = requests.post(
    "http://localhost:8000/api/v1/pharmacokinetics/analyze",
    json={
        "patient_id": "P-2026-001",
        "tumor_volume_mm3": 2450.5,
        "ki67_index": 45.2,
        "body_surface_area": 1.73
    }
)

pk = pk_response.json()
print(f"ìµì  ì©ë: {pk['optimal_dose_mg_m2']} mg/mÂ²")
print(f"ë°ê°ê¸°: {pk['half_life_hours']:.1f}ìê°")
print(f"í¬ì¬ ê°ê²©: {pk['dosing_interval_hours']}ìê°")
```

---

## ð§ª ì°êµ¬ ë°°ê²½ / Research Background

### PrPc ë°ì´ì¤ë§ì»¤ ë°ê²¬ ì¬ì 

| ë²ì  | ì ëµ | ì½í¸í¸ | ëª©í | ê²°ê³¼ |
|------|------|--------|------|------|
| v1.0 | ë¨ì¼ ë§ì»¤ (íì²­) | n=63 | Stage III CRC | â ê°­ ë°ê²¬ |
| v2.0 | ë©í°ë§ì»¤ í¨ë | 20â30ê° | ì¼ë° GI ì | ð ì ëµ ì í |
| **v3.0** | **AI-First / êµ­ê° ë°ì´ì¤ë°ì´í°** | **n=300â800** | **ì¡°ê¸° ê²ì¶** | â **ì§í ì¤** |

### ì§ì ë² ì´ì¤ êµ¬ì± (2026ë 2ì ê¸°ì¤)

```
ë¬¸í ì§ì ë² ì´ì¤ v2.0
âââ Tier 1 (100í¸): Nature / Cell / Science / Nature Medicine
âââ Tier 2 (100í¸): JCO / Cancer Research
âââ Tier 3: The Biology of Cancer (Weinberg)

íµê³:
â¢ 311í¸ ë¼ë¬¸ (ì´ë¡ ê¸°ë° GPT-4 ì¶ì¶)
â¢ 2,285 ì¤ì  TCGA ìí (BRCA, STAD, COAD, PAAD, READ)
â¢ 113ì¢ ì½ë¬¼ / 90ê° ê¸°ì  / 69ê° ë°ì´ì¤ë§ì»¤
â¢ 59ê° ìëì§ ì¡°í© ê²ì¦
```

### ìì íì¼ë¿ íë¡í ì½

```
íì¼ë¿ ì°êµ¬ ì¤ê³ (v1.0)
â¢ ëìì¸: ì í¥ì  íì¼ë¿, N=100 (ì¦ë¡ 50, ëì¡° 50)
â¢ ëª©í: Stage I 30% + Stage II 30% (ì¡°ê¸° ê²ì¶)
â¢ Go/No-Go ê¸°ì¤: AUC â¥ 0.75

3ê°ì ë¡ëë§µ:
â¢ Month 1: IRB ì ì¶ + ê³ì  ì¤ì 
â¢ Month 2: ì¹ì¸ íë³´ + ì¬ì´í¸ íì±í
â¢ Month 3: ë±ë¡ + Go/No-Go ê²°ì 
```

---

## â ï¸ ë°ì´í° ê°ì©ì± / Data Availability

íì CT ë°ì´í° ë° ìì íë¯¸ê²½ ì´ë¯¸ì§ë ì´ ë í¬ì§í ë¦¬ì **í¬í¨ëì§ ììµëë¤:**

- ð **PHI ê·ì ** (Protected Health Information): ê°ì¸ê±´ê°ì ë³´ ë³´í¸ë²
- ð **íì¼ í¬ê¸° ì í**: GitHub 100MB ì í (CT ë³¼ë¥¨ì ì GB)
- ð¥ **ê¸°ê´ ì¹ì¸ íì**: ì´íëíêµë³ì IRB ì¹ì¸ ë°ì´í°

ì¬íì ìí ë°ì´í° ì ê·¼ì ì ììê² ë¬¸ìíì¸ì.  
`data/samples/` ëë í ë¦¬ìë ìµëªíë ìê·ëª¨ ìíë§ í¬í¨ë©ëë¤.

---

## ð ì¸ì© / Citation

ì´ ì½ëë¥¼ ì°êµ¬ì ì¬ì©íì ë¤ë©´ ë¤ìì ì¸ì©í´ ì£¼ì¸ì:

```bibtex
@misc{adds2026,
  title     = {ADDS: AI-Driven Drug Synergy and Diagnostic System â 
               A Multimodal Precision Oncology Platform},
  author    = {Lee, Jaeyoung and others},
  year      = {2026},
  url       = {https://github.com/leejaeyoung-cpu/ADDS},
  note      = {Inha University Hospital, Incheon, Korea}
}
```

---

## ð¤ ê¸°ì¬ / Contributing

ê¸°ì¬ë¥¼ íìí©ëë¤! ì¸ë¶ ê°ì´ëë¼ì¸ì [CONTRIBUTING.md](.github/CONTRIBUTING.md)ë¥¼ ì°¸ì¡°íì¸ì.

**ë¹ ë¥¸ ê¸°ì¬ ê°ì´ë:**
1. `Fork` â `Feature Branch` ìì± (`feat/my-feature`)
2. ë³ê²½ì¬í­ ìì± + íì¤í¸ ì¶ê°
3. `Pull Request` ìì± (PR ííë¦¿ ìì± íì)

---

## ð ë³´ì / Security

ë³´ì ì·¨ì½ì  ë°ê²¬ ì ê³µê° ì´ìë¥¼ ìì±íì§ ë§ê³ , [SECURITY.md](.github/SECURITY.md)ì ê°ì´ëë¼ì¸ì ë°ë¼ ë¹ê³µê° ë³´ê³ í´ ì£¼ì¸ì.

---

## â ï¸ Methodological Notes / ë°©ë²ë¡  ì£¼ì

> **Transparency Statement**: All performance metrics are reported with their methodological context and limitations. This section is intended to support scientific reproducibility and honest evaluation.

### CT Tumor Detection (98.65% Accuracy)

| Item | Detail |
|------|--------|
| **Dataset** | Inha University Hospital CRC cohort |
| **Sample size** | N = 74 CT slices (single patient, arterial phase) |
| **Method** | HU-threshold (60â120 HU) + morphological filtering + connected-component analysis |
| **Ground truth** | Manual annotation by clinical radiologist |
| **Metric** | Slice-level detection accuracy (correct slices / total slices) |
| **95% CI** | [0.949, 1.000] (Wilson score interval) |
| **â ï¸ Limitation** | Single-patient pilot study. Multi-center validation with Nâ¥200 patients is ongoing. This metric does NOT represent patient-level diagnostic accuracy. |

### Cell Morphometry (N = 43,190 cells)

| Item | Detail |
|------|--------|
| **Instrument** | Brightfield microscopy |
| **Cell lines** | HUVEC (Human Umbilical Vein Endothelial Cells) |
| **Conditions** | 4 groups: Control Â· Healthy Serum Â· HGPS Serum Â· HGPS + MT-Exosome |
| **Images analyzed** | 80 brightfield images |
| **Segmentation** | Cellpose v3 (cyto3 model), GPU-accelerated |
| **â ï¸ Limitation** | In vitro model only. Clinical relevance requires PDO (Patient-Derived Organoid) validation. |

### Drug Synergy Models (TCGA N = 2,285)

| Item | Detail |
|------|--------|
| **Training data** | TCGA-COAD + DrugComb + OncoKB |
| **Synergy metrics** | Bliss Independence, Loewe Additivity, HSA, ZIP |
| **Model architecture** | DeepSynergy v2 (DNN) + XGBoost ensemble |
| **Validation** | 5-fold cross-validation on held-out TCGA subset |
| **â ï¸ Limitation** | Synergy predictions are based on genomic/transcriptomic features. Prospective clinical validation has not been conducted. Not for clinical use without regulatory approval. |

### Reproducibility

```bash
# Verify core scientific logic (no GPU required)
pip install -r requirements-ci.txt
python -m pytest tests/test_science_core.py -v
# Expected: 18 passed
```

All statistical tests, synergy formulas, and data integrity checks in `tests/test_science_core.py` pass with zero external dependencies.

---

## ð¬ ì°ë½ì² / Contact

| í­ëª© | ë´ì© |
|------|------|
| **ë í¬ì§í ë¦¬** | [github.com/leejaeyoung-cpu/ADDS](https://github.com/leejaeyoung-cpu/ADDS) |
| **ê¸°ê´** | ì´íëíêµë³ì, ì¸ì²ê´ì­ì, ëíë¯¼êµ­ |
| **ì°êµ¬ ë¶ì¼** | ì ë° ì¢ìí / AI ìë£ê¸°ê¸° (SaMD) |
| **ëª©í ì ë** | Nature Communications |

---

<div align="center">

**ADDS v3.5.0** â Built with â¤ï¸ for Precision Oncology  
Inha University Hospital Ã AI Research Team | 2026

</div>
