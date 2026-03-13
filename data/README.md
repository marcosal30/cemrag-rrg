# Data

CEMRAG is evaluated on two publicly available chest X-ray datasets.
**Raw data is not included in this repository** and must be downloaded separately.

---

## MIMIC-CXR

**Paper:** Johnson et al., 2019 — [MIMIC-CXR: A large publicly available database of labeled chest radiographs](https://doi.org/10.1038/s41597-019-0322-0)
**Access:** Requires PhysioNet credentialed account — [https://physionet.org/content/mimic-cxr/](https://physionet.org/content/mimic-cxr/)

### Download
```bash
# Requires PhysioNet credentials
wget -r -N -c -np --user <your-username> --ask-password \
    https://physionet.org/files/mimic-cxr-jpg/2.0.0/ \
    -P data/mimic_cxr/
```

### Expected structure
```
data/mimic_cxr/
├── files/                   # chest X-ray images (JPEG)
│   └── p10/.../CXR*.jpg
├── mimic-cxr-2.0.0-split.csv
├── mimic-cxr-2.0.0-metadata.csv
└── mimic-cxr-2.0.0-chexpert.csv
```

### Notes
- We use **frontal views only** (posteroanterior and anteroposterior projections): 156,344 images.
- Official train/val/test split from `mimic-cxr-2.0.0-split.csv` is used.
- Only the **Findings** section of each report is used as generation target.

---

## IU X-Ray

**Paper:** Demner-Fushman et al., 2015 — [Preparing a collection of radiology examinations for distribution and retrieval](https://doi.org/10.1093/jamia/ocv080)
**Access:** Open access — [https://openi.nlm.nih.gov/](https://openi.nlm.nih.gov/)

### Download
```bash
# Images
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz -P data/iu_xray/
tar -xzf data/iu_xray/NLMCXR_png.tgz -C data/iu_xray/

# Reports (XML)
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz -P data/iu_xray/
tar -xzf data/iu_xray/NLMCXR_reports.tgz -C data/iu_xray/
```

### Expected structure
```
data/iu_xray/
├── images/                  # PNG chest X-ray images
│   └── CXR*.png
└── reports/                 # XML report files
    └── *.xml
```

### Notes
- We use **frontal projections only**: 3,307 images.
- Samples without a Findings section are excluded.
- Split: 80% train / 10% validation / 10% test, with **patient-level separation**.
- **Retrieval is cross-domain**: IU X-Ray test images retrieve from the MIMIC-CXR training index.

---

## Preprocessing

Preprocessing scripts are in [`data/preprocessing/`](preprocessing/).

Images are resized according to each encoder's specification:
- **CXR-CLIP**: 224×224, normalized to `[-1, 1]`
- **LLaVA-Med**: 336×336 center crop with encoder-specific normalization
