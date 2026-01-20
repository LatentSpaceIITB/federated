# Dataset Documentation

## Overview

We use **3 real-world chest X-ray datasets** from different institutions to simulate a realistic federated learning scenario where hospitals cannot share data but want to collaboratively train a model.

---

## 1. NIH ChestX-ray14

### Source Information
- **Institution**: National Institutes of Health (NIH) Clinical Center
- **Download**: Kaggle - `nih-chest-xrays/data`
- **Size**: ~45 GB (112,120 images)
- **Patients**: 30,805 unique patients

### Download Command
```bash
kaggle datasets download nih-chest-xrays/data
unzip data.zip -d data/nih_chestxray/
```

### Data Structure
```
data/nih_chestxray/
├── images_001/images/     # ~10K images per folder
├── images_002/images/
├── ...
├── images_012/images/
├── Data_Entry_2017.csv    # Labels and metadata
├── BBox_List_2017.csv     # Bounding boxes (not used)
└── train_val_list.txt     # Official splits (if available)
```

### Label Format
```csv
Image Index,Finding Labels,Patient ID,...
00000001_000.png,Cardiomegaly|Emphysema,1,...
00000002_000.png,No Finding,2,...
```

### Label Mapping to CheXpert Tasks
| NIH Label | CheXpert Task |
|-----------|---------------|
| Atelectasis | Atelectasis |
| Cardiomegaly | Cardiomegaly |
| Consolidation | Consolidation |
| Edema | Edema |
| Effusion | Pleural Effusion |

### Characteristics
- Largest public chest X-ray dataset
- Extracted from clinical PACS
- NLP-extracted labels (some noise)
- Frontal view only

---

## 2. CheXpert

### Source Information
- **Institution**: Stanford University Hospital
- **Download**: Kaggle - `ashery/chexpert`
- **Size**: ~11 GB (224,316 images)
- **Patients**: 65,240 unique patients

### Download Command
```bash
kaggle datasets download -d ashery/chexpert
unzip chexpert.zip -d data/chexpert/
```

### Data Structure
```
data/chexpert/
├── train/
│   ├── patient00001/
│   │   └── study1/
│   │       └── view1_frontal.jpg
│   └── ...
├── valid/
│   └── ...
├── train.csv
└── valid.csv
```

### Label Format
```csv
Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Atelectasis,Cardiomegaly,...
train/patient00001/study1/view1_frontal.jpg,Female,68,Frontal,AP,1.0,,,,...
```

### Uncertainty Labels
CheXpert has **uncertainty labels** (-1) which we handle as:
- **U-Ones**: Treat uncertain as positive (our default)
- **U-Zeros**: Treat uncertain as negative
- **U-Ignore**: Exclude uncertain samples

### Competition Tasks (Our Target)
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Pleural Effusion

### Characteristics
- High-quality expert annotations
- Includes uncertainty labels
- Both frontal and lateral views
- Standard benchmark dataset

---

## 3. RSNA Pneumonia Detection Challenge

### Source Information
- **Institution**: Radiological Society of North America
- **Download**: Kaggle Competition - `rsna-pneumonia-detection-challenge`
- **Size**: ~3.7 GB (26,684 images)
- **Format**: DICOM files

### Download Command
```bash
# Requires accepting competition rules on Kaggle website first
kaggle competitions download -c rsna-pneumonia-detection-challenge
unzip rsna-pneumonia-detection-challenge.zip -d data/rsna_pneumonia/
```

### Data Structure
```
data/rsna_pneumonia/
├── stage_2_train_images/
│   ├── 0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm
│   └── ...
├── stage_2_train_labels.csv
└── stage_2_detailed_class_info.csv
```

### Label Format
```csv
# stage_2_train_labels.csv
patientId,x,y,width,height,Target
0004cfab-14fd-4e49-80ba-63a80b6bddd6,,,,,0
00313ee0-9eaa-42f4-b0ab-c148ed3241cd,264,152,213,379,1

# stage_2_detailed_class_info.csv
patientId,class
0004cfab-14fd-4e49-80ba-63a80b6bddd6,No Lung Opacity / Not Normal
003d8fa0-6bf1-40ed-b54c-ac657f8495c5,Normal
```

### Label Mapping
| RSNA Class | CheXpert Task |
|------------|---------------|
| Lung Opacity (Target=1) | Consolidation |
| Lung Opacity present | Pleural Effusion (proxy) |

### Characteristics
- DICOM format (requires pydicom)
- Binary pneumonia detection
- Bounding box annotations available
- Competition dataset (high quality)

---

## 4. Dataset Comparison

| Property | NIH ChestX-ray14 | CheXpert | RSNA |
|----------|------------------|----------|------|
| Institution | NIH | Stanford | RSNA/Kaggle |
| Images | 112,120 | 224,316 | 26,684 |
| Patients | 30,805 | 65,240 | ~26,000 |
| Labels | 14 findings | 14 findings | Binary |
| Format | PNG | JPG | DICOM |
| Size | 45 GB | 11 GB | 3.7 GB |
| Annotation | NLP-extracted | Expert + NLP | Expert |

---

## 5. Data Preprocessing

### Image Preprocessing Pipeline
```python
transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to MedCLIP input
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation (train only)
    transforms.RandomRotation(10),           # Augmentation (train only)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.586], std=[0.279]),  # ChestX-ray stats
    transforms.Lambda(lambda x: x.repeat(3,1,1)),     # Grayscale to RGB
])
```

### Label Preprocessing
1. Map dataset-specific labels to 5 CheXpert competition tasks
2. Handle missing labels as negative (0)
3. Handle uncertainty labels based on policy
4. Create multi-hot encoding: `[Atel, Card, Cons, Edem, PlEf]`

---

## 6. Train/Test Splits

### Strategy
We create splits that ensure:
- No patient overlap between train and test
- Approximately 80% train, 20% test
- Consistent splits across experiments (seed=42)

### Implementation
```python
# Patient-based split
unique_patients = df['patient_id'].unique()
np.random.shuffle(unique_patients)
split_idx = int(len(unique_patients) * 0.8)
train_patients = unique_patients[:split_idx]
test_patients = unique_patients[split_idx:]
```

---

## 7. Storage Requirements

| Dataset | Compressed | Extracted | With Cache |
|---------|------------|-----------|------------|
| NIH | 45 GB | ~50 GB | ~55 GB |
| CheXpert | 11 GB | ~15 GB | ~18 GB |
| RSNA | 3.7 GB | ~5 GB | ~7 GB |
| **Total** | **~60 GB** | **~70 GB** | **~80 GB** |

---

## 8. Kaggle Setup

### Prerequisites
1. Create Kaggle account: https://www.kaggle.com
2. Get API credentials: Account → API → Create New Token
3. Save to `~/.kaggle/kaggle.json`:
```json
{"username":"your_username","key":"your_api_key"}
```
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### For Competition Data (RSNA)
1. Go to competition page
2. Accept competition rules
3. Then download via API

---

## Quick Links

- [Project Overview](./01_PROJECT_OVERVIEW.md)
- [Training Methodology](./03_TRAINING_METHODOLOGY.md)
- [Novel Contributions](./04_NOVEL_CONTRIBUTIONS.md)
- [Experimental Setup](./05_EXPERIMENTAL_SETUP.md)
