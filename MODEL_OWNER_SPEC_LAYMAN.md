# LungLens Model Handoff (Simple Version)

This is the easy version for teammates using Colab.

You do **not** need perfect documentation.  
Just send the minimum items below so backend can integrate your model.

## Quick Step Navigation

[Step 1](#step-1-send-model-file) | [Step 2](#step-2-send-class-mapping) | [Step 3](#step-3-send-preprocessing) | [Step 4](#step-4-send-one-sample-prediction)

## Step 1: Send model file

- Send your model artifact.
- Example: `.h5`, `.pth`, `.pt`, `.safetensors`

[Back to Steps](#quick-step-navigation) | [Next: Step 2](#step-2-send-class-mapping)

## Step 2: Send class mapping

- Tell us class names in exact index order.
- Example:
  - `0 = Normal`
  - `1 = Lung Opacity`
  - `2 = Viral Pneumonia`

[Back: Step 1](#step-1-send-model-file) | [Next: Step 3](#step-3-send-preprocessing)

## Step 3: Send preprocessing

- image size (224x224?)
- RGB or BGR
- normalization (`/255`, `[-1,1]`, etc.)

[Back: Step 2](#step-2-send-class-mapping) | [Next: Step 4](#step-4-send-one-sample-prediction)

## Step 4: Send one sample prediction

- one test image
- raw model output (example: `[0.1, 0.8, 0.1]`)
- final predicted class

[Back: Step 3](#step-3-send-preprocessing) | [Back to Steps](#quick-step-navigation)

If you send Step 1 to Step 4, we can already start integration.

## Model type quick notes

### A) Binary model (Normal vs Pneumonia)

Expected:
- output a pneumonia probability, or
- output 2 values `[Normal, Pneumonia]`

### B) 3-class model

Expected classes:
- `Normal`
- `Lung Opacity`
- `Viral Pneumonia`

Important: tell us exact index order (0/1/2).

### C) Clinical/tabular model (questionnaire)

Expected outputs:
- severity: `low/moderate/high`
- risk: `low/medium/high`
- recovery: `favorable/guarded/uncertain`

## Optional but helpful

If available, also share:
- TensorFlow / PyTorch version
- Colab notebook link
- thresholds used for final class decision
- Grad-CAM code (if you already have it)

## Copy-paste reply template

```text
Model owner:
Model type: (binary / 3-class / clinical)
Model file name:

Class mapping:
0 =
1 =
2 = (if any)

Preprocessing:
- size:
- color mode:
- normalization:

Sample prediction:
- image:
- raw output:
- predicted class:

Framework version:
Notebook link (optional):
```

## Why this matters

If class order or preprocessing is wrong, model can run but give wrong results.  
So the most important things are:

1. exact class mapping  
2. exact preprocessing  
3. one real sample output
