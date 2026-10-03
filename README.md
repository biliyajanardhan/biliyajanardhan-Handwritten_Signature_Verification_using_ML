# ✍️ AI-Based Handwritten Signature Verification

An AI-based handwritten signature verification system that combines a **Siamese Neural Network with ResNet-18** for feature extraction and a **Support Vector Machine (SVM)** for final signature classification.

The system compares a genuine/reference signature with a signature to be verified and classifies the pair as:

- ✅ **Genuine**
- ❌ **Forged**

The application is implemented using **Python, PyTorch, Scikit-learn, and Streamlit**.

---

## 🚀 Project Overview

Handwritten signature verification is an important biometric authentication technique used in applications such as:

- Banking and financial transactions
- Legal documents
- Identity verification
- Document authentication
- Automated signature screening

Instead of directly comparing raw signature images, this project uses a deep learning model to extract meaningful feature representations from the signatures.

The extracted features are then processed by a machine learning classifier to determine whether the two signatures belong to the same class.

---

## 🧠 Model Architecture

The project uses a **Siamese-style ResNet-18 feature extraction network combined with an SVM classifier**.

### Complete Pipeline

```text
                 Reference Signature
                         │
                         ▼
                    Image Preprocessing
                         │
                         ▼
                    ResNet-18 Backbone
                         │
                         ▼
                 128-Dimensional Embedding
                         │
                         │
                         │
                         │
                         │
                         │
                         │
                 ┌───────┴────────┐
                 │                │
                 │                │
                 ▼                ▼
        Signature Embedding 1  Signature Embedding 2
                 │                │
                 └───────┬────────┘
                         │
                         ▼
              Pair Feature Generation
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Absolute Difference    Element-wise
       |Embedding1 - Embedding2|   Multiplication
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                  256-D Feature Vector
                         │
                         ▼
                   StandardScaler
                         │
                         ▼
                    SVM Classifier
                         │
                         ▼
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
             Genuine            Forged
