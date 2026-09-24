# ✍️ AI-Based Handwritten Signature Verification

An AI-powered handwritten signature verification system that uses a **Siamese Neural Network with ResNet-18** to determine whether two handwritten signatures are sufficiently similar or likely forged.

## 🚀 Project Overview

Signature verification is an important biometric authentication technique used in banking, legal documents, financial transactions, and identity verification.

This project uses **deep learning and metric learning** to compare two signature images and calculate their similarity using learned feature embeddings.

The system takes:

- 📝 Reference Signature
- 📝 Signature to Verify

and produces:

- **Similarity Distance**
- **Verification Result**
- **Genuine / Forged prediction**

---

## 🧠 Model Architecture

The project uses a **Siamese Neural Network** consisting of:

```text
Input Signature 1 ──┐
                    ├── ResNet-18 ── Feature Embedding ──┐
Input Signature 2 ──┘                                     │
                                                          ↓
                                                Euclidean Distance
                                                          ↓
                                                   Threshold
                                                          ↓
                                             Genuine / Forged
