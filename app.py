
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision import transforms
from PIL import Image

import numpy as np
import pickle
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Signature Verification",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================
# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN APPLICATION
       ===================================================== */

    .stApp {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef0ff 100%);
        color: #1f2937;
    }

    /* Main content width */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       TITLE
       ===================================================== */

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #29235c !important;
        margin-top: 10px;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #667085 !important;
        font-size: 18px;
        margin-bottom: 35px;
    }


    /* =====================================================
       UPLOAD CARDS
       ===================================================== */

    .upload-card {
        background: #ffffff !important;
        padding: 22px 24px;
        border-radius: 16px;
        border: 1px solid #e2e4f0;
        box-shadow: 0 8px 25px rgba(42, 35, 92, 0.08);
        margin-bottom: 12px;
        min-height: 120px;
    }

    .upload-card h3 {
        color: #29235c !important;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .upload-card p {
        color: #667085 !important;
        font-size: 15px;
        margin-bottom: 0;
    }


    /* =====================================================
       STREAMLIT FILE UPLOADER
       ===================================================== */

    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 1px solid #dedff0 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        box-shadow: 0 4px 12px rgba(42, 35, 92, 0.05);
    }

    div[data-testid="stFileUploader"] section {
        background: #ffffff !important;
        border: none !important;
    }

    div[data-testid="stFileUploader"] label {
        color: #4b5563 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] small {
        color: #6b7280 !important;
    }

    div[data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #423a8e !important;
        border: 1px solid #423a8e !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: #423a8e !important;
        color: #ffffff !important;
    }


    /* =====================================================
       UPLOAD LABELS
       ===================================================== */

    .stFileUploader label,
    [data-testid="stFileUploader"] label {
        color: #4b5563 !important;
    }


    /* =====================================================
       COMPARE BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;
        background: linear-gradient(
            135deg,
            #423a8e,
            #5549b8
        ) !important;

        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;

        padding: 13px 20px !important;
        font-size: 17px !important;
        font-weight: 700 !important;

        box-shadow: 0 6px 15px rgba(66, 58, 142, 0.25);

        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(
            135deg,
            #342d75,
            #423a8e
        ) !important;

        transform: translateY(-2px);

        box-shadow: 0 8px 20px rgba(66, 58, 142, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }


    /* =====================================================
       UPLOADED IMAGE SECTION
       ===================================================== */

    .uploaded-title {
        color: #29235c !important;
        font-size: 24px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* =====================================================
       SUCCESS RESULT
       ===================================================== */

    .result-genuine {
        background: linear-gradient(
            135deg,
            #ecfdf3,
            #dff8e9
        );

        border: 2px solid #22c55e;

        color: #166534 !important;

        padding: 28px;
        border-radius: 16px;

        text-align: center;

        font-size: 30px;
        font-weight: 800;

        margin-top: 25px;

        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.12);
    }


    /* =====================================================
       FORGED RESULT
       ===================================================== */

    .result-forged {
        background: linear-gradient(
            135deg,
            #fff1f2,
            #fee2e2
        );

        border: 2px solid #ef4444;

        color: #991b1b !important;

        padding: 28px;
        border-radius: 16px;

        text-align: center;

        font-size: 30px;
        font-weight: 800;

        margin-top: 25px;

        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.12);
    }


    /* =====================================================
       INFORMATION CARD
       ===================================================== */

    .info {
        background: #ffffff !important;

        padding: 24px;

        border-radius: 16px;

        border: 1px solid #e2e4f0;

        margin-top: 25px;

        box-shadow: 0 8px 25px rgba(42, 35, 92, 0.07);

        color: #374151 !important;
    }

    .info h3,
    .info h4 {
        color: #29235c !important;
    }

    .info p {
        color: #4b5563 !important;
    }


    /* =====================================================
       METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        background: #f8f8ff;
        border: 1px solid #e4e2f5;
        border-radius: 12px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #667085 !important;
    }

    [data-testid="stMetricValue"] {
        color: #423a8e !important;
        font-weight: 700 !important;
    }


    /* =====================================================
       SPINNER
       ===================================================== */

    .stSpinner > div {
        color: #423a8e !important;
    }


    /* =====================================================
       WARNING / ERROR
       ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #6b7280 !important;

        margin-top: 45px;
        padding: 25px;

        border-top: 1px solid #d9dbea;

        font-size: 14px;
        line-height: 1.7;
    }

    .footer b {
        color: #423a8e !important;
    }


    /* =====================================================
       HORIZONTAL DIVIDER
       ===================================================== */

    hr {
        border: none !important;
        border-top: 1px solid #dedff0 !important;
        margin: 30px 0;
    }


    /* =====================================================
       MOBILE RESPONSIVE
       ===================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .title {
            font-size: 30px;
        }

        .subtitle {
            font-size: 15px;
        }

        .upload-card {
            padding: 18px;
        }

        .result-genuine,
        .result-forged {
            font-size: 24px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SiameseResNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Same ResNet-18 architecture used during training
        self.backbone = models.resnet18(
            weights=None
        )

        num_features = self.backbone.fc.in_features

        # Remove original classification layer
        self.backbone.fc = nn.Identity()

        # 128-dimensional embedding head
        self.fc_head = nn.Sequential(

            nn.Linear(
                num_features,
                512
            ),

            nn.BatchNorm1d(512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                128
            )
        )


    def forward_once(self, x):

        features = self.backbone(x)

        embedding = self.fc_head(features)

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESNET_PATH = os.path.join(
    BASE_DIR,
    "siamese_resnet_signature_final.pth"
)

SVM_PATH = os.path.join(
    BASE_DIR,
    "signature_svm_classifier.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "signature_feature_scaler.pkl"
)


# ============================================================
# LOAD RESNET MODEL
# ============================================================

@st.cache_resource
def load_resnet():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SiameseResNet().to(device)

    checkpoint = torch.load(
        RESNET_PATH,
        map_location=device
    )

    # Support both normal state_dict
    # and checkpoint format

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model, device


# ============================================================
# LOAD SVM
# ============================================================

@st.cache_resource
def load_svm():

    with open(
        SVM_PATH,
        "rb"
    ) as file:

        svm_model = pickle.load(file)

    return svm_model


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    with open(
        SCALER_PATH,
        "rb"
    ) as file:

        scaler = pickle.load(file)

    return scaler


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    model, device = load_resnet()

    svm_model = load_svm()

    scaler = load_scaler()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "❌ Model files could not be loaded."
    )

    st.error(
        str(e)
    )


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# GET RESNET EMBEDDING
# ============================================================

def get_embedding(image):

    image = image.convert(
        "RGB"
    )

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        embedding = model.forward_once(
            image_tensor
        )

    # Convert to NumPy
    embedding = embedding.cpu().numpy()

    # Shape:
    # (1, 128)

    return embedding[0]


# ============================================================
# CREATE SVM FEATURES
# ============================================================

def create_pair_features(
    embedding1,
    embedding2
):

    # Absolute difference
    difference = np.abs(
        embedding1 - embedding2
    )

    # Element-wise multiplication
    product = (
        embedding1 * embedding2
    )

    # Combine both
    features = np.concatenate(
        [
            difference,
            product
        ]
    )

    return features


# ============================================================
# PREDICT SIGNATURE
# ============================================================

def predict_signature(
    image1,
    image2
):

    # ---------------------------------
    # Generate embeddings
    # ---------------------------------

    embedding1 = get_embedding(
        image1
    )

    embedding2 = get_embedding(
        image2
    )


    # ---------------------------------
    # Create pair features
    # ---------------------------------

    features = create_pair_features(
        embedding1,
        embedding2
    )


    # ---------------------------------
    # Reshape for Scaler
    # ---------------------------------

    features = features.reshape(
        1,
        -1
    )


    # ---------------------------------
    # Apply same scaler
    # used during SVM training
    # ---------------------------------

    scaled_features = scaler.transform(
        features
    )


    # ---------------------------------
# SVM prediction
# ---------------------------------

    prediction = svm_model.predict(
        scaled_features
    )[0]


    # ---------------------------------
    # SVM probability
    # ---------------------------------

    probability = None

    if hasattr(
        svm_model,
        "predict_proba"
    ):

        probabilities = svm_model.predict_proba(
            scaled_features
        )

        class_index = list(
            svm_model.classes_
        ).index(prediction)

        probability = float(
            probabilities[0][class_index]
        ) * 100


    return prediction, probability


# ============================================================
# HOME PAGE
# ============================================================

st.markdown(
    """
    <div class="title">
        ✍️ Handwritten Signature Verification
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Compare two signatures using ResNet-18 + SVM
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODELS
# ============================================================

if not models_loaded:

    st.stop()


# ============================================================
# UPLOAD SECTION
# ============================================================

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# SIGNATURE 1
# ------------------------------------------------------------

with col1:

    st.markdown(
        """
        <div class="upload-card">

        <h3>📄 Reference Signature</h3>

        <p>
        Upload the genuine/reference signature.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    image1_file = st.file_uploader(
        "Choose reference signature",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="image1"
    )


# ------------------------------------------------------------
# SIGNATURE 2
# ------------------------------------------------------------

with col2:

    st.markdown(
        """
        <div class="upload-card">

        <h3>✍️ Signature to Verify</h3>

        <p>
        Upload the signature you want to verify.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    image2_file = st.file_uploader(
        "Choose signature to verify",
        type=[
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "tif",
            "tiff"
        ],
        key="image2"
    )


# ============================================================
# DISPLAY IMAGES
# ============================================================

if image1_file and image2_file:

    image1 = Image.open(
        image1_file
    )

    image2 = Image.open(
        image2_file
    )

    st.markdown(
        "### Uploaded Signatures"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image1,
            caption="Reference Signature",
            use_container_width=True
        )

    with col2:

        st.image(
            image2,
            caption="Signature to Verify",
            use_container_width=True
        )


# ============================================================
# COMPARE BUTTON
# ============================================================

st.markdown("")


if st.button(
    "🔍 COMPARE SIGNATURES",
    use_container_width=True
):

    if not image1_file or not image2_file:

        st.warning(
            "⚠️ Please upload both signatures."
        )

    else:

        image1 = Image.open(
            image1_file
        )

        image2 = Image.open(
            image2_file
        )

        try:

            with st.spinner(
                "Analyzing signatures..."
            ):

                prediction, probability = predict_signature(
                    image1,
                    image2
                )


            # ==================================================
            # CONVERT SVM OUTPUT TO RESULT
            # ==================================================


            prediction_string = str(
                prediction
            ).lower()

            # SVM training labels:
            # 0 = Genuine
            # 1 = Forged

            if prediction_string == "0":
                result = "GENUINE"
            else:
                result = "FORGED"


            # ==================================================
            # DISPLAY RESULT
            # ==================================================

            if result == "GENUINE":

                st.markdown(
                    """
                    <div class="result-genuine">

                    ✅ GENUINE SIGNATURE

                    <br>

                    <span style="font-size:17px;">
                    The signatures are classified as similar.
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-forged">

                    ❌ FORGED SIGNATURE

                    <br>

                    <span style="font-size:17px;">
                    The signatures are classified as different.
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ==================================================
            # RESULT INFORMATION
            # ==================================================

            st.markdown(
                '<div class="info">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Prediction Details"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "SVM Prediction",
                    str(prediction)
                )

            with col2:

                if probability is not None:

                    st.metric(
                        "SVM Confidence",
                        f"{probability:.2f}%"
                    )

                else:

                    st.metric(
                        "Classifier",
                        "SVM"
                    )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)


