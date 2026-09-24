import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Signature Verification System",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

# ============================================================
# CUSTOM CSS - FIXED COLORS
# ============================================================

st.markdown("""
<style>

    /* ==============================
       MAIN APP
       ============================== */

    .stApp {
        background-color: #f5f7fb;
        color: #1f2937;
    }


    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #171b2e;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }


    /* ==============================
       MAIN TITLE
       ============================== */

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #202a55 !important;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #5b6475 !important;
        font-size: 18px;
        margin-bottom: 35px;
    }


    /* ==============================
       CARDS
       ============================== */

    .card {
        background-color: #ffffff !important;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        color: #1f2937 !important;
    }

    .card h2,
    .card h3 {
        color: #202a55 !important;
    }

    .card p {
        color: #4b5563 !important;
        font-size: 16px;
        line-height: 1.6;
    }


    /* ==============================
       ALL MAIN TEXT
       ============================== */

    .main .block-container {
        color: #1f2937;
    }

    .main h1,
    .main h2,
    .main h3,
    .main h4 {
        color: #202a55 !important;
    }

    .main p,
    .main li {
        color: #374151 !important;
    }


    /* ==============================
       HOW IT WORKS
       ============================== */

    .main strong {
        color: #202a55 !important;
    }


    /* ==============================
       RESULT - ORIGINAL
       ============================== */

    .result-original {
        background-color: #dcfce7 !important;
        border: 2px solid #22c55e;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: #166534 !important;
        font-size: 30px;
        font-weight: bold;
    }

    .result-original span {
        color: #166534 !important;
    }


    /* ==============================
       RESULT - FORGED
       ============================== */

    .result-forged {
        background-color: #fee2e2 !important;
        border: 2px solid #ef4444;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: #991b1b !important;
        font-size: 30px;
        font-weight: bold;
    }

    .result-forged span {
        color: #991b1b !important;
    }


    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {
        background-color: #423a8e !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    .stButton > button:hover {
        background-color: #342d75 !important;
        color: #ffffff !important;
    }


    /* ==============================
       FILE UPLOADER
       ============================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 10px;
    }

    [data-testid="stFileUploader"] * {
        color: #374151 !important;
    }


    /* ==============================
       METRICS
       ============================== */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    [data-testid="stMetricValue"] {
        color: #202a55 !important;
    }


    /* ==============================
       INFO BOX
       ============================== */

    [data-testid="stAlert"] {
        color: #374151 !important;
    }


    /* ==============================
       CODE BLOCK
       ============================== */

    code {
        color: #202a55;
    }


    /* ==============================
       FOOTER
       ============================== */

    .footer {
        text-align: center;
        color: #6b7280 !important;
        padding: 30px 0 10px 0;
        margin-top: 50px;
        border-top: 1px solid #d1d5db;
    }

    .footer b {
        color: #202a55 !important;
    }
    
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SiameseResNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        # IMPORTANT:
        # Same architecture used during training
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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SiameseResNet().to(device)

    checkpoint = torch.load(
        "siamese_resnet_signature_final.pth",
        map_location=device
    )

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
# LOAD MODEL SAFELY
# ============================================================

try:

    model, device = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "Model could not be loaded. "
        "Make sure 'siamese_resnet_signature_final.pth' "
        "is in the same folder as app.py."
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
# GET EMBEDDING
# ============================================================

def get_embedding(
    image,
    model,
    device
):

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

    return embedding


# ============================================================
# COMPARE SIGNATURES
# ============================================================

def compare_signatures(
    image1,
    image2,
    model,
    device
):

    embedding1 = get_embedding(
        image1,
        model,
        device
    )

    embedding2 = get_embedding(
        image2,
        model,
        device
    )

    distance = torch.norm(
        embedding1 - embedding2,
        p=2
    ).item()

    return distance


# ============================================================
# THRESHOLD
# ============================================================

# Temporary threshold from current testing.
# Replace this after calculating the optimal validation threshold.

THRESHOLD = 1.0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## ✍️ Signature AI"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Verification",
            "📖 Project Explanation",
            "ℹ️ About"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        **AI-Based Signature Verification**

        Siamese Neural Network  
        + ResNet-18  
        + Contrastive Learning
        """
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '✍️ AI-Based Signature Verification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Verify handwritten signatures using Deep Learning'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="card">
            <h3>🧠 AI Model</h3>
            <p>
            Siamese Neural Network with
            ResNet-18 architecture.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="card">
            <h3>🔎 Verification</h3>
            <p>
            Compare two handwritten signatures
            using embedding distance.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="card">
            <h3>⚡ Fast Prediction</h3>
            <p>
            Generate embeddings and calculate
            Euclidean distance instantly.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("## How It Works")

    st.markdown(
        """
        ### 1️⃣ Upload Reference Signature

        Upload a known genuine signature.

        ### 2️⃣ Upload Signature to Verify

        Upload the signature that needs verification.

        ### 3️⃣ AI Comparison

        The Siamese Neural Network converts both
        signatures into numerical embeddings.

        ### 4️⃣ Distance Calculation

        Euclidean distance is calculated between
        the two embeddings.

        ### 5️⃣ Final Result

        The distance is compared with a threshold.

        **Small distance → Similar signatures**

        **Large distance → Different signatures**
        """
    )


# ============================================================
# VERIFICATION
# ============================================================

elif page == "🔍 Verification":

    st.markdown(
        '<div class="main-title">'
        '🔍 Signature Verification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload two signatures to compare them'
        '</div>',
        unsafe_allow_html=True
    )


    if not model_loaded:

        st.stop()


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 📄 Reference Signature"
        )

        reference_file = st.file_uploader(
            "Upload original/reference signature",
            type=[
                "png",
                "jpg",
                "jpeg",
                "bmp",
                "tif",
                "tiff"
            ],
            key="reference"
        )


    with col2:

        st.markdown(
            "### ✍️ Signature to Verify"
        )

        test_file = st.file_uploader(
            "Upload signature to verify",
            type=[
                "png",
                "jpg",
                "jpeg",
                "bmp",
                "tif",
                "tiff"
            ],
            key="test"
        )


    if reference_file and test_file:

        reference_image = Image.open(
            reference_file
        )

        test_image = Image.open(
            test_file
        )


        col1, col2 = st.columns(2)


        with col1:

            st.image(
                reference_image,
                caption="Reference Signature",
                use_container_width=True
            )


        with col2:

            st.image(
                test_image,
                caption="Signature to Verify",
                use_container_width=True
            )


        st.markdown("")


        if st.button(
            "🔍 VERIFY SIGNATURE",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing signatures..."
            ):

                distance = compare_signatures(

                    reference_image,

                    test_image,

                    model,

                    device
                )


            if distance <= THRESHOLD:

                st.markdown(
                    """
                    <div class="result-original">
                    ✅ ORIGINAL (O)
                    <br>
                    <span style="font-size:18px;">
                    Signatures are sufficiently similar
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-forged">
                    ❌ FORGED (F)
                    <br>
                    <span style="font-size:18px;">
                    Signatures are sufficiently different
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown("")


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Euclidean Distance",
                    f"{distance:.4f}"
                )


            with col2:

                st.metric(
                    "Threshold",
                    f"{THRESHOLD:.4f}"
                )


            with col3:

                if distance <= THRESHOLD:

                    st.metric(
                        "Decision",
                        "ORIGINAL"
                    )

                else:

                    st.metric(
                        "Decision",
                        "FORGED"
                    )


            st.info(
                "Note: The current threshold is a "
                "temporary threshold used for demonstration. "
                "For final evaluation, it should be optimized "
                "using validation data."
            )


# ============================================================
# PROJECT EXPLANATION
# ============================================================

elif page == "📖 Project Explanation":

    st.markdown(
        '<div class="main-title">'
        '📖 Project Explanation'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown("## 1. Project Overview")

    st.write(
        """
        This project is an AI-based handwritten signature
        verification system developed using a Siamese Neural
        Network and ResNet-18.

        The objective is to determine whether two handwritten
        signatures are sufficiently similar or different.

        Instead of treating signature verification as a normal
        image classification problem, the system learns a
        similarity relationship between two signatures.
        """
    )


    st.markdown("## 2. Problem Statement")

    st.write(
        """
        Traditional signature verification can depend on
        manual inspection, which can be time-consuming and
        subjective.

        The proposed system uses Deep Learning to automatically
        compare handwritten signatures and identify whether the
        signatures appear to belong to the same writer.
        """
    )


    st.markdown("## 3. Dataset")

    st.write(
        """
        The model was trained using handwritten signature
        datasets, including the CEDAR signature dataset.

        The CEDAR dataset contains genuine signatures and
        forged signatures from multiple writers.

        These signatures are used to construct pairs for
        Siamese Network training.
        """
    )


    st.markdown("## 4. Siamese Neural Network")

    st.write(
        """
        A Siamese Neural Network consists of two identical
        branches that share the same weights.

        Both signature images are passed through the same
        feature extraction network.

        The network generates an embedding for each signature.

        The embeddings are then compared using Euclidean
        distance.
        """
    )


    st.markdown("## 5. ResNet-18")

    st.write(
        """
        ResNet-18 is used as the feature extraction backbone.

        ResNet uses residual connections that help deep neural
        networks learn useful image features effectively.

        In this project, the original classification layer
        is replaced with a custom embedding head.
        """
    )


    st.markdown("## 6. Embedding")

    st.write(
        """
        Each signature is converted into a 128-dimensional
        numerical representation called an embedding.

        Signatures that are visually and structurally similar
        should produce embeddings that are closer together.

        Different signatures should produce embeddings that
        are farther apart.
        """
    )


    st.markdown("## 7. Contrastive Loss")

    st.write(
        """
        Contrastive Loss is used to train the Siamese Network.

        The model learns to reduce the distance between
        similar signature pairs and increase the distance
        between dissimilar pairs.

        This makes Contrastive Loss suitable for signature
        verification.
        """
    )


    st.markdown("## 8. Euclidean Distance")

    st.write(
        """
        After generating embeddings for two signatures,
        Euclidean distance is calculated.

        A smaller distance indicates greater similarity.

        A larger distance indicates greater difference.
        """
    )


    st.code(
        """
Distance = ||Embedding 1 - Embedding 2||
        """,
        language="text"
    )


    st.markdown("## 9. Verification Process")

    st.write(
        """
        The complete verification process is:

        1. Upload a reference signature.
        2. Upload the signature to verify.
        3. Preprocess both images.
        4. Generate embeddings using ResNet-18.
        5. Calculate Euclidean distance.
        6. Compare the distance with the threshold.
        7. Display the final verification result.
        """
    )


    st.markdown("## 10. Decision Logic")

    st.code(
        """
if distance <= threshold:
    ORIGINAL (O)
else:
    FORGED (F)
        """,
        language="python"
    )


    st.markdown("## 11. Technologies Used")

    st.markdown(
        """
        - 🐍 Python
        - 🧠 PyTorch
        - 🔥 ResNet-18
        - 🔗 Siamese Neural Network
        - 📉 Contrastive Loss
        - 🖼️ PIL
        - 🎨 Streamlit
        - 📊 Euclidean Distance
        - ☁️ Google Colab
        - 📁 CEDAR Dataset
        """
    )


    st.markdown("## 12. Training Results")

    st.write(
        """
        During training, the model achieved its best observed
        validation accuracy of approximately 89.83% at Epoch 3,
        with a validation loss of approximately 0.2755.

        Later epochs showed some fluctuation in validation
        performance, so the best validation checkpoint should
        be preferred for final evaluation when available.
        """
    )


    st.markdown("## 13. Applications")

    st.markdown(
        """
        - Banking and financial document verification
        - Legal document verification
        - Cheque processing
        - Identity verification
        - Document authentication
        - Academic certificate verification
        - Automated signature screening
        """
    )


    st.markdown("## 14. Limitations")

    st.markdown(
        """
        - Performance depends on the quality of input images.
        - Threshold selection affects the final decision.
        - Very different writing conditions can affect similarity.
        - A limited test set may not represent real-world performance.
        - The system should be evaluated using larger unseen datasets
          before deployment in high-stakes applications.
        """
    )


    st.markdown("## 15. Future Improvements")

    st.markdown(
        """
        - Optimize the verification threshold using validation data.
        - Evaluate using FAR, FRR and EER.
        - Increase the size and diversity of the training dataset.
        - Add image-quality checking.
        - Build a database of authorized reference signatures.
        - Deploy the system as a web application.
        - Add authentication and secure storage for signatures.
        """
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">'
        'ℹ️ About the Project'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        <h2>AI-Based Handwritten Signature Verification</h2>

        <p>
        This application demonstrates an AI-powered handwritten
        signature verification system using a Siamese Neural
        Network with a ResNet-18 feature extractor.
        </p>

        <h3>Model</h3>

        <p>
        Siamese ResNet-18
        </p>

        <h3>Embedding Size</h3>

        <p>
        128 dimensions
        </p>

        <h3>Comparison Method</h3>

        <p>
        Euclidean Distance
        </p>

        <h3>Training Approach</h3>

        <p>
        Contrastive Learning
        </p>

        <h3>Interface</h3>

        <p>
        Streamlit
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("### How to Use")

    st.markdown(
        """
        **Step 1:** Go to **Verification**.

        **Step 2:** Upload the genuine/reference signature.

        **Step 3:** Upload the signature you want to verify.

        **Step 4:** Click **VERIFY SIGNATURE**.

        **Step 5:** The application calculates the embedding
        distance and displays the result.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>AI-Based Handwritten Signature Verification</b>
    <br><br>

    Siamese Neural Network • ResNet-18 • Contrastive Learning
    <br>

    Developed using Python, PyTorch and Streamlit
    <br><br>

    © 2026 Signature Verification System

    </div>
    """,
    unsafe_allow_html=True
)