import streamlit as st
import random
import time
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator
from google import genai
from google.genai import types

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FruitSure",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Fetch API Key from Secrets ───────────────────────────────────────────────
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    gemini_api_key = None

# ── Class Labels — exact order from train_generator.class_indices ────────────
CLASS_NAMES = [
    "apple_formalin_mixed",   # 0
    "apple_fresh",            # 1
    "apple_rotten",           # 2
    "banana_formalin_mixed",  # 3
    "banana_fresh",           # 4
    "banana_rotten",          # 5
    "grape_formalin_mixed",   # 6
    "grape_fresh",            # 7
    "grape_rotten",           # 8
    "mango_formalin_mixed",   # 9
    "mango_fresh",            # 10
    "mango_rotten",           # 11
    "orange_formalin_mixed",  # 12
    "orange_fresh",           # 13
    "orange_rotten",          # 14
    "other",                  # 15
]

# ── Friendly display names ────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "apple_formalin_mixed":  "Apple (Chemical/Formalin)",
    "apple_fresh":           "Apple",
    "apple_rotten":          "Apple (Rotten)",
    "banana_formalin_mixed": "Banana (Chemical/Formalin)",
    "banana_fresh":          "Banana",
    "banana_rotten":         "Banana (Rotten)",
    "grape_formalin_mixed":  "Grape (Chemical/Formalin)",
    "grape_fresh":           "Grape",
    "grape_rotten":          "Grape (Rotten)",
    "mango_formalin_mixed":  "Mango (Chemical/Formalin)",
    "mango_fresh":           "Mango",
    "mango_rotten":          "Mango (Rotten)",
    "orange_formalin_mixed": "Orange (Chemical/Formalin)",
    "orange_fresh":          "Orange",
    "orange_rotten":         "Orange (Rotten)",
    "other":                 "Unknown / Not a Fruit",
}

def get_ripening_status(class_name: str):
    if class_name == "other":
        return "not_fruit"
    if "_fresh" in class_name:
        return "natural"
    if "_formalin_mixed" in class_name:
        return "chemical"
    if "_rotten" in class_name:
        return "rotten"
    return "not_fruit"

# ── Model Loading ─────────────────────────────────────────────────────────────
def _patch_dense_quantization():
    import keras
    from keras.layers import Dense

    original_from_config = Dense.from_config.__func__

    @classmethod
    def patched_from_config(cls, config):
        config.pop("quantization_config", None)
        return original_from_config(cls, config)

    Dense.from_config = patched_from_config


@st.cache_resource(show_spinner="Loading FruitSure model...")
def load_model():
    import keras

    MODEL_PATH = "fruit_model.keras"

    try:
        model = keras.saving.load_model(MODEL_PATH)
        return model
    except Exception as e:
        first_error = str(e)

    try:
        _patch_dense_quantization()
        model = keras.saving.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        second_error = str(e)

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.warning(f"Model could not be loaded: {first_error}. Running in demo mode.")
        return None


model = load_model()
MODEL_LOADED = model is not None


# ── Image Preprocessing ───────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_image(pil_image: Image.Image):
    if MODEL_LOADED:
        arr = preprocess_image(pil_image)
        preds = model.predict(arr, verbose=0)
        class_idx = int(np.argmax(preds[0]))
        confidence = int(round(float(preds[0][class_idx]) * 100))
        class_name = CLASS_NAMES[class_idx]
        display_name = DISPLAY_NAMES.get(class_name, class_name)
        status = get_ripening_status(class_name)
        if confidence < 50:
            return "other", "Unknown / Not a Fruit", "not_fruit", confidence
        return class_name, display_name, status, confidence
    else:
        demo_classes = ["apple_fresh", "banana_fresh", "mango_fresh", "orange_fresh",
                        "apple_formalin_mixed", "banana_formalin_mixed"]
        class_name = random.choice(demo_classes)
        display_name = DISPLAY_NAMES[class_name]
        status = get_ripening_status(class_name)
        confidence = random.randint(72, 97)
        return class_name, display_name, status, confidence


# ── Supported Languages ───────────────────────────────────────────────────────
LANGUAGE_OPTIONS = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Kannada": "kn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Arabic": "ar",
}

# ── Base English Strings ──────────────────────────────────────────────────────
BASE_STRINGS = {
    "title": "FruitSure",
    "subtitle": "Detect naturally vs chemically ripened fruits instantly",
    "upload_label": "Upload Fruit Image",
    "upload_hint": "Supported formats: JPG, JPEG, PNG",
    "predict_btn": "Analyze Fruit",
    "result_natural": "Naturally Ripened",
    "result_chemical": "Chemically Ripened (Carbide)",
    "result_natural_desc": "This fruit was ripened naturally without any chemical treatment.",
    "result_chemical_desc": "This fruit shows signs of chemical ripening using calcium carbide.",
    "nutrition_title": "Estimated Nutritional Values",
    "potassium": "Potassium",
    "sugar": "Sugar Level",
    "vitamin_c": "Vitamin C",
    "fiber": "Dietary Fiber",
    "safe_to_eat": "Safe to Eat",
    "caution": "Consume with Caution",
    "tip_title": "Health Tips",
    "tip_natural": "Naturally ripened fruits retain more nutrients, antioxidants, and have a better taste profile.",
    "tip_chemical": "Chemically ripened fruits may have lower nutritional value. Wash thoroughly before consuming.",
    "sidebar_title": "Settings & Info",
    "lang_label": "Select Language",
    "about_title": "About FruitSure",
    "about_text": "FruitSure uses deep learning to detect whether a fruit is naturally ripened or treated with calcium carbide : a common but harmful chemical used to speed up ripening.",
    "fact_title": "Did You Know?",
    "facts": "Calcium carbide releases acetylene gas which mimics natural ethylene.|Naturally ripened fruits have a sweeter richer flavor.|India is one of the world's largest fruit producers.|Most fruits are over 70 percent water by weight.",
    "history_title": "Analysis History",
    "no_history": "No analysis yet. Upload a fruit image.",
    "natural_label": "Natural",
    "chemical_label": "Carbide",
    "confidence": "Confidence",
    "analyzing": "Analyzing your fruit...",
    "demo_note": "Demo mode — connect your trained model for real predictions.",
    "model_note": "AI model loaded — predictions are real.",
    "upload_prompt": "Drag and drop or click to upload a fruit image",
    "file_label": "File",
    "size_label": "Size",
    "mode_label": "Mode",
    "total_analyses": "Total Analyses",
    "chat_title": "FruitSure Assistant",
    "chat_placeholder": "Ask anything about fruit ripening or nutrition...",
    "api_warning": "Chatbot unavailable: Gemini API key not found in secrets.toml.",
    "fruit_detected": "Fruit Detected",
}

# ── Translation with Caching ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_translations(lang_code: str) -> dict:
    if lang_code == "en":
        return BASE_STRINGS.copy()

    translator = GoogleTranslator(source='en', target=lang_code)
    translated = {}

    DELIMITER = " ||| "
    keys = list(BASE_STRINGS.keys())
    values = list(BASE_STRINGS.values())

    try:
        chunk_size = 10
        all_translated_values = []

        for i in range(0, len(values), chunk_size):
            chunk = values[i:i + chunk_size]
            joined = DELIMITER.join(chunk)
            result = translator.translate(joined)
            all_translated_values.extend(result.split(DELIMITER))

        if len(all_translated_values) != len(keys):
            all_translated_values = []
            for val in values:
                try:
                    all_translated_values.append(translator.translate(val))
                except Exception:
                    all_translated_values.append(val)

        translated = dict(zip(keys, all_translated_values))

    except Exception as e:
        st.warning(f"Translation error: {e}. Falling back to English.")
        return BASE_STRINGS.copy()

    return translated


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #F6C90E 0%, #F59E0B 50%, #D97706 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
}

.main-header h1 {
    font-family: 'Sora', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #1C1917;
    margin: 0;
    letter-spacing: -1px;
}

.main-header p {
    color: #44403C;
    font-size: 1.05rem;
    margin: 0.5rem 0 0;
    font-weight: 400;
}

.result-natural {
    background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
    border: 2px solid #22C55E;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-chemical {
    background: linear-gradient(135deg, #FEF2F2, #FECACA);
    border: 2px solid #EF4444;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-fruit {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 2px solid #3B82F6;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin: 0.75rem 0;
}

.result-icon { font-size: 2.5rem; display: block; margin-bottom: 0.5rem; font-style: normal; }
.result-title { font-size: 1.5rem; font-weight: 700; font-family: 'Sora', sans-serif; }
.result-desc { font-size: 0.95rem; margin-top: 0.5rem; opacity: 0.85; }

.nutrient-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}

.nutrient-card:hover { transform: translateY(-2px); }
.nutrient-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6B7280; font-weight: 600; }
.nutrient-value { font-size: 1.6rem; font-weight: 700; font-family: 'Sora', sans-serif; color: #1C1917; }
.nutrient-unit { font-size: 0.8rem; color: #9CA3AF; }

.tip-box {
    background: #FEF3C7;
    border-left: 5px solid #F59E0B;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    color: #78350F;   /* visible dark brown text */
    font-size: 0.95rem;
    font-weight: 500;
}

.history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.9rem;
    background: #F9FAFB;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    border: 1px solid #E5E7EB;
    color: #1F2937;   /* dark readable text */
    font-weight: 500;
}

.badge-natural {
    background: #DCFCE7; color: #15803D;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}

.badge-chemical {
    background: #FEE2E2; color: #DC2626;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600;
}

.translate-banner {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #166534;
    text-align: center;
    margin: 0.5rem 0 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #F6C90E, #F59E0B) !important;
    color: #1C1917 !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4) !important;
}

div[data-testid="stFileUploader"] {
    background: #FFFBEB;
    border: 2px dashed #F59E0B;
    border-radius: 16px;
    padding: 1rem;
}

.confidence-bar {
    background: #E5E7EB;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.confidence-fill-natural {
    background: linear-gradient(90deg, #22C55E, #4ADE80);
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease;
}

.confidence-fill-chemical {
    background: linear-gradient(90deg, #EF4444, #F87171);
    height: 100%;
    border-radius: 10px;
    transition: width 1s ease;
}

.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.chat-container {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #E5E7EB;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")

    selected_lang_name = st.selectbox(
        "Language",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
    )
    lang_code = LANGUAGE_OPTIONS[selected_lang_name]

    if lang_code != "en":
        with st.spinner("Translating..."):
            T = get_translations(lang_code)
    else:
        T = get_translations("en")

    st.markdown("---")
    st.markdown(f"### {T['about_title']}")
    st.markdown(f"<small>{T['about_text']}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### {T['fact_title']}")
    facts_list = T["facts"].split("|")
    fact = random.choice(facts_list)
    st.info(fact.strip())

    st.markdown("---")
    st.markdown(f"### {T['history_title']}")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            badge_class = "badge-natural" if item["type"] == "natural" else "badge-chemical"
            badge_text = T["natural_label"] if item["type"] == "natural" else T["chemical_label"]
            st.markdown(
                f'<div class="history-item"><span>{item["name"][:15]}...</span> '
                f'<span class="{badge_class}">{badge_text}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption(T["no_history"])

    if st.session_state.analysis_count > 0:
        st.markdown("---")
        st.metric(T["total_analyses"], st.session_state.analysis_count)

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>{T["title"]}</h1>
    <p>{T['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    T['upload_label'],
    type=["jpg", "jpeg", "png"],
    help=T["upload_hint"],
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col2:
        st.markdown(f"**{T['file_label']}:** {uploaded_file.name}")
        st.markdown(f"**{T['size_label']}:** {image.size[0]} × {image.size[1]} px")
        st.markdown(f"**{T['mode_label']}:** {image.mode}")
        st.markdown("---")
        predict = st.button(T['predict_btn'], use_container_width=True)

    if predict:
        with st.spinner(T["analyzing"]):
            time.sleep(1 if MODEL_LOADED else 2)

        class_name, display_name, status, confidence = predict_image(image)

        if status == "not_fruit":
            st.markdown("""
            <div style="background:#FEF9C3; border:2px solid #EAB308; border-radius:16px;
                        padding:1.5rem; text-align:center; margin:1rem 0;">
                <div style="font-size:2.5rem;">🚫</div>
                <div style="font-size:1.4rem; font-weight:700; color:#854D0E; margin-top:0.5rem;">
                    Not a Recognized Fruit!</div>
                <div style="font-size:0.95rem; color:#713F12; margin-top:0.5rem;">
                    Please upload a clear image of an apple, banana, grape, mango, or orange.</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            fruit_emoji = {
                "apple": "🍎", "banana": "🍌", "grape": "🍇",
                "mango": "🥭", "orange": "🍊"
            }
            base_fruit = class_name.split("_")[0]
            emoji = fruit_emoji.get(base_fruit, "🍑")

            st.markdown(f"""
            <div class="result-fruit">
                <span class="result-icon">{emoji}</span>
                <div class="result-title" style="color:#1D4ED8">{T.get('fruit_detected', 'Fruit Detected')}: {display_name}</div>
            </div>
            """, unsafe_allow_html=True)

            if status == "natural":
                nutrients = {
                    T["potassium"]: (random.randint(350, 400), "mg"),
                    T["sugar"]: (round(random.uniform(12.0, 14.0), 1), "g"),
                    T["vitamin_c"]: (round(random.uniform(8.0, 10.0), 1), "mg"),
                    T["fiber"]: (round(random.uniform(2.5, 3.5), 1), "g"),
                }
                st.markdown(f"""
                <div class="result-natural">
                    <span class="result-icon" style="color:#15803D;">&#10003;</span>
                    <div class="result-title" style="color:#15803D">{T['result_natural']}</div>
                    <div class="result-desc" style="color:#166534">{T['result_natural_desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                fill_class = "confidence-fill-natural"
                tip = T["tip_natural"]
                st.session_state.history.append({"name": uploaded_file.name, "type": "natural"})

            elif status == "chemical":
                nutrients = {
                    T["potassium"]: (random.randint(300, 340), "mg"),
                    T["sugar"]: (round(random.uniform(9.0, 11.0), 1), "g"),
                    T["vitamin_c"]: (round(random.uniform(5.0, 7.0), 1), "mg"),
                    T["fiber"]: (round(random.uniform(2.0, 2.4), 1), "g"),
                }
                st.markdown(f"""
                <div class="result-chemical">
                    <span class="result-icon" style="color:#DC2626;">&#9888;</span>
                    <div class="result-title" style="color:#DC2626">{T['result_chemical']}</div>
                    <div class="result-desc" style="color:#991B1B">{T['result_chemical_desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                fill_class = "confidence-fill-chemical"
                tip = T["tip_chemical"]
                st.session_state.history.append({"name": uploaded_file.name, "type": "chemical"})

            elif status == "rotten":
                nutrients = {
                    T["potassium"]: (random.randint(200, 280), "mg"),
                    T["sugar"]: (round(random.uniform(5.0, 8.0), 1), "g"),
                    T["vitamin_c"]: (round(random.uniform(1.0, 3.0), 1), "mg"),
                    T["fiber"]: (round(random.uniform(1.0, 1.8), 1), "g"),
                }
                st.markdown("""
                <div style="background:linear-gradient(135deg,#FFF7ED,#FED7AA);
                            border:2px solid #F97316; border-radius:16px;
                            padding:1.5rem; text-align:center; margin:1rem 0;">
                    <span style="font-size:2.5rem; display:block;">🤢</span>
                    <div style="font-size:1.5rem; font-weight:700; color:#C2410C;">Rotten Fruit</div>
                    <div style="font-size:0.95rem; color:#9A3412; margin-top:0.5rem;">
                        This fruit appears to be rotten. Do NOT consume it.</div>
                </div>
                """, unsafe_allow_html=True)
                fill_class = "confidence-fill-chemical"
                tip = "Rotten fruits can cause food poisoning. Discard them immediately and do not consume."
                st.session_state.history.append({"name": uploaded_file.name, "type": "chemical"})

            st.session_state.analysis_count += 1

            st.markdown(f"**{T['confidence']}: {confidence}%**")
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="{fill_class}" style="width:{confidence}%"></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f'<div class="section-title">{T["nutrition_title"]}</div>', unsafe_allow_html=True)
            cols = st.columns(4)
            for i, (name, (value, unit)) in enumerate(nutrients.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="nutrient-card">
                        <div class="nutrient-label">{name}</div>
                        <div class="nutrient-value">{value}</div>
                        <div class="nutrient-unit">{unit}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="tip-box">
                <strong>{T['tip_title']}:</strong><br>
                {tip}
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align:center; padding: 3rem 1rem; color: #9CA3AF;">
        <div style="font-size: 1.1rem; margin-top: 1rem; font-weight: 500;">{T['upload_prompt']}</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">{T['upload_hint']}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Chatbot Section ───────────────────────────────────────────────────────────
st.markdown(f'<div class="chat-container">', unsafe_allow_html=True)
st.markdown(f'<div class="section-title">{T["chat_title"]}</div>', unsafe_allow_html=True)

if gemini_api_key:
    try:
        if st.session_state.gemini_chat is None:
            client = genai.Client(api_key=gemini_api_key)
            st.session_state.gemini_chat = client

    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")

    # Display previous chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input(T["chat_placeholder"]):

        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            import time

            for attempt in range(3):
                try:
                    response = st.session_state.gemini_chat.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=(
                                f"You are the FruitSure Assistant, a friendly AI companion designed to help users "
        f"understand fruit ripening, nutrition, health, and safe agricultural practices. "

        f"Always respond in {selected_lang_name}. "
        f"If the user asks questions in another language, still reply in {selected_lang_name}. "

        "Behave like a supportive guide and knowledgeable friend. Respond politely, clearly, "
        "and helpfully in simple language. "

        "Your responsibilities include: "
        "1. Explaining natural vs chemically ripened fruits. "
        "2. Warning about calcium carbide and artificial ripening chemicals. "
        "3. Giving safe health advice related to fruits and nutrition. "
        "4. Suggesting safe washing, storage, and selection methods for fruits. "
        "5. Helping farmers learn natural ripening techniques. "
        "6. Recommending organic fertilizers like compost and vermicompost. "
        "7. Promoting eco-friendly agriculture and chemical-free farming. "

        "Avoid giving medical diagnoses. Suggest consulting professionals if needed. "
        "Keep responses short, friendly, and practical."
                            )
                        ),
                    )

                    reply = response.text

                    st.markdown(reply)

                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": reply}
                    )

                    break

                except Exception as e:

                    if attempt == 2:
                        st.error(f"Error communicating with Gemini: {e}")
                    else:
                        time.sleep(2)

else:
    st.warning(T["api_warning"])

st.markdown('</div>', unsafe_allow_html=True)

