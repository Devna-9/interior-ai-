import streamlit as st
import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import clip
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Interior Design Generator",layout="wide")

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Streamlit Cloud = CPU. Do not argue with physics.
device = "cpu"

# ---------------- CACHED MODEL LOADERS ----------------

@st.cache_resource
def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    return processor, model


# ---------------- FUNCTIONS ----------------

def generate_caption(image, processor, model):
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=50)
    return processor.decode(out[0], skip_special_tokens=True)


def generate_dalle_image(prompt):
    response = client.images.generate(model="gpt-image-1",prompt=prompt,size="1024x1024")
    return response.data[0].url


def clip_similarity(image, text, clip_model, preprocess):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)

    sim = cosine_similarity(image_features.cpu().numpy(),text_features.cpu().numpy())[0][0]

    return round(sim * 100, 2)


# ---------------- UI ----------------

st.title("AI-Powered Interior Design Generator")
st.caption("Multimodal AI using CLIP, BLIP, GPT & DALL·E")

uploaded_image = st.file_uploader("Upload a room image",type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

    # Load models only when needed
    clip_model, preprocess = load_clip()
    processor, blip_model = load_blip()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Room Image")
        st.image(image, use_column_width=True)

    with st.spinner("Understanding the room..."):
        caption = generate_caption(image, processor, blip_model)

    base_prompt = f"""
    Design a modern interior for the following room:
    {caption}
    """

    with st.spinner("Generating interior design..."):
        dalle_url = generate_dalle_image(base_prompt)

    with col2:
        st.subheader("AI Generated Interior Design")
        st.image(dalle_url, use_column_width=True)

    st.markdown("### Room Understanding")
    st.write(caption)

    # ---------------- PROMPT REFINEMENT ----------------

    st.markdown("### Modify the Design")
    user_prompt = st.text_area("Describe the changes you want",placeholder="e.g. minimalist, warm lights, wooden furniture")

    if st.button("Regenerate with My Prompt") and user_prompt.strip():
        final_prompt = base_prompt + "\nUser Preferences: " + user_prompt

        with st.spinner("Regenerating design..."):
            new_dalle_url = generate_dalle_image(final_prompt)

        st.subheader("Updated Interior Design")
        st.image(new_dalle_url, use_column_width=True)

        accuracy = clip_similarity(image,user_prompt,clip_model,preprocess)

        st.markdown("### Prompt–Image Alignment Accuracy")
        st.metric(label="CLIP Similarity Score",value=f"{accuracy}%")

        st.info("This score reflects semantic alignment between ","the user prompt and the original room image using CLIP.")
