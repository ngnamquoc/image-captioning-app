# Import Gradio for building the web interface
import gradio as gr

# Import the BLIP image captioning model and processor from Hugging Face Transformers
from transformers import BlipProcessor, BlipForConditionalGeneration

# Import PIL for image handling
from PIL import Image

# Load the pre-trained BLIP processor which handles image preprocessing and tokenization
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the pre-trained BLIP model for image captioning
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    """
    Generates a caption for the given image using the BLIP model.
    
    Parameters:
        image (PIL.Image): The input image to caption.
    
    Returns:
        str: Generated caption text.
    """
    # Preprocess the image into the format expected by the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate caption tokens using the model
    outputs = model.generate(**inputs)
    
    # Decode the token output into human-readable text
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

def caption_image(image):
    """
    Wraps the caption generation logic and handles exceptions gracefully.

    Parameters:
        image (PIL.Image): The input image.

    Returns:
        str: The generated caption or an error message.
    """
    try:
        # Try to generate a caption from the image
        caption = generate_caption(image)
        return caption
    except Exception as e:
        # Return the error message if something goes wrong
        return f"An error occurred: {str(e)}"

# Define a Gradio interface for the web app
iface = gr.Interface(
    fn=caption_image,              # Function to call when user submits an image
    inputs=gr.Image(type="pil"),   # Input component: upload an image (as a PIL object)
    outputs="text",                # Output component: display caption as text
    title="Image Captioning App",             # Title of the interface
    description="Upload an image to generate a caption."  # Short description
)

# Launch the Gradio app locally on port 7860
iface.launch(share=True)
