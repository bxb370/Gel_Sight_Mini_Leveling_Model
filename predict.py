"""Create/export a single-image mean-pixel ONNX model and preprocessing utilities.

Backend contract for this ONNX model:
1. Decode the image file (PNG/JPG/etc.) outside ONNX.
2. Convert image to grayscale (single channel).
3. Flatten grayscale pixels to a 1D vector of length P.
4. Cast to float32 and keep shape [P] (no batch dimension).
5. Feed this tensor to ONNX input name ``pixels``.

The model output ``avg_pixel`` has shape [1] (single score).
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from PIL import Image


def predict_average_pixel(pixel_values):
    """Return the average pixel value for input values in [0, 255]."""
    arr = np.asarray(pixel_values, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("pixel_values must not be empty")
    return float(arr.mean())


def load_image_pixels(image_path):
    """Load an image as grayscale and return flattened pixel values."""
    img = Image.open(image_path).convert("L")
    img_array = np.asarray(img, dtype=np.float32)
    return img_array.ravel()


def prepare_onnx_input_from_image(image_path):
    """Prepare one image for ONNX inference as float32 tensor [P].

    What backend needs to do before calling the ONNX model with an image:
    1. Read/decode image bytes with an image library.
    2. Convert to grayscale to produce one pixel value per location (0-255).
    3. Flatten to a 1D pixel vector.
    4. Cast to float32 and keep a 1D vector [P].

    Returns:
        np.ndarray: float32 array with shape [P] for input ``pixels``.
    """
    pixels = load_image_pixels(image_path)
    return pixels.astype(np.float32)


def predict_average_pixel_from_image(image_path):
    """Compute average pixel value directly from an image file path."""
    pixels = load_image_pixels(image_path)
    return predict_average_pixel(pixels)


def load_color_image_pixels(image_path):
    """Load an image in its native mode and return flattened pixel values."""
    img = Image.open(image_path)
    img_array = np.asarray(img, dtype=np.float32)
    return img_array.ravel()


def load_rgb_image_pixels(image_path):
    """Load an image as RGB and return flattened pixel values."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.asarray(img, dtype=np.float32)
    return img_array.ravel()


def predict_average_color_pixel_from_image(image_path):
    """Compute average pixel value from image pixels without RGB conversion."""
    pixels = load_color_image_pixels(image_path)
    return predict_average_pixel(pixels)


def predict_average_rgb_pixel_from_image(image_path):
    """Compute average pixel value after converting the image to RGB."""
    pixels = load_rgb_image_pixels(image_path)
    return predict_average_pixel(pixels)


def run_mean_pixel_onnx(image_path, model_path="models/mean_pixel.onnx"):
    """Run ONNX mean-pixel model directly on a single input image.

    Steps:
    1. Convert image to grayscale and flatten to [P].
    2. Feed tensor to ONNX input "pixels".
    3. Return scalar output from ONNX output "avg_pixel".
    """
    pixels = prepare_onnx_input_from_image(image_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: pixels})[0]

    return float(np.asarray(output).reshape(-1)[0])


def export_mean_pixel_onnx(output_path="models/mean_pixel.onnx"):
    """Create an ONNX model that computes mean grayscale pixel for one image.

    Input:
        name: ``pixels``
        type: float32
        shape: [P]
            P = flattened grayscale pixel count for one image

    Output:
        name: ``avg_pixel``
        type: float32
        shape: [1]

    Note for backend integration:
    ONNX does not decode PNG files directly in this model. Decode image and
    preprocess first (grayscale -> flatten -> float32 -> [P]), then call
    ONNX runtime with input name ``pixels``.
    """
    input_tensor = helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["P"])
    output_tensor = helper.make_tensor_value_info("avg_pixel", TensorProto.FLOAT, [1])

    reduce_mean_node = helper.make_node(
        "ReduceMean",
        inputs=["pixels"],
        outputs=["avg_pixel"],
        axes=[0],
        keepdims=1,
    )

    graph = helper.make_graph(
        nodes=[reduce_mean_node],
        name="MeanPixelGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    model = helper.make_model(
        graph,
        producer_name="mean-pixel-model",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_file)
    return output_file


if __name__ == "__main__":
    requested_name = "iestImage0.png"
    candidates = [Path(requested_name), Path("testImage0.png")]
    image_path = next((p for p in candidates if p.exists()), None)

    if image_path is None:
        names = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Could not find test image. Checked: {names}")

    onnx_path = Path("models/mean_pixel.onnx")
    if not onnx_path.exists():
        onnx_path = export_mean_pixel_onnx(onnx_path)

    avg = run_mean_pixel_onnx(image_path, onnx_path)
    print(f"ONNX average pixel value for {image_path}: {avg:.2f}")

    print(f"ONNX model used: {onnx_path}")

    color_avg = predict_average_color_pixel_from_image(image_path)
    native_mode = Image.open(image_path).mode
    print(f"Native-mode ({native_mode}) average pixel value for {image_path}: {color_avg:.2f}")

    native_pixels = load_color_image_pixels(image_path)
    print(f"First 10 native-mode pixel values: {native_pixels[:10]}")
    print(f"Last 10 native-mode pixel values:  {native_pixels[-10:]}")

    rgb_avg = predict_average_rgb_pixel_from_image(image_path)
    rgb_pixels = load_rgb_image_pixels(image_path)
    print(f"RGB-converted average pixel value for {image_path}: {rgb_avg:.2f}")
    print(f"First 10 RGB pixel values: {rgb_pixels[:10]}")
    print(f"Last 10 RGB pixel values:  {rgb_pixels[-10:]}")




    

    
