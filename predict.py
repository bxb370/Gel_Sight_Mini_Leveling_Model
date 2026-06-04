"""Minimal mean-pixel model + ONNX export utilities."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def predict_average_pixel(pixel_values):
    """Return the average pixel value for input values in [0, 255]."""
    arr = np.asarray(pixel_values, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("pixel_values must not be empty")
    return float(arr.mean())


def export_mean_pixel_onnx(output_path="models/mean_pixel.onnx"):
    """Create an ONNX model that computes mean(pixel_values) per sample.

    Input shape:  [N, P] where N=batch size and P=number of pixels.
    Output shape: [N, 1] mean pixel value for each sample.
    """
    input_tensor = helper.make_tensor_value_info("pixels", TensorProto.FLOAT, ["N", "P"])
    output_tensor = helper.make_tensor_value_info("avg_pixel", TensorProto.FLOAT, ["N", 1])

    reduce_mean_node = helper.make_node(
        "ReduceMean",
        inputs=["pixels"],
        outputs=["avg_pixel"],
        axes=[1],
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
    example_pixels = [0, 128, 255, 100, 50]
    avg = predict_average_pixel(example_pixels)
    print(f"Python average pixel: {avg:.2f}")

    onnx_path = export_mean_pixel_onnx()
    print(f"ONNX model saved to: {onnx_path}")
