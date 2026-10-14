import os
import torch
from transformers import AutoModel, AutoTokenizer


def export_to_onnx():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "onnx_model"
    os.makedirs(output_dir, exist_ok=True)

    print(f" Downloading model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Dummy input untuk menentukan tensor shape saat export
    dummy_text = "Operating profit increased by 10 percent."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f" Exporting model to ONNX format at '{onnx_path}'...")

    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    # Save tokenizer agar bisa dipakai saat inference
    tokenizer.save_pretrained(output_dir)
    print(" Export ONNX & Tokenizer berhasil!")


if __name__ == "__main__":
    export_to_onnx()