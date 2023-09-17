import torch
import torch.jit
import argparse
from my_model import MyModel  # Assuming you have defined your model in 'my_model.py'

def main(model_path, torchscript_path):
    # 1. Load the PyTorch model
    model = MyModel()  # Create an instance of your model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # 2. Trace the model using an example input
    example_input = torch.randn(1, 3, 224, 224)  # Example input (change according to your model's input shape)
    traced_model = torch.jit.trace(model, example_input)

    # 3. Save the traced model
    traced_model.save(torchscript_path)
    print(f'Saved TorchScript model to {torchscript_path}')

    # 4. Compare PyTorch and TorchScript results
    with torch.no_grad():
        pytorch_output = model(example_input)
        torchscript_output = traced_model(example_input)

    # Check if the outputs are close
    if torch.allclose(pytorch_output, torchscript_output, atol=0.001):
        print("The PyTorch and TorchScript outputs match!")
    else:
        print("The outputs do not match! Please investigate further.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to TorchScript and compare results.")
    parser.add_argument("model_path", type=str, help="Path to the PyTorch model file.")
    parser.add_argument("torchscript_path", type=str, help="Path to save the TorchScript model file.")
    args = parser.parse_args()
    
    main(args.model_path, args.torchscript_path)
