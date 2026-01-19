import importlib
import os
import shutil
import subprocess
import sys


def check_package(package_name, import_name=None, silent_on_failure=False):
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        # Try different attributes for version
        version = getattr(module, "__version__", getattr(module, "VERSION", "unknown"))
        print(f"✅ {package_name} installed (version: {version})")
        return True
    except ImportError as e:
        if not silent_on_failure:
            print(f"❌ {package_name} failed to import: {e}")
        return False


def check_cli_tool(tool_name):
    # Check in the same directory as the python executable (e.g., .venv/bin)
    bin_dir = os.path.dirname(sys.executable)
    venv_tool_path = os.path.join(bin_dir, tool_name)

    path_to_check = None
    # Check venv bin dir first
    if os.path.exists(venv_tool_path) and os.access(venv_tool_path, os.X_OK):
        path_to_check = venv_tool_path
    else:
        # Check system PATH if not found in venv
        path_to_check = shutil.which(tool_name)

    if not path_to_check:
        print(f"❌ {tool_name} NOT found")
        return False

    version_output = "version unknown"
    try:
        result = subprocess.run(
            [path_to_check, "--version"], capture_output=True, text=True
        )
        if result.stdout:
            version_output = result.stdout.strip().split("\n")[0]
    except Exception:
        pass  # Silently ignore version check errors

    print(f"✅ {tool_name} found ({version_output})")
    return True


def main():
    print(
        "============================================================================"
    )
    print("Verifying Machine Learning Course Environment Installation")
    print(f"Python Executable: {sys.executable}")
    print(
        "============================================================================"
    )

    all_passed = True

    print("\n--- Python Libraries ---")
    packages = [
        ("ua473 (Course Utils)", "utils"),  # Added check for local package
        ("streamlit", "streamlit"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("datasets", "datasets"),
        ("scikit-learn", "sklearn"),
        ("pytorch-lightning", "lightning"),
        ("torchvision", "torchvision"),
        ("Pillow", "PIL"),
        ("einops", "einops"),
        ("matplotlib", "matplotlib"),
        ("plotly", "plotly"),
        ("gymnasium", "gymnasium"),
        ("requests", "requests"),
    ]

    for pkg_name, import_name in packages:
        # Special handling for pytorch-lightning which might be imported as pytorch_lightning
        if pkg_name == "pytorch-lightning":
            # Try lightning first (newer versions style)
            if not check_package(pkg_name, "lightning", silent_on_failure=True):
                # Try pytorch_lightning (older versions style or explicit name)
                if not check_package(pkg_name, "pytorch_lightning"):
                    all_passed = False
        elif not check_package(pkg_name, import_name):
            all_passed = False

    print("\n--- Development Tools (CLI) ---")
    tools = ["pre-commit", "black", "flake8", "isort"]
    for tool in tools:
        if not check_cli_tool(tool):
            all_passed = False

    print("\n--- Hardware Acceleration ---")
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available (macOS GPU)")
        else:
            print("⚠️ No GPU acceleration detected (running on CPU)")
    except ImportError:
        pass

    print(
        "\n============================================================================"
    )
    if all_passed:
        print("🎉 SUCCESS: All dependencies appear to be installed correctly!")
    else:
        print("⚠️ WARNING: Some dependencies are missing or failed to load.")
    print(
        "============================================================================"
    )


if __name__ == "__main__":
    main()
