import torch
from sklearn.decomposition import PCA


def check_step_1_pca(local_vars):
    """
    Step 1: Standard PCA with scikit-learn.
    Expected: 'pca' (PCA object), 'reduced_data' (numpy array)
    """
    if "PCA" not in local_vars:
        return False, "⚠️ Did you forget to import `PCA` from `sklearn.decomposition`?"

    if "pca" not in local_vars:
        return False, "⚠️ Variable `pca` not found. Initialize the PCA model."

    if "reduced_data" not in local_vars:
        return (
            False,
            "⚠️ Variable `reduced_data` not found. Store the transformed data here.",
        )

    pca = local_vars["pca"]
    reduced = local_vars["reduced_data"]

    if not isinstance(pca, PCA):
        return False, "⚠️ `pca` is not an instance of sklearn.decomposition.PCA."

    if pca.n_components != 2:
        return False, f"⚠️ Expected n_components=2, got {pca.n_components}."

    if reduced.shape[1] != 2:
        return (
            False,
            f"⚠️ Reduced data should have 2 dimensions, got {reduced.shape[1]}.",
        )

    return True, "✅ PCA implemented! You've successfully compressed the data."


def check_step_2_autoencoder_arch(local_vars):
    """
    Step 2: Define Autoencoder Architecture.
    Expected: 'Autoencoder' class with encoder/decoder.
    """
    if "Autoencoder" not in local_vars:
        return False, "⚠️ Class `Autoencoder` not found."

    Autoencoder = local_vars["Autoencoder"]

    # Try to instantiate to check structure
    try:
        # Check if nn is in local_vars before trying to instantiate
        if "nn" not in local_vars:
            return False, "⚠️ `nn` is not defined. Please add `import torch.nn as nn`."

        # CRITICAL FIX: When instantiating, we need to make sure 'nn' is available
        # in the current scope. The class was defined in local_vars, so its methods
        # will look for 'nn' in that scope. We need to temporarily inject it.

        # Actually, the better approach: use the nn from local_vars if it exists
        # But we're in a different function scope, so we need to make it available
        # Let's try a different approach: instantiate within a context where nn is available

        # Create a temporary namespace that has both the class and nn
        dict(local_vars)
        if "nn" in local_vars:
            # Dynamically find the embedding dimension
            emb_dim = 768
            if "embeddings" in local_vars:
                emb_dim = local_vars["embeddings"].shape[1]

            # Use the nn from local_vars
            model = Autoencoder(input_dim=emb_dim, latent_dim=2)
        else:
            return False, "⚠️ `nn` is not defined. Please add `import torch.nn as nn`."

    except NameError as e:
        # More detailed error: check if it's specifically about nn
        if "'nn'" in str(e) or "name 'nn'" in str(e):
            return (
                False,
                "⚠️ NameError: `nn` is not accessible. Make sure `import torch.nn as nn` is at the TOP of your code block, before the class definition.",
            )
        return (
            False,
            "⚠️ NameError during instantiation: {e}. Did you import torch.nn as nn?",
        )
    except Exception as e:
        return False, f"⚠️ Could not instantiate Autoencoder: {e}"

    if not hasattr(model, "encoder") or not hasattr(model, "decoder"):
        return False, "⚠️ Model must have `self.encoder` and `self.decoder` attributes."

    # Check if they used nn.Sequential or layers
    if not isinstance(model.encoder, torch.nn.Module):
        return False, "⚠️ `encoder` must be a PyTorch Module (e.g. nn.Sequential)."

    return True, "✅ Architecture defined! You have a neural network structure."


def check_step_3_training_loop(local_vars):
    """
    Step 3: Training Loop.
    Expected: 'losses' list and trained 'model'. Optionally 'val_losses'.
    """
    if "losses" not in local_vars:
        return False, "⚠️ Variable `losses` not found."

    if "model" not in local_vars:
        return False, "⚠️ Variable `model` not found."

    losses = local_vars["losses"]
    if len(losses) == 0:
        return False, "⚠️ Loss list is empty. Did training run?"

    if losses[-1] > losses[0] * 0.95:
        return (
            False,
            "⚠️ Loss didn't seem to decrease. Check your optimizer or learning rate?",
        )

    # Check val_losses if present
    val_losses = local_vars.get("val_losses", [])
    if val_losses and len(val_losses) != len(losses):
        return (
            False,
            f"⚠️ `val_losses` has {len(val_losses)} entries but `losses` has {len(losses)}. They should match (one per epoch).",
        )

    return True, "✅ Training complete! The model has learned to compress."
