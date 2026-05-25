
import json
import os

def create_notebook():
    # Files to include
    files_to_embed = [
        ('models.py', 'models.py'),
        ('train_x2_finetune.py', 'train_x2_finetune.py'),
        ('test_freezing.py', 'test_freezing.py')
    ]
    
    cells = []
    
    # 1. Setup Cell
    setup_source = [
        "# Install dependencies\n",
        "!pip install timm diffusers"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_source
    })
    
    # 2. Clone Instructions
    clone_source = [
        "# IMPORTANT: This notebook is designed for YOUR CUSTOM DiT fork with x2 modifications.\n",
        "# \n",
        "# Option 1: If you've pushed your code to GitHub, replace the URL below:\n",
        "# !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git\n",
        "# %cd YOUR_REPO\n",
        "#\n",
        "# Option 2: For now, we clone the base DiT repo and overwrite with your custom files:\n",
        "!git clone -b vit-side https://github.com/mrmc-speco/DiT.git\n",
        "%cd DiT\n",
        "\n",
        "# The next cells will overwrite models.py (with x2 modifications) and add new scripts"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": clone_source
    })

    # 3. Embed Files
    for local_path, notebook_path in files_to_embed:
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                content = f.read()
            
            source = [f"%%writefile {notebook_path}\n"]
            source.append(content)
            
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source
            })
            print(f"Added {local_path} to notebook.")
        else:
            print(f"WARNING: Could not find {local_path}")

    # 4. Verify Freezing
    test_source = [
        "# Verify that the freezing logic works correctly\n",
        "!python test_freezing.py"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": test_source
    })
    
    # 5. Download and Organize ImageNet Validation Dataset
    imagenet_source = [
        "# Download ImageNet Validation Dataset (ILSVRC2012)\n",
        "# This will download ~6.3GB and organize it into class folders\n",
        "\n",
        "import os\n",
        "import tarfile\n",
        "from pathlib import Path\n",
        "import xml.etree.ElementTree as ET\n",
        "\n",
        "# Download validation set\n",
        "print('Downloading ImageNet validation set (~6.3GB)...')\n",
        "!wget -nc https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar\n",
        "\n",
        "# Download validation ground truth annotations\n",
        "print('Downloading validation ground truth...')\n",
        "!wget -nc https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz\n",
        "\n",
        "# Extract validation images\n",
        "print('Extracting validation images...')\n",
        "os.makedirs('imagenet_val', exist_ok=True)\n",
        "!tar -xf ILSVRC2012_img_val.tar -C imagenet_val\n",
        "\n",
        "# Extract devkit to get ground truth\n",
        "print('Extracting devkit...')\n",
        "!tar -xzf ILSVRC2012_devkit_t12.tar.gz\n",
        "\n",
        "# Parse ground truth from devkit\n",
        "print('Parsing validation ground truth...')\n",
        "gt_file = 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'\n",
        "with open(gt_file, 'r') as f:\n",
        "    # Ground truth file has 1-indexed class IDs, convert to 0-indexed\n",
        "    labels = [int(line.strip()) - 1 for line in f]\n",
        "\n",
        "# Organize into class folders\n",
        "print('Organizing into class folders...')\n",
        "val_dir = Path('imagenet_val')\n",
        "organized_dir = Path('imagenet_val_organized')\n",
        "\n",
        "# Create class directories\n",
        "for class_id in set(labels):\n",
        "    (organized_dir / str(class_id)).mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "# Move images to class folders\n",
        "val_images = sorted(val_dir.glob('ILSVRC2012_val_*.JPEG'))\n",
        "for idx, img_path in enumerate(val_images):\n",
        "    class_id = labels[idx]\n",
        "    target_path = organized_dir / str(class_id) / img_path.name\n",
        "    img_path.rename(target_path)\n",
        "\n",
        "print(f'✓ ImageNet validation set organized into {len(set(labels))} class folders')\n",
        "print(f'Total images: {len(val_images)}')\n",
        "print('Dataset ready at: ./imagenet_val_organized')"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": imagenet_source
    })
    
    # 6. Run Training
    train_source = [
        "# Run Fine-Tuning on x2 block only\n",
        "# Using ImageNet validation set (adjust --classes to your desired classes)\n",
        "!torchrun --nnodes=1 --nproc_per_node=1 train_x2_finetune.py \\\n",
        "    --model DiT-XL/2 \\\n",
        "    --data-path ./imagenet_val_organized \\\n",
        "    --classes 0 1 2 3 4 5 6 7 8 9 \\\n",
        "    --epochs 2 \\\n",
        "    --global-batch-size 4 \\\n",
        "    --log-every 10"
    ]
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": train_source
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open('x2_finetune_colab.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("Notebook created: x2_finetune_colab.ipynb")

if __name__ == "__main__":
    create_notebook()
