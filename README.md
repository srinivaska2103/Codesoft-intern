# Image Captioning Demo

An intelligent image captioning system that uses ResNet50 pre-trained model to detect objects and generate natural language descriptions of images.

## Features

- **Object Detection**: Identifies objects in images using ResNet50 trained on ImageNet
- **Feature Extraction**: Extracts 2048-dimensional feature vectors from images
- **Caption Generation**: Creates natural language captions based on detected objects
- **Multiple Captions**: Generates multiple caption candidates for each image
- **Category-Aware**: Recognizes different object categories (animals, vehicles, people, furniture, food, nature, buildings)

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy
- Pillow

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Image_Caption
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the demo with a default image:
```bash
python image_captioning_demo.py
```

### Custom Image

Provide your own image path:
```bash
python image_captioning_demo.py path/to/your/image.jpg
```

### Example Output

```
Analyzing image: ./sample_images/tigerpic.jpg...

==================================================
OBJECT DETECTION RESULTS
==================================================
1. tiger: 85.23% confidence
2. tiger_cat: 8.45% confidence
3. jaguar: 3.21% confidence

==================================================
SINGLE CAPTION GENERATION
==================================================
Generated Caption: a tiger sitting in the wild

==================================================
MULTIPLE CAPTION GENERATION
==================================================
Caption 1: a tiger sitting in the wild
Caption 2: a cat lying on grass
Caption 3: a jaguar standing in nature

==================================================
FEATURE STATISTICS
==================================================
Feature vector shape: (1, 2048)
Feature mean: 0.3421
Feature std: 0.5234
```

## How It Works

1. **Object Detection**: Uses ResNet50 with ImageNet weights to classify objects in the image
2. **Categorization**: Maps detected objects to semantic categories (animals, vehicles, etc.)
3. **Caption Construction**: Combines object names with appropriate actions and locations based on category
4. **Multiple Hypotheses**: Generates several caption candidates using top-N predictions

## Project Structure

```
Image_Caption/
├── image_captioning_demo.py  # Main script
├── sample_images/             # Sample test images
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Supported Object Categories

- **Animals**: dog, cat, bird, tiger, elephant, etc.
- **Vehicles**: car, truck, bus, motorcycle, bicycle, etc.
- **People**: person, man, woman, child, etc.
- **Furniture**: chair, table, couch, bed, etc.
- **Food**: pizza, burger, sandwich, cake, etc.
- **Nature**: tree, flower, mountain, beach, etc.
- **Buildings**: house, building, church, castle, etc.

## Limitations

- Captions are template-based and may not capture complex scenes
- Works best with single dominant objects
- Limited to ImageNet object classes
- Action and location words are randomly selected from predefined lists

## Future Improvements

- Implement LSTM/Transformer-based caption generation
- Train on MS COCO or Flickr30k datasets
- Add beam search for better caption selection
- Support for multi-object scenes
- Fine-tune on domain-specific images

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
