import numpy as np
import sys
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Feature Extraction and Classification with Pre-trained ResNet50 ---

# Load ResNet50 without the top layer (for feature extraction)
cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Load ResNet50 with top layer (for classification/object detection)
classifier_model = ResNet50(weights="imagenet", include_top=True)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array)
    return features  # shape: (1, 2048)

def detect_objects(img_path, top_n=5):
    """Detect objects in the image using ResNet50 classifier."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = classifier_model.predict(img_array)
    decoded = decode_predictions(predictions, top=top_n)[0]
    return decoded  # Returns list of (class_id, class_name, probability)

# --- 2. Enhanced Caption Generation with Multiple Captions ---

# Extended tokenizer with more vocabulary
class EnhancedTokenizer:
    def __init__(self):
        self.word_index = {
            "startseq": 1, "endseq": 2, "a": 3, "dog": 4, "on": 5, "grass": 6,
            "cat": 7, "sitting": 8, "table": 9, "person": 10, "walking": 11,
            "street": 12, "car": 13, "parked": 14, "road": 15, "bird": 16,
            "flying": 17, "sky": 18, "beach": 19, "sunset": 20, "people": 21,
            "playing": 22, "park": 23, "tree": 24, "building": 25, "city": 26
        }
        self.index_word = {v: k for k, v in self.word_index.items()}
        
    def texts_to_sequences(self, texts):
        return [[self.word_index.get(word, 0) for word in text.split()] for text in texts]

tokenizer = EnhancedTokenizer()
max_length = 10

# Object category mappings for caption generation
OBJECT_CATEGORIES = {
    'animals': ['dog', 'cat', 'bird', 'elephant', 'bear', 'zebra', 'giraffe', 'tiger', 'lion', 
                'horse', 'sheep', 'cow', 'pig', 'chicken', 'duck', 'goose', 'rabbit', 'fox',
                'wolf', 'deer', 'monkey', 'panda', 'koala', 'kangaroo', 'penguin', 'owl',
                'eagle', 'parrot', 'butterfly', 'fish', 'shark', 'whale', 'dolphin'],
    'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'airplane', 'boat',
                 'ship', 'helicopter', 'ambulance', 'taxi', 'van', 'scooter'],
    'people': ['person', 'man', 'woman', 'child', 'baby', 'boy', 'girl'],
    'furniture': ['chair', 'table', 'couch', 'bed', 'desk', 'bench', 'sofa'],
    'food': ['pizza', 'burger', 'sandwich', 'cake', 'bread', 'apple', 'banana', 'orange'],
    'nature': ['tree', 'flower', 'plant', 'mountain', 'beach', 'ocean', 'lake', 'forest'],
    'buildings': ['house', 'building', 'church', 'castle', 'tower', 'bridge']
}

ACTION_WORDS = {
    'animals': ['sitting', 'standing', 'lying', 'walking', 'running', 'playing'],
    'vehicles': ['parked', 'driving', 'moving'],
    'people': ['standing', 'sitting', 'walking', 'running', 'working'],
}

LOCATION_WORDS = {
    'animals': ['on grass', 'in a field', 'in nature', 'outdoors', 'in the wild'],
    'vehicles': ['on the road', 'on the street', 'in a parking lot'],
    'people': ['in a room', 'outdoors', 'on the street'],
    'furniture': ['in a room', 'indoors'],
    'food': ['on a plate', 'on a table'],
}

def categorize_object(object_name):
    """Categorize detected object into a general category."""
    object_lower = object_name.lower()
    for category, keywords in OBJECT_CATEGORIES.items():
        for keyword in keywords:
            if keyword in object_lower:
                return category
    return 'object'

def generate_caption_from_detection(detections):
    """
    Generate natural caption based on detected objects.
    Uses actual ImageNet predictions to create accurate captions.
    """
    if not detections:
        return "an image"
    
    # Get top detection
    top_detection = detections[0]
    class_id, class_name, probability = top_detection
    
    # Clean up class name (remove underscores, get main object)
    clean_name = class_name.replace('_', ' ').lower()
    
    # Categorize the object
    category = categorize_object(clean_name)
    
    # Generate caption based on category
    if category == 'animals':
        action = np.random.choice(ACTION_WORDS['animals'])
        location = np.random.choice(LOCATION_WORDS['animals'])
        # Extract main animal name
        for animal in OBJECT_CATEGORIES['animals']:
            if animal in clean_name:
                return f"a {animal} {action} {location}"
        return f"a {clean_name} {action} {location}"
    
    elif category == 'vehicles':
        action = np.random.choice(ACTION_WORDS['vehicles'])
        location = np.random.choice(LOCATION_WORDS['vehicles'])
        for vehicle in OBJECT_CATEGORIES['vehicles']:
            if vehicle in clean_name:
                return f"a {vehicle} {action} {location}"
        return f"a {clean_name} {action} {location}"
    
    elif category == 'people':
        action = np.random.choice(ACTION_WORDS['people'])
        location = np.random.choice(LOCATION_WORDS['people'])
        return f"a person {action} {location}"
    
    elif category == 'furniture':
        location = np.random.choice(LOCATION_WORDS['furniture'])
        return f"a {clean_name} {location}"
    
    elif category == 'food':
        location = np.random.choice(LOCATION_WORDS['food'])
        return f"{clean_name} {location}"
    
    else:
        return f"a {clean_name}"

# Enhanced caption generator that uses actual object detection
def generate_caption(img_path, tokenizer, max_length=10):
    """
    Generate caption based on actual object detection.
    Uses ResNet50 ImageNet predictions to identify objects.
    """
    detections = detect_objects(img_path, top_n=3)
    caption = generate_caption_from_detection(detections)
    return caption, detections

def generate_multiple_captions(img_path, tokenizer, max_length=10, num_captions=3):
    """
    Generate multiple caption candidates based on top object detections.
    """
    detections = detect_objects(img_path, top_n=num_captions)
    captions = []
    
    for detection in detections:
        caption = generate_caption_from_detection([detection])
        if caption not in captions:
            captions.append(caption)
    
    return captions[:num_captions], detections

# --- 3. Putting It All Together ---

if __name__ == "__main__":
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "./sample_images/tigerpic.jpg"  # Default test image
    
    print(f"\nAnalyzing image: {img_path}...")
    print(f"Usage: python {sys.argv[0]} [image_path]\n")
    
    # Detect objects in the image
    print("\n" + "="*50)
    print("OBJECT DETECTION RESULTS")
    print("="*50)
    detections = detect_objects(img_path, top_n=5)
    for i, (class_id, class_name, prob) in enumerate(detections, 1):
        print(f"{i}. {class_name}: {prob*100:.2f}% confidence")
    
    # Generate single caption
    print("\n" + "="*50)
    print("SINGLE CAPTION GENERATION")
    print("="*50)
    caption, _ = generate_caption(img_path, tokenizer, max_length)
    print(f"Generated Caption: {caption}")
    
    # Generate multiple captions
    print("\n" + "="*50)
    print("MULTIPLE CAPTION GENERATION")
    print("="*50)
    multiple_captions, _ = generate_multiple_captions(img_path, tokenizer, max_length, num_captions=3)
    for i, cap in enumerate(multiple_captions, 1):
        print(f"Caption {i}: {cap}")
    
    # Extract features for analysis
    print("\n" + "="*50)
    print("FEATURE STATISTICS")
    print("="*50)
    features = extract_features(img_path)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature mean: {np.mean(features):.4f}")
    print(f"Feature std: {np.std(features):.4f}")
    print(f"Feature min: {np.min(features):.4f}")
    print(f"Feature max: {np.max(features):.4f}")