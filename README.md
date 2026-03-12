# Skin Disease Detection System

## 📌 High‑Level Idea (what you should say first)
> "We built a web‑based AI system that lets a user upload a skin image and automatically classifies it into one of six common skin diseases using a ResNet50 deep learning model, then generates Gemini‑based medical‑style recommendations."

- **Diseases covered**: Acne, Carcinoma, Eczema, Keratosis, Milia, Rosacea  
- **Core model**: ResNet50 (transfer learning, ImageNet weights)  
- **Accuracy**: **92% test accuracy** on 250 held‑out images  
- **Interface**: Flask web app (login → upload image → view results + recommendations)


## 1. Overall Architecture

### Components
- **Frontend (UI)**
  - `templates/login.html`, `signup.html`, `detection.html`, `results.html`
  - Tailwind CSS + Bootstrap Icons for modern UI
  - JavaScript (`static/js/detection.js`, `static/js/results.js`) for image upload and showing results

- **Backend (Flask)** – `app.py`
  - User auth with SQLite database (`users.db`)
  - Image upload and validation
  - Calls the ResNet50 model for prediction
  - Calls Google Gemini to generate human‑readable advice

- **ML Model** – `models/best_model_finetuned.keras`
  - ResNet50 backbone (pre‑trained on ImageNet)
  - Custom classification head for 6 skin diseases

- **Dataset** – `dataset/`
  - 6 folders: `Acne/`, `Carcinoma/`, `Eczema/`, `Keratosis/`, `Milia/`, `Rosacea/`
  - **399 images per class** → **2394 images total** (balanced dataset)

### Request Flow (from user’s point of view)
1. User signs up / logs in (credentials stored in `users.db`).  
2. On `detection.html`, user clicks **Skin Disease Detection** and uploads an image.  
3. JS (`detection.js`) sends the image to `/predict` (Flask) via `fetch` POST.  
4. Flask saves the image, preprocesses it to `(1, 224, 224, 3)`, and calls the **ResNet50 model**.  
5. Model outputs probabilities for the 6 classes → we take the **argmax** as predicted disease + confidence %.  
6. Flask sends disease name + confidence to **Gemini**, gets a detailed recommendation text.  
7. Flask returns JSON → JS stores it in `sessionStorage` → `results.html` reads it and shows:
   - Uploaded image
   - Predicted disease
   - Confidence bar
   - All 6 class probabilities
   - Gemini recommendations + medical disclaimer

---

## 2. Dataset & Preprocessing

### Dataset (Skin Disease Only)
- Source: **Augmented Skin Conditions Image Dataset**  
- Classes (6): `['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']`  
- Images per class: **399 each** (balanced)  
- Total images: **2394**

### Train / Validation / Test Split
Done in `skindisorder.ipynb` using `image_dataset_from_directory`:
- Train: **80%** (~1915 images → 60 batches)
- Validation: **10%** (~239 images → 7 batches)
- Test: **10%** (250 images → 8 batches)

### Image Preprocessing
- All images are resized to **224×224×3** (height × width × RGB).  
- For training: tensors from `image_dataset_from_directory`.  
- For inference (Flask):
  ```python
  img = image.load_img(filepath, target_size=(224, 224))
  img_array = image.img_to_array(img)          # (224, 224, 3)
  img_array = np.expand_dims(img_array, 0)     # (1, 224, 224, 3)
  # NO normalization for this model (trained on 0–255 range)
  ```

### Data Augmentation (Training)
Implemented in `skindisorder.ipynb`:
```python
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),   # flip left/right
    RandomRotation(0.1),        # rotate ±10°
    RandomZoom(0.1)             # zoom ±10%
])

train_dataset = train_dataset.map(lambda img, label: (data_augmentation(img), label))
```

Intuition: each epoch the network sees **slightly modified versions** of the same images (flipped / rotated / zoomed). This:
- Increases effective dataset size
- Makes model robust to camera angle and position
- Reduces overfitting

You can say in viva:  
> "We applied on‑the‑fly augmentation: horizontal flip, ±10° rotation, and ±10% zoom to make the model more robust and prevent overfitting."

---

## 3. Model Architecture (Skin Disease Only)

### Base: ResNet50 (Transfer Learning)
- Input: `(224, 224, 3)`
- Pre‑trained on ImageNet (over 1M images, 1000 classes)
- We **remove the original classifier head** and freeze its weights:

```python
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False   # feature extractor only
```

Intuition to explain:  
> "Instead of training from scratch, we reuse ResNet50 as a powerful feature extractor and only train a small classification head for our 6 diseases."

### Custom Classification Head
```python
x = base_model.output
x = GlobalAveragePooling2D()(x)    # 7×7×2048 → 2048
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)                # regularization
outputs = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)
```

- **GlobalAveragePooling2D**: converts 2D feature maps into a single 2048‑dimensional vector by averaging each channel.  
- **Dense(1024, ReLU)**: learns a non‑linear combination of features.  
- **Dropout(0.5)**: randomly drops 50% of neurons during training to reduce overfitting.  
- **Dense(6, Softmax)**: outputs probabilities for the 6 skin diseases (sum to 1).

### Training Configuration
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)
```

- **Batch size**: 32  
- **Learning rate**: `1e-4` (0.0001) → small, stable steps, good for transfer learning  
- **Loss**: Sparse Categorical Crossentropy (integer labels 0–5 for 6 classes)  
- **Metric**: Accuracy

### Training Callbacks
1. **EarlyStopping** – stop if validation loss doesn’t improve for 5 epochs and restore best weights.  
2. **ReduceLROnPlateau** – if val loss plateaus for 3 epochs, multiply learning rate by 0.5 (fine‑tuning).  
3. **ModelCheckpoint** – whenever validation accuracy improves, save model as `best_model.keras` (later exported as `best_model_finetuned.keras`).

What actually happened (from notebook):
- Epoch 1: **Train Acc ~39%**, **Val Acc ~76%**, Val Loss ~0.70  
- Epoch 2: **Train Acc ~71%**, **Val Acc ~80%**  
- Epoch 3: **Train Acc ~80%**, **Val Acc ~84%**, Val Loss ~0.49  
- Training stopped early around epoch 10–15 with **best Val Acc ~84%**.

Then evaluated on the **test set (250 images)**:
- **Test Accuracy: 92%**  
- Per‑class performance (from classification report):
  - Acne: Precision 0.85, Recall 0.89
  - Carcinoma: Precision 0.97, Recall 0.97
  - Eczema: Precision 0.94, Recall 0.75
  - Keratosis: Precision 0.94, Recall 1.00
  - Milia: Precision 0.89, Recall 0.98
  - Rosacea: Precision 0.90, Recall 0.90

One‑line takeaway:  
> "Our ResNet50‑based model achieves 92% accuracy and very strong per‑class precision/recall on 6 balanced skin disease classes."

---

### How We Calculated the Final 92% Test Accuracy

**Step-by-step process** (from `skindisorder.ipynb` Cell 8):

#### 1. **Test Set Preparation**
- We had **250 images** in the test set (10% of 2394 total images).
- These images were **never seen during training** – completely held out.
- Test set distribution: ~38–45 images per class (balanced).

#### 2. **Model Evaluation**
After training completed, we loaded the **best saved model** (saved by ModelCheckpoint based on validation accuracy) and ran:

```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
```

**Output from notebook:**
```
Test Accuracy: 0.9200  (which is 92%)
Test Loss: 0.2340
```

#### 3. **How `model.evaluate()` Calculates Accuracy**

Internally, TensorFlow does this for each test image:

1. **Forward pass**: Image → ResNet50 → Dense layers → Softmax probabilities `[p1, p2, p3, p4, p5, p6]`
2. **Prediction**: `predicted_class = argmax([p1, p2, p3, p4, p5, p6])` (class with highest probability)
3. **Compare**: Check if `predicted_class == true_label`
4. **Count**: Accuracy = (Number of correct predictions) / (Total test images)

**Mathematical formula:**
\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Test Images}} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}}
\]

For our case:
- Total test images: **250**
- Correct predictions: **230** (from 92% accuracy: 0.92 × 250 = 230)
- Wrong predictions: **20** (250 - 230 = 20)

#### 4. **Detailed Per-Class Analysis**

We also generated a **classification report** to see performance per disease:

```python
y_true = []  # actual labels
y_pred = []  # predicted labels

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())                    # true labels: [0, 1, 2, 3, 4, 5]
    y_pred.extend(np.argmax(predictions, axis=1))    # predicted: [0, 1, 2, 3, 4, 5]

print(classification_report(y_true, y_pred, target_names=class_names))
```

**Results:**
```
              precision    recall  f1-score   support

        Acne       0.85      0.89      0.87        38
   Carcinoma       0.97      0.97      0.97        39
      Eczema       0.94      0.75      0.84        44
   Keratosis       0.94      1.00      0.97        45
       Milia       0.89      0.98      0.93        43
     Rosacea       0.90      0.90      0.90        41

    accuracy                           0.92       250
```

**What this means:**
- **Overall accuracy**: 92% (230 out of 250 correct)
- **Per-class breakdown**:
  - **Keratosis**: Best performance (100% recall – found all 45 cases)
  - **Carcinoma**: Excellent (97% precision, 97% recall)
  - **Eczema**: Lower recall (75% – missed some cases)
  - Other classes: 85–98% performance

#### 5. **Confusion Matrix**

We also created a confusion matrix to see **which classes were confused with each other**:

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ...)
```

This shows:
- **Diagonal values** (TP for each class): correct predictions
- **Off-diagonal values**: misclassifications (e.g., some Acne predicted as Eczema)

#### 6. **Why This is the "Final" Accuracy**

- **Test set is held out**: Never used during training or validation
- **Best model used**: We evaluated the model saved by ModelCheckpoint (best validation accuracy)
- **Unbiased estimate**: This 92% represents how well the model will perform on **new, unseen images**

**For your viva, you can say:**
> "After training, we evaluated the best model on a completely held-out test set of 250 images. The model correctly predicted 230 out of 250 images, giving us a final test accuracy of 92%. We also generated a classification report showing per-class precision and recall, and a confusion matrix to identify which diseases were sometimes confused with each other."

---

## 4. Inference Pipeline in `app.py` (Skin Disease Path)

### Step 1: Model Loading
```python
from tensorflow.keras.models import load_model

skin_disease_model = load_model('models/best_model_finetuned.keras')
skin_disease_classes = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']
```
- Loaded **once at server start** (not on every request).  
- Stays in memory for fast predictions.

### Step 2: Image Upload & Validation
In `/predict` route:
```python
if 'image' not in request.files:
    return jsonify({'error': 'No image uploaded'}), 400

file = request.files['image']
if file.filename == '':
    return jsonify({'error': 'No image selected'}), 400

if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
```
- Only allows `png`, `jpg`, `jpeg`.  
- Max size: **16 MB** (`MAX_CONTENT_LENGTH`).  
- Uses `secure_filename` to avoid path‑traversal attacks.  
- Saves file under `static/uploads/` for later display.

### Step 3: Preprocessing (exactly like training) – **Detailed Step-by-Step**

This is the **critical transformation** from a user's uploaded image file to the tensor format ResNet50 expects. Here's exactly what happens:

#### **Step 3.1: Load Image from File**
```python
img = image.load_img(filepath, target_size=(224, 224))
```

**What happens:**
- Reads the image file from disk (e.g., `static/uploads/user_image.jpg`)
- **Resizes** the image to exactly **224×224 pixels** (height × width)
- Converts to RGB format (3 color channels: Red, Green, Blue)
- Returns a **PIL Image object** (Python Imaging Library format)

**Example transformation:**
```
Original image: 800×600 pixels (or any size)
         ↓
    Resize to 224×224
         ↓
PIL Image: 224×224×3 (RGB)
```

**Why 224×224?**
- ResNet50 was pre-trained on ImageNet with 224×224 images
- This is the **standard input size** for ResNet50
- All images must be the same size for batch processing

---

#### **Step 3.2: Convert PIL Image to NumPy Array**
```python
img_array = image.img_to_array(img)
```

**What happens:**
- Converts the PIL Image object into a **NumPy array**
- Shape: `(224, 224, 3)` 
- Data type: `float32` or `uint8` (depending on TensorFlow version)
- Pixel values: **0 to 255** (for each RGB channel)

**Array structure:**
```
img_array shape: (224, 224, 3)

Where:
- First dimension (224): Height (rows)
- Second dimension (224): Width (columns)  
- Third dimension (3): RGB channels
  - Channel 0: Red values (0-255)
  - Channel 1: Green values (0-255)
  - Channel 2: Blue values (0-255)
```

**Visual representation:**
```
img_array[0, 0, :] = [R, G, B]  # Top-left pixel RGB values
img_array[0, 1, :] = [R, G, B]  # Next pixel to the right
...
img_array[223, 223, :] = [R, G, B]  # Bottom-right pixel
```

**Example values:**
```python
# A pixel might look like:
img_array[100, 150, :] = [128, 64, 192]  
# Meaning: R=128, G=64, B=192 (purplish color)
```

---

#### **Step 3.3: Add Batch Dimension**
```python
img_array = np.expand_dims(img_array, axis=0)
```

**What happens:**
- Adds a **batch dimension** at position 0 (the first dimension)
- Transforms shape from `(224, 224, 3)` → `(1, 224, 224, 3)`

**Why add batch dimension?**
- Neural networks process images in **batches** (multiple images at once)
- Even for a single image, we need to maintain the batch structure
- The "1" means: "batch size = 1" (one image in this batch)

**Shape transformation:**
```python
Before: img_array.shape = (224, 224, 3)
         ↓ expand_dims(axis=0)
After:  img_array.shape = (1, 224, 224, 3)
```

**What each dimension means:**
```
(1, 224, 224, 3)
 │   │    │    │
 │   │    │    └─ RGB channels (3)
 │   │    └────── Width (224 pixels)
 │   └─────────── Height (224 pixels)
 └─────────────── Batch size (1 image)
```

**Visual representation:**
```
Before expand_dims:
┌─────────────────┐
│  224 × 224 × 3  │  ← Single image array
└─────────────────┘

After expand_dims:
┌─────────────────┐
│ 1 × 224 × 224 × 3 │  ← Batch of 1 image
└─────────────────┘
```

---

#### **Step 3.4: No Normalization (Important!)**
```python
# No normalization here because model was trained on 0–255 scale
# img_array = img_array / 255.0  ← We DON'T do this!
```

**What this means:**
- Pixel values stay in range **0 to 255** (not normalized to 0–1)
- This matches how the model was trained
- If we normalized to 0–1, predictions would be wrong!

**Why no normalization?**
- Our ResNet50 model was trained with images in 0–255 range
- The pre-trained ImageNet weights expect 0–255 pixel values
- Normalizing would shift the distribution and hurt accuracy

---

#### **Complete Preprocessing Pipeline Summary**

```
User uploads: user_image.jpg (800×600 pixels, JPEG format)
         ↓
Step 1: Save file to static/uploads/user_image.jpg
         ↓
Step 2: Load & resize
   image.load_img() → PIL Image (224×224×3)
         ↓
Step 3: Convert to array
   image.img_to_array() → NumPy array (224, 224, 3)
         ↓
Step 4: Add batch dimension
   np.expand_dims(axis=0) → NumPy array (1, 224, 224, 3)
         ↓
Step 5: Ready for model!
   model.predict(img_array) → Predictions [p1, p2, p3, p4, p5, p6]
```

---

#### **Code Reference (from `app.py` lines 205-207)**

```python
# Skin Disease Detection - NO NORMALIZATION
img = image.load_img(filepath, target_size=(224, 224))      # Step 1: Load & resize
img_array = image.img_to_array(img)                          # Step 2: PIL → NumPy
img_array = np.expand_dims(img_array, axis=0)               # Step 3: Add batch dim
# Result: img_array.shape = (1, 224, 224, 3)
# Pixel values: 0-255 (no normalization)
```

---

#### **For Your Viva - Quick Explanation**

You can say:
> "When a user uploads an image, Flask first saves it to disk. Then we use TensorFlow's `image.load_img()` to load and resize it to 224×224 pixels, converting it to RGB format. Next, `image.img_to_array()` converts the PIL Image into a NumPy array of shape (224, 224, 3) with pixel values from 0 to 255. Finally, `np.expand_dims()` adds a batch dimension at the front, transforming it to (1, 224, 224, 3) - where the '1' represents batch size of 1 image. We don't normalize the pixel values because our model was trained on the 0-255 range. This tensor is then fed directly into the ResNet50 model for prediction."

### Step 4: Prediction
```python
predictions = skin_disease_model.predict(img_array)

predicted_class_idx = np.argmax(predictions[0])
predicted_class = skin_disease_classes[predicted_class_idx]
confidence = float(np.max(predictions[0]) * 100)

all_predictions = [
    {
        'class': skin_disease_classes[idx],
        'confidence': float(prob * 100)
    }
    for idx, prob in enumerate(predictions[0])
]
all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
```
- Output is a **softmax vector** of length 6 → probabilities for each class.  
- `argmax` → **predicted label**.  
- `max(probabilities) × 100` → **confidence %**.  
- We also build a sorted list of all 6 class confidences for the UI.

### Step 5: Gemini Recommendations
```python
prompt = f"""You are a medical AI assistant specializing in dermatology...
Detected Skin Condition: {predicted_class}
Confidence Level: {confidence:.2f}%
[asks for overview, symptoms, treatment, lifestyle, prevention, warning signs]
"""

model = genai.GenerativeModel("gemini-2.5-flash-lite")
response = model.generate_content(prompt)
```
- Uses your **GEMINI_API_KEY** (set in environment).  
- Prompt includes: disease name + confidence.  
- Asks Gemini to return: overview, symptoms, causes, treatment options, lifestyle tips, prevention, warning signs.  
- Text is passed to the frontend and rendered nicely in `results.html`.

### Step 6: Response JSON
```python
return jsonify({
    'success': True,
    'detection_type': 'disease',
    'condition_name': predicted_class,
    'skin_condition': predicted_class,
    'confidence': confidence,
    'skin_confidence': confidence,
    'all_predictions': all_predictions,
    'recommendations': response.text,
    'image_path': filename
})
```

The frontend (`results.js`) then:
- Shows the uploaded image.  
- Fills a **confidence progress bar**.  
- Lists all six diseases with their confidence bars.  
- Inserts Gemini’s markdown‑style text into the recommendations section.

---

## 5. Web Application Flow (What to Explain in Demo)

### 1. Authentication
- **Sign Up** (`/signup`):
  - User enters name, email, username, password, age, purpose.
  - Password is hashed with **Werkzeug** (`generate_password_hash`).
  - Stored securely in `users.db` (SQLite).

- **Login** (`/login`):
  - User enters username & password.
  - Password verified with `check_password_hash`.
  - On success, we store `user_id`, `username`, `first_name` in Flask `session`.

### 2. Detection Page (`/detection`)
- Shows **skin disease card** and (optionally) skin cancer card in UI.  
- For your exam, you can say: *"For now we are focusing on the skin disease part; the UI also has a placeholder for future cancer detection extension."*
- User clicks **Skin Disease Detection** → UI switches to upload step.  
- User uploads image via drag‑and‑drop; `detection.js` handles preview and validation.

### 3. Prediction (`/predict` – disease path)
- JS sends the selected image to Flask.  
- Flask runs the **ResNet50 model** and Gemini prompt.  
- Returns JSON with predictions + recommendations.

### 4. Results Page (`/results`)
- Shows:
  - Uploaded image.  
  - Predicted disease name.  
  - Confidence percentage and colored bar:  
    - Green ≥ 80%, Yellow 60–79%, Red < 60%.  
  - All six class confidences.  
  - Gemini‑generated recommendations + **medical disclaimer**.

---

## 6. Key Talking Points for Viva

If you get nervous, you can structure your explanation like this:

1. **Problem**: Manual diagnosis of skin diseases is time‑consuming and subjective; we want an AI assistant to give a **quick, preliminary assessment**.

2. **Dataset**: 2394 images, 6 balanced classes (399 each), from an augmented skin condition dataset.

3. **Model**: ResNet50 transfer learning:
   - Pre‑trained on ImageNet, `include_top=False`, input 224×224×3.  
   - Added GlobalAveragePooling + Dense(1024, ReLU) + Dropout(0.5) + Dense(6, Softmax).  
   - Trained with Adam (lr=1e‑4), sparse categorical crossentropy, accuracy metric.  
   - Data augmentation: horizontal flip, ±10° rotation, ±10% zoom.  
   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.  
   - Result: **92% test accuracy**.

4. **Application**:
   - Flask web app with login and image upload.  
   - Backend endpoint `/predict` loads the finetuned ResNet50 model and runs inference.  
   - Uses Google Gemini to turn the raw prediction into **human‑readable medical‑style guidance**.

5. **Limitations**:
   - AI is **not a doctor**; serves as screening/educational tool only.  
   - Performance depends on image quality and dataset distribution.

6. **Future Work**:
   - Improve recall for Eczema (currently 75%).  
   - Add explainability (heatmaps) to show which part of the skin the model focused on.  
   - Extend to more diseases / multi‑label detection.

---

## 7. How to Run the Project (Quick)

```bash
cd /Users/divyansh/Desktop/skin_Cancer
source venv/bin/activate       # activate virtualenv
pip install -r requirements_updated.txt
export GEMINI_API_KEY='your_key_here'
python app.py                  # runs on http://127.0.0.1:5001
```

Then:
1. Open `http://127.0.0.1:5001` in browser.  
2. Sign up, log in.  
3. Choose **Skin Disease Detection**, upload an image, and view results.

---

This README now focuses **only on the skin disease (ResNet50) module** and is structured to be easy to revise quickly before your review.


## 8. Model Internals & Layer‑by‑Layer Workflow (Skin Disease Model)

### 8.1 Big Picture – 5 Main Steps in the Network

You can remember the model as **5 logical steps**:

1. **Input & Initial Convolution** – 224×224×3 image → low‑level edges & textures.  
2. **Deep ResNet Blocks** – many Conv+ReLU layers with skip connections → complex skin patterns.  
3. **Global Average Pooling** – compresses feature maps into a single 2048‑dimensional vector.  
4. **Dense (1024) + Dropout** – learns a compact 1024‑dimensional representation and regularizes.  
5. **Dense (6, Softmax)** – outputs probabilities for the 6 skin disease classes.

In short: **Image → Convs (ResNet) → 2048‑d features → 1024‑d features → 6‑class softmax.**

---

### 8.2 How Many Layers? (Explainable Version)

ResNet50 is a 50‑layer convolutional neural network. Rough breakdown you can quote:

- 1 **initial convolution + max‑pool** layer group.  
- 4 **stages** of residual blocks: 
  - Stage 1 (Conv2_x): 3 blocks × 3 conv layers each = 9 conv layers.  
  - Stage 2 (Conv3_x): 4 blocks × 3 conv layers each = 12 conv layers.  
  - Stage 3 (Conv4_x): 6 blocks × 3 conv layers each = 18 conv layers.  
  - Stage 4 (Conv5_x): 3 blocks × 3 conv layers each = 9 conv layers.  
- 1 **Global Average Pooling**.  
- Our custom head: **Dense(1024) + Dropout + Dense(6)**.

Total ≈ 1 (initial) + 9 + 12 + 18 + 9 + 2 (our dense layers) = **51 trainable layers** in the full pipeline, but we **freeze the ResNet part** and train only the last 2 dense layers.

For viva, keep it simple:
> "ResNet50 has ~50 convolutional layers grouped into 4 residual stages; on top of that we add GlobalAveragePooling and 2 dense layers (1024 and 6 units)."

---

### 8.3 What Each Type of Layer Does

#### 1) Convolution (Conv) Layer
- **Operation**: Slides filters over the image/feature maps and computes weighted sums.  
- **Effect**: Detects local patterns – edges, corners, textures, color blobs.

**Filter Sizes Used in ResNet50:**
- **7×7 filter** (initial layer, stride 2): Large receptive field for initial feature extraction
- **3×3 filter** (most common, stride 1 or 2): Standard convolution for pattern detection
- **1×1 filter** (in residual blocks, stride 1): Dimensionality reduction/expansion

**Strides Used:**
- **Stride 2**: Downsampling (reduces spatial dimensions by half)
- **Stride 1**: No downsampling (preserves spatial dimensions)

**ResNet50 Architecture Details:**
1. **Initial Layer**: 7×7 conv, stride 2 → reduces 224×224 to 112×112
2. **Max Pool**: 3×3 max-pool, stride 2 → reduces 112×112 to 56×56
3. **Residual Blocks**: 
   - Each block uses: 1×1 conv → 3×3 conv → 1×1 conv
   - First block in each stage: stride 2 (downsampling)
   - Remaining blocks: stride 1 (same size)

Mathematically:
\[
(I * K)(i, j) = \sum_m \sum_n I(i + m, j + n)\, K(m,n)
\]
Where \(I\) is the input image/feature map and \(K\) is the kernel.

**In skin disease context:**
- **7×7 initial conv**: Captures large-scale skin texture patterns
- **3×3 convs**: Detect edges, lesions, color variations at different scales
- **1×1 convs**: Efficiently adjust feature map dimensions between stages

**Complete Filter & Stride Breakdown:**

| Layer/Stage | Filter Size | Stride | Output Size | Purpose |
|-------------|-------------|--------|-------------|---------|
| **Input** | - | - | 224×224×3 | Original image |
| **Initial Conv** | 7×7 | 2 | 112×112×64 | Large receptive field, initial downsampling |
| **Max Pool** | 3×3 | 2 | 56×56×64 | Further downsampling, translation invariance |
| **Stage 1 (Conv2_x)** | 1×1, 3×3, 1×1 | 1 (first block: 1) | 56×56×256 | Low-level features |
| **Stage 2 (Conv3_x)** | 1×1, 3×3, 1×1 | 2 (first block: 2) | 28×28×512 | Mid-level features |
| **Stage 3 (Conv4_x)** | 1×1, 3×3, 1×1 | 2 (first block: 2) | 14×14×1024 | High-level features |
| **Stage 4 (Conv5_x)** | 1×1, 3×3, 1×1 | 2 (first block: 2) | 7×7×2048 | Very high-level semantic features |

**Key Points:**
- **7×7 filter**: Only in the initial layer (stride 2) for initial feature extraction
- **3×3 filters**: Most common, used in every residual block (stride 1 or 2)
- **1×1 filters**: Used in bottleneck design (stride 1) for efficiency
- **Stride 2**: Used for downsampling (reduces size by half)
- **Stride 1**: Used to preserve spatial dimensions

#### 2) ReLU Activation
- **Definition**: \( f(x) = \max(0, x) \).  
- **Why**: Introduces non‑linearity, helps network learn complex functions, and reduces vanishing gradients.

Flow: **Conv → ReLU** means: "detect a pattern, then keep only positive evidence of it".

#### 3) Pooling (Max‑Pool)
- **Operation**: Takes the maximum value in a small window (e.g., 2×2).  
- **Effect**: Reduces spatial size (downsampling), keeps strongest activations, makes model translation‑invariant.

In ResNet50:
- Initial **7×7 conv** with stride 2 + **3×3 max‑pool** with stride 2 → quickly reduce 224×224 to 56×56 while keeping meaningful features.

#### 4) Residual (Skip) Connections
Each residual block does:
\[
\text{Output} = F(x) + x
\]
Where \(F(x)\) is a stack of conv+BN+ReLU layers.

Benefits:
- Makes optimization easier for very deep networks (like 50 layers).  
- Allows gradients to flow directly through skip paths.  
- Helps prevent vanishing gradients.

You can say:  
> "Instead of learning a full mapping, residual blocks learn only the *difference* (residual) from the input." 

#### 5) Global Average Pooling
- **Input**: 7×7×2048 feature map (after last ResNet stage).  
- **Operation**: Averages each 7×7 feature map to a single number.  
- **Output**: 2048‑dimensional vector.

Why:
- Replaces large fully‑connected layers from classic CNNs.  
- Reduces parameters → less overfitting.  
- Still keeps global information about the image.

#### 6) Dense (Fully Connected) Layers – 1024 and 6
- **Dense(1024, ReLU)**:
  - Input: 2048‑dim vector from pooling.  
  - Learns a compressed representation specific to skin diseases.  
- **Dropout(0.5)**:
  - Randomly sets 50% of activations to 0 during training.  
  - Forces network not to rely on any single neuron → regularization.
- **Dense(6, Softmax)**:
  - Outputs a probability for each disease class.  
  - Softmax formula for class \(i\):
    \[
    P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
    \]

---

### 8.4 End‑to‑End Forward Pass (Skin Disease Workflow)

Given an input image **X** of size 224×224×3, the model does:

1. **Initial Conv + MaxPool**  
   - 7×7 conv (stride 2) → basic edges/textures.  
   - 3×3 max‑pool (stride 2) → reduce resolution.

2. **Residual Stage 1 (Conv2_x)** – 3 residual blocks  
   - Each block: 1×1 conv → 3×3 conv → 1×1 conv + skip connection.  
   - Learns low‑level patterns of skin texture.

3. **Residual Stage 2 (Conv3_x)** – 4 residual blocks  
   - Learns more abstract features (blotches, color variations).  

4. **Residual Stage 3 (Conv4_x)** – 6 residual blocks  
   - Deepest and widest stage; captures complex lesion structures.

5. **Residual Stage 4 (Conv5_x)** – 3 residual blocks  
   - High‑level semantic features: combination of shapes, textures, and colors.

6. **Global Average Pooling**  
   - Converts 7×7×2048 → 2048.  
   - Now we have a compact feature vector describing the whole image.

7. **Dense(1024, ReLU) + Dropout(0.5)**  
   - Learns a task‑specific embedding for the 6 diseases.  
   - Dropout prevents overfitting.

8. **Dense(6, Softmax)**  
   - Outputs probabilities for: `[Acne, Carcinoma, Eczema, Keratosis, Milia, Rosacea]`.  
   - Prediction = class with maximum probability, confidence = max(probabilities) × 100.

You can summarize the whole forward path in one sentence:  
> "The image passes through many Conv+ReLU+Residual blocks (ResNet50) to extract features, then GlobalAveragePooling + Dense(1024) + Dropout + Dense(6) turn those features into a 6‑class probability vector."

---

### 8.5 Strategies We Used (Easy to List in Review)

1. **Transfer Learning**  
   - Use ResNet50 pre‑trained on ImageNet.  
   - Freeze all ResNet layers and only train the top dense layers.

2. **Data Augmentation**  
   - Horizontal flip, ±10° rotation, ±10% zoom.  
   - Makes the model robust to orientation, minor scale and position changes.

3. **Regularization**  
   - Dropout(0.5) before final layer.  
   - Balanced dataset (399 images per class).  
   - GlobalAveragePooling instead of huge fully‑connected layers.

4. **Optimization Strategy**  
   - Adam optimizer with small learning rate (1e‑4) for stable fine‑tuning.  
   - EarlyStopping to avoid overfitting.  
   - ReduceLROnPlateau to lower learning rate when validation loss plateaus.  
   - ModelCheckpoint to always keep the best performing model.

These points give you a clear, layer‑wise and strategy‑wise explanation of **what happens inside the network at each step** and **why we designed it this way**.

