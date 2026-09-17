import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("dataset.csv")

# Basic sanity check
print(f"   Rows: {len(df)}  |  Columns: {list(df.columns)}")
print(f"   Roles found: {df['role'].unique()}")

# 2. Preprocess text
def clean_text(text):
    """Lowercase and strip extra whitespace."""
    text = str(text).lower().strip()
    return text

df["resume_text"] = df["resume_text"].apply(clean_text)

# 3. TF-IDF Vectorization
print("\n📊 Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=500,   # keep top 500 words – enough for our small dataset
    ngram_range=(1, 2)  # unigrams + bigrams capture "machine learning" etc.
)

X = vectorizer.fit_transform(df["resume_text"]).toarray()
print(f"   Feature matrix shape: {X.shape}")

# 4. Encode labels
print("\n🏷️  Encoding labels...")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(df["role"])
num_classes = len(encoder.classes_)
print(f"   Classes ({num_classes}): {list(encoder.classes_)}")

# One-hot encode for categorical_crossentropy
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

#  5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n   Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# 6. Build ANN
print("\n🧠 Building ANN model...")
model = Sequential([
    # Input layer (size = TF-IDF features)
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),   # randomly drop 30 % of neurons to reduce overfitting

    # Hidden layer 1
    Dense(128, activation="relu"),
    Dropout(0.3),

    # Hidden layer 2
    Dense(64, activation="relu"),

    # Output layer – softmax gives a probability per class
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. Train
print("\n🚀 Training...")
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,          # stop if val_loss doesn't improve for 10 epochs
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 8. Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy : {accuracy * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")


print("\n💾 Saving model artifacts...")

model.save("resume_model.keras")
print("   Saved: resume_model.keras")

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("   Saved: vectorizer.pkl")

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("   Saved: encoder.pkl")

print("\n🎉 Training complete! All artifacts saved.")
