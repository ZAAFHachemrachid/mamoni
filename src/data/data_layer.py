import os
import numpy as np
import csv
from PIL import Image
# ======================================
# Data Layer
# ======================================

class ImageDataset:
    """Handles loading, preprocessing, and feature extraction from image datasets."""

    def __init__(self):
        self.images = []
        self.labels = []
        self.processed_images = []
        self.features = None
        self.encoded_labels = None
        self.num_classes = 0
        self.feature_method = 'average' # Default feature method
        self.current_feature_size = (5, 5) # Default feature size
        # Split datasets
        self.train_features = None
        self.train_labels = None
        self.val_features = None
        self.val_labels = None
        self.test_features = None
        self.test_labels = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.processed_images[idx], self.labels[idx]

    def crop_and_resize_image(self, image, target_size=(50, 50)):
        """Crop image to content and resize to target dimensions."""
        bbox = self.find_content_bounding_box(image)
        if not bbox:
            return None

        l, t, r, b = bbox
        cropped_image = image[t:b+1, l:r+1]
        return np.array(Image.fromarray(cropped_image.astype('uint8')).resize(target_size, Image.LANCZOS))

    def find_content_bounding_box(self, image):
        """Find the bounding box of non-zero content in an image."""
        rows = np.any(image > 0, axis=1)
        cols = np.any(image > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min, y_min, x_max, y_max) if y_max >= y_min and x_max >= x_min else None

    def load_dataset(self, root_dir, max_num_img_per_label, progress_callback=None, heatmap_callback=None):
        """Load images from directory structure and preprocess them."""
        self.images, self.labels, self.processed_images = [], [], []
        self.features, self.encoded_labels = None, None
        self.num_classes = 0

        dataset_dir = os.path.join(root_dir, 'train')
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Directory not found: {dataset_dir}")

        unique_labels = set()
        label_dirs = sorted(d for d in os.listdir(dataset_dir)
                           if os.path.isdir(os.path.join(dataset_dir, d))
                           and d.isdigit())

        total_images_to_load = sum(min(max_num_img_per_label,
                                       len(os.listdir(os.path.join(dataset_dir, label))))
                                  if max_num_img_per_label else
                                  len(os.listdir(os.path.join(dataset_dir, label)))
                                  for label in label_dirs)

        loaded_images_count = 0

        for label in label_dirs:
            numeric_label = int(label)
            unique_labels.add(numeric_label)
            img_files = sorted(os.listdir(os.path.join(dataset_dir, label)))[:max_num_img_per_label] if max_num_img_per_label else sorted(os.listdir(os.path.join(dataset_dir, label)))

            print(f"Loading {len(img_files)}/{len(os.listdir(os.path.join(dataset_dir, label)))} images for label '{numeric_label}'.")

            for img_file in img_files:
                try:
                    img_array = np.array(Image.open(os.path.join(dataset_dir, label, img_file)).convert('L'))
                    self.images.append(img_array)
                    self.labels.append(numeric_label)

                    processed_image = self.crop_and_resize_image(img_array)
                    self.processed_images.append(processed_image)
                    if heatmap_callback:
                        heatmap_callback(processed_image)

                    loaded_images_count += 1
                    if progress_callback:
                        progress_callback(loaded_images_count, total_images_to_load,
                                         f"Loading images: {loaded_images_count}/{total_images_to_load}")
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")

        self.num_classes = len(unique_labels)
        print(f"Loaded and processed {len(self.images)} images from '{dataset_dir}'.")

    def prepare_features(self, progress_callback=None, heatmap_callback=None, feature_method='average', feature_size=(5, 5)):
        """Extract features from processed images for neural network training."""
        if not self.processed_images or not self.labels:
            raise ValueError("No images loaded to prepare features from.")

        features_list, labels_list = [], []
        total_images = len(self.processed_images)
        self.current_feature_size = feature_size # Store current feature size

        for i in range(total_images):
            processed_image = self.processed_images[i] # This is 50x50
            if processed_image is None:
                print(f"Skipping empty image {i} for feature extraction.")
                continue

            if heatmap_callback:
                heatmap_callback(processed_image) # Show 50x50 image first

            feature_vector = self.image_to_features(processed_image, method=feature_method, feature_size=feature_size)
            if feature_vector is None:
                print(f"Skipping image {i} - no features extracted.")
                continue

            if heatmap_callback and feature_size != (50, 50): # Don't show 5x5 if no prep is selected
                feature_image_5x5 = feature_vector.reshape(feature_size)
                feature_image_50x50 = np.array(Image.fromarray(feature_image_5x5.astype('uint8')).resize((50, 50), Image.NEAREST))
                heatmap_callback(feature_image_50x50) # Show upscaled feature map

            features_list.append(feature_vector)
            labels_list.append(self.labels[i])

            if progress_callback:
                progress_callback(i + 1, total_images,
                                  f"Preparing features: {i+1}/{total_images}")

        self.features = np.array(features_list)
        self.encoded_labels = np.array(labels_list)

        # Convert labels to one-hot encoding
        num_classes = len(np.unique(self.encoded_labels))
        one_hot_labels = np.eye(num_classes)[self.encoded_labels]
        self.encoded_labels = one_hot_labels

        print(f"Prepared {len(self.features)} feature vectors for NN using {feature_method} method and feature size {feature_size}.")

    def image_to_features(self, image, method='average', feature_size=(5, 5)):
        """Convert image to feature vector by downsampling using different methods."""
        if image is None:
            return None

        if feature_size == (50, 50): # No preparation, return flattened 50x50 image
            return image.flatten()
        elif method == 'average':
            return self._average_pooling_features(image, feature_size)
        elif method == 'sum':
            return self._sum_pooling_features(image, feature_size)
        elif method == 'max':
            return self._max_pooling_features(image, feature_size)
        else:
            raise ValueError(f"Unknown feature method: {method}")

    def _average_pooling_features(self, image, feature_size):
        """Average pooling to create feature vector."""
        block_size_x = image.shape[1] // feature_size[1]  # block size for width
        block_size_y = image.shape[0] // feature_size[0]  # block size for height
        features = np.zeros(feature_size)
        for i in range(feature_size[0]):
            for j in range(feature_size[1]):
                block = image[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
                features[i, j] = np.mean(block)
        return features.flatten()

    def _sum_pooling_features(self, image, feature_size):
        """Sum pooling to create feature vector."""
        block_size_x = image.shape[1] // feature_size[1]
        block_size_y = image.shape[0] // feature_size[0]
        features = np.zeros(feature_size)
        for i in range(feature_size[0]):
            for j in range(feature_size[1]):
                block = image[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
                features[i, j] = np.sum(block)
        return features.flatten()

    def _max_pooling_features(self, image, feature_size):
        """Max pooling (emptiness/grayness) to create feature vector."""
        block_size_x = image.shape[1] // feature_size[1]
        block_size_y = image.shape[0] // feature_size[0]
        features = np.zeros(feature_size)
        for i in range(feature_size[0]):
            for j in range(feature_size[1]):
                block = image[i*block_size_y:(i+1)*block_size_y, j*block_size_x:(j+1)*block_size_x]
                non_zero_count = np.count_nonzero(block)
                zero_count = block.size - non_zero_count
                features[i, j] = 255 if non_zero_count > zero_count else 0 # 255 if more gray, 0 if more empty
        return features.flatten()

    def export_to_csv(self, csv_filename):
        """Export features and labels to CSV file for external use."""
        if self.features is None or self.encoded_labels is None:
            raise ValueError("Features not prepared. Call prepare_features first.")

        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"feature_{i}" for i in range(self.features.shape[1])] + ["label"])

                # Use original labels, not one-hot encoded
                original_labels = np.argmax(self.encoded_labels, axis=1)

                for i, (feature_vec, label_val) in enumerate(zip(self.features, original_labels)):
                    writer.writerow(list(feature_vec) + [label_val])

            print(f"Dataset saved to {csv_filename}")
        except Exception as e:
            print(f"CSV write error: {e}")

    def split_dataset(self, train_ratio, val_ratio, test_ratio):
        """Splits the dataset into training, validation, and testing sets."""
        if self.features is None or self.encoded_labels is None:
            raise ValueError("Features must be prepared before splitting dataset.")

        if not (train_ratio + val_ratio + test_ratio) == 1.0:
            raise ValueError("Dataset split ratios must sum to 1.0")

        indices = np.random.permutation(len(self.features))
        train_size = int(len(self.features) * train_ratio)
        val_size = int(len(self.features) * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        self.train_features = self.features[train_indices]
        self.train_labels = self.encoded_labels[train_indices]
        self.val_features = self.features[val_indices]
        self.val_labels = self.encoded_labels[val_indices]
        self.test_features = self.features[test_indices]
        self.test_labels = self.encoded_labels[test_indices]

        print(f"Dataset split: Train={len(self.train_features)}, Val={len(self.val_features)}, Test={len(self.test_features)}")

