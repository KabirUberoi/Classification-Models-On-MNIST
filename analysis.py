from re import escape
from config import *
from svm import *
from ensembling import *
from utils import *
import warnings
from tqdm import tqdm  
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import time
warnings.filterwarnings('ignore')

train_processor = MNISTPreprocessor('./dataset/train', PRE_PROCESSING_CONFIG)
train_X, train_y = train_processor.get_all_data()
train_X, train_y = filter_dataset(train_X, train_y, ENTRY_NUMBER_LAST_DIGIT)
train_y = convert_labels_to_svm_labels(train_y, ENTRY_NUMBER_LAST_DIGIT)

val_processor = MNISTPreprocessor('./dataset/val', PRE_PROCESSING_CONFIG)
val_X, val_y = val_processor.get_all_data()
val_X, val_y = filter_dataset(val_X, val_y, ENTRY_NUMBER_LAST_DIGIT)
val_y = convert_labels_to_svm_labels(val_y, ENTRY_NUMBER_LAST_DIGIT)

test_processor = MNISTPreprocessor('./dataset/test', PRE_PROCESSING_CONFIG)
test_X, test_y = test_processor.get_all_data()
test_X, test_y = filter_dataset(test_X, test_y, ENTRY_NUMBER_LAST_DIGIT)
test_y = convert_labels_to_svm_labels(test_y, ENTRY_NUMBER_LAST_DIGIT)

def grid_search_svm(X_train, y_train, X_val, y_val, C_values, kernel_type):
    best_C = None
    best_f1_score = -1
    results = []

    for C in tqdm(C_values, desc=f"Grid search for {kernel_type} kernel", unit="C value"):
        model = SoftMarginSVMQP(C=C, kernel=kernel_type)
        model.fit(X_train, y_train)
        
        y_val_pred = model.predict(X_val)
        
        f1 = f1_score(y_val, y_val_pred)
        print("The F1 score is: ")
        print(f1)
        results.append((C, f1))
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_C = C

    return best_C, best_f1_score, results

def test_svm():
    C_values = [0.001,0.01,0.1,1,10,100,1000]

    best_C_linear, best_f1_linear, linear_results = grid_search_svm(train_X, train_y, val_X, val_y, C_values, kernel_type='linear')
    best_C_rbf, best_f1_rbf, rbf_results = grid_search_svm(train_X, train_y, val_X, val_y, C_values, kernel_type='rbf')

    print("Grid Search Results for Linear Kernel:")
    for C, f1 in linear_results:
        print(f"C = {C}: F1 Score = {f1:.4f}")
    print(f"\nBest C for Linear Kernel: {best_C_linear} with F1 Score = {best_f1_linear:.4f}\n")

    print("Grid Search Results for RBF Kernel:")
    for C, f1 in rbf_results:
        print(f"C = {C}: F1 Score = {f1:.4f}")
    print(f"\nBest C for RBF Kernel: {best_C_rbf} with F1 Score = {best_f1_rbf:.4f}")


def grid_search_rf(X_train, y_train, X_val, y_val, min_samples_values, max_depth_values, pca_components_values):
    best_score = 0
    best_params = None
    all_results = []  
    
    for min_samples in min_samples_values:
        for pca_components in pca_components_values:
            for max_depth in max_depth_values:
                start_time = time.time()
                
                rf_model = RandomForestClassifier(num_trees=30, min_samples=min_samples, max_depth=max_depth, pca_components=pca_components)
                rf_model.fit(X_train, y_train)
                
                y_train_pred = rf_model.predict(X_train)
                train_score = f1_score(y_train, y_train_pred)
                
                y_val_pred = rf_model.predict(X_val)
                val_score = f1_score(y_val, y_val_pred)
                
                end_time = time.time()
                time_taken = end_time - start_time
                
                print(f"Min Samples: {min_samples}, Max Depth: {max_depth}, PCA Components: {pca_components} -> Train F1 Score: {train_score:.4f}, Val F1 Score: {val_score:.4f}, Time Taken: {time_taken:.4f} seconds")
                
                all_results.append((min_samples, max_depth, pca_components, train_score, val_score, time_taken))
                
                if val_score > best_score:
                    best_score = val_score
                    best_params = {
                        'min_samples': min_samples,
                        'max_depth': max_depth,
                        'pca_components': pca_components
                    }

    print("\nAll Grid Search Results:")
    for min_samples, max_depth, pca_components, train_score, val_score, time_taken in all_results:
        print(f"Min Samples: {min_samples}, Max Depth: {max_depth}, PCA Components: {pca_components} -> Train F1 Score: {train_score:.4f}, Val F1 Score: {val_score:.4f}, Time Taken: {time_taken:.4f} seconds")
    
    print("\nBest F1 Score (Validation):", best_score)
    print("Best Parameters:", best_params)
    
    return best_params, best_score


# min_samples_values = [10]  
# max_depth_values = [15]  
# pca_components_values = [150]

# best_params, best_score = grid_search_rf(train_X, train_y, val_X, val_y, min_samples_values, max_depth_values, pca_components_values)

#This was part of svm to retrieve the top 6 support vectors
def get_top_support_vectors_image(self, top_n=6, img_shape=(28, 28), output_filename="top_support_vectors.png"):
    top_indices = np.argsort(-self.lamda)[:top_n]
    top_sv = self.sv[top_indices]

    # Plot the top support vectors in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(8, 5))
    for i, ax in enumerate(axs.flatten()):
        if i < top_n:
            img = top_sv[i].reshape(img_shape)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"λ={self.lamda[top_indices[i]]:.2f}")
        else:
            ax.axis('off')  # Hide any extra subplots if top_n < 6

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.show()

# I added this piece of code to svm and then ran the test.py file to generate the required stats
def report(self, X_train, y_train, X_test, y_test, output_dir="SVM_reports"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_file_path = os.path.join(output_dir, f"svm_report_{timestamp}.txt")
        with open(report_file_path, 'w') as report_file:
            n_support_vectors = len(self.sv)
            report_file.write(f"Number of support vectors: {n_support_vectors}\n")
            
            # Training accuracy and F1-Score
            y_train_pred = self.predict(X_train)
            y_test_pred = self.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            report_file.write(f"Training Accuracy: {train_accuracy}\n")
            report_file.write(f"Test Accuracy: {test_accuracy}\n")
            report_file.write(f"Training F1-Score: {train_f1}\n")
            report_file.write(f"Test F1-Score: {test_f1}\n")

            # Misclassified Instances
            misclassified_train_idx = np.where(y_train != y_train_pred)[0]
            misclassified_test_idx = np.where(y_test != y_test_pred)[0]

            misclassified_train_images = X_train[misclassified_train_idx[:4]]
            misclassified_test_images = X_test[misclassified_test_idx[:4]]

            report_file.write(f"Number of misclassified instances (train): {len(misclassified_train_idx)}\n")
            report_file.write(f"Number of misclassified instances (test): {len(misclassified_test_idx)}\n")

            # Plot misclassified images (Train)
            plt.figure(figsize=(12, 6))
            for i, img in enumerate(misclassified_train_images):
                plt.subplot(2, 2, i + 1)
                plt.imshow(img.reshape(28, 28), cmap='gray')
                plt.title(f"Misclassified Train {i + 1}")
            plt.savefig(os.path.join(output_dir, f"misclassified_train_images_{timestamp}.png"))
            plt.close()

            # Plot misclassified images (Test)
            plt.figure(figsize=(12, 6))
            for i, img in enumerate(misclassified_test_images):
                plt.subplot(2, 2, i + 1)
                plt.imshow(img.reshape(28, 28), cmap='gray')
                plt.title(f"Misclassified Test {i + 1}")
            plt.savefig(os.path.join(output_dir, f"misclassified_test_images_{timestamp}.png"))
            plt.close()

            print(f"Report generated: {report_file_path}")
            

def get_metrics(self, X_train, y_train, X_test, y_test, filename="ada_boost_metrics.txt"):
    y_train_pred = self.predict(X_train)
    y_test_pred = self.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    # Misclassified instances (train and test)
    train_misclassified = np.sum(y_train != y_train_pred)
    test_misclassified = np.sum(y_test != y_test_pred)

    # Misclassified images (first 4)
    misclassified_train_indices = np.where(y_train != y_train_pred)[0][:4]
    misclassified_test_indices = np.where(y_test != y_test_pred)[0][:4]

    # Create a folder for saving metrics and images
    folder_name = 'AdaBoost_Results'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Write the metrics to a text file
    metrics_file = os.path.join(folder_name, filename)
    with open(metrics_file, 'w') as f:
        f.write(f"AdaBoost Classifier Metrics:\n")
        f.write(f"Training Accuracy: {train_accuracy:.4f}\n")
        f.write(f"Training F1-Score: {train_f1:.4f}\n")
        f.write(f"Number of Misclassified Instances (Train): {train_misclassified}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test F1-Score: {test_f1:.4f}\n")
        f.write(f"Number of Misclassified Instances (Test): {test_misclassified}\n")
        f.write(f"Misclassified Train Indices: {misclassified_train_indices}\n")
        f.write(f"Misclassified Test Indices: {misclassified_test_indices}\n")

    # Save misclassified images as PNG files and combine them into one image
    train_images = []
    for idx in misclassified_train_indices:
        img = X_train[idx].reshape(28, 28)  
        train_images.append(img)

    test_images = []
    for idx in misclassified_test_indices:
        img = X_test[idx].reshape(28, 28) 
        test_images.append(img)

    # Create the folder for images
    images_folder = os.path.join(folder_name, 'misclassified_images')
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Combine the misclassified train images into one image
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(train_images[i], cmap='gray')
        ax.set_title(f'Misclassified Train {i+1}')
        ax.axis('off')

    plt.tight_layout()
    train_image_path = os.path.join(images_folder, 'misclassified_train_combined.png')
    plt.savefig(train_image_path)
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(test_images[i], cmap='gray')
        ax.set_title(f'Misclassified Test {i+1}')
        ax.axis('off')

    plt.tight_layout()
    test_image_path = os.path.join(images_folder, 'misclassified_test_combined.png')
    plt.savefig(test_image_path)
    plt.close()

    print(f"Metrics saved in {metrics_file}")
    print(f"Misclassified images saved in {images_folder}")