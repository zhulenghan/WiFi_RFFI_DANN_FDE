# WiFi_RFFI_DANN_FDE
# Device Authentication for Wi-Fi Based on Deep Learning and Radio Frequency Fingerprint

## Introduction

This project explores the application of Radio Frequency Fingerprint Identification (RFFI) to 40MHz packages of Wi-Fi 4, aiming to enhance IoT security by leveraging unique device identifiers at the physical layer. Unlike traditional methods, RFFI offers a novel approach to security without heavy computational demands. The project aligns RFFI with the latest Wi-Fi standards and addresses challenges such as the Wi-Fi dataset offset problem.

## Project Motivation

Convolutional Neural Network (CNN)-based radio fingerprinting algorithms often suffer from significant accuracy degradation due to wireless channel effects. This project investigates domain adversarial training as a promising solution for domain generalization, mitigating these channel effects and improving RFFI system robustness.

## Dataset

The dataset was collected using Software Defined Radio (SDR) USRP B210 and 8 Wi-Fi dongles across various locations and dates. It includes raw I/Q samples, Channel State Information (CSI), Carrier Frequency Offset (CFO), and timestamps in HT-mixed format with 40MHz bandwidth.

| #  | Place           | Date       | Packages × STA |
|----|-----------------|------------|----------------|
| 0  | Bedroom         | 13/12/2023 | 2500×8         |
| 1  | Kitchen         | 05/03/2024 | 2000×8         |
| 2  | Harold Cohen    | 06/03/2024 | 2000×8         |
| 3  | Sydney Jones    | 07/03/2024 | 1500×8         |
| 4  | EEE Lab         | 08/03/2024 | 2000×8         |

## RFFI Algorithm

The proposed RFFI algorithm consists of three stages: signal preprocessing, training, and inference. 

### Signal Preprocessing

- **Compensation**: CFO Compensation
- **Equalization**: Frequency Domain Equalization (FDE)
- **Trimming**: Removing DC and 0-value subcarriers
- **Normalization**: Using RMS, Z-score, and Max

### Feature Extraction

A CNN-based feature extractor processes the preprocessed I/Q samples to maintain physical meaning. Domain-adversarial training (DANN) is utilized for training the RFFI extractor, followed by finetuning for higher accuracy.

## Results

### Performance

- **I/Q CNN**: Highest accuracy of 36.4%, with instability and significant drop in worst-case scenarios.
- **FDE + DANN**: Decent and stable accuracy, further improved with finetuning.

| Method         | 0,1,2,3->4 | 1,2,3,4->0 | 0,2,3,4->1 | 0,1,3,4->2 | 0,1,2,4->3 |
|----------------|------------|------------|------------|------------|------------|
| I/Q CNN        | 21.2       | 36.4       | 11.9       | 0.4        | 15.9       |
| FDE + DANN     | 60.0       | 55.2       | 44.0       | 60.5       | 59.5       |
| +Finetune      | 78.1       | 92.0       | 86.2       | 84.9       | 80.8       |


## Source Code Overview

The repository contains the following files:

1. **dataloader.py**: Functions and classes for loading and preprocessing the dataset.
2. **dataset_create.py**: Script for creating the dataset, including data collection and initial preprocessing.
3. **demo.py**: Demonstrates how to use the trained models for inference on new data.
4. **demo**: Directory likely containing sample data, pre-trained models, or additional demo-related scripts.
5. **iqcnn.py**: Implementation of the CNN-based model for I/Q sample processing.
6. **knn.py**: K-Nearest Neighbors (KNN) algorithm implementation for baseline comparison.
7. **mac_addresses.json**: JSON file containing MAC addresses for labeling or filtering the dataset.
8. **main.py**: Main entry point for training and evaluating the models.
9. **models.py**: Definitions of various models used in the project, including neural network architectures.
10. **picoscenes.cpython-311-x86_64-linux-gnu.so**: Compiled shared object file providing optimized routines or bindings to lower-level libraries.
11. **requirements.txt**: Lists the Python dependencies required to run the project.
12. **tsne.py**: Code for t-SNE visualization to analyze high-dimensional feature embeddings.
13. **utils.py**: Utility functions used across various parts of the project, such as logging, data manipulation, and evaluation metrics.

### Setup and Usage

To set up the project environment, you can use the `requirements.txt` file to install necessary dependencies:

```sh
pip install -r requirements.txt
```

You can then run the main script to start training and evaluation:

```sh
python main.py
```

For a practical demonstration of the model's capabilities, you can use the demo script:

```sh
python demo.py
```

## Conclusion

This project presents the first Wi-Fi 4 RFFI dataset, suitable for testing the channel robustness of RFFI systems. The proposed preprocessing methods and domain-adversarial training significantly enhance the system's accuracy and stability. Future work will explore non-parametric methods such as Cosine Similarity for further improvements.

## References

1. IDC, December 2019
2. J. Zhang et al., “Radio frequency fingerprint identification for device authentication in the internet of things,” IEEE Communications Magazine, 2023.
3. A. Al-Shawabka et al., “Exposing the fingerprint: Dissecting the impact of the wireless channel on radio fingerprinting,” in IEEE INFOCOM 2020-IEEE Conference on Computer Communications, 2020, pp. 646–655.
4. Y. Li et al., “Deep domain generalization via conditional invariant adversarial networks,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 624–639.

## Contact

For more information, contact Lenghan Zhu at sglzhu9@liverpool.ac.uk.
```

This updated README provides a comprehensive overview of the project, including details about the dataset, algorithm, results, source code, and usage instructions.
