# AuthNet: Neural Network with Integrated Authentication Logic

This repository contains the official implementation of **AuthNet: Neural Network with Integrated Authentication Logic**.  

We provide a trained **AuthNet VGG13** model together with its corresponding **secret key** for testing.  

---

## 🚀 Quick Start

To test with the pretrained model, simply run:

```bash
python Effectiveness.py
```

---

## 🛠️ Training Your Own AuthNet Model

If you want to train your own AuthNet from scratch, please follow the steps below:

1. Train a clean model:
   ```bash
   python train_clean_models.py --model=VGG13 --epochs=50
   ```

2. Generate inverse mask:
   ```bash
   python inverse_mask.py --model=VGG13 --epsilon_M=0.1 --epsilon_AE=0.03
   ```

3. Train the AuthNet-tail model:
   ```bash
   python train_tail.py --model=VGG13 --epochs=50
   ```

---

## 📂 Project Structure

```
.
├── checkpoints/              # Saved model checkpoints
├── mask/
│   └── effectiveness/VGG13/  # Precomputed masks for VGG13
├── models/                   # Model architectures
├── Effectiveness.py          # Test effectiveness of pretrained AuthNet
├── inverse_mask.py           # Generate inverse mask
├── train_clean_models.py     # Train clean base models
├── train_tail.py             # Train AuthNet-tail models
└── utils.py                  # Utility functions
```

---

## 📌 Requirements

- python>=3.8
- torch>=1.10
- torchvision>=0.11
- numpy
- pillow

---

## ✨ Citation

If you find this repository useful, please consider citing our paper:  

```
@article{AuthNet2025,
  title   = {AuthNet: Neural Network with Integrated Authentication Logic},
  author  = {Cai Yuling, Xiang Fan, Meng Guozhu, Cao Yinzhi, Chen Kai},
  conference = {European Conference on Artificial Intelligence},
  year    = {2025}
}
```

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
