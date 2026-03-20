# 🎧 Voice Profile Restoration Engine

A system that learns how a specific speaker sounds on a high-quality microphone chain, then restores toward that reference using spectral profile matching and automated EQ correction.

---

## 🚀 Why this matters

Most recording artists create audio in untreated environments using low-quality recording gear.

This system improves:

- 🎯 Vocal clarity   
- 🔊 Tonal balance  
- 📈 Perceived production quality  

---

## ⚙️ What this system does

This pipeline:

1. Extracts spectral features from clean reference audio  
2. Builds a target voice profile  
3. Detects tonal imbalance in degraded recordings  
4. Maps imbalance to EQ correction  
5. Applies iterative restoration  
6. Applies final polish processing  
7. Scores audio quality before and after  
8. Outputs improved audio + reports  

---

## 📊 Example Output

- Before Score: 23.60  
- After Score: 25.54  
- Improvement: +1.94  

---

## 🧠 Key idea

Instead of manual EQ tweaking, this system:

👉 **learns what “good audio” looks like**  
👉 and automatically pushes bad audio toward that target  

---

## 🛠 Tech Stack

- Python  
- NumPy / SciPy  
- Librosa  
- SoundFile  

---

## ▶️ Usage

```bash
PYTHONPATH=. python3 cli.py \
  --input_dirs \
  "/path/to/folder_a" \
  "/path/to/folder_b"

