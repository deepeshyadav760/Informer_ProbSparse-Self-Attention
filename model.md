# Informer Architecture and Data Flow

Here is the complete architectural flow for the **Informer Model** and how the data moves through the Encoder and Decoder during both training and testing.

### 🧩 1. The Setup (Inputs)
In time-series forecasting, we deal with sequences. The Informer splits the timeline into specific chunks:
*   **Encoder Input (`x_enc`)**: A long historical sequence (e.g., the past 96 days). This contains all our environmental variables (`temperature`, `soil_moisture`, `ndvi`, etc.) and the target (`nee`).
*   **Decoder Input (`x_dec`)**: This is divided into two parts:
    1.  **Label Length:** The recent known past (e.g., the last 48 days from `x_enc`). It gives the Decoder a starting point.
    2.  **Prediction Length:** The future horizon we want to predict (e.g., the next 24 days).

---

### 🏢 2. The Encoder: Summarizing the Past
The Encoder's job is to look at the past 96 days and compress it into a dense, highly informative memory state.
1.  **Embedding:** The continuous variables and time features (day-of-year sines/cosines) are embedded into a high-dimensional space (`d_model`).
2.  **ProbSparse Self-Attention:** Instead of standard attention where every day looks at every other day $O(L^2)$ (which uses too much memory for long time-series), the Informer calculates an empirical "sparsity score". It only allows the most "important" or "active" queries to attend to all keys, dropping the lazy ones. This runs much faster at $O(L \log L)$.
3.  **Self-Attention Distilling:** Between the attention layers, the sequence goes through a 1D Convolution and Max-Pooling. This halves the length of the sequence (e.g., 96 days $\rightarrow$ 48 days $\rightarrow$ 24 days). It forcefully extracts only the most dominant seasonal and trend features.
4.  **Output (`enc_out`):** A heavily compressed, feature-rich memory of the past.

---

### 🏭 3. The Decoder: Generative Forecasting
The Decoder's job is to take the "Label Length" (recent past) and generate the "Prediction Length" (future). The Informer uses a **Generative Style Decoder**, which predicts the *entire future sequence in a single step* (unlike typical auto-regressive models which generate step-by-step).
1.  **Masked Self-Attention:** The Decoder looks at its own sequence (`x_dec`). A mask is applied so that when it is processing Day $T$, it cannot "cheat" and look at Day $T+1$.
2.  **Cross-Attention (The Merger):** The Decoder takes what it learned about itself and compares it against `enc_out` (the compressed memory from the Encoder). This is where the model asks: *"Given the recent trend (decoder), how does this align with the long-term historical patterns (encoder)?"*
3.  **Projection:** A final Linear layer takes the high-dimensional output and squashes it down to `1` value: our predicted Carbon Flux (`nee`).

---

### 🔄 4. Complete Flow: Training vs. Testing

#### 🏋️‍♂️ During Training
*   **Input Prep:** We chop our historical dataset into sliding windows. We know the "true future" for each window.
*   **The Decoder Trick:** The `x_dec` prediction portion is filled with **zeros**.
*   **Forward Pass:** The model looks at the past (Encoder) and the recent past + zeros (Decoder) and spits out a 24-day prediction all at once.
*   **Loss & Update:** We calculate the Mean Squared Error (MSE) between the model's 24-day prediction and the *actual* 24 days that happened. The error is backpropagated to adjust the weights of the ProbSparse and Cross-Attention layers.

#### 🎯 During Testing / Real-World Inference
*   **Input Prep:** We take the *most recent* 96 days of satellite data. We have absolutely no idea what the future holds.
*   **The Decoder Trick:** We take the last 48 days of the known data, append 24 days of **zeros**, and feed this to the Decoder.
*   **Forward Pass:** The Encoder distills the 96 days. The Decoder cross-attends to this memory and, in **one single forward pass**, predicts the `nee` values to replace those 24 zeros. 
*   **Result:** We get our 24-day forecast instantly without the compounding errors common in auto-regressive (step-by-step) models!
