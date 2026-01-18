Great question — this is **exactly** about **PyTorch (NumPy-style) broadcasting**, and your intuition is already very close 👍
Let’s break it down **slowly and precisely**, with shapes and an example.

---

## 1️⃣ The tensors you have

You said:

```python
token_embeddings.shape == (8, 4, 64)
pos_embeddings.shape   == (4, 64)
```

Let’s interpret these dimensions (typical in NLP): 

| Dimension | Meaning                  |
| --------- | ------------------------ |
| `8`       | batch size               |
| `4`       | sequence length (tokens) |
| `64`      | embedding dimension      |

So:

* **token_embeddings** → embedding for each token in each batch
* **pos_embeddings** → embedding for each *position* in the sequence

---

## 2️⃣ How can tensors of different ranks be added?

This works because of **broadcasting rules**.

### 🔹 Broadcasting rule (important)

When operating on two tensors:

1. Compare shapes **from right to left**
2. Dimensions are compatible if:

   * they are equal, or
   * one of them is `1`
3. Missing leading dimensions are treated as `1`

---

## 3️⃣ Apply broadcasting step-by-step

### Step 1: Align dimensions

PyTorch internally treats:

```text
token_embeddings: (8, 4, 64)
pos_embeddings:        (4, 64)
```

as:

```text
token_embeddings: (8, 4, 64)
pos_embeddings:   (1, 4, 64)   ← implicit expansion
```

Why?

* `pos_embeddings` is missing the **batch dimension**
* PyTorch **assumes batch size = 1** and broadcasts it

---

### Step 2: Check compatibility (right → left)

| Dimension | token | position | Compatible? |
| --------- | ----- | -------- | ----------- |
| 64        | 64    | 64       | ✅ same      |
| 4         | 4     | 4        | ✅ same      |
| 8         | 8     | 1        | ✅ broadcast |

So broadcasting is valid ✅

---

## 4️⃣ What actually happens during addition?

This line:

```python
input_embeddings = token_embeddings + pos_embeddings
```

is **conceptually equivalent to**:

```python
pos_embeddings_expanded = pos_embeddings.unsqueeze(0)  # (1, 4, 64)
pos_embeddings_expanded = pos_embeddings_expanded.repeat(8, 1, 1)

input_embeddings = token_embeddings + pos_embeddings_expanded
```

### 🔹 Meaning in plain English

> **The same position embedding is added to every batch element**

So for:

* token at position `j`
* embedding dimension `k`

You get:

```text
input_embeddings[b, j, k]
= token_embeddings[b, j, k]
+ pos_embeddings[j, k]
```

✔ **Same position vector added across all batches**

---

## 5️⃣ Your intuition — was it correct?

You said:

> “I think the fixed, the third dimension will be simply added as a fixed value to all.”

✅ **Yes — but more precisely:**

* The **entire `(4, 64)` position matrix** is added
* It is **copied across the batch dimension**
* Not just a single value — but a **full vector per position**

So:

* Position 0 has its own 64-dim vector
* Position 1 has its own 64-dim vector
* …
* These are shared across all batches

---

## 6️⃣ Why Transformers are designed this way

This is intentional:

* **Token embedding** → *what* the word is
* **Position embedding** → *where* the word is
* Adding them gives:

  > “word meaning + position information”

And batching should **not change positional meaning**, so the same position embedding is reused for all batches.

---

## 7️⃣ Quick sanity check (small example)

```python
token = torch.zeros(2, 3, 5)   # batch=2, seq=3, emb=5
pos   = torch.ones(3, 5)

out = token + pos
print(out.shape)
```

Output:

```text
torch.Size([2, 3, 5])
```

Each batch receives **identical position embeddings**.

---

## 8️⃣ Key takeaway (one-liner)

> **PyTorch broadcasts `(4, 64)` to `(1, 4, 64)` and then to `(8, 4, 64)`, adding the same position embeddings to every batch.**

If you want, next we can:

* Visualize broadcasting with actual numbers
* Explain why concatenation is *not* used instead of addition
* Relate this to `nn.Embedding` and `register_buffer` usage in Transformers
