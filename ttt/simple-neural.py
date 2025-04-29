import numpy as np, random

# ------------- 1.  synthetic “rotate-90°” tasks -----------------
np.random.seed(0);  random.seed(0)

def rot90(a):            # 90° clockwise
    return np.rot90(a, -1)

def rand_grid(p=0.4):
    return (np.random.rand(3, 3) < p).astype(np.float32)

def make_tasks(n=2000):
    tasks = []
    for _ in range(n):
        tr_in = rand_grid()
        te_in = rand_grid()
        tasks.append((tr_in, rot90(tr_in), te_in, rot90(te_in)))
    return tasks

tasks = make_tasks()

# ------------- 2.  tiny fully-connected ReLU net ----------------
IN, H, OUT = 27, 64, 9
W1 = (0.1 * np.random.randn(H, IN)).astype(np.float32);  b1 = np.zeros(H,  dtype=np.float32)
W2 = (0.1 * np.random.randn(OUT, H)).astype(np.float32); b2 = np.zeros(OUT, dtype=np.float32)

relu = lambda x: np.maximum(0., x)

LR         = 2e-3          # learning-rate
EPOCHS     = 8_000
CLIP       = 1.0           # clip range for all grads
WEIGHT_DEC = 1e-4          # L2 weight-decay

for _ in range(EPOCHS):
    tr_in, tr_out, te_in, te_out = random.choice(tasks)
    x = np.concatenate([tr_in.ravel(), tr_out.ravel(), te_in.ravel()]).astype(np.float32)
    y = te_out.ravel().astype(np.float32)

    # forward
    h0 = W1 @ x + b1
    h  = relu(h0)
    y_hat = W2 @ h + b2

    # gradients (MSE)
    g = (2.0 / OUT) * (y_hat - y)

    # --- clip & update W2 / b2 ---
    grad_W2 = np.outer(g, h)
    np.clip(grad_W2, -CLIP, CLIP, out=grad_W2)
    grad_b2 = np.clip(g, -CLIP, CLIP)

    W2 -= LR * (grad_W2 + WEIGHT_DEC * W2)   # weight-decay
    b2 -= LR * grad_b2

    # --- back-prop into layer 1 ---
    dh   = W2.T @ g
    dh0  = dh * (h0 > 0)
    grad_W1 = np.outer(dh0, x)
    np.clip(grad_W1, -CLIP, CLIP, out=grad_W1)
    grad_b1 = np.clip(dh0, -CLIP, CLIP)

    W1 -= LR * (grad_W1 + WEIGHT_DEC * W1)
    b1 -= LR * grad_b1

# ------------- 3.  test on your specific example ----------------
train_in = np.array([[1,0,0],
                     [1,0,0],
                     [1,0,0]], dtype=np.float32)

train_out = np.array([[1,1,1],
                      [0,0,0],
                      [0,0,0]], dtype=np.float32)

test_in  = np.array([[0,1,0],
                     [0,1,0],
                     [0,1,0]], dtype=np.float32)

x_ex = np.concatenate([train_in.ravel(),
                       train_out.ravel(),
                       test_in.ravel()]).astype(np.float32)

h_ex = relu(W1 @ x_ex + b1)
y_ex = W2 @ h_ex + b2
pred  = (y_ex.reshape(3,3) > 0.5).astype(int)

print(pred)
