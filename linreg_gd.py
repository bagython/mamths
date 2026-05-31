import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def make_data(n: int, w: float, b: float, noise: float, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 10.0, n)
    return xs, w * xs + b + rng.normal(0.0, noise, n)


def mse(xs, ys, w: float, b: float):
    return float(np.mean((w * xs + b - ys) ** 2))


def descend(xs, ys, alpha: float, steps: int, w=0.0, b=0.0):
    history = [(w, b)]
    for _ in range(steps):
        err = w * xs + b - ys
        w -= alpha * 2.0 * np.mean(err * xs)
        b -= alpha * 2.0 * np.mean(err)
        history.append((w, b))
    return history


def animate_fit(xs, ys, history, alpha: float, path: str, frames=120):
    step = max(1, len(history) // frames)
    shots = history[::step]

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=12, color="tab:blue", label="data")
    line_x = np.array([xs.min(), xs.max()])
    (line,) = ax.plot([], [], color="tab:red", lw=2, label="fit")
    text = ax.text(0.05, 0.92, "", transform=ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Linear regression via gradient descent (alpha={alpha})")
    ax.legend(loc="lower right")

    def update(i):
        w, b = shots[i]
        line.set_data(line_x, w * line_x + b)
        text.set_text(f"iter {i * step}: w={w:.3f} b={b:.3f}")
        return line, text

    anim = animation.FuncAnimation(
        fig, update, frames=len(shots), interval=60, blit=True
    )
    anim.save(path, writer=animation.PillowWriter(fps=20))
    plt.close(fig)


def plot_alphas(xs, ys, alphas: list[float], steps: int, path: str):
    fig, ax = plt.subplots()
    for alpha in alphas:
        losses = [mse(xs, ys, w, b) for w, b in descend(xs, ys, alpha, steps)]
        ax.plot(losses, label=f"alpha={alpha}")
    ax.set(
        yscale="log",
        xlabel="iteration",
        ylabel="MSE",
        title="Effect of step size on convergence",
    )
    ax.legend()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    xs, ys = make_data(n=60, w=2.5, b=-4.0, noise=2.0, seed=1)

    alpha, steps = 0.01, 400
    history = descend(xs, ys, alpha, steps)
    w, b = history[-1]
    print(f"fit: w={w:.4f} b={b:.4f}  (true w=2.5 b=-4.0)")
    print(f"final MSE: {mse(xs, ys, w, b):.4f}")

    animate_fit(xs, ys, history, alpha, "linreg_fit.gif")
    plot_alphas(xs, ys, [0.002, 0.01, 0.031], steps, "linreg_alphas.png")
    print("wrote linreg_fit.gif and linreg_alphas.png")


if __name__ == "__main__":
    main()
