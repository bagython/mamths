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


def loss_surface(xs, ys, ws, bs):
    W, B = np.meshgrid(ws, bs)
    return np.mean((W[:, :, None] * xs + B[:, :, None] - ys) ** 2, axis=2)


def frames_by_distance(ws, bs, frames: int):
    """make it smootj (not snap)"""
    pts = np.column_stack([ws, bs])
    cum = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    )
    targets = np.linspace(0.0, cum[-1], frames)

    def _interp(v):
        return np.interp(targets, cum, v)

    return _interp(ws), _interp(bs), _interp(np.arange(len(ws)))


def animate_fit(
    xs, ys, history: list[tuple[float, float]], alpha: float, path: str, frames=120
):
    ws, bs = np.array(history).T
    wf, bf, itf = frames_by_distance(ws, bs, frames)

    fig, (ax_fit, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: data with the current fit line.
    ax_fit.scatter(xs, ys, s=12, color="tab:blue", label="data")
    line_x = np.array([xs.min(), xs.max()])
    (line,) = ax_fit.plot([], [], color="tab:red", lw=2.5, label="fit")
    text = ax_fit.text(0.05, 0.9, "", transform=ax_fit.transAxes)
    ax_fit.set(xlabel="x", ylabel="y", title=f"Fit (alpha={alpha})")
    ax_fit.legend(loc="lower right")

    # Right: MSE loss surface with the descent path rolling toward the minimum.
    wg = np.linspace(ws.min() - 0.5, ws.max() + 0.5, 120)
    bg = np.linspace(bs.min() - 0.5, bs.max() + 0.5, 120)
    Z = loss_surface(xs, ys, wg, bg)
    ax_loss.contourf(wg, bg, Z, levels=30, cmap="viridis")
    ax_loss.contour(wg, bg, Z, levels=30, colors="white", linewidths=0.3, alpha=0.4)
    (path_line,) = ax_loss.plot([], [], color="tab:red", lw=1.5)
    (dot,) = ax_loss.plot([], [], "o", color="white", ms=7, mec="black")
    ax_loss.set(xlabel="w", ylabel="b", title="Descent on the MSE surface")

    def update(i):
        w, b = wf[i], bf[i]
        line.set_data(line_x, w * line_x + b)
        text.set_text(
            f"iter {int(round(itf[i]))}: w={w:.3f} b={b:.3f}\nMSE={mse(xs, ys, w, b):.3f}"
        )
        path_line.set_data(wf[: i + 1], bf[: i + 1])
        dot.set_data([w], [b])
        return [line, text, path_line, dot]

    anim = animation.FuncAnimation(fig, update, frames=len(wf), interval=60, blit=True)
    anim.save(path, writer=animation.PillowWriter(fps=30))
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
