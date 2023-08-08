import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings


np.random.seed(42)
warnings.filterwarnings("ignore", category=np.RankWarning)


class PolynomialFittingApp:
    def __init__(self, coefficients, x_range, y_range, num_points, noise_scale):
        self.coefficients, self.x_range, self.y_range = coefficients, x_range, y_range
        self.num_points, self.noise_scale = num_points, noise_scale
        self.x, self.y, self.y_true = self.generate_data()
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax_sample_size = plt.axes([0.35, 0.02, 0.3, 0.04])
        self.ax_degree = plt.axes([0.35, 0.07, 0.3, 0.04])
        self.ax_noise_scale = plt.axes([0.35, 0.12, 0.3, 0.04])
        self.sample_size_slider = Slider(self.ax_sample_size, "Amostra", 1, len(self.x), valinit=100, valstep=1)
        self.degree_slider = Slider(self.ax_degree, "Grau de Ajuste", 1, 100, valinit=4, valstep=1)
        self.noise_scale_slider = Slider(self.ax_noise_scale, "Ruído", 1, 15, valinit=self.noise_scale, valstep=0.1)
        self.sample_size_slider.on_changed(self.update_plot)
        self.degree_slider.on_changed(self.update_degree)
        self.noise_scale_slider.on_changed(self.update_noise_scale)
        self.update_plot(None)
        plt.subplots_adjust(bottom=0.2, top=0.95)

    def generate_data(self):
        x = np.linspace(*self.x_range, self.num_points)
        noise = np.random.normal(0, self.noise_scale, len(x))
        y_true = np.polyval(self.coefficients, x)
        y = y_true + noise
        return x, y, y_true

    def fit_and_plot_model(self, x_sample, y_sample, degree):
        coeffs = np.polyfit(x_sample, y_sample, degree)
        model = np.poly1d(coeffs)
        y_pred = model(self.x)
        self.ax.clear()
        self.ax.scatter(x_sample, y_sample, label=f"Amostra com ruído", color="blue", alpha=0.5, s=20)
        self.ax.plot(self.x, self.y_true, color="red", label="Polinômio Original", linestyle="--", alpha=0.5, linewidth=1.5)
        self.ax.plot(self.x, y_pred, color="green", label=f"Modelo Ajustado ({degree})")
        self.ax.legend(loc="upper right", fontsize="medium")
        self.ax.set_title(f"Ajuste Polinomial de Grau {degree}", fontsize=12)
        self.ax.set_xlim(*self.x_range)
        self.ax.set_ylim(*self.y_range)
        plt.draw()

    def update_plot(self, event):
        sample_size = int(self.sample_size_slider.val)
        degree = int(self.degree_slider.val)
        xy = list(zip(self.x, self.y))
        np.random.shuffle(xy)
        self.x_sample, self.y_sample = zip(*xy[:sample_size])
        self.fit_and_plot_model(self.x_sample, self.y_sample, degree)

    def update_degree(self, event):
        degree = int(self.degree_slider.val)
        self.fit_and_plot_model(self.x_sample, self.y_sample, degree)

    def update_noise_scale(self, event):
        self.noise_scale = int(self.noise_scale_slider.val)
        self.x, self.y, self.y_true = self.generate_data()
        self.update_plot(None)


if __name__ == "__main__":
    coefficients = [1, -4, -1, 10, 0]
    x_range = [-2.5, 4.5]
    y_range = [-30, 40]
    num_points = 500
    noise_scale = 1
    app = PolynomialFittingApp(coefficients, x_range, y_range, num_points, noise_scale)
    plt.show()
