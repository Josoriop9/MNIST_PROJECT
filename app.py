import numpy as np
import tensorflow as tf
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Cargar modelo
model = tf.keras.models.load_model('mnist_model.keras')

class MnistApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognition")

        # Configuración del área de dibujo
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, 28)
        self.ax.set_ylim(28, 0)
        self.ax.axis('off')  # Oculta los ejes para un aspecto más limpio
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=TOP, fill=BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)  # Capturar movimiento del ratón

        # Botón para predecir
        self.predict_button = Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(side=LEFT)

        # Botón para limpiar
        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=RIGHT)

        self.image = np.zeros((28, 28))

    def on_click(self, event):
        self.draw(event.xdata, event.ydata)

    def on_drag(self, event):
        self.draw(event.xdata, event.ydata)

    def draw(self, x, y):
        if x is not None and y is not None:
            x, y = int(x), int(y)
            # Asegurarse de que el trazo sea grueso y visible
            radius = 1  # Puedes aumentar el radio para un pincel más grueso
            self.image[max(y-radius, 0):min(y+radius+1, 28), max(x-radius, 0):min(x+radius+1, 28)] = 1
            self.ax.clear()
            self.ax.imshow(self.image, cmap='gray')
            self.canvas.draw()

    def predict(self):
        # Redimensionar imagen para el modelo
        img_reshaped = self.image.reshape(1, 28, 28, 1)
        prediction = model.predict(img_reshaped)
        pred_digit = np.argmax(prediction)
        print(f"Predicted Digit: {pred_digit}")

    def clear_canvas(self):
        # Limpiar el lienzo y la matriz de imagen
        self.image = np.zeros((28, 28))
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        self.canvas.draw()

root = Tk()
app = MnistApp(root)
root.mainloop()
 