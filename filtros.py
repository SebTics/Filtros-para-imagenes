from PIL import Image, ImageOps
import numpy as np
import tkinter as tk  # Interfaz gráfica
from tkinter import filedialog, ttk, messagebox  # Widgets y texto para abrir archivos
from PIL import Image, ImageTk, ImageFilter, ImageDraw, ImageFont  # Manipulación de imágenes
import threading  # Ejecución concurrente
from collections import Counter

# Función para aplicar convoluciones, usada en los filtros como Sharpen y Promedio
def apply_convolution(img_array, kernel):
    h, w, d = img_array.shape  # Dimensiones de la imagen
    kh, kw = kernel.shape  # Dimensiones del kernel
    pad_h, pad_w = kh // 2, kw // 2  # Padding necesario para aplicar el kernel
    # Se añade un padding reflejado alrededor de la imagen
    padded_img = np.pad(img_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='reflect')
    new_img = np.zeros_like(img_array)  # Imagen resultante inicializada en ceros

    # Convolución: se aplica el kernel en cada píxel
    for i in range(h):
        for j in range(w):
            for c in range(d):
                new_img[i, j, c] = np.sum(kernel * padded_img[i:i+kh, j:j+kw, c])

    new_img = np.clip(new_img, 0, 255)  # Se asegura que los valores estén en el rango válido [0, 255]
    return new_img

# Filtros específicos

def apply_red(img_array):
    img_array[:, :, 1] = 0  # Elimina el canal verde
    img_array[:, :, 2] = 0  # Elimina el canal azul
    return img_array

def apply_blue(img_array):
    img_array[:, :, 0] = 0  # Elimina el canal rojo
    img_array[:, :, 1] = 0  # Elimina el canal verde
    return img_array

def apply_grey_scale(img_array):
    # Convierte la imagen a escala de grises usando ponderaciones estándar para los canales RGB
    grey_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return grey_array

def increase_contrast(img_array):
    # Incrementa el contraste multiplicando los valores de píxeles y limitándolos entre 0 y 255
    img_array = np.clip(img_array * 1.5, 0, 255).astype(np.uint8)
    return img_array

def high_contrast(img_array):
    # Convierte la imagen a un formato de alto contraste: valores menores a 128 se vuelven 0 y mayores a 255
    img_array = np.where(img_array < 128, 0, 255).astype(np.uint8)
    return img_array

def apply_mosaic(img_array, block_size=10):
    # Aplica un efecto de mosaico promediando los colores dentro de bloques de tamaño block_size
    for i in range(0, img_array.shape[0], block_size):
        for j in range(0, img_array.shape[1], block_size):
            block = img_array[i:i+block_size, j:j+block_size]
            avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            img_array[i:i+block_size, j:j+block_size] = avg_color
    return img_array

def apply_blur(img):
    return img.filter(ImageFilter.BLUR)  # Aplica un filtro de desenfoque (blur) usando PIL

def apply_motion_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=5))  # Simula motion blur usando Gaussian Blur

def apply_find_edges(img):
    return img.filter(ImageFilter.FIND_EDGES)  # Detecta bordes en la imagen usando PIL

def apply_sharpen(img_array):
    # Aplica un filtro de agudizado usando un kernel el cual es definido manualmente
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return apply_convolution(img_array, kernel)

def apply_emboss(img):
    return img.filter(ImageFilter.EMBOSS)  # Aplica un efecto tipo relieve usando PIL

def apply_average(img_array, kernel_size=5):
    # Aplica un filtro de promediado, suavizando la imagen dada
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return apply_convolution(img_array, kernel)

# Filtros nuevos de escala de grises y mosaico

# Función para convertir la imagen a escala de grises
def convertir_a_gris(imagen):
    return ImageOps.grayscale(imagen)

# Función para aplicar un filtro recursivo en tonos de gris
def filtro_recursivo_gris(imagen, iteraciones):
    imagen_original = imagen
    for i in range(iteraciones):
        imagen = convertir_a_gris(imagen)
        imagen = ImageOps.autocontrast(imagen)
    return imagen

# Función para generar una serie de imágenes con diferentes niveles de gris
def generar_imagenes_grises(imagen, niveles=20):
    imagenes = []
    for i in range(niveles):
        factor = i / niveles
        imagen_modificada = ImageOps.colorize(imagen.convert('L'), (int(factor*255), int(factor*255), int(factor*255)), (255, 255, 255))
        imagenes.append(imagen_modificada)
    return imagenes

# Función para dividir la imagen en una rejilla
def dividir_en_rejilla(imagen, tamanio_rejilla):
    ancho, alto = imagen.size
    rejilla = []
    for y in range(0, alto, tamanio_rejilla):
        for x in range(0, ancho, tamanio_rejilla):
            caja = (x, y, x + tamanio_rejilla, y + tamanio_rejilla)
            subimagen = imagen.crop(caja)
            rejilla.append((subimagen, caja))
    return rejilla

# Función para calcular el tono de gris promedio de una imagen
def tono_gris_promedio(subimagen):
    np_imagen = np.array(subimagen)
    return np.mean(np_imagen)

# Función para aplicar el segundo filtro en base a los tonos de gris de una rejilla
def filtro_gris_reemplazo(imagen, imagenes_grises, tamanio_rejilla):
    imagen_gris = convertir_a_gris(imagen)
    rejilla = dividir_en_rejilla(imagen_gris, tamanio_rejilla)

    nueva_imagen = Image.new('RGB', imagen.size)
    for subimagen, caja in rejilla:
        tono = tono_gris_promedio(subimagen)
        indice_imagen_gris = int((tono / 255) * (len(imagenes_grises) - 1))
        imagen_gris_cercana = imagenes_grises[indice_imagen_gris]
        nueva_imagen.paste(imagen_gris_cercana.resize((caja[2] - caja[0], caja[3] - caja[1])), caja)

    return nueva_imagen

# Función para generar una serie de imágenes coloreadas según el promedio de color
def generar_imagenes_coloreadas(imagen, niveles=20):
    imagenes_coloreadas = []
    for i in range(niveles):
        # Factor para ajustar el nivel de color (i.e., de oscuro a brillante)
        factor = i / niveles
        # Modifica los colores de la imagen ajustando el brillo y saturación
        imagen_modificada = ImageOps.colorize(imagen.convert('L'), (int(factor*255), 0, 0), (255, 255, 255))
        imagenes_coloreadas.append(imagen_modificada)
    return imagenes_coloreadas

def color_promedio(imagen):
    # Convierte la imagen a un array de numpy (para procesar más rápido los colores)
    np_imagen = np.array(imagen)
    # Calcula el promedio de los colores en los canales (R, G, B) por separado
    promedio_color = np.mean(np_imagen, axis=(0, 1))  # Promedio a lo largo de las dimensiones x, y
    # Redondea y convierte a enteros los valores de color promedio
    return tuple(promedio_color.astype(int))

def filtro_mosaico_color_real(imagen, imagenes_coloreadas, tamanio_rejilla):
    ancho, alto = imagen.size
    nueva_imagen = Image.new('RGB', imagen.size)
    
    for x in range(0, ancho, tamanio_rejilla):
        for y in range(0, alto, tamanio_rejilla):
            subimagen = imagen.crop((x, y, x + tamanio_rejilla, y + tamanio_rejilla))
            color = color_promedio(subimagen)
            tono_promedio = int(sum(color) / 3)  # Calcula el tono promedio basado en RGB
            indice_imagen_coloreada = int((tono_promedio / 255) * (len(imagenes_coloreadas) - 1))
            imagen_coloreada_cercana = imagenes_coloreadas[indice_imagen_coloreada]
            nueva_imagen.paste(imagen_coloreada_cercana.resize((tamanio_rejilla, tamanio_rejilla)), (x, y))

    return nueva_imagen

def add_watermark(base_image, watermark_image, position=(0, 0)):
    # Asegúrate de que ambas imágenes están en modo RGBA
    base_image = base_image.convert("RGBA")
    watermark_image = watermark_image.convert("RGBA")

    # Obtén las dimensiones de la imagen base y la marca de agua
    base_width, base_height = base_image.size
    watermark_width, watermark_height = watermark_image.size

    # Ajusta la posición si está fuera del límite de la imagen base
    x, y = position
    if x + watermark_width > base_width:
        x = base_width - watermark_width
    if y + watermark_height > base_height:
        y = base_height - watermark_height

    # Añade la marca de agua en la posición deseada
    base_image.paste(watermark_image, (x, y), watermark_image)

    return base_image

def create_circular_mask(h, w):
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    mask = x**2 + y**2 <= (h // 2) ** 2
    return mask

def apply_semitone(image_array, block_size=10):
    # Si la imagen es RGB o RGBA, conviértela a escala de grises
    if len(image_array.shape) == 3:
        image_array = np.array(Image.fromarray(image_array).convert('L'))

    # Dimensiones de la imagen
    height, width = image_array.shape

    # Crear una copia de la imagen para modificar
    new_image = np.copy(image_array)

    # Iterar sobre bloques de la imagen
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_array[i:i + block_size, j:j + block_size]
            if block.shape[0] != block_size or block.shape[1] != block_size:
                continue

            # Calcular el promedio de los valores del bloque
            mean_value = np.mean(block)

            # Crear un nuevo bloque con el promedio en lugar de un círculo
            new_block = np.ones_like(block) * mean_value

            # Aplicar la máscara circular
            mask = create_circular_mask(block_size, block_size)
            new_block[mask] = 0  # Dibujar círculo negro en el bloque

            # Asignar el bloque nuevo a la imagen nueva
            new_image[i:i + block_size, j:j + block_size] = new_block

    return new_image


def apply_random_dithering(img_array):
    h, w = img_array.shape[:2]
    grey_image = apply_grey_scale(img_array.copy())  # Convertimos a escala de grises
    noise = np.random.random((h, w)) * 255  # Generamos ruido aleatorio
    
    # Comparación del nivel de gris con el ruido
    dithered_img = np.where(grey_image > noise, 255, 0).astype(np.uint8)
    
    return dithered_img

def apply_floyd_steinberg(img_array):
    grey_image = apply_grey_scale(img_array.copy())
    h, w = grey_image.shape
    new_img = np.copy(grey_image)

    for y in range(h):
        for x in range(w):
            old_pixel = new_img[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            new_img[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Difusión del error según el algoritmo Floyd-Steinberg
            if x + 1 < w:
                new_img[y, x + 1] = np.clip(new_img[y, x + 1] + error * 7 / 16, 0, 255)
            if y + 1 < h:
                if x > 0:
                    new_img[y + 1, x - 1] = np.clip(new_img[y + 1, x - 1] + error * 3 / 16, 0, 255)
                new_img[y + 1, x] = np.clip(new_img[y + 1, x] + error * 5 / 16, 0, 255)
                if x + 1 < w:
                    new_img[y + 1, x + 1] = np.clip(new_img[y + 1, x + 1] + error * 1 / 16, 0, 255)

    return new_img

def apply_ordered_dithering(img_array):
    grey_image = apply_grey_scale(img_array.copy())  # Escala de grises
    threshold_map = np.array([[1, 9, 3, 11], [13, 5, 15, 7], [4, 12, 2, 10], [16, 8, 14, 6]]) / 17.0 * 255
    h, w = grey_image.shape
    new_img = np.zeros_like(grey_image)

    for i in range(h):
        for j in range(w):
            threshold = threshold_map[i % 4, j % 4]  # Se aplica la matriz de Bayer
            new_img[i, j] = 255 if grey_image[i, j] > threshold else 0

    return new_img


# Clase principal de la app de filtros
class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplicador de Filtros")
        self.root.geometry("600x700")
        
        style = ttk.Style()
        style.theme_use('clam')

        # Botón para abrir la imagen
        self.open_button = ttk.Button(root, text="Abrir Imagen", command=self.open_image)
        self.open_button.pack(pady=10)

        # Área para mostrar la imagen cargada
        self.image_label = ttk.Label(root)
        self.image_label.pack(pady=10)

        # Entradas para ancho y alto de la imagen
        self.width_entry = ttk.Entry(root, width=10)
        self.width_entry.insert(0, "Ancho")
        self.width_entry.pack(pady=5)

        self.height_entry = ttk.Entry(root, width=10)
        self.height_entry.insert(0, "Alto")
        self.height_entry.pack(pady=5)

        # Botón para reescalar la imagen
        self.resize_button = ttk.Button(root, text="Reescalar Imagen", command=self.resize_image)
        self.resize_button.pack(pady=10)

        # Entrada de texto para escribir sobre la imagen
        self.text_entry = ttk.Entry(root, width=20)
        self.text_entry.insert(0, "Texto")
        self.text_entry.pack(pady=5)

        # Entradas para las coordenadas x, y
        self.x_entry = ttk.Entry(root, width=10)
        self.x_entry.insert(0, "X")
        self.x_entry.pack(pady=5)
        
        self.y_entry = ttk.Entry(root, width=10)
        self.y_entry.insert(0, "Y")
        self.y_entry.pack(pady=5)
        
        # Entrada para el tamaño del texto
        self.font_size_entry = ttk.Entry(root, width=10)
        self.font_size_entry.insert(0, "Tamaño")
        self.font_size_entry.pack(pady=5)

        # Botón para agregar texto
        self.text_button = ttk.Button(root, text="Agregar Texto", command=self.add_text)
        self.text_button.pack(pady=10)

        # Menú desplegable para seleccionar el filtro que se quiera aplicar
        self.filter_var = tk.StringVar(root)
        self.filter_var.set("Seleccionar Filtro")
        
        # Opciones de filtros disponibles en el menú desplegable
        self.filter_menu = ttk.OptionMenu(root, self.filter_var, "Original", "Rojo", "Azul", 
                                "Escala de Grises", "Incrementar Contraste", 
                                "Alto Contraste", "Mosaico", "Blur", "Motion Blur", 
                                "Encontrar Bordes", "Sharpen", "Emboss", "Promedio",
                                "Filtro Gris Recursivo", "Filtro Color Real Recursivo",
                                "Semitonos", "Dithering Aleatorio", "Floyd-Steinberg", "Dithering Ordenado/Disperso",
                                "Aplicar Marca de Agua", "Óleo", "Erosión Máxima", "Erosión Mínima",
                                command=self.apply_filter)

        self.filter_menu.pack(pady=10)

        # Barra de progreso que indica el estado de la aplicación del filtro
        self.progress = ttk.Progressbar(root, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=20, pady=10)

        # Botón para ver la imagen original sin filtros
        self.show_original_button = ttk.Button(root, text="Ver Imagen Original", command=self.show_original)
        self.show_original_button.pack(pady=10)
        
        # Variables para almacenar la imagen, su array y la marca de agua
        self.img = None
        self.img_array = None
        self.original_img = None
        self.watermark_img = None

    # Función para abrir y cargar una imagen
    def open_image(self):
        img_path = filedialog.askopenfilename()  # Texto para seleccionar el archivo de imagen
        if img_path:
            self.img = Image.open(img_path).convert("RGB")  # Carga la imagen en modo RGB
            self.original_img = self.img.copy()  # Guarda una copia de la imagen original
            self.img_array = np.array(self.img)  # Convierte la imagen en un array de NumPy
            self.display_image(self.img)  # Muestra la imagen en la interfaz

    # Función para abrir y cargar la imagen de la marca de agua
    def open_watermark(self):
        watermark_path = filedialog.askopenfilename()  # Texto para seleccionar el archivo de la marca de agua
        if watermark_path:
            self.watermark_img = Image.open(watermark_path)  # Carga la marca de agua
            messagebox.showinfo("Marca de Agua", "Marca de agua cargada exitosamente")  # Muestra mensaje de éxito

    # Función para aplicar el filtro seleccionado
    def apply_filter(self, selection):
        if self.img_array is None:
            return  # Si no hay imagen cargada, no se hace nada

        def process_filter():
            self.progress.start()  # Inicia la barra de progreso
            img_filtered = self.original_img  # Imagen filtrada inicializada como la original

            # Aplicación del filtro seleccionado
            if selection == "Original":
                img_filtered = self.original_img
            elif selection == "Rojo":
                img_filtered = Image.fromarray(apply_red(self.img_array.copy()))
            elif selection == "Azul":
                img_filtered = Image.fromarray(apply_blue(self.img_array.copy()))
            elif selection == "Escala de Grises":
                img_filtered = Image.fromarray(apply_grey_scale(self.img_array.copy()))
            elif selection == "Incrementar Contraste":
                img_filtered = Image.fromarray(increase_contrast(self.img_array.copy()))
            elif selection == "Alto Contraste":
                img_filtered = Image.fromarray(high_contrast(self.img_array.copy()))
            elif selection == "Mosaico":
                img_filtered = Image.fromarray(apply_mosaic(self.img_array.copy()))
            elif selection == "Blur":
                img_filtered = apply_blur(self.original_img)
            elif selection == "Motion Blur":
                img_filtered = apply_motion_blur(self.original_img)
            elif selection == "Encontrar Bordes":
                img_filtered = apply_find_edges(self.original_img)
            elif selection == "Sharpen":
                img_filtered = Image.fromarray(apply_sharpen(self.img_array.copy()))
            elif selection == "Emboss":
                img_filtered = apply_emboss(self.original_img)
            elif selection == "Promedio":
                img_filtered = Image.fromarray(apply_average(self.img_array.copy(), kernel_size=7))  # Kernel de tamaño 7x7
            elif selection == "Filtro Gris Recursivo":
                imagenes_grises = generar_imagenes_grises(self.original_img, niveles=10)
                img_filtered = filtro_gris_reemplazo(self.original_img, imagenes_grises, tamanio_rejilla=25)  # Filtro mosaico gris
            elif selection == "Filtro Color Real Recursivo":
                # Genera las imágenes coloreadas con diferentes niveles (puedes ajustar el nivel según tu necesidad)
                imagenes_coloreadas = generar_imagenes_coloreadas(self.original_img, niveles=10)
    
                # Aplica el filtro mosaico de color real
                img_filtered = filtro_mosaico_color_real(self.original_img, imagenes_coloreadas, tamanio_rejilla=25)

            elif selection == "Aplicar Marca de Agua":
                if self.watermark_img is not None:
                    img_filtered = add_watermark(self.original_img, self.watermark_img)
                else:
                    messagebox.showwarning("Marca de Agua", "No se ha cargado ninguna marca de agua.")
                    return
            elif selection == "Semitonos":
                img_filtered = Image.fromarray(apply_semitone(self.img_array.copy(), block_size=10))
            elif selection == "Dithering Aleatorio":
                img_filtered = Image.fromarray(apply_random_dithering(self.img_array.copy()))
            elif selection == "Floyd-Steinberg":
                img_filtered = Image.fromarray(apply_floyd_steinberg(self.img_array.copy()))
            elif selection == "Dithering Ordenado/Disperso":
                img_filtered = Image.fromarray(apply_ordered_dithering(self.img_array.copy()))
            elif selection == "Óleo":
                img_filtered = Image.fromarray(self.oil_paint())
            elif selection == "Erosión Máxima":
                img_filtered = Image.fromarray(self.max_filter())
            elif selection == "Erosión Mínima":
                img_filtered = Image.fromarray(self.min_filter())

            self.display_image(img_filtered)  # Muestra la imagen con filtro
            self.progress.stop()  # Detiene la barra de progreso

        # Ejecuta el proceso de filtrado en un hilo separado para no bloquear la interfaz
        threading.Thread(target=process_filter).start()
        
    def oil_paint(self, radius=3, intensity_levels=20):
        image = self.original_img.copy().convert("RGB")
        pixels = np.array(image)
        new_img = np.zeros_like(pixels)
        height, width, _ = pixels.shape

        # Crear un array de intensidades para un procesamiento más rápido
        intensity_bin = np.linspace(0, 255, intensity_levels, endpoint=False).astype(int)

        for y in range(height):
            for x in range(width):
                color_count = np.zeros((intensity_levels, 3), dtype=int)
                count = np.zeros(intensity_levels, dtype=int)

                for ny in range(max(0, y - radius), min(height, y + radius + 1)):
                    for nx in range(max(0, x - radius), min(width, x + radius + 1)):
                        r, g, b = pixels[ny, nx]
                        intensity = (int(r) + int(g) + int(b)) // 3
                        intensity_idx = np.searchsorted(intensity_bin, intensity, side="right") - 1
                        color_count[intensity_idx] += [r, g, b]
                        count[intensity_idx] += 1

                max_intensity_idx = np.argmax(count)
                if count[max_intensity_idx] > 0:
                    new_img[y, x] = color_count[max_intensity_idx] // count[max_intensity_idx]
                else:
                    new_img[y, x] = pixels[y, x]

        return new_img

    def get_neighborhood(self, img_array, x, y, kernel_size):
        half_size = kernel_size // 2
        neighbors = []
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                if 0 <= x + i < img_array.shape[0] and 0 <= y + j < img_array.shape[1]:
                    neighbors.append(img_array[x + i, y + j])
        return neighbors

    def max_filter(self, kernel_size=3):
        img_array = np.array(self.original_img.convert('L'))
        dilated_image = np.copy(img_array)
        for x in range(img_array.shape[0]):
            for y in range(img_array.shape[1]):
                neighborhood = self.get_neighborhood(img_array, x, y, kernel_size)
                dilated_image[x, y] = max(neighborhood)
        return dilated_image

    def min_filter(self, kernel_size=3):
        img_array = np.array(self.original_img.convert('L'))
        eroded_image = np.copy(img_array)
        for x in range(img_array.shape[0]):
            for y in range(img_array.shape[1]):
                neighborhood = self.get_neighborhood(img_array, x, y, kernel_size)
                eroded_image[x, y] = min(neighborhood)
        return eroded_image

    def resize_image(self):
        if self.img is None:
            messagebox.showwarning("Error", "No se ha cargado ninguna imagen.")
            return

        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
            resized_img = self.original_img.resize((width, height), Image.LANCZOS)
            self.img = resized_img
            self.display_image(self.img)
        except ValueError:
            messagebox.showwarning("Error", "Por favor, ingresa valores válidos para el ancho y alto.")
    
    def add_text(self):
        if self.img is None:
            messagebox.showwarning("Error", "No se ha cargado ninguna imagen.")
            return
        
        text = self.text_entry.get()
        try:
            x = int(self.x_entry.get())
            y = int(self.y_entry.get())
            font_size = int(self.font_size_entry.get())
        except ValueError:
            messagebox.showwarning("Error", "Por favor, ingresa valores válidos para X, Y y tamaño del texto.")
            return

        draw = ImageDraw.Draw(self.img)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text((x, y), text, fill="white", font=font)
        self.display_image(self.img)


    # Función para mostrar la imagen original sin filtros
    def show_original(self):
        if self.original_img:
            self.display_image(self.original_img)

    # Función para mostrar la imagen en la interfaz, ajustada a un tamaño fijo
    def display_image(self, img):
        img_resized = img.resize((300, 300), Image.Resampling.LANCZOS)  # Redimensiona la imagen a 300x300 píxeles
        img_tk = ImageTk.PhotoImage(img_resized)  # Convierte la imagen en un formato compatible con Tkinter
        self.image_label.config(image=img_tk)  # Actualiza la etiqueta para mostrar la imagen
        self.image_label.image = img_tk  # Guarda una referencia a la imagen para evitar que se elimine

# Inicializa la aplicación
root = tk.Tk()
app = ImageFilterApp(root)
root.mainloop()
