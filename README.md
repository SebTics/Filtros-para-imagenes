# 📷 Aplicación de Filtros y Marca de Agua en Imágenes  

## 📂 Archivos Incluidos  

- **`filtros.py`** → Archivo de Python ejecutable.  
- **`img.jpg`** → Imagen de prueba (puedes usar cualquier imagen de tu computadora).  

## 📝 Descripción  

Este programa cuenta con una librería de filtros para aplicar a una imagen y la opción de agregar una marca de agua. Utiliza:  
- **Tkinter** → Para la interfaz gráfica.  
- **Pillow** → Para la manipulación de imágenes.  

### 📌🎨 Filtros del programa 
- **Rojo** → Elimina los canales verde y azul, dejando solo el canal rojo.  
- **Azul** → Elimina los canales rojo y verde, dejando solo el canal azul.  
- **Escala de Grises** → Convierte la imagen a escala de grises.  
- **Incrementar Contraste** → Aumenta el contraste de la imagen.  
- **Alto Contraste** → Convierte la imagen a una versión en blanco y negro con alto contraste.  
- **Mosaico** → Aplica un efecto de mosaico dividiendo la imagen en bloques y promediando sus colores.  
- **Blur** → Aplica un desenfoque suave a la imagen.  
- **Motion Blur** → Simula un desenfoque de movimiento con un desenfoque gaussiano.  
- **Encontrar Bordes** → Detecta los bordes en la imagen.  
- **Sharpen** → Realza los bordes y detalles en la imagen.  
- **Emboss** → Aplica un efecto de relieve a la imagen.  
- **Promedio** → Aplica un filtro de promediado para suavizar la imagen.  
- **Filtro Gris Recursivo**  
- **Filtro Color Real Recursivo**  
- **Aplicar Marca de Agua** → Aplica una marca de agua previamente cargada con el botón 'Abrir Marca de Agua'.  
- **Semitonos** → Crea la ilusión de tonos continuos usando patrones de puntos.  
- **Dithering Azar** → Aplica ruido aleatorio para suavizar transiciones de color.  
- **Floyd-Steinberg** → Mejora la calidad de imagen redistribuyendo el error a píxeles vecinos.  
- **Dithering Ordenado y Disperso** → Utiliza patrones fijos y dispersos para simular colores.  
- **Erosión Máxima** → Resalta las áreas claras, expandiendo los píxeles más brillantes y eliminando detalles oscuros.  
- **Erosión Mínima** → Resalta las áreas oscuras, expandiendo los píxeles más oscuros y eliminando detalles claros.  
- **Óleo** → Transforma la imagen en escala de grises para que parezca pintada con pinceladas gruesas y mezclas suaves.  
- **Imágenes con Texto** → Agrega texto a la imagen, eligiendo la posición en los ejes **X** y **Y**, además del tamaño de la letra.  

## 🔧 Requisitos  

- **Python 3.x**  
- Instalación de las siguientes bibliotecas:  
  ```bash
  pip install pillow numpy
